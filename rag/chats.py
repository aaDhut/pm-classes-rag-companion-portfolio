"""Saved conversations: one JSON file per chat, in data/chats/ locally or in a GCS bucket
when CHATS_BUCKET is set (Cloud Run's disk is wiped on every restart)."""
import copy
import hashlib
import json
import threading
import uuid
from datetime import datetime
from functools import lru_cache

from . import config

CHATS_DIR = config.DATA_DIR / "chats"
AUDIO_DIR = config.DATA_DIR / "audio"


@lru_cache
def _bucket():
    from google.cloud import storage
    return storage.Client(project=config.GCP_PROJECT).bucket(config.CHATS_BUCKET)


# Bucket chats already downloaded, by blob name: (generation, parsed chat). Listing the
# bucket is one fast call that returns each blob's generation, so only chats that changed
# since the last run are downloaded again.
_cache: dict[str, tuple[int, dict]] = {}


def _read(chat_id: str) -> dict | None:
    if config.CHATS_BUCKET:
        name = f"chats/{chat_id}.json"
        blob = _bucket().get_blob(name)
        if blob is None:
            _cache.pop(name, None)
            return None
        hit = _cache.get(name)
        if hit and hit[0] == blob.generation:
            return hit[1]
        data = json.loads(blob.download_as_text())
        _cache[name] = (blob.generation, data)
        return data
    path = CHATS_DIR / f"{chat_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _write(chat_id: str, data: dict):
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if config.CHATS_BUCKET:
        blob = _bucket().blob(f"chats/{chat_id}.json")
        blob.upload_from_string(text, content_type="application/json")
        _cache[blob.name] = (blob.generation, data)
        return
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    (CHATS_DIR / f"{chat_id}.json").write_text(text)


def _all() -> list[dict]:
    if config.CHATS_BUCKET:
        out, seen = [], set()
        for b in _bucket().list_blobs(prefix="chats/"):
            seen.add(b.name)
            hit = _cache.get(b.name)
            if not hit or hit[0] != b.generation:
                try:
                    hit = (b.generation, json.loads(b.download_as_text()))
                except json.JSONDecodeError:
                    continue
                _cache[b.name] = hit
            out.append(hit[1])
        for name in set(_cache) - seen:  # deleted from another instance
            _cache.pop(name, None)
        return out
    out = []
    for p in CHATS_DIR.glob("*.json") if CHATS_DIR.exists() else []:
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def new_id() -> str:
    return uuid.uuid4().hex[:12]


_lock = threading.Lock()  # chat files are also updated from the voice route's threads


def save(chat_id: str, messages: list[dict]):
    if not messages:
        return
    with _lock:
        old = _read(chat_id)
        now = datetime.now().isoformat(timespec="seconds")
        created = old["created"] if old else now
        # The voice route records an answer's audio cost in the file; keep it if this
        # session's copy of the messages doesn't have it yet.
        for m, o in zip(messages, old["messages"] if old else []):
            if "tts" in o and "tts" not in m and o["content"] == m["content"]:
                m["tts"] = o["tts"]
        first = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
        title = first if len(first) <= 60 else first[:57].rsplit(" ", 1)[0] + "…"
        _write(chat_id, {"id": chat_id, "title": title, "created": created,
                         "updated": now, **totals(messages), "messages": messages})


def set_voice(chat_id: str, idx: int, content: str, tts: dict):
    """Record the cost of reading message idx aloud. Plays add up: one stopped early and a
    later full play are both billed."""
    with _lock:
        data = copy.deepcopy(_read(chat_id))
        if not data or idx >= len(data["messages"]) or data["messages"][idx]["content"] != content:
            return
        old = data["messages"][idx].get("tts")
        if old:
            tts = dict(tts, plays=old.get("plays", 1) + 1,
                       text_tokens=old["text_tokens"] + tts["text_tokens"],
                       audio_tokens=old["audio_tokens"] + tts["audio_tokens"])
            for k in ("cost", "cost_inr"):
                if old.get(k) is not None and tts.get(k) is not None:
                    tts[k] = round(old[k] + tts[k], 6)
        data["messages"][idx]["tts"] = tts
        data.update(totals(data["messages"]))
        _write(chat_id, data)


def totals(messages: list[dict]) -> dict:
    """Questions asked, tokens, dollars and rupees for a chat (answers + voice).
    cost is None when nothing has usage saved (chats from before costs were tracked)."""
    from .usage import inr
    used = [m["usage"] for m in messages if m.get("usage")]
    voice = [m["tts"] for m in messages if m.get("tts")]
    costs = [(u["cost"], inr(u)) for u in used + voice if u.get("cost") is not None]
    return {"questions": sum(m["role"] == "user" for m in messages),
            "tokens": sum(u["input"] + u["output"] + u.get("embed", 0) for u in used)
                      + sum(t["text_tokens"] + t["audio_tokens"] for t in voice),
            "cost": round(sum(c for c, _ in costs), 6) if costs else None,
            "cost_inr": round(sum(r for _, r in costs), 4) if costs else None}


def load(chat_id: str) -> list[dict]:
    data = _read(chat_id)
    # Callers change the list they get, so never hand out the cached copy itself.
    return json.loads(json.dumps(data["messages"])) if data else []


def cached(chat_id: str) -> list[dict]:
    """Like load(), but with no bucket call: the copy read by the last list_chats(),
    which runs at the end of every page run."""
    if not config.CHATS_BUCKET:
        return load(chat_id)
    hit = _cache.get(f"chats/{chat_id}.json")
    return json.loads(json.dumps(hit[1]["messages"])) if hit else []


def list_chats() -> list[dict]:
    """Newest first: [{id, title, updated, questions, tokens, cost, cost_inr}]."""
    chats = []
    for d in _all():
        try:
            chats.append({"id": d["id"], "title": d["title"], "updated": d["updated"],
                          **totals(d.get("messages", []))})
        except KeyError:
            continue
    return sorted(chats, key=lambda c: c["updated"], reverse=True)


# Saved voice recordings, so replaying an answer is free.
def _audio_name(chat_id: str, idx: int, content: str) -> str:
    # VOICE_VERSION in the hash: changing how audio is made ignores older saved files.
    h = hashlib.sha1(f"{config.VOICE_VERSION}:{content}".encode()).hexdigest()[:10]
    return f"{chat_id}-{idx}-{h}.mp3"


def read_audio(chat_id: str, idx: int, content: str) -> bytes | None:
    name = _audio_name(chat_id, idx, content)
    if config.CHATS_BUCKET:
        blob = _bucket().blob(f"audio/{name}")
        return blob.download_as_bytes() if blob.exists() else None
    path = AUDIO_DIR / name
    return path.read_bytes() if path.exists() else None


def write_audio(chat_id: str, idx: int, content: str, mp3: bytes):
    name = _audio_name(chat_id, idx, content)
    if config.CHATS_BUCKET:
        _bucket().blob(f"audio/{name}").upload_from_string(bytes(mp3), content_type="audio/mpeg")
        return
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / name).write_bytes(bytes(mp3))


def delete(chat_id: str):
    if config.CHATS_BUCKET:
        from google.api_core.exceptions import NotFound
        for blob in _bucket().list_blobs(prefix=f"audio/{chat_id}-"):
            blob.delete()
        _cache.pop(f"chats/{chat_id}.json", None)
        try:
            _bucket().blob(f"chats/{chat_id}.json").delete()
        except NotFound:
            pass
        return
    for path in AUDIO_DIR.glob(f"{chat_id}-*.mp3"):
        path.unlink(missing_ok=True)
    (CHATS_DIR / f"{chat_id}.json").unlink(missing_ok=True)
