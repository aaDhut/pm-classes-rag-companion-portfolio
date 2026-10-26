"""Read answers aloud with Gemini TTS, streamed so playback starts in about a second.

Each answer is generated once: a background thread streams raw audio to whoever is
listening and, when done, saves an MP3 (replays are free) and records the exact cost.
"""
import hashlib
import hmac
import os
import queue
import re
import secrets
import threading
import time

import lameenc
from google.genai import types

from . import chats, config, llm
from .usage import price, usd_to_inr

RATE = 24000  # Gemini TTS returns 16-bit mono PCM at 24 kHz
# Shorter requests keep the voice steady; generation runs ~3x faster than playback, so
# the next part is ready before the current one finishes.
PART_CHARS = 1200
# Make the next part only when the listener has less than this much unheard audio left.
# A part (~80 s of speech) takes ~25 s to make and its first sound arrives in ~1 s,
# so 40 s is a safe lead with no gaps. Stopping early then skips the rest of the bill.
AHEAD_SECONDS = 40
# Seconds with no one listening before the remaining parts are dropped.
LISTENER_GRACE = 5

# Signs voice links so only answers this app showed can be read aloud (each read costs).
# Must be the same on every Cloud Run instance: VOICE_SECRET, else derived from APP_PASSWORD,
# else (local runs) random per process.
_SECRET = os.getenv("VOICE_SECRET") or config.APP_PASSWORD
# With no secret (local runs) the key is random, but kept in the process environment:
# Streamlit reloads this module when project files change, and server.py's route would
# otherwise keep checking links against the key from before the reload.
_SECRET = _SECRET or os.environ.setdefault("_LOCAL_LINK_SECRET", secrets.token_hex(32))
_KEY = hashlib.sha256(("voice:" + _SECRET).encode()).digest()


def sign(chat_id: str, idx: int) -> str:
    return hmac.new(_KEY, f"{chat_id}:{idx}".encode(), "sha256").hexdigest()[:32]


def verify(chat_id: str, idx: int, sig: str) -> bool:
    return hmac.compare_digest(sign(chat_id, idx), sig)


def plain(md: str) -> str:
    """Markdown answer -> text worth speaking (no symbols, links or [1] citations)."""
    t = re.sub(r"```.*?```", " ", md, flags=re.S)
    t = re.sub(r"\[(\d+(,\s*\d+)*)\]", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^\s*#+\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\|?[-:| ]+\|?\s*$", "", t, flags=re.M)  # table separator rows
    t = re.sub(r"[*_`>|~]", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# Measured on 37 past plays (Oct 2026): per character of spoken text, Gemini bills
# about 1.65 audio tokens and 0.24 text tokens. Real plays land within about ±20%.
AUDIO_PER_CHAR = 1.65
TEXT_PER_CHAR = 0.24


def estimate(md: str) -> tuple[float, float] | None:
    """(usd, inr) it would cost to read this answer aloud once, or None if unpriced."""
    p = price(config.TTS_MODEL)
    if not p:
        return None
    n = len(plain(md))
    usd = (n * TEXT_PER_CHAR * p["in"] + n * AUDIO_PER_CHAR * p["out"]) / 1e6
    return usd, usd * usd_to_inr()


def parts(text: str) -> list[str]:
    """Split at paragraph (then sentence) breaks into pieces of about PART_CHARS."""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        pieces = [para] if len(para) <= PART_CHARS else re.split(r"(?<=[.!?])\s+", para)
        for p in pieces:
            if cur and len(cur) + len(p) > PART_CHARS:
                out.append(cur)
                cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _stream_part(text: str):
    """Yields PCM bytes; returns the usage metadata."""
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        # Accent comes from language_code, not a prompt: any instruction put in the text
        # can get read out loud. The model only ever receives the answer itself.
        speech_config=types.SpeechConfig(
            language_code=config.TTS_LANGUAGE,
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.TTS_VOICE))))
    meta = None
    for chunk in llm.gen_client().models.generate_content_stream(
            model=config.TTS_MODEL, contents=text, config=cfg):
        meta = chunk.usage_metadata or meta
        cand = chunk.candidates[0] if chunk.candidates else None
        for p in (cand.content.parts if cand and cand.content and cand.content.parts else []):
            if p.inline_data and p.inline_data.data:
                yield p.inline_data.data
    return meta


class Job:
    """One answer being turned into speech. Listeners read chunks from their own queue."""

    def __init__(self, key, chat_id: str, idx: int, content: str):
        self.key, self.chat_id, self.idx, self.content = key, chat_id, idx, content
        self.chunks: list[bytes] = []
        self.listeners: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.done = False
        self.error: Exception | None = None
        self.stopped = False  # True if everyone left before the end
        self.first_at: float | None = None
        # Set when the last listener leaves. Starts "now": the page connects right after.
        self.alone_since: float | None = time.monotonic()

    def subscribe(self) -> queue.Queue:
        """A queue that gets every chunk so far, then each new one; None marks the end."""
        q = queue.Queue()
        with self.lock:
            for c in self.chunks:
                q.put(c)
            if self.done:
                q.put(None)
            else:
                self.listeners.append(q)
                self.alone_since = None
        return q

    def unsubscribe(self, q: queue.Queue):
        """Called when a listener stops or closes the page."""
        with self.lock:
            if q in self.listeners:
                self.listeners.remove(q)
            if not self.listeners:
                self.alone_since = time.monotonic()

    def _ahead(self) -> float:
        """Seconds of audio made but not yet heard (assumes playback since the first chunk)."""
        made = sum(len(c) for c in self.chunks) / 2 / RATE
        return made - (time.monotonic() - self.first_at) if self.first_at else 0.0

    def _wanted(self) -> bool:
        """Wait until the listener nears the end of what's made; False if nobody is listening.
        Each part is billed, so parts nobody will hear are never made."""
        while True:
            with self.lock:
                gone = (self.alone_since is not None
                        and time.monotonic() - self.alone_since > LISTENER_GRACE)
            if gone:
                return False
            if self._ahead() <= AHEAD_SECONDS:
                return True
            time.sleep(0.5)

    def _emit(self, c):
        with self.lock:
            if c is not None:
                self.chunks.append(c)
                self.first_at = self.first_at or time.monotonic()
            else:
                self.done = True
            for q in self.listeners:
                q.put(c)

    def run(self):
        text_tokens = audio_tokens = 0
        try:
            for part in parts(plain(self.content)):
                if not self._wanted():
                    self.stopped = True
                    break
                gen = _stream_part(part)
                while True:
                    try:
                        self._emit(next(gen))
                    except StopIteration as stop:
                        meta = stop.value
                        break
                if meta:
                    text_tokens += meta.prompt_token_count or 0
                    audio_tokens += meta.candidates_token_count or 0
        except Exception as e:  # a part failed: keep what played, don't save a cut-off file
            self.error = e
        finally:
            self._emit(None)
            with _jobs_lock:
                _jobs.pop(self.key, None)
        if text_tokens or audio_tokens:
            p = price(config.TTS_MODEL)
            usd = (text_tokens * p["in"] + audio_tokens * p["out"]) / 1e6 if p else None
            rate = usd_to_inr()
            chats.set_voice(self.chat_id, self.idx, self.content, {
                "model": config.TTS_MODEL, "text_tokens": text_tokens,
                "audio_tokens": audio_tokens,
                "cost": round(usd, 6) if usd is not None else None,
                "inr_rate": round(rate, 4),
                "cost_inr": round(usd * rate, 4) if usd is not None else None,
                "complete": self.error is None and not self.stopped})
        if self.error is None and not self.stopped and self.chunks:
            enc = lameenc.Encoder()
            enc.set_bit_rate(48)
            enc.set_in_sample_rate(RATE)
            enc.set_channels(1)
            enc.set_quality(5)
            pcm = b"".join(self.chunks)
            chats.write_audio(self.chat_id, self.idx, self.content, enc.encode(pcm) + enc.flush())


_jobs: dict[tuple, Job] = {}
_jobs_lock = threading.Lock()


def audio_for(chat_id: str, idx: int) -> tuple[str, object] | None:
    """("mp3", bytes) if saved, ("pcm", (job, queue)) while generating, None if no such answer.
    The caller reads the queue until None and must call job.unsubscribe(queue) when done."""
    msgs = chats.load(chat_id)
    if idx >= len(msgs) or msgs[idx]["role"] != "assistant":
        return None
    content = msgs[idx]["content"]
    if (mp3 := chats.read_audio(chat_id, idx, content)) is not None:
        return "mp3", mp3
    key = (chat_id, idx, hashlib.sha1(content.encode()).hexdigest())
    with _jobs_lock:
        job = _jobs.get(key)
        if job is None:
            job = _jobs[key] = Job(key, chat_id, idx, content)
            threading.Thread(target=job.run, daemon=True).start()
    return "pcm", (job, job.subscribe())
