"""Class video lectures: Drive video -> audio + screen frames -> Deepgram transcript + OCR.

Everything a video produces is kept in data/videos/<file id>/ so each video is
downloaded, transcribed and OCR'd only once:
    audio.ogg        the lecture audio (small; the .mp4 itself is deleted)
    deepgram.json    raw Deepgram response (words, speakers, timestamps)
    frames/          one JPEG each time the shared screen changes (t<seconds>.jpg)
    ocr.json         {frame name: text read off it}
"""
import json
import re
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image, ImageChops, ImageStat

from . import config, drive_sync

SESSION_RE = re.compile(r"session\s*(\d+)", re.I)
DATE_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-]+([A-Za-z]{3})")


def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


# ---------- which class a video belongs to ----------

def _class_titles() -> dict:
    """class_no -> title and workshop (day, mon) -> title, from the notes already indexed."""
    manifest = json.loads(config.MANIFEST.read_text()) if config.MANIFEST.exists() else {}
    from .classes import class_no
    by_no, workshops = {}, {}
    for e in manifest.values():
        if e["class_no"] is not None:
            by_no[class_no(e["class_no"])] = e["class_title"]
        elif m := DATE_RE.search(e["class_title"]):
            workshops[(int(m.group(1)), m.group(2).lower())] = e["class_title"]
    return {"by_no": by_no, "workshops": workshops}


def match_class(name: str, titles: dict) -> tuple[int | None, str]:
    """'Academic Session 9.mp4' -> (9, 'PRD, concept Note'); workshop -> (None, its folder title)."""
    if m := SESSION_RE.search(name):
        no = int(m.group(1))
        return no, titles["by_no"].get(no, "")  # "" = no notes; title comes from the lecture
    if m := DATE_RE.search(name):
        key = (int(m.group(1)), m.group(2).lower())
        if key in titles["workshops"]:
            return None, titles["workshops"][key]
    return None, Path(name).stem


def list_videos(svc=None) -> list[dict]:
    svc = svc or drive_service()
    titles = _class_titles()
    out = []
    for f in drive_sync.list_children(svc, config.VIDEO_FOLDER_ID):
        if not f["mimeType"].startswith("video/"):
            continue
        no, title = match_class(f["name"], titles)
        out.append({"id": f["id"], "name": f["name"], "modifiedTime": f["modifiedTime"],
                    "size": int(f.get("size") or 0), "link": f.get("webViewLink", ""),
                    "class_no": no, "class_title": title})
    out.sort(key=lambda v: (v["class_no"] is None, v["class_no"] or 0, v["name"]))
    return out


def drive_service():
    return build("drive", "v3", credentials=drive_sync.get_credentials(), cache_discovery=False)


def video_dir(v: dict) -> Path:
    return config.VIDEO_DIR / v["id"]


# ---------- step 1: download, then keep only audio + screen frames ----------

def download(svc, v: dict) -> Path:
    config.VIDEO_TMP.mkdir(parents=True, exist_ok=True)
    dest = config.VIDEO_TMP / f"{v['id']}.mp4"
    if dest.exists() and dest.stat().st_size == v["size"]:
        return dest
    req = svc.files().get_media(fileId=v["id"], supportsAllDrives=True)
    with open(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=100 * 1024 * 1024)
        done = False
        while not done:
            status, done = dl.next_chunk()
            pct = int(status.progress() * 100) if status else 0
            if status and pct // 20 != getattr(dl, "_last", -1):
                dl._last = pct // 20
                print(f"    {v['name']}: downloaded {pct}%", flush=True)
    return dest


FRAME_EVERY = 5      # seconds between sampled frames
FRAME_WIDTH = 1280
CHANGE_MIN = 4.0     # mean pixel difference (0-255) that counts as a new screen


def split_media(mp4: Path, out: Path) -> None:
    """One ffmpeg pass: mono 16 kHz Opus audio, plus a frame every FRAME_EVERY seconds."""
    raw = out / "frames_raw"
    raw.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        ffmpeg(), "-v", "error", "-y", "-i", str(mp4),
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", str(out / "audio.ogg"),
        "-map", "0:v:0", "-vf", f"fps=1/{FRAME_EVERY},scale={FRAME_WIDTH}:-2", "-q:v", "4",
        str(raw / "%06d.jpg"),
    ], check=True)


def keep_changed_frames(out: Path) -> list[str]:
    """Keep a frame only when the screen differs from the last kept one. Names carry the time."""
    raw, frames = out / "frames_raw", out / "frames"
    frames.mkdir(exist_ok=True)
    kept, last = [], None
    for i, p in enumerate(sorted(raw.glob("*.jpg"))):
        small = Image.open(p).convert("L").resize((96, 54))
        if last is None or ImageStat.Stat(ImageChops.difference(small, last)).mean[0] >= CHANGE_MIN:
            t = i * FRAME_EVERY
            name = f"t{t:05d}.jpg"
            p.rename(frames / name)
            kept.append(name)
            last = small
        else:
            p.unlink()
    raw.rmdir()
    return kept


# ---------- step 2: transcript ----------

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def transcribe(out: Path) -> dict:
    dest = out / "deepgram.json"
    if dest.exists():
        return json.loads(dest.read_text())
    params = {"model": config.DEEPGRAM_MODEL, "language": "multi", "diarize": "true",
              "smart_format": "true", "punctuate": "true", "paragraphs": "true", "utterances": "true"}
    with open(out / "audio.ogg", "rb") as fh:
        r = requests.post(DEEPGRAM_URL, params=params, data=fh, timeout=1800,
                          headers={"Authorization": f"Token {config.DEEPGRAM_API_KEY}",
                                   "Content-Type": "audio/ogg"})
    r.raise_for_status()
    dest.write_text(r.text)
    return r.json()


# ---------- step 3: OCR of the shared screen ----------

def ocr_frames(out: Path) -> dict:
    from .ocr import ocr_images
    dest = out / "ocr.json"
    done = json.loads(dest.read_text()) if dest.exists() else {}
    todo = [p for p in sorted((out / "frames").glob("*.jpg")) if p.name not in done]
    if todo:
        texts = ocr_images([p.read_bytes() for p in todo])
        done.update({p.name: t for p, t in zip(todo, texts)})
        dest.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    return done


def process(svc, v: dict, keep_mp4: bool = False) -> Path:
    """Download -> audio + frames -> transcript -> OCR. Safe to re-run; finished steps are skipped."""
    out = video_dir(v)
    out.mkdir(parents=True, exist_ok=True)
    (out / "info.json").write_text(json.dumps(v, indent=1))
    if not (out / "audio.ogg").exists() or not (out / "frames").exists():
        mp4 = download(svc, v)
        split_media(mp4, out)
        n = len(keep_changed_frames(out))
        print(f"    {v['name']}: audio + {n} screen frames", flush=True)
        if not keep_mp4:
            mp4.unlink()
    transcribe(out)
    ocr_frames(out)
    print(f"    {v['name']}: transcribed and OCR'd", flush=True)
    return out


# ---------- step 4: who is speaking ----------

def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# Names the transcription gets wrong, checked against the slides and notes:
# (pattern, correct spelling, only in this class number or None for all classes).
TERM_FIXES = [
    (r"\bnama? ?yatri\b", "Namma Yatri", None),
    (r"\bjucero\b", "Juicero", None),
    (r"\bmeta ?raven\b", "Meta Ray-Ban", None),
    (r"\bstazilla\b", "Stayzilla", None),
    (r"\bhickey act\b", "Yik Yak", None),
    (r"\bsnap ?it\b", "Snabbit", None),
    (r"\bdifferent track\b", "Tiffin Track", 14),
]


def _fix_terms(text: str, class_no) -> str:
    for pat, right, only in TERM_FIXES:
        if only is None or only == class_no:
            text = re.sub(pat, right, text, flags=re.I)
    return text


def segments(out: Path) -> list[dict]:
    """Deepgram paragraphs as [{start, end, speaker (Deepgram number), text}], with
    misheard names corrected (TERM_FIXES)."""
    info = json.loads((out / "info.json").read_text()) if (out / "info.json").exists() else {}
    dg = json.loads((out / "deepgram.json").read_text())
    alt = dg["results"]["channels"][0]["alternatives"][0]
    segs = []
    for p in alt.get("paragraphs", {}).get("paragraphs", []):
        text = _fix_terms(" ".join(s["text"] for s in p["sentences"]).strip(), info.get("class_no"))
        if text:
            segs.append({"start": p["start"], "end": p["end"], "speaker": p.get("speaker", 0), "text": text})
    return segs


SPEAKER_PROMPT = """Below is the transcript of a recorded online product management class
(an Indian cohort; teachers are addressed as "sir"). Deepgram split the voices into numbered
speakers, but the numbering can be imperfect: one person may appear under two numbers.

Known teachers of this course: {teachers}. A class is usually taught by one of them.
Text read off the shared screen (meeting name tags, slide titles, presenter names) is
included because names often appear there.

For EVERY speaker number, decide who it is:
- "name": the teacher's name exactly as in the known list (e.g. "Instructor A"), or a
  student's first name ONLY if it is clearly theirs (they are called by it right before
  speaking, or introduce themselves), otherwise "Student". Use "Unknown" for noise or
  very short fragments you can't place.
- "role": "teacher", "student" or "other" (a guest speaker or host; then give their name
  if known).
- "evidence": one short line saying why.
Also say who taught this class overall ("teacher").

Screen text samples:
{screen}

Talk time per speaker (minutes): {talk}

Transcript:
{transcript}"""

_SPEAKER_SCHEMA = {"type": "OBJECT", "properties": {
    "teacher": {"type": "STRING"},
    "speakers": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "speaker": {"type": "INTEGER"}, "name": {"type": "STRING"}, "role": {"type": "STRING"},
        "evidence": {"type": "STRING"}}, "required": ["speaker", "name", "role", "evidence"]}}},
    "required": ["teacher", "speakers"]}

TEACHERS = ["Instructor A", "Instructor B"]
OVERRIDES = config.DATA_DIR / "video_speakers.json"  # hand corrections: {file id: {"0": "Instructor A", ...}}


def screen_sample(out: Path, limit: int = 6000) -> str:
    ocr = json.loads((out / "ocr.json").read_text()) if (out / "ocr.json").exists() else {}
    seen, lines = set(), []
    for name, text in sorted(ocr.items()):
        for line in text.splitlines():
            line = line.strip()
            if 3 <= len(line) <= 80 and line.lower() not in seen:
                seen.add(line.lower())
                lines.append(line)
    return "\n".join(lines)[:limit]


def identify_speakers(out: Path, usage=None) -> dict:
    """{"teacher": name, "speakers": {"<n>": {"name", "role", "evidence"}}}; cached in speakers.json."""
    from google.genai import types
    from . import llm
    dest = out / "speakers.json"
    if dest.exists():
        return json.loads(dest.read_text())
    segs = segments(out)
    talk = {}
    for s in segs:
        talk[s["speaker"]] = talk.get(s["speaker"], 0) + (s["end"] - s["start"]) / 60
    transcript = "\n".join(f"[{fmt_time(s['start'])}] S{s['speaker']}: {s['text']}" for s in segs)
    prompt = SPEAKER_PROMPT.format(teachers=", ".join(TEACHERS), screen=screen_sample(out) or "(none)",
                                   talk=", ".join(f"S{k}: {v:.1f}" for k, v in sorted(talk.items())),
                                   transcript=transcript[:400_000])
    resp = llm._retry(llm.gen_client().models.generate_content)(
        model=config.ANSWER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=4096,
                                           response_mime_type="application/json",
                                           response_schema=_SPEAKER_SCHEMA,
                                           thinking_config=types.ThinkingConfig(thinking_level="low")))
    if usage is not None:
        usage.add_generation(config.ANSWER_MODEL, resp.usage_metadata)
    data = json.loads(resp.text)
    result = {"teacher": data["teacher"],
              "speakers": {str(s["speaker"]): {k: s[k] for k in ("name", "role", "evidence")}
                           for s in data["speakers"]},
              "talk_minutes": {str(k): round(v, 1) for k, v in talk.items()}}
    dest.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def speaker_names(v: dict) -> dict:
    """Deepgram speaker number (str) -> display name, with hand corrections applied."""
    found = identify_speakers(video_dir(v))
    names = {}
    for k, s in found["speakers"].items():
        name, role = s["name"], s["role"]
        if name in ("Unknown", "Student") or role == "teacher":
            names[k] = name
        elif role == "student":
            names[k] = f"{name} (student)"
        else:
            names[k] = f"{name} (host/TA)"
    if OVERRIDES.exists():
        names.update(json.loads(OVERRIDES.read_text()).get(v["id"], {}))
    return names


# ---------- step 5: searchable pieces ----------

CHUNK_CHARS = 1400        # spoken text per piece (same size as the notes' chunks)
SCREEN_CHARS = 700        # screen text added to a piece
SLIDE_MIN_CHARS = 60      # a frame with less text than this is a camera view, not a slide


def lecture_label(v: dict) -> str:
    if v["class_no"] is None:
        return v["class_title"]
    return f"Class {v['class_no']} - {v['class_title']}"


def time_link(v: dict, sec: float) -> str:
    return f"https://drive.google.com/file/d/{v['id']}/view?t={int(sec)}"


def _screen_between(ocr: dict, start: float, end: float) -> tuple[str, list[str]]:
    """Text on screen during [start, end) and the frames that are slides (not camera views)."""
    texts, frames = [], []
    for name in sorted(ocr):
        t = int(name[1:6])
        if start - 30 <= t < end and len(ocr[name]) >= SLIDE_MIN_CHARS:
            frames.append(name)
            texts.append(ocr[name])
    seen, lines = set(), []
    for text in texts:
        for line in text.splitlines():
            key = line.strip().lower()
            if len(key) > 2 and key not in seen:
                seen.add(key)
                lines.append(line.strip())
    return "\n".join(lines)[:SCREEN_CHARS], frames


def make_chunks(v: dict, title: str | None = None) -> list[dict]:
    """Transcript -> pieces of ~CHUNK_CHARS spoken text, each with time, speakers and screen text."""
    out = video_dir(v)
    names = speaker_names(v)
    teacher = identify_speakers(out)["teacher"]
    ocr = json.loads((out / "ocr.json").read_text()) if (out / "ocr.json").exists() else {}
    v = {**v, "class_title": title or v["class_title"]}
    label = lecture_label(v)

    groups, cur = [], []
    for s in segments(out):
        line = f"[{fmt_time(s['start'])}] {names.get(str(s['speaker']), 'Student')}: {s['text']}"
        if cur and sum(len(x[1]) for x in cur) + len(line) > CHUNK_CHARS:
            groups.append(cur)
            cur = cur[-1:] if len(cur[-1][1]) < 400 else []   # carry the last line for context
        cur.append((s, line))
    if cur:
        groups.append(cur)

    chunks = []
    for i, g in enumerate(groups):
        start, end = g[0][0]["start"], g[-1][0]["end"]
        speech = "\n".join(line for _, line in g)
        screen, frames = _screen_between(ocr, start, end)
        text = speech + (f"\n\n[On screen]\n{screen}" if screen else "")
        who = sorted({names.get(str(s["speaker"]), "Student") for s, _ in g})
        header = f"{label} | Lecture video ({teacher}) | {fmt_time(start)}–{fmt_time(end)}"
        chunks.append({
            "key": f"{v['id']}:lecture:{i}",
            "embed_text": f"{header}\n\n{text}",
            "frames": frames,
            "payload": {
                "text": text, "class_no": v["class_no"], "class_title": v["class_title"],
                "class_label": label, "file_id": v["id"], "file_name": f"Lecture video ({teacher})",
                "page": None, "via": "deepgram", "link": time_link(v, start), "teacher": teacher,
                "modified": v["modifiedTime"], "source": "lecture", "start": int(start), "end": int(end),
                "time": f"{fmt_time(start)}–{fmt_time(end)}", "speakers": who,
                "images": [], "has_images": False,
            },
        })
    return chunks


# ---------- step 6: class summary, topic title, related notes ----------

SUMMARY_PROMPT = """You are writing study notes from the full transcript of one recorded class
of a product management course ({label}). Speakers are named; lines start with [time].
Text read off the shared screen (slides, tools shown) is given too. Use only what happened
in this class. Times must come from the transcript.
{notes_hint}
Return:
- "title": the class topic in 2-6 words (e.g. "Agile Ceremonies and Jira").
- "teacher": who taught it.
- "overview": 2-3 sentences on what the class was about.
- "sections": the class in order, 4-10 parts, each with "time" (start, as in the transcript),
  "heading" and 2-5 "points" (concrete: definitions, steps, frameworks, examples, tools).
- "frameworks": frameworks / models / templates taught, each with a one-line explanation.
- "examples": case studies, companies or products discussed, one line each.
- "student_questions": notable questions, each with "time", "who", "question", "answer"
  (the gist of the teacher's reply).
- "homework": assignments, readings or tasks given (empty if none).
- "takeaways": 3-6 key takeaways.

Screen text:
{screen}

Transcript:
{transcript}"""

_LIST = {"type": "ARRAY", "items": {"type": "STRING"}}
_SUMMARY_SCHEMA = {"type": "OBJECT", "properties": {
    "title": {"type": "STRING"}, "teacher": {"type": "STRING"}, "overview": {"type": "STRING"},
    "sections": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "time": {"type": "STRING"}, "heading": {"type": "STRING"}, "points": _LIST},
        "required": ["time", "heading", "points"]}},
    "frameworks": _LIST, "examples": _LIST,
    "student_questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "time": {"type": "STRING"}, "who": {"type": "STRING"}, "question": {"type": "STRING"},
        "answer": {"type": "STRING"}}, "required": ["time", "who", "question", "answer"]}},
    "homework": _LIST, "takeaways": _LIST},
    "required": ["title", "teacher", "overview", "sections", "frameworks", "examples",
                 "student_questions", "homework", "takeaways"]}


def _hms(sec: float) -> str:
    """Always with hours ("0:10:51"), so the model can't drop the hour when copying a time."""
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def named_transcript(v: dict) -> str:
    names = speaker_names(v)
    return "\n".join(f"[{_hms(s['start'])}] {names.get(str(s['speaker']), 'Student')}: {s['text']}"
                     for s in segments(video_dir(v)))


def _secs(t: str) -> int | None:
    parts = [int(x) for x in re.findall(r"\d+", t or "")][-3:]
    return sum(x * 60 ** i for i, x in enumerate(reversed(parts))) if parts else None


_STOPWORDS = set("""this that with from have they their there what when which will would about into
your some more than then them were been also just like only very much such each other these those
using used make made does done over under because where while class student students teacher sir""".split())


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _readings(t: str, given: int) -> list[int]:
    """Other ways to read a mangled time: the hour dropped ("10:51" = 1:10:51), or the hour
    merged into the minutes with a digit lost ("13:47" = 1:3x:47)."""
    out = [given + h * 3600 for h in range(1, 3)]
    m = re.fullmatch(r"\D*(\d)(\d):(\d\d)\D*", t or "")
    if m:
        h, tens, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out += [h * 3600 + (tens * 10 + x) * 60 + sec for x in range(10)]
    return out


def fix_times(v: dict, summary: dict) -> tuple[dict, list[str]]:
    """Repair summary times the model got wrong, usually by dropping the hour ("10:51" for
    1:10:51). Sections are in class order, so a section time earlier than the one before it
    (or past the end of the video) is impossible; it is moved by whole hours to the reading
    that fits the order and best matches the transcript there. Student questions are only
    moved when out of order AND the transcript an hour later clearly matches them better.
    Times that are possible are left alone. Returns (summary, changes)."""
    segs = segments(video_dir(v))
    if not segs:
        return summary, []
    end = segs[-1]["end"]

    def score(sec: float, words: set[str], before: float, after: float) -> float:
        near = " ".join(s["text"] for s in segs if sec - before <= s["start"] <= sec + after)
        return len(words & _keywords(near)) / max(len(words), 1)

    def snap(sec: float, words: set[str], lo: float, hi: float) -> int:
        """A repaired time moved onto the start of the best-matching transcript line near it."""
        near = [x for x in segs if max(lo, sec - 60) <= x["start"] <= min(hi, sec + 120)]
        if not near:
            return int(sec)
        best = max(near, key=lambda x: (len(words & _keywords(x["text"])), -abs(x["start"] - sec)))
        return int(best["start"]) if words & _keywords(best["text"]) else int(sec)

    changes = []
    prev = 0.0
    sections = summary.get("sections", [])
    for s in sections:
        given = _secs(s["time"])
        if given is None:
            continue
        if prev <= given <= end:
            prev = given
            continue
        words = _keywords(s["heading"] + " " + " ".join(s["points"]))
        later = [_secs(x["time"]) for x in sections[sections.index(s) + 1:]]
        nxt = next((t for t in later if t is not None and t > prev), end)
        cands = [c for c in _readings(s["time"], given) if prev <= c <= min(nxt, end)]
        if cands:
            sec = max(cands, key=lambda c: score(c, words, 20, 240))
            sec = snap(sec, words, prev, min(nxt, end))
            changes.append(f"section '{s['heading']}': {s['time']} -> {_hms(sec)}")
            s["time"] = _hms(sec)
            prev = sec
    prev = 0.0
    for q in summary.get("student_questions", []):
        given = _secs(q["time"])
        if given is None:
            continue
        if given > end or given < prev:
            words = _keywords(q["question"] + " " + q.get("who", ""))
            cands = [c for c in _readings(q["time"], given) if prev <= c <= end]
            if cands:
                mine = score(given, words, 60, 120) if given <= end else 0.0
                other, sec = max((score(c, words, 60, 120), c) for c in cands)
                if given > end or (other >= 0.25 and other >= 1.5 * mine):
                    sec = snap(sec, words, prev, end)
                    changes.append(f"question '{q['question'][:50]}': {q['time']} -> {_hms(sec)}")
                    q["time"] = _hms(sec)
                    given = sec
        prev = max(prev, given)
    return summary, changes


def summarize(v: dict, usage=None) -> dict:
    """Study-notes summary of one class video; cached in summary.json."""
    from google.genai import types
    from . import llm
    out = video_dir(v)
    dest = out / "summary.json"
    if dest.exists():
        return json.loads(dest.read_text())
    notes_hint = ("This class has its own notes in the course folder; the summary is of the lecture."
                  if v["class_title"] else
                  "This class has no notes in the course folder, so the title you give will name it.")
    prompt = SUMMARY_PROMPT.format(label=lecture_label(v) if v["class_title"] else f"Class {v['class_no']}",
                                   notes_hint=notes_hint, screen=screen_sample(out, 8000) or "(none)",
                                   transcript=named_transcript(v)[:400_000])
    resp = llm._retry(llm.gen_client().models.generate_content)(
        model=config.ANSWER_MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=config.TEMPERATURE, seed=42, max_output_tokens=16384,
                                           response_mime_type="application/json",
                                           response_schema=_SUMMARY_SCHEMA,
                                           thinking_config=types.ThinkingConfig(thinking_level="low")))
    if usage is not None:
        usage.add_generation(config.ANSWER_MODEL, resp.usage_metadata)
    data, _ = fix_times(v, json.loads(resp.text))
    dest.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return data


def related_notes(v: dict, summary: dict, per_part: int = 4, limit: int = 6) -> dict:
    """Notes (PDFs, Docs, Sheets) closest to what was taught: this class's own files, and
    the closest files from other classes. Cached in related.json."""
    from qdrant_client import models
    from . import llm, store
    dest = video_dir(v) / "related.json"
    if dest.exists():
        return json.loads(dest.read_text())
    not_lecture = [models.FieldCondition(key="source", match=models.MatchAny(any=["lecture", "summary"]))]
    files = {}
    for sec in summary["sections"]:
        q = f"{sec['heading']}\n" + "\n".join(sec["points"])
        res = store.qdrant().query_points(
            config.COLLECTION, query=llm.embed_query(q), using=store.DENSE, limit=per_part * 3,
            query_filter=models.Filter(must_not=not_lecture), with_payload=True)
        seen_here = set()
        for p in res.points:
            pl = p.payload
            if pl["file_id"] in seen_here or len(seen_here) >= per_part:
                continue
            seen_here.add(pl["file_id"])
            f = files.setdefault(pl["file_id"], {
                "class_label": pl["class_label"], "class_no": pl.get("class_no"), "file_name": pl["file_name"],
                "link": pl["link"], "pages": set(), "score": 0.0, "topics": []})
            if pl.get("page"):
                f["pages"].add(pl["page"])
            f["score"] += p.score
            f["topics"].append(sec["heading"])
    ranked = sorted(files.values(), key=lambda f: -f["score"])
    for f in ranked:
        f["pages"] = sorted(f["pages"])[:6]
        f["topics"] = list(dict.fromkeys(f["topics"]))[:3]
        f["score"] = round(f["score"], 3)
    same = [f for f in ranked if f["class_label"] == lecture_label(v)]
    other = [f for f in ranked if f["class_label"] != lecture_label(v)][:limit]
    data = {"own": same, "other": other}
    dest.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return data


def summary_markdown(v: dict, summary: dict, related: dict | None = None) -> str:
    """The summary as readable text, with times linking into the video."""
    def t_link(t: str) -> str:
        parts = [int(x) for x in re.findall(r"\d+", t)][-3:]
        sec = sum(x * 60 ** i for i, x in enumerate(reversed(parts)))
        return f"[{t}]({time_link(v, sec)})"

    title = v["class_title"] or summary["title"]
    head = f"Class {v['class_no']} - {title}" if v["class_no"] is not None else title
    md = [f"### {head}", f"**Taught by:** {summary['teacher']}  ·  [Watch the lecture]({v['link']})", "",
          summary["overview"], "", "**What was covered**"]
    for s in summary["sections"]:
        md.append(f"- {t_link(s['time'])} **{s['heading']}**")
        md += [f"    - {p}" for p in s["points"]]
    for key, name in (("frameworks", "Frameworks taught"), ("examples", "Examples and case studies"),
                      ("homework", "Homework / tasks"), ("takeaways", "Key takeaways")):
        if summary.get(key):
            md += ["", f"**{name}**"] + [f"- {x}" for x in summary[key]]
    if summary.get("student_questions"):
        md += ["", "**Questions from students**"]
        md += [f"- {t_link(q['time'])} {q['who']}: *{q['question']}* — {q['answer']}"
               for q in summary["student_questions"]]
    if related:
        def line(f):
            pages = f" (pages {', '.join(map(str, f['pages']))})" if f["pages"] else ""
            return f"- [{f['file_name']}]({f['link']}){pages} — {f['class_label']}; on {', '.join(f['topics'])}"
        if related["own"]:
            md += ["", "**This class's notes**"] + [line(f) for f in related["own"]]
        if related["other"]:
            md += ["", "**Closest notes from other classes**"] + [line(f) for f in related["other"]]
    return "\n".join(md)


# ---------- step 7: into Qdrant ----------

PICS_PER_CHUNK = 2
VIDEO_VERSION = 3   # bump when lecture chunks change, so videos are re-indexed
VIDEO_STATE = config.DATA_DIR / "video_indexed.json"


def _attach_frames(v: dict, chunks: list[dict]) -> None:
    """Each piece gets up to PICS_PER_CHUNK slide frames (the most text-heavy, distinct ones)."""
    from .images import _save
    ocr = json.loads((video_dir(v) / "ocr.json").read_text())
    used = set()
    for c in chunks:
        best, texts = [], []
        for name in sorted(c.pop("frames"), key=lambda n: -len(ocr[n])):
            words = set(ocr[name].lower().split())
            if name in used or any(len(words & t) / max(len(words | t), 1) > 0.6 for t in texts):
                continue  # same slide as one already picked (scrolling, cursor moves)
            best.append(name)
            texts.append(words)
            if len(best) == PICS_PER_CHUNK:
                break
        used.update(best)
        pics = [_save(v, name[:-4], 0, (video_dir(v) / "frames" / name).read_bytes()) for name in best]
        c["payload"]["images"] = pics
        c["payload"]["has_images"] = bool(pics)


def summary_chunks(v: dict, md: str, title: str) -> list[dict]:
    from .chunk import _split
    vv = {**v, "class_title": title}
    label = lecture_label(vv)
    out = []
    for i, piece in enumerate(_split(md)):
        out.append({
            "key": f"{v['id']}:summary:{i}",
            "embed_text": f"{label} | Class summary\n\n{piece}",
            "payload": {
                "text": piece, "class_no": v["class_no"], "class_title": title, "class_label": label,
                "file_id": v["id"], "file_name": "Class summary (from the lecture video)", "page": None,
                "via": "summary", "link": v["link"], "teacher": "", "modified": v["modifiedTime"],
                "source": "summary", "part": i, "images": [], "has_images": False,
            },
        })
    out[0]["payload"]["full_text"] = md  # read back whole by store.class_summary()
    return out


def index(v: dict) -> int:
    """Lecture pieces + described slides + summary into Qdrant (replacing anything indexed for this video before)."""
    from . import store
    summary = summarize(v)
    title = v["class_title"] or summary["title"]
    teacher = identify_speakers(video_dir(v))["teacher"]
    for c in (chunks := make_chunks(v, title)):
        c["payload"]["teacher"] = teacher
    slides = slide_chunks(v, title) if slides_described(v) else []
    if not slides:  # no slide descriptions yet: lecture pieces carry the frames instead
        _attach_frames(v, chunks)
    related = related_notes({**v, "class_title": title}, summary)
    md = summary_markdown({**v, "class_title": title}, summary, related)
    (video_dir(v) / "summary.md").write_text(md)
    allc = chunks + slides + summary_chunks(v, md, title)
    for c in allc:
        c["payload"]["teacher"] = teacher
    store.delete_file(v["id"])
    store.upsert_chunks(allc)
    return len(allc)


def check_summary(v: dict, usage=None, model: str = "gemini-3.1-pro-preview") -> list[dict]:
    """Fact-check a stored class summary against the full transcript and screen text, and
    correct wrong points in summary.json (then refresh_summary(force=True) rebuilds the
    page and its index). Returns the fixes applied. Run once per summary (ingest time)."""
    from . import verify
    out = video_dir(v)
    dest = out / "summary.json"
    summary = json.loads(dest.read_text())
    title = v["class_title"] or summary["title"]
    md = summary_markdown({**v, "class_title": title}, summary)
    material = (f"[Full transcript of {lecture_label({**v, 'class_title': title})}]\n" + named_transcript(v)
                + "\n\n[Text read off the shared screen]\n" + screen_sample(out, 30_000))
    old_model = config.VERIFY_MODEL
    try:
        config.VERIFY_MODEL = model
        _, fixes = verify.check(md, material, "Study-notes summary of this whole class (check every point "
                                "and every [time]: a section's time is where it starts in the transcript)", usage)
    finally:
        config.VERIFY_MODEL = old_model
    applied = []

    def fix_in(obj):
        """Swap the wrong text inside whichever string field holds it."""
        if isinstance(obj, dict):
            for k, val in obj.items():
                if isinstance(val, str) and f["wrong"] in val:
                    obj[k] = val.replace(f["wrong"], f["right"], 1)
                    return True
                if isinstance(val, (dict, list)) and fix_in(val):
                    return True
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                if isinstance(val, str) and f["wrong"] in val:
                    obj[i] = val.replace(f["wrong"], f["right"], 1)
                    return True
                if isinstance(val, (dict, list)) and fix_in(val):
                    return True
        return False

    for f in fixes:
        for strip in ("", "**", "*"):
            f = {**f, "wrong": f["wrong"].strip(strip).strip(), "right": f["right"].strip(strip).strip()}
            if f["wrong"] and fix_in(summary):
                applied.append(f)
                break
    # Drop points emptied by a fix.
    for sec in summary.get("sections", []):
        sec["points"] = [p for p in sec["points"] if p.strip()]
    for key in ("frameworks", "examples", "homework", "takeaways"):
        summary[key] = [x for x in summary.get(key, []) if x.strip()]
    if applied:
        dest.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return applied


def refresh_summary(v: dict, force: bool = False) -> list[str]:
    """Repair the times in a cached summary and re-index only its summary points.
    force: rebuild summary.md and the summary points even when no time changed."""
    from . import store
    dest = video_dir(v) / "summary.json"
    summary, changes = fix_times(v, json.loads(dest.read_text()))
    if not changes and not force:
        return []
    dest.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    title = v["class_title"] or summary["title"]
    related = related_notes({**v, "class_title": title}, summary)
    md = summary_markdown({**v, "class_title": title}, summary, related)
    (video_dir(v) / "summary.md").write_text(md)
    teacher = identify_speakers(video_dir(v))["teacher"]
    chunks = summary_chunks(v, md, title)
    for c in chunks:
        c["payload"]["teacher"] = teacher
    store.delete_summary(v["id"])
    store.upsert_chunks(chunks)
    return changes


def stamp(v: dict) -> str:
    return f"{v['modifiedTime']}|v{VIDEO_VERSION}"


REVIEW = config.DATA_DIR / "video_review.md"


def write_review(videos: list[dict]) -> None:
    """One readable page to check: each class video, its topic, who taught it, who spoke,
    and how well the lecture matches the notes in its class folder."""
    lines = ["# Class videos: check teachers and speakers", "",
             f"Fix a wrong name by adding it to {OVERRIDES.name}, e.g.",
             '`{"<video id>": {"3": "Instructor A"}}`, then run `ingest.py --videos-only` again.', ""]
    for v in videos:
        out = video_dir(v)
        if not (out / "speakers.json").exists():
            lines += [f"## {v['name']}", "not processed yet", ""]
            continue
        sp = json.loads((out / "speakers.json").read_text())
        sm = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        rel = json.loads((out / "related.json").read_text()) if (out / "related.json").exists() else {}
        title = v["class_title"] or f"{sm.get('title', '?')} (title from the lecture; no notes)"
        head = f"Class {v['class_no']} - {title}" if v["class_no"] is not None else title
        lines += [f"## {head}", f"- Video: [{v['name']}]({v['link']}) · id `{v['id']}`",
                  f"- Taught by: **{sp['teacher']}**"]
        if sm:
            lines.append(f"- Lecture topic: {sm['title']}")
        if v["class_title"]:
            own = len(rel.get("own", []))
            lines.append(f"- Matches its notes folder: {'yes' if own else '**no — check the class number**'}"
                         f" ({own} of its files came up as related)")
        for k, s in sorted(sp["speakers"].items(), key=lambda x: -sp["talk_minutes"].get(x[0], 0)):
            lines.append(f"  - S{k} · {sp['talk_minutes'].get(k, 0)} min · {s['name']} ({s['role']}) — {s['evidence']}")
        lines.append("")
    REVIEW.write_text("\n".join(lines))


# ---------- slides: each distinct thing shown on screen, described and searchable ----------

SAME_SLIDE = 0.6     # word overlap at or above which two frames show the same slide
CHROME_SHARE = 0.3   # words on this share of a video's frames are app menus, names, tabs
LOOK_BACK = 3        # a frame is compared with this many recent slides (switching back and forth)
_WORD = re.compile(r"[a-z0-9]{2,}")


def distinct_slides(v: dict) -> list[dict]:
    """Frames showing the same slide are merged; the last one is kept, since slides often
    build up piece by piece. Words that sit on most frames (Zoom/Canva menus, participant
    names, browser tabs) are left out of the comparison, and a frame is also matched against
    the few slides before, so going back to an earlier slide doesn't count as a new one.
    [{name, ocr, start, end}] in time order of first appearance."""
    ocr = json.loads((video_dir(v) / "ocr.json").read_text())
    names = [n for n in sorted(ocr) if len(ocr[n]) >= SLIDE_MIN_CHARS]  # skip camera views
    words = {n: set(_WORD.findall(ocr[n].lower())) for n in names}
    df = Counter(w for n in names for w in words[n])
    chrome = {w for w, c in df.items() if c >= CHROME_SHARE * len(names)}
    groups = []
    for name in names:
        w, t = words[name] - chrome, int(name[1:6])
        if len(w) < 4:
            continue
        g = next((g for g in reversed(groups[-LOOK_BACK:])
                  if len(w & g["words"]) / max(len(w | g["words"]), 1) >= SAME_SLIDE), None)
        if g:
            g.update(name=name, ocr=ocr[name], words=w, end=t + FRAME_EVERY)
        else:
            groups.append({"name": name, "ocr": ocr[name], "words": w, "start": t, "end": t + FRAME_EVERY})
    return [{k: g[k] for k in ("name", "ocr", "start", "end")} for g in groups]


SLIDE_PROMPT = """This is a frame from a recorded online product management class (a screen
share in a Zoom call). The text read off it by OCR is below; it also contains app menus,
browser tabs and the meeting window, which you should ignore.

Describe what the frame TEACHES visually, for a student searching their class notes:
- a diagram / framework / flow / matrix / chart: name it, its parts, and how they connect
  (axes, quadrants, steps, arrows, what leads to what); hand-drawn sketches too;
- a table: what it compares and the main rows and columns;
- a tool screenshot (Jira, Figma, Excel, a website or app): which tool and screen, what is
  on it and what is highlighted or being done;
- a text slide: its heading and main points, in a few words.
Do not copy long text word for word. Ignore the webcam tile, toolbars, slide thumbnails
and menus.

"kind": diagram, table, chart, screenshot, text_slide, title_slide (only a heading or
section name), or other. Use other for anything with nothing to learn from: camera view,
desktop, blank or loading screen, a browser home or search page, a Drive/Docs file list,
an upload or download dialog, a sign-in page, the meeting window sharing itself.
"title": the slide or screen title in a few words.
"description": 2-6 short bullet lines ("" for title_slide and other).

OCR text:
{ocr}"""

_SLIDE_SCHEMA = {"type": "OBJECT", "properties": {
    "kind": {"type": "STRING", "enum": ["diagram", "table", "chart", "screenshot", "text_slide",
                                        "title_slide", "other"]},
    "title": {"type": "STRING"}, "description": {"type": "STRING"}},
    "required": ["kind", "title", "description"]}


def describe_slide(image: bytes, ocr: str, model: str | None = None, usage=None) -> dict:
    """Gemini's {kind, title, description} for one frame."""
    from google.genai import types
    from . import llm
    model = model or config.VISION_MODEL
    thinking = (types.ThinkingConfig(thinking_level="low") if model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=512))
    resp = llm._retry(llm.gen_client().models.generate_content)(
        model=model,
        contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"),
                  SLIDE_PROMPT.format(ocr=ocr[:3000] or "(none)")],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=2048,
                                           response_mime_type="application/json",
                                           response_schema=_SLIDE_SCHEMA, thinking_config=thinking))
    if usage is not None:
        usage.add_generation(model, resp.usage_metadata)
    return json.loads(resp.text)


SLIDE_WORKERS = 8
SLIDE_KEEP = {"diagram", "table", "chart", "screenshot", "text_slide"}


def describe_slides(v: dict, usage=None, model: str | None = None) -> list[dict]:
    """Every distinct slide of a video with Gemini's description; cached in slides.json
    (by frame name and model, so an interrupted run picks up where it stopped).
    [{name, ocr, start, end, kind, title, description}]"""
    out = video_dir(v)
    dest = out / "slides.json"
    model = model or config.VISION_MODEL
    cache = json.loads(dest.read_text()) if dest.exists() else {}
    cache = {k: d for k, d in cache.items() if d.get("model") == model}
    slides = distinct_slides(v)
    todo = [s for s in slides if s["name"] not in cache]
    lock = threading.Lock()

    def one(s):
        d = describe_slide((out / "frames" / s["name"]).read_bytes(), s["ocr"], model, usage)
        with lock:
            cache[s["name"]] = {**d, "model": model}
            if len(cache) % 20 == 0:
                dest.write_text(json.dumps(cache, indent=1, ensure_ascii=False))

    if todo:
        with ThreadPoolExecutor(SLIDE_WORKERS) as pool:
            list(pool.map(one, todo))
        dest.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return [{**s, **{k: cache[s["name"]][k] for k in ("kind", "title", "description")}} for s in slides]


def slides_described(v: dict) -> bool:
    dest = video_dir(v) / "slides.json"
    if not dest.exists():
        return False
    cache = json.loads(dest.read_text())
    return all(s["name"] in cache for s in distinct_slides(v))


SLIDE_SPEECH_CHARS = 700   # what was said while the slide was up (about 30-45 s)
SLIDE_OCR_CHARS = 700


def _clean_ocr(text: str) -> str:
    """Slide text without short menu/toolbar lines."""
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(dict.fromkeys(ln for ln in lines if len(ln) > 3))[:SLIDE_OCR_CHARS]


def slide_chunks(v: dict, title: str | None = None) -> list[dict]:
    """One searchable point per described slide (title slides and empty screens left out):
    its title, description, on-screen text and the words spoken as it came up."""
    from .images import _save
    out = video_dir(v)
    names = speaker_names(v)
    teacher = identify_speakers(out)["teacher"]
    segs = segments(out)
    v = {**v, "class_title": title or v["class_title"]}
    label = lecture_label(v)
    chunks = []
    for s in describe_slides(v):
        if s["kind"] not in SLIDE_KEEP:
            continue
        said = []
        for seg in segs:
            if seg["end"] >= s["start"] - 5 and seg["start"] <= s["end"]:
                said.append(f"{names.get(str(seg['speaker']), 'Student')}: {seg['text']}")
        speech = " ".join(said)[:SLIDE_SPEECH_CHARS]
        text = f"Slide: {s['title']} ({s['kind'].replace('_', ' ')})\n{s['description']}".strip()
        if screen := _clean_ocr(s["ocr"]):
            text += f"\n\n[On screen]\n{screen}"
        if speech:
            text += f"\n\n[Said while it was shown]\n{speech}"
        when = fmt_time(s["start"])
        pic = _save(v, s["name"][:-4], 0, (out / "frames" / s["name"]).read_bytes())
        chunks.append({
            "key": f"{v['id']}:slide:{s['name']}",
            "embed_text": f"{label} | Slide shown in the lecture ({teacher}) at {when}\n\n{text}",
            "payload": {
                "text": text, "class_no": v["class_no"], "class_title": v["class_title"],
                "class_label": label, "file_id": v["id"], "file_name": f"Slide shown at {when}",
                "page": None, "via": "slide", "link": time_link(v, s["start"]), "teacher": teacher,
                "modified": v["modifiedTime"], "source": "slide", "start": int(s["start"]),
                "end": int(s["end"]), "time": when, "kind": s["kind"], "slide_title": s["title"],
                "images": [pic], "has_images": True,
            },
        })
    return chunks
