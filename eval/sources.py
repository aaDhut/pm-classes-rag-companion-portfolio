"""The ground truth an eval case points at: lecture transcript windows, notes pages
(with their rebuilt tables) and the stored class summaries, read straight from disk."""
import functools
import json
import re

from rag import config, video
from rag.extract import extract

WORKSHOPS = {"w27sep": "27 Sep", "w4oct": "4 Oct", "w17oct": "17 Oct"}


@functools.lru_cache(maxsize=None)
def _videos() -> dict:
    """class key (int class number, or "w27sep"/"w4oct"/"w17oct") -> (video info, folder)."""
    out = {}
    for d in (config.DATA_DIR / "videos").iterdir():
        if not (d / "info.json").exists():
            continue
        info = json.loads((d / "info.json").read_text())
        if info.get("class_no") is not None:
            out[info["class_no"]] = (info, d)
        for key, date in WORKSHOPS.items():
            if date in info["name"]:
                out[key] = (info, d)
    return out


def video_of(key) -> tuple[dict, object]:
    return _videos()[key]


def class_key_of_video(file_id: str):
    return next((k for k, (info, _) in _videos().items() if info["id"] == file_id), None)


def secs(t: str) -> int:
    parts = [int(x) for x in re.findall(r"\d+", t)][-3:]
    return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))


@functools.lru_cache(maxsize=None)
def _segments(key) -> list[dict]:
    info, d = video_of(key)
    names = video.speaker_names(info)
    return [{**s, "who": names.get(str(s["speaker"]), "Student")} for s in video.segments(d)]


def transcript(key, start: float = 0, end: float = 1e9) -> str:
    lines = [f"[{video._hms(s['start'])}] {s['who']}: {s['text']}" for s in _segments(key)
             if start <= s["start"] <= end]
    return "\n".join(lines)


def label(key) -> str:
    info, _ = video_of(key)
    return f"Class {key}" if isinstance(key, int) else info["name"].replace(".mp4", "")


def summary_json(key) -> dict:
    _, d = video_of(key)
    return json.loads((d / "summary.json").read_text())


def summary_md(key) -> str:
    _, d = video_of(key)
    return (d / "summary.md").read_text()


def section_window(key, heading: str) -> tuple[int, int]:
    """(start, end) seconds of the summary section whose heading contains `heading`."""
    secs_ = summary_json(key)["sections"]
    for i, s in enumerate(secs_):
        if heading.lower() in s["heading"].lower():
            start = secs(s["time"])
            end = secs(secs_[i + 1]["time"]) if i + 1 < len(secs_) else 1e9
            return start, int(end)
    raise KeyError(f"no section '{heading}' in {label(key)}")


@functools.lru_cache(maxsize=None)
def _manifest() -> dict:
    return json.loads(config.MANIFEST.read_text())


def notes(name: str, page: int | None = None) -> str:
    """A notes file's page (with its tables), or the whole file, by (part of) its name."""
    matches = [e for e in _manifest().values() if name.lower() in e["name"].lower()]
    if not matches:
        raise KeyError(f"no notes file named like '{name}'")
    e = min(matches, key=lambda e: len(e["name"]))
    units = extract(e)
    if page is not None:
        units = [u for u in units if u.get("page") == page]
    return "\n\n".join(u["text"] for u in units)[:60_000]


def evidence(spec: dict) -> tuple[str, str]:
    """(title, text) for one evidence item of a case."""
    if "file" in spec:
        where = spec["file"] + (f", page {spec['page']}" if spec.get("page") else "")
        return f"Notes: {where}", notes(spec["file"], spec.get("page"))
    if "summary" in spec:
        return f"Stored summary of {label(spec['summary'])}", summary_md(spec["summary"])
    key = spec["class"]
    if spec.get("full"):
        _, d = video_of(key)
        screen = video.screen_sample(d, 30_000)
        return (f"{label(key)} — full lecture transcript, then text read off the shared screen",
                transcript(key) + "\n\n[Text on the shared screen during the class]\n" + screen)
    if "section" in spec:
        start, end = section_window(key, spec["section"])
    else:
        start, end = secs(spec["from"]), secs(spec["to"])
    pad = spec.get("pad", 30)
    text = transcript(key, start - pad, end + pad)
    return f"{label(key)} — lecture transcript {video._hms(max(start - pad, 0))}–{video._hms(end + pad)}", text


def lecture_window(file_id: str, start: float, end: float, pad: float = 60) -> str:
    """Transcript around a retrieved lecture piece, so the grader sees what was really said."""
    key = class_key_of_video(file_id)
    return transcript(key, start - pad, end + pad) if key is not None else ""


def key_of_label(label_: str):
    """Class key for a class label ("Class 12 - …" -> 12, "Sunday Workshop - 27-Sept - PRD" -> "w27sep")."""
    m = re.match(r"Class (\d+) - ", label_)
    if m:
        return int(m.group(1))
    for key, date in WORKSHOPS.items():
        day, mon = date.split()
        if re.search(rf"\b{day}(?:st|nd|rd|th)?[\s-]+{mon}", label_, re.I):
            return key
    return None
