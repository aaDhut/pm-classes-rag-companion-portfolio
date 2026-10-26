"""Pictures from the class material (slides, screenshots, diagrams) shown next to answers.

ingest.py cuts them out of each file and stores them as JPEGs in data/images/ (and in
IMAGES_BUCKET/images/ for Cloud Run). Each chunk lists its pictures in its "images"
payload field. At answer time find() searches all of them for the question, has Gemini
score each for fit, and only those that fit are shown (served by the /img route in server.py).

Only pages that extraction already flagged as visual (OCR'd or described by Gemini) get a
picture, so this adds no OCR or Gemini calls.
"""
import hashlib
import hmac
import html
import io
import json
import os
import re
import secrets
import zipfile
from functools import lru_cache

import pymupdf

from . import config, llm, store

MAX_WIDTH = 1400          # px; enough to read a slide when opened full size
MAX_DPI = 150
JPEG_QUALITY = 80
MIN_BOX_SHARE = 0.05      # same icon/logo cut-off as extraction
PAD = 8                   # points of margin around a cropped picture
MAX_PER_ANSWER = 3
NAME_RE = re.compile(r"^[\w-]{1,120}\.jpg$")


# ---------- cutting pictures out of files (ingest time) ----------

def _jpeg(pix: pymupdf.Pixmap) -> bytes:
    if pix.alpha:  # transparency -> drop it (JPEG has none)
        pix = pymupdf.Pixmap(pix, 0)
    if pix.n not in (1, 3):  # CMYK etc. -> RGB
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.width > MAX_WIDTH:
        pix = pymupdf.Pixmap(pix, MAX_WIDTH, round(pix.height * MAX_WIDTH / pix.width), None)
    return pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)


def _save(entry: dict, slot, k: int, data: bytes) -> str:
    h = hashlib.sha1(data).hexdigest()[:10]
    name = f"{entry['id'][:16]}-{slot}-{k}-{h}.jpg"
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.IMAGES_DIR / name
    if not path.exists():
        path.write_bytes(data)
    return name


def _render(page, clip=None) -> bytes:
    width = (clip or page.rect).width
    dpi = min(MAX_DPI, int(MAX_WIDTH * 72 / max(width, 1)))
    return _jpeg(page.get_pixmap(dpi=dpi, clip=clip))


def _page_crops(page) -> list:
    """Areas of a page worth showing. None = the whole page.

    Slides (landscape) and image-only pages are shown whole. On document pages
    (Google Docs exported as PDF) only the screenshots/diagrams are cut out, so the
    picture isn't a page of text."""
    r = page.rect
    if r.width >= r.height or len(page.get_text("text").strip()) < config.VISION_MIN_CHARS:
        return [None]
    area = r.width * r.height
    boxes = []
    for info in page.get_image_info():
        b = pymupdf.Rect(info["bbox"]) & r
        if not b.is_empty and b.width * b.height / area >= MIN_BOX_SHARE:
            boxes.append(b)
    if not boxes:  # a diagram drawn with shapes: take the area the shapes cover
        drawn = pymupdf.Rect()
        for d in page.get_drawings():
            drawn |= d["rect"]
        drawn &= r
        if drawn.is_empty or drawn.width * drawn.height / area < MIN_BOX_SHARE:
            return []
        boxes = [drawn]
    # Merge overlapping boxes, then pad.
    merged = []
    for b in sorted(boxes, key=lambda b: (b.y0, b.x0)):
        for m in merged:
            if m.intersects(b):
                m |= b
                break
        else:
            merged.append(pymupdf.Rect(b))
    out = []
    for m in merged[:3]:
        m = pymupdf.Rect(m.x0 - PAD, m.y0 - PAD, m.x1 + PAD, m.y1 + PAD) & r
        out.append(None if m.width * m.height / area > 0.7 else m)
    return [None] if None in out else out


def _pdf(entry: dict, path, units: list[dict]) -> dict:
    doc = pymupdf.open(path)
    out = {}
    for u in units:
        if not u["page"] or not re.search(r"ocr|visual", u["via"]):
            continue
        page = doc[u["page"] - 1]
        names = [_save(entry, f"p{u['page']}", k, _render(page, clip))
                 for k, clip in enumerate(_page_crops(page))]
        if names:
            out[u["page"]] = [{"name": n} for n in names]
    return out


def _picture(entry, slot, k, blob: bytes) -> str | None:
    try:
        pix = pymupdf.Pixmap(blob)
    except Exception:  # EMF/WMF and other formats browsers can't show anyway
        return None
    if pix.width < 120 or pix.height < 80:
        return None
    return _save(entry, slot, k, _jpeg(pix))


def _xlsx(entry: dict, path, units: list[dict]) -> dict:
    from .extract import MIN_SHEET_PICTURE, _sheet_drawings
    z = zipfile.ZipFile(path)
    out = {}
    for sheet, parts in _sheet_drawings(z).items():
        n = next((i for i, u in enumerate(units) if u["text"].startswith(f"Sheet: {sheet}\n")
                  or u["text"] == f"Sheet: {sheet}"), None)
        if n is None:
            continue
        # Same pictures, same order as extract._xlsx, so "[Picture k]" in the text matches.
        pics = [p for p in parts if "/media/" in p and z.getinfo(p).file_size >= MIN_SHEET_PICTURE]
        found = []
        for k, p in enumerate(pics, 1):
            if name := _picture(entry, f"u{n}", k, z.read(p)):
                found.append({"name": name, "marker": f"[Picture {k}]"})
        if found:
            out[f"u{n}"] = found
    return out


def for_file(entry: dict, units: list[dict]) -> dict:
    """{slot: [{"name", "marker"?}]} for one file; slot is the page number, or "u<n>" for
    units without a page. Cached in data/images/<file id>.json."""
    from .extract import IMAGE_EXT, _sniff_ext
    cache = config.IMAGES_DIR / f"{entry['id']}.json"
    stamp = f"{entry['modifiedTime']}|{entry['path']}|v{config.IMAGES_VERSION}"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("stamp") == stamp and all((config.IMAGES_DIR / i["name"]).exists()
                                                for v in cached["images"].values() for i in v):
            return {(int(k) if k.isdigit() else k): v for k, v in cached["images"].items()}
    path = config.ROOT / entry["path"]
    ext = _sniff_ext(path)
    if ext == ".pdf":
        found = _pdf(entry, path, units)
    elif ext == ".xlsx" or entry["name"].lower().endswith(".xlsx"):
        found = _xlsx(entry, path, units)
    elif ext in IMAGE_EXT and units:
        name = _picture(entry, "u0", 0, path.read_bytes())
        found = {"u0": [{"name": name}]} if name else {}
    else:
        found = {}
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"stamp": stamp, "images": {str(k): v for k, v in found.items()}}))
    return found


def for_chunk(unit_images: list[dict], piece: str) -> list[str]:
    """Pictures of one chunk: all of its unit's, except pictures tied to a "[Picture k]"
    block that landed in another chunk."""
    return [i["name"] for i in unit_images if not i.get("marker") or i["marker"] in piece]


# ---------- storage ----------

@lru_cache
def _bucket():
    from google.cloud import storage
    return storage.Client(project=config.GCP_PROJECT).bucket(config.IMAGES_BUCKET)


def upload_missing() -> int:
    """Copy local pictures that aren't in IMAGES_BUCKET yet. Returns how many were sent."""
    if not config.IMAGES_BUCKET or not config.IMAGES_DIR.exists():
        return 0
    there = {b.name.removeprefix("images/") for b in _bucket().list_blobs(prefix="images/")}
    todo = [p for p in config.IMAGES_DIR.glob("*.jpg") if p.name not in there]
    for p in todo:
        blob = _bucket().blob(f"images/{p.name}")
        blob.cache_control = "private, max-age=31536000, immutable"
        blob.upload_from_filename(str(p), content_type="image/jpeg")
    return len(todo)


def read(name: str) -> bytes | None:
    if not NAME_RE.match(name):
        return None
    path = config.IMAGES_DIR / name
    if path.exists():
        return path.read_bytes()
    if config.IMAGES_BUCKET:
        blob = _bucket().blob(f"images/{name}")
        return blob.download_as_bytes() if blob.exists() else None
    return None


# ---------- picking and showing pictures (answer time) ----------

# Signed links, so the /img route only serves pictures this app handed out. The key must
# be the same on every Cloud Run instance (see voice.py).
_SECRET = os.getenv("VOICE_SECRET") or config.APP_PASSWORD
# With no secret (local runs) the key is random, but kept in the process environment:
# Streamlit reloads this module when project files change, and server.py's route would
# otherwise keep checking links against the key from before the reload.
_SECRET = _SECRET or os.environ.setdefault("_LOCAL_LINK_SECRET", secrets.token_hex(32))
_KEY = hashlib.sha256(("img:" + _SECRET).encode()).digest()


def sign(name: str) -> str:
    return hmac.new(_KEY, name.encode(), "sha256").hexdigest()[:24]


def verify(name: str, sig: str) -> bool:
    return hmac.compare_digest(sign(name), sig)


def url(name: str) -> str:
    return f"/img/{name}?sig={sign(name)}"


SEARCH_POOL = 15        # passages with pictures fetched per question before scoring
PAGE_TEXT_CHARS = 1500  # of each candidate's text sent for scoring


def _fingerprint(name: str) -> str:
    """Content hash in the file name: the same picture in two files has the same one."""
    return name.rsplit("-", 1)[-1]


def _candidates(hits: list[dict]) -> list[dict]:
    """One candidate per distinct picture set (a slide page, a doc's screenshot, a sheet's
    picture), in search order, with the full text of every passage that shares it."""
    by_file = {}
    for p in store.picture_passages(sorted({h["file_id"] for h in hits})):
        by_file.setdefault((p["file_id"], tuple(p["images"])), []).append(p)
    out, seen = [], set()
    for h in hits:
        key = (h["file_id"], tuple(h["images"]))
        prints = {_fingerprint(n) for n in h["images"]}
        if key in seen or prints & seen:  # same page twice, or a copy of a picture already in
            continue
        seen.add(key)
        seen |= prints
        parts = by_file.get(key) or [h]
        text = "\n".join(dict.fromkeys(x["text"] for x in parts))  # pieces of one page, deduped
        where = f", page {h['page']}" if h.get("page") else ""
        out.append({"images": list(h["images"]), "label": f"{h['class_label']} — {h['file_name']}{where}",
                    "link": h.get("link", ""), "text": text})
    return out


def _snippet(c: dict) -> str:
    """Start of the text plus the picture description (which sits at the end)."""
    text = c["text"]
    if len(text) > PAGE_TEXT_CHARS:
        cut = max(text.rfind("Visual:"), text.rfind("[Text in screenshots/images]"))
        tail = text[cut:] if cut > 0 else ""
        tail = tail[:PAGE_TEXT_CHARS // 2]
        text = text[:PAGE_TEXT_CHARS - len(tail)] + " …\n" + tail
    return f"{c['label']}\n{text}"


def find(question: str, class_labels: list[str] | None = None, usage=None,
         limit: int = MAX_PER_ANSWER) -> list[dict]:
    """Pictures from all the class material that fit the question, best first; [] when none
    fits. question should carry the topic (for follow-ups, the earlier questions too)."""
    try:
        hits = store.search(question, class_labels=class_labels, limit=SEARCH_POOL,
                            usage=usage, pictures_only=True)
        cands = _candidates(hits)
        if not cands:
            return []
        scores = llm.score_pictures(question, [_snippet(c) for c in cands], usage)
    except Exception as e:  # pictures are extra: never break the answer over them
        print(f"picture search failed: {e}")
        return []
    ranked = sorted((i for i in range(len(cands)) if scores.get(i + 1, 0) >= config.PICTURE_MIN_SCORE),
                    key=lambda i: (-scores[i + 1], i))
    out = []
    for i in ranked:
        c = cands[i]
        for name in c["images"]:
            out.append({"name": name, "label": c["label"], "link": c["link"], "score": scores[i + 1]})
            if len(out) >= limit:
                return out
    return out


# Clicking a thumbnail opens it large over the page (same tab); clicking anywhere on the
# overlay or its ✕ closes it. Built from <details>/<summary> because Streamlit's markdown
# can't run scripts: the thumbnail and the overlay both sit inside <summary>, so any click
# on them toggles it.
_CSS = """<style>
.cls-pics{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px;margin:4px 0 8px}
.cls-pics figure{margin:0}
.cls-pics summary{list-style:none;cursor:zoom-in}
.cls-pics summary::-webkit-details-marker{display:none}
.cls-pics .thumb{width:100%;aspect-ratio:16/10;object-fit:contain;background:#fff;border-radius:6px;border:1px solid rgba(128,128,128,.35);display:block}
.cls-pics figcaption{font-size:12px;opacity:.75;line-height:1.3;margin-top:3px}
.cls-pics .ov{display:none}
.cls-pics details[open] .ov{display:flex;position:fixed;inset:0;z-index:1000000;cursor:zoom-out;
  background:rgba(0,0,0,.85);align-items:center;justify-content:center;flex-direction:column;padding:24px}
.cls-pics .ov img{max-width:96vw;max-height:84vh;object-fit:contain;border-radius:6px;background:#fff}
.cls-pics .ov .cap{color:#eee;font-size:14px;margin-top:10px;text-align:center;max-width:90vw}
.cls-pics .ov .x{position:fixed;top:14px;right:18px;width:40px;height:40px;border-radius:50%;
  background:rgba(255,255,255,.15);color:#fff;font-size:24px;line-height:40px;text-align:center;cursor:pointer}
.cls-pics .ov .x:hover{background:rgba(255,255,255,.3)}
</style>"""


def render(pics: list[dict]):
    """A row of thumbnails; clicking one shows it large on the same page."""
    import streamlit as st
    if not pics:
        return
    cells = []
    for p in pics:
        u = html.escape(url(p["name"]))
        # Older saved chats numbered pictures by the source they came from.
        label = html.escape(f"[{p['source']}] {p['label']}" if p.get("source") else p["label"])
        drive = f' · <a href="{html.escape(p["link"])}" target="_blank">Drive</a>' if p.get("link") else ""
        cells.append(f'<figure><details><summary title="Click to enlarge">'
                     f'<img class="thumb" src="{u}" loading="lazy" alt="slide">'
                     f'<div class="ov"><span class="x" title="Close">✕</span>'
                     f'<img src="{u}" alt="slide"><div class="cap">{label}</div></div>'
                     f'</summary></details><figcaption>{label}{drive}</figcaption></figure>')
    st.markdown(_CSS + '<div style="font-size:13px;opacity:.8">🖼️ From the class material · click to enlarge</div>'
                f'<div class="cls-pics">{"".join(cells)}</div>', unsafe_allow_html=True)
