"""Tables in PDF notes, kept as rows and columns.

The plain text of a PDF page loses the grid: a table comes out as a loose list of words
("Mutual funds or Stocks / High / Low / Low"), and the model then guesses which value
belongs to which column. Here each table is rebuilt and added to its page as a markdown
table:
  1. PyMuPDF's table finder, when its result is clean (no merged or repeated cells);
  2. otherwise Gemini reads the table off the page image. Every word it returns must
     appear in the page's own text (text layer or OCR), so it cannot invent cells.
Results are cached in data/tables/<file id>.json, keyed by Drive modifiedTime and
config.TABLES_VERSION.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

from . import config, llm

CACHE_DIR = config.DATA_DIR / "tables"
LAYOUT = 2  # bump when the way tables are put into the page text changes (re-indexes, no Gemini calls)
MAX_BLOCK = 1200          # chars per table block, so one chunk holds the header with its rows
MIN_COVERAGE = 0.97       # share of a Gemini table's words that must be on the page
TABLE_DPI = 130
_WORD = re.compile(r"[a-z0-9]+")


def _cell(v) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _clean(rows: list[list[str]]) -> bool:
    """A PyMuPDF table that can be used as is: a real grid, not a slide layout cut into boxes."""
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [c for r in rows for c in r]
    filled = [c for c in cells if c and not re.fullmatch(r"Col\d+", c)]
    if len(filled) < 0.5 * len(cells) or any(len(c) > 200 for c in filled):
        return False
    if any(re.fullmatch(r"[a-z]{1,2}", c) for c in filled):  # a word cut across two cells ("D" | "o")
        return False
    long = [c for c in filled if len(c) > 15]
    return len(long) == len(set(long))  # repeated long cells = merged-cell artefacts


def _render(header: list[str], rows: list[list[str]], title: str = "") -> list[str]:
    """Markdown blocks; a long table is split with its header repeated in each block."""
    width = max(len(header), *(len(r) for r in rows)) if rows else len(header)
    header = [("" if re.fullmatch(r"Col\d+", h) else h) for h in header] + [""] * (width - len(header))

    def line(r):
        r = [c.replace("|", "/") for c in r] + [""] * (width - len(r))
        return "| " + " | ".join(r) + " |"

    head = (f"[Table{': ' + title if title else ''} — read each value by its row and column]\n"
            + line(header) + "\n|" + "---|" * width)
    blocks, cur = [], []
    for r in rows:
        if cur and len(head) + sum(len(x) + 1 for x in cur) + len(line(r)) > MAX_BLOCK:
            blocks.append(head + "\n" + "\n".join(cur))
            cur = []
        cur.append(line(r))
    if cur:
        blocks.append(head + "\n" + "\n".join(cur))
    return blocks


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def _supported(table: dict, page_text: str) -> bool:
    """Every (or nearly every) word of the table is on the page; every number exactly."""
    have = set(_words(page_text))
    words = _words(" ".join(table["header"] + [c for r in table["rows"] for c in r] + [table.get("title", "")]))
    if not words:
        return False
    missing = [w for w in words if w not in have]
    if any(w.isdigit() for w in missing):
        return False
    return len(missing) <= (1 - MIN_COVERAGE) * len(words)


def _from_gemini(page, page_text: str) -> tuple[list[str], int]:
    """(table blocks, tables rejected for words not on the page)."""
    image = page.get_pixmap(dpi=TABLE_DPI).tobytes("jpeg", jpg_quality=85)
    blocks, rejected = [], 0
    for t in llm.read_tables(image, page_text):
        rows = [[_cell(c) for c in r] for r in t.get("rows", []) if any(_cell(c) for c in r)]
        t = {"title": _cell(t.get("title")), "header": [_cell(h) for h in t.get("header", [])], "rows": rows}
        if not rows or len(t["header"]) < 2:
            continue
        if not _supported(t, page_text):
            rejected += 1
            continue
        blocks += _render(t["header"], rows, t["title"])
    return blocks, rejected


def _page_tables(doc, i: int, unit_text: str) -> tuple[list[str], str]:
    """(markdown blocks, how they were made) for page i."""
    page = doc[i]
    try:
        found = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    except Exception:
        found = []
    image_page = not page.get_text("text").strip() or len(page.get_text("text").strip()) < config.VISION_MIN_CHARS
    looks_like_table = image_page and re.search(r"\btable\b|\bcolumns?\b|\brows?\b", unit_text, re.I)
    if not found and not looks_like_table:
        return [], ""
    grids = [[[_cell(c) for c in r] for r in t.extract()] for t in found]
    if grids and all(_clean(g) for g in grids):
        return [b for g in grids for b in _render(g[0], g[1:])], "pymupdf"
    # Text layer + OCR of the page. The "Visual:" part is Gemini's own description, so it
    # doesn't count as proof that a word is on the page.
    reference = page.get_text("text") + "\n" + unit_text.split("\n\nVisual:\n")[0]
    blocks, rejected = _from_gemini(page, reference)
    return blocks, f"gemini{f' ({rejected} rejected)' if rejected else ''}"


def for_pdf(entry: dict, path: Path, units: list[dict]) -> dict[int, list[str]]:
    """{page: [markdown table blocks]} for a PDF, cached."""
    cache = CACHE_DIR / f"{entry['id']}.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c["modifiedTime"] == entry["modifiedTime"] and c["version"] == config.TABLES_VERSION:
            return {int(k): v for k, v in c["pages"].items()}
    doc = pymupdf.open(path)
    text_of = {u["page"]: u["text"] for u in units if u.get("page")}

    def one(i):
        try:
            return i, *_page_tables(doc, i, text_of.get(i + 1, ""))
        except Exception as e:
            print(f"    ! tables on page {i + 1} of {entry['name']}: {e}")
            return i, [], "error"

    pages, log = {}, {}
    with ThreadPoolExecutor(6) as pool:
        for i, blocks, how in pool.map(one, range(len(doc))):
            if blocks:
                pages[i + 1] = blocks
            if how:
                log[i + 1] = how
    if log:
        print(f"    Tables: {sum(len(v) for v in pages.values())} block(s) on {len(pages)} page(s) "
              f"of {entry['name']} ({', '.join(sorted(set(log.values())))})")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"modifiedTime": entry["modifiedTime"], "version": config.TABLES_VERSION,
                                 "pages": pages, "log": log}, ensure_ascii=False, indent=1))
    return pages


def _useful(block: str) -> bool:
    """False for an empty shell: no row has a value outside its first column."""
    rows = [l for l in block.splitlines()[3:] if l.startswith("|")]
    return any(any(c.strip() for c in r.strip().strip("|").split("|")[1:]) for r in rows)


def add_to_units(entry: dict, path: Path, units: list[dict]) -> list[dict]:
    """Units with each page's tables appended (the cached units are not changed)."""
    pages = {p: [b for b in blocks if _useful(b)] for p, blocks in for_pdf(entry, path, units).items()}
    if not pages:
        return units
    out = []
    for u in units:
        blocks = pages.get(u.get("page"), [])
        if not blocks:
            out.append(u)
            continue
        text = _drop_table_lines(u["text"], blocks)
        out.append({**u, "text": text + "\n\n" + "\n\n".join(blocks), "via": u["via"] + "+table"})
    return out


def _norm(t: str) -> str:
    return " ".join(_words(t))


def _drop_table_lines(text: str, blocks: list[str]) -> str:
    """The page text without the loose lines that are only pieces of a rebuilt table, so the
    model reads each value from the table (where rows and columns are right), not from a
    jumbled list. OCR text and the "Visual:" description are kept as they are."""
    cut = len(text)
    for marker in ("\n\n[Text in screenshots/images]", "\n\nVisual:\n"):
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    body, rest = text[:cut], text[cut:]
    table_text = " " + _norm(" ".join(blocks)) + " "
    kept, dropped = [], 0
    for line in body.splitlines():
        n = _norm(line)
        if n and f" {n} " in table_text:
            dropped += 1
            continue
        kept.append(line)
    if dropped:
        kept.append("(The rest of this page's text is in the table below.)")
    return "\n".join(kept) + rest
