"""Split extracted units into retrieval chunks that carry their class/file/page."""
import re

MAX_CHARS = 1400
OVERLAP = 200
MIN_CHARS = 25


def _split(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= MAX_CHARS:
        return [text]
    paras = re.split(r"\n\s*\n|\n(?=[#•\-\*\d]+[\.\)]?\s)", text)
    chunks, cur = [], ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        while len(p) > MAX_CHARS:  # one giant paragraph
            cut = p.rfind(" ", 0, MAX_CHARS)
            cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut])
            p = p[max(cut - OVERLAP, 0):].strip()
        if len(cur) + len(p) + 2 > MAX_CHARS and cur:
            chunks.append(cur)
            cur = cur[-OVERLAP:].split(" ", 1)[-1] + "\n\n" + p  # carry a little context forward
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def class_label(entry: dict) -> str:
    """Label of the class these notes belong to (numbered by lecture video, see rag/classes.py)."""
    from .classes import class_no, label
    return label(class_no(entry["class_no"]), entry["class_title"])


def make_chunks(entry: dict, units: list[dict], images: dict | None = None) -> list[dict]:
    """images: rag.images.for_file() output, {slot: [{"name", "marker"?}]}."""
    from .classes import class_no, folder_label
    from .images import for_chunk
    out = []
    for n, u in enumerate(units):
        # Units without a page (spreadsheet tabs) need their position in the key, or they overwrite each other.
        slot = u["page"] if u["page"] else f"u{n}"
        unit_images = (images or {}).get(slot, [])
        for i, piece in enumerate(_split(u["text"])):
            if len(piece) < MIN_CHARS:
                continue
            where = f"page {u['page']}" if u["page"] else f"part {i + 1}"
            header = f"{class_label(entry)} | {entry['name']} | {where}"
            pics = for_chunk(unit_images, piece)
            out.append({
                "key": f"{entry['id']}:{slot}:{i}",
                "embed_text": f"{header}\n\n{piece}",
                "payload": {
                    "text": piece, "class_no": class_no(entry["class_no"]), "class_title": entry["class_title"],
                    "class_label": class_label(entry), "notes_folder": folder_label(entry),
                    "file_id": entry["id"], "file_name": entry["name"],
                    "page": u["page"], "via": u["via"], "link": entry["link"], "teacher": entry["teacher"],
                    "modified": entry["modifiedTime"], "images": pics, "has_images": bool(pics),
                },
            })
    return out
