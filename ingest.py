"""Sync Drive -> extract text -> embed -> Qdrant.

    uv run python ingest.py            # normal run: only new/changed files
    uv run python ingest.py --no-sync  # re-index from files already downloaded
    uv run python ingest.py --rebuild  # re-embed everything (e.g. after changing chunking)
    uv run python ingest.py --no-sync --images  # also (re)attach pictures to files already indexed
    uv run python ingest.py --videos   # also add new class videos (transcript, OCR, summary)
    uv run python ingest.py --no-sync --videos-only  # only the videos
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from rag import config, drive_sync, images, netfix, store, tables, video
from rag.usage import Usage, fmt_money, usd_to_inr

netfix.install()
from rag.chunk import make_chunks
from rag.extract import extract

STATE = config.DATA_DIR / "indexed.json"
VIDEO_WORKERS = 3  # videos processed at once (each needs its .mp4 on disk while it runs)  # file id -> version stamp already in Qdrant


def stamp(e: dict) -> str:
    """Changes when the Drive file, its downloaded form, or the extraction logic changes."""
    return f"{e['modifiedTime']}|{e['path']}|v{config.EXTRACT_VERSION}|t{config.TABLES_VERSION}.{tables.LAYOUT}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true", help="skip the Drive download step")
    ap.add_argument("--rebuild", action="store_true", help="re-embed every file")
    ap.add_argument("--images", action="store_true",
                    help="cut out pictures for files already indexed and attach them (no re-embedding)")
    ap.add_argument("--videos", action="store_true", help="also add new/changed class videos")
    ap.add_argument("--videos-only", action="store_true", help="only add class videos, skip the notes")
    args = ap.parse_args()
    if args.videos_only:
        store.ensure_collection()
        index_videos()
        return

    if args.no_sync:
        manifest = json.loads(config.MANIFEST.read_text())
    else:
        manifest = drive_sync.sync()["manifest"]

    store.ensure_collection()
    indexed = {} if args.rebuild or not STATE.exists() else json.loads(STATE.read_text())

    for fid in [f for f in indexed if f not in manifest]:
        store.delete_file(fid)
        indexed.pop(fid)
        print(f"  - removed from index: {fid}")

    todo = [e for e in manifest.values() if indexed.get(e["id"]) != stamp(e)]
    todo.sort(key=lambda e: (e["class_no"] is None, e["class_no"] or 0, e["name"]))
    print(f"Indexing {len(todo)} file(s); {len(manifest) - len(todo)} already up to date.")
    total = 0
    for e in todo:
        label = f"[Class {e['class_no']}] {e['name']}" if e["class_no"] is not None else f"[{e['class_title']}] {e['name']}"
        try:
            units = extract(e)
            chunks = make_chunks(e, units, images.for_file(e, units))
            store.delete_file(e["id"])
            store.upsert_chunks(chunks)
        except Exception as err:  # one bad file must not stop the rest
            print(f"  ! {label}: {err}")
            continue
        indexed[e["id"]] = stamp(e)
        STATE.write_text(json.dumps(indexed, indent=2))
        total += len(chunks)
        print(f"  ✓ {label}: {len(chunks)} chunks")

    if args.images:
        done = [e for e in manifest.values() if e not in todo and indexed.get(e["id"]) == stamp(e)]
        print(f"Attaching pictures to {len(done)} already-indexed file(s)…")
        for e in done:
            try:
                units = extract(e)  # cached, so no OCR/Gemini calls
                chunks = make_chunks(e, units, images.for_file(e, units))
                store.set_images(chunks)
            except Exception as err:
                print(f"  ! {e['name']}: {err}")
                continue
            n = len({x for c in chunks for x in c["payload"]["images"]})
            if n:
                print(f"  🖼  {e['name']}: {n} picture(s)")

    if args.videos:
        index_videos()

    if sent := images.upload_missing():
        print(f"Uploaded {sent} picture(s) to gs://{config.IMAGES_BUCKET}/images/")

    count = store.qdrant().count(config.COLLECTION).count
    print(f"Done. Added {total} chunks; collection '{config.COLLECTION}' now holds {count}.")


def index_videos():
    """Download, transcribe, OCR, summarise and index each new or changed class video.
    One video at a time, so only one .mp4 is ever on disk; finished steps are never redone."""
    svc = video.drive_service()
    videos = video.list_videos(svc)
    state = json.loads(video.VIDEO_STATE.read_text()) if video.VIDEO_STATE.exists() else {}
    for vid in [i for i in state if i not in {v["id"] for v in videos}]:
        store.delete_file(vid)
        state.pop(vid)
        print(f"  - removed video from index: {vid}")
    todo = [v for v in videos if state.get(v["id"]) != video.stamp(v)]
    print(f"Videos: {len(todo)} to add; {len(videos) - len(todo)} already indexed.", flush=True)
    lock = threading.Lock()
    usage = Usage()  # Gemini cost of describing slides (the big one-time cost)

    def one(n_v):
        n, v = n_v
        name = video.lecture_label(v) if v["class_title"] else v["name"]
        for attempt in range(1, 6):
            try:
                video.process(video.drive_service(), v)  # own Drive client: they aren't thread-safe
                sp = video.identify_speakers(video.video_dir(v))
                video.describe_slides(v, usage)  # cached: only new slides cost anything
                count = video.index(v)
                break
            except Exception as err:  # one bad video must not stop the rest
                # This Mac's DNS sometimes fails a lookup for a minute; finished steps are
                # cached, so trying again only redoes the step that failed.
                if "nodename nor servname" in str(err) and attempt < 5:
                    print(f"    {name}: network lookup failed, retrying in 60 s", flush=True)
                    time.sleep(60)
                    continue
                print(f"  ! [{n}/{len(todo)}] {name}: {err}", flush=True)
                return
        with lock:
            state[v["id"]] = video.stamp(v)
            video.VIDEO_STATE.write_text(json.dumps(state, indent=1))
        print(f"  ✓ [{n}/{len(todo)}] {name}: taught by {sp['teacher']}; {count} pieces", flush=True)

    # Downloads are the slow part, so a few videos run side by side.
    with ThreadPoolExecutor(VIDEO_WORKERS) as pool:
        list(pool.map(one, enumerate(todo, 1)))
    video.write_review(videos)
    if usage.cost:
        print(f"Slide descriptions cost {fmt_money(usage.cost, usage.cost * usd_to_inr())}")
    if sent := images.upload_missing():
        print(f"Uploaded {sent} picture(s) to gs://{config.IMAGES_BUCKET}/images/")


if __name__ == "__main__":
    main()
