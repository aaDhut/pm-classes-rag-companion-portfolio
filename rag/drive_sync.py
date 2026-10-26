"""Mirror the class folder from Google Drive into data/raw/.

Each file is recorded in data/manifest.json with its class number/title,
so later steps know where every chunk came from. Only new or changed
files (by Drive modifiedTime) are downloaded again.
"""
import io
import json
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from . import config

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"

# Google-native files have no bytes of their own; they are exported.
EXPORTS = {
    # PDF keeps screenshots/diagrams so they can be OCR'd; text is the fallback for huge docs.
    "application/vnd.google-apps.document": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.presentation": ("application/pdf", ".pdf"),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
}
SKIP_PREFIXES = ("video/", "audio/")
SKIP_TYPES = {"application/vnd.google-apps.form", "application/vnd.google-apps.site",
              "application/vnd.google-apps.map", "application/zip"}

CLASS_RE = re.compile(r"\b(?:class|session)\s*[-#]?\s*(\d+)", re.I)


def get_credentials():
    """Own OAuth client if secrets/client_secret.json exists, else gcloud application-default login."""
    if not config.CLIENT_SECRET.exists() and not config.TOKEN_FILE.exists():
        import google.auth
        creds, _ = google.auth.default(scopes=SCOPES)
        return creds
    creds = None
    if config.TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(config.TOKEN_FILE), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        if not config.CLIENT_SECRET.exists():
            raise SystemExit(f"Missing {config.CLIENT_SECRET}. Download the OAuth Desktop client JSON first.")
        flow = InstalledAppFlow.from_client_secrets_file(str(config.CLIENT_SECRET), SCOPES)
        creds = flow.run_local_server(port=0)
    config.TOKEN_FILE.write_text(creds.to_json())
    config.TOKEN_FILE.chmod(0o600)
    return creds


def parse_class(folder_name: str) -> tuple[int | None, str]:
    """'Class 9 - PRD, concept Note' -> (9, 'PRD, concept Note')."""
    m = CLASS_RE.search(folder_name)
    if not m:
        return None, folder_name.strip()
    title = folder_name[m.end():].strip(" -–:_")
    return int(m.group(1)), title or folder_name.strip()


def safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", name).strip()[:120]


def list_children(svc, folder_id: str) -> list[dict]:
    items, token = [], None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink, size, shortcutDetails)",
            pageSize=1000, pageToken=token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        items += resp.get("files", [])
        token = resp.get("nextPageToken")
        if not token:
            return items


def resolve_shortcut(svc, f: dict) -> dict:
    target = f["shortcutDetails"]["targetId"]
    return svc.files().get(fileId=target, supportsAllDrives=True,
                           fields="id, name, mimeType, modifiedTime, webViewLink, size").execute()


def walk(svc, folder_id: str, class_no=None, class_title="General", path=()):
    """Yield (file, class_no, class_title, folder_path). A top-level class folder sets the class for everything under it."""
    for f in list_children(svc, folder_id):
        if f["mimeType"] == SHORTCUT:
            try:
                f = resolve_shortcut(svc, f)
            except HttpError:
                continue
        if f["mimeType"] == FOLDER:
            if not path:  # direct child of the course folder = one class
                no, title = parse_class(f["name"])
            else:
                no, title = class_no, class_title
            yield from walk(svc, f["id"], no, title, path + (f["name"],))
        else:
            yield f, class_no, class_title, path


def export_mime(f: dict) -> str | None:
    return EXPORTS[f["mimeType"]][0] if f["mimeType"] in EXPORTS else None


def download(svc, f: dict, dest_base) -> str | None:
    mime = f["mimeType"]
    if mime.startswith(SKIP_PREFIXES) or mime in SKIP_TYPES:
        return None
    if mime in EXPORTS:
        target, ext = EXPORTS[mime]
        try:
            return _fetch(svc.files().export_media(fileId=f["id"], mimeType=target), dest_base, ext)
        except HttpError as e:
            if "exportSizeLimitExceeded" not in str(e):
                raise
            # files.export caps output at 10 MB; the file's exportLinks URL has no such cap.
            if rel := _fetch_export_link(svc, f, target, dest_base, ext):
                return rel
            if mime != "application/vnd.google-apps.document":
                raise
            print(f"    {f['name']} is too large to export as PDF; using plain text")
            return _fetch(svc.files().export_media(fileId=f["id"], mimeType="text/plain"), dest_base, ".txt")
    if mime.startswith("application/vnd.google-apps."):
        return None
    return _fetch(svc.files().get_media(fileId=f["id"], supportsAllDrives=True), dest_base, "")


def _fetch_export_link(svc, f: dict, target: str, dest_base, ext: str) -> str | None:
    url = svc.files().get(fileId=f["id"], fields="exportLinks").execute().get("exportLinks", {}).get(target)
    if not url:
        return None
    resp, content = svc._http.request(url)
    if resp.status != 200:
        return None
    dest = dest_base.with_name(dest_base.name + ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return str(dest.relative_to(config.ROOT))


def _fetch(request, dest_base, ext: str) -> str:
    dest = dest_base.with_name(dest_base.name + ext)
    dest.parent.mkdir(parents=True, exist_ok=True)
    buf = io.FileIO(dest, "wb")
    downloader = MediaIoBaseDownload(buf, request, chunksize=20 * 1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.close()
    return str(dest.relative_to(config.ROOT))


def sync() -> dict:
    svc = build("drive", "v3", credentials=get_credentials(), cache_discovery=False)
    old = json.loads(config.MANIFEST.read_text()) if config.MANIFEST.exists() else {}
    new, changed = {}, []
    for f, class_no, class_title, path in walk(svc, config.DRIVE_FOLDER_ID):
        prev = old.get(f["id"])
        entry = {
            "id": f["id"], "name": f["name"], "mimeType": f["mimeType"],
            "modifiedTime": f["modifiedTime"], "link": f.get("webViewLink", ""),
            "class_no": class_no, "class_title": class_title,
            "folder": "/".join(path), "teacher": config.TEACHER,
            "exportMime": export_mime(f),
        }
        if prev and prev["modifiedTime"] == f["modifiedTime"] and prev.get("path") \
                and prev.get("exportMime") == entry["exportMime"]:
            entry["path"] = prev["path"]
            new[f["id"]] = entry
            continue
        folder = config.RAW_DIR / safe_name(path[0] if path else "_root")
        try:
            rel = download(svc, f, folder / f"{safe_name(f['name'])}__{f['id'][:8]}")
        except HttpError as e:
            print(f"  ! could not download {f['name']}: {e.reason}")
            continue
        if rel is None:
            print(f"  - skipped {f['name']} ({f['mimeType']})")
            continue
        entry["path"] = rel
        new[f["id"]] = entry
        changed.append(f["id"])
        print(f"  + {'updated' if prev else 'new'}: [Class {class_no}] {f['name']}")
    removed = [fid for fid in old if fid not in new]
    config.DATA_DIR.mkdir(exist_ok=True)
    config.MANIFEST.write_text(json.dumps(new, indent=2))
    print(f"Drive sync: {len(new)} files, {len(changed)} new/changed, {len(removed)} removed.")
    return {"manifest": new, "changed": changed, "removed": removed}
