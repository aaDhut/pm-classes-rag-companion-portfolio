"""Central settings. Values come from .env (see .env in the project root)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GCP_PROJECT = os.getenv("GCP_PROJECT", "")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")  # embeddings
# Gemini 3.x models are only served from the global endpoint.
GEN_LOCATION = os.getenv("GEN_LOCATION", "global")
# Answers and slide descriptions. Set GEN_PROJECT to another project to run them there
# via the service account at GEN_CREDENTIALS; embeddings always stay on GCP_PROJECT so
# the existing vectors keep matching.
GEN_PROJECT = os.getenv("GEN_PROJECT", GCP_PROJECT)

QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "pm_classes")

# The course notes folder on Google Drive (set DRIVE_FOLDER_ID in .env).
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "")
TEACHER = os.getenv("TEACHER", "Instructor A")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemini-3.5-flash")
# Cheaper model for plain questions; "where was it taught" and number questions stay on
# ANSWER_MODEL (see rag/answer.py pick_model). "" = off: every answer uses ANSWER_MODEL.
ANSWER_MODEL_LITE = os.getenv("ANSWER_MODEL_LITE", "")
# Used when ANSWER_MODEL keeps returning 429. Set to "" to disable.
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")
VISION_MODEL = os.getenv("VISION_MODEL", "gemini-3.5-flash")
# Scores how well each candidate slide/picture fits the question before it is shown.
PICTURE_MODEL = os.getenv("PICTURE_MODEL", "gemini-3.5-flash-lite")
# Pictures scoring below this (0-10) are not shown. No picture beats a wrong one.
PICTURE_MIN_SCORE = int(os.getenv("PICTURE_MIN_SCORE", "7"))
# Checks every answer against its excerpts and fixes wrong details (rag/verify.py).
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "gemini-3.5-flash")
# "low" tested 2026-10-26 on 50 questions: same accuracy as "medium" (95.6% vs 95.9%),
# ~₹1.5 cheaper per answer and done in 4 s instead of 10 s. Set "medium" to go back.
VERIFY_THINKING = os.getenv("VERIFY_THINKING", "low")
SPARSE_MODEL = "Qdrant/bm25"

# Vertex AI standard (global) prices in USD per 1M tokens, from
# https://cloud.google.com/vertex-ai/generative-ai/pricing (checked Oct 2026).
# "out" covers both the answer and the model's thinking tokens. "cached_in" is what
# Google charges for prompt tokens it has cached from a recent similar request.
# A model with several entries changes price on the given date (first match wins).
PRICES = {
    "gemini-3.5-flash": [("", {"in": 1.50, "cached_in": 0.15, "audio_in": 1.50, "out": 9.00})],
    # Flash-Lite price from third-party price lists (not yet on Google's page, Oct 2026).
    "gemini-3.5-flash-lite": [("", {"in": 0.30, "cached_in": 0.03, "out": 2.50})],
    "gemini-2.5-flash": [("", {"in": 0.30, "cached_in": 0.03, "audio_in": 1.00, "out": 2.50})],
    # Only used by the eval's grader (eval/accuracy.py). Prompts up to 200k tokens.
    "gemini-3.1-pro-preview": [("", {"in": 2.00, "cached_in": 0.20, "out": 12.00})],
    "gemini-embedding-001": [("", {"in": 0.15})],
    # Voice for the Play button. "out" = audio tokens (25 per second of speech).
    "gemini-3.8-flash-tts": [("2027-01-01", {"in": 1.00, "out": 18.00}),
                             ("", {"in": 0.50, "out": 9.00})],
    "gemini-3.8-flash-lite-tts": [("2027-01-01", {"in": 1.00, "out": 12.00}),
                                  ("", {"in": 0.50, "out": 6.00})],
}

# Voice used to read answers aloud.
# Flash-Lite chosen 2026-10-26 after a side-by-side listen: sounds as good, ~40% cheaper.
# "gemini-3.8-flash-tts" is the earlier, slightly more natural voice.
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-3.8-flash-lite-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "Kore")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en-IN")  # Indian English accent
# Bump when the way audio is generated changes, so saved recordings are made again.
VOICE_VERSION = 2

# Rupee conversion. The live rate is fetched once a day; this is used if that fails.
USD_INR_FALLBACK = float(os.getenv("USD_INR_FALLBACK", "94.59"))
FX_URL = "https://open.er-api.com/v6/latest/USD"
# BM25 files are committed in models/bm25 so nothing is downloaded from Hugging Face,
# which blocks Cloud Run and Cloud Build IPs.
BM25_MODEL_PATH = os.getenv("BM25_MODEL_PATH", str(ROOT / "models" / "bm25"))
# Low temperature = consistent, factual answers (user asked for "not very creative").
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

# A PDF page with less text-layer text than this is OCR'd as a whole image.
VISION_MIN_CHARS = int(os.getenv("VISION_MIN_CHARS", "200"))
# Pages whose images cover at least this share of the page get a Gemini "Visual:" description.
VISUAL_MIN_AREA = float(os.getenv("VISUAL_MIN_AREA", "0.15"))
# Bump when extraction logic changes so cached extractions are redone.
EXTRACT_VERSION = 2
# Bump when the way PDF tables are rebuilt changes (rag/tables.py has its own cache).
TABLES_VERSION = 1

SECRETS_DIR = ROOT / "secrets"
CLIENT_SECRET = SECRETS_DIR / "client_secret.json"
TOKEN_FILE = SECRETS_DIR / "token.json"
# On Cloud Run the key is mounted from Secret Manager at the path in GEN_CREDENTIALS.
GEN_CREDENTIALS = Path(os.getenv("GEN_CREDENTIALS", SECRETS_DIR / "gen_service_account.json"))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
MANIFEST = DATA_DIR / "manifest.json"

# Set on Cloud Run: chats go to this GCS bucket instead of data/chats/.
CHATS_BUCKET = os.getenv("CHATS_BUCKET", "")
# Slide/screenshot images shown with answers: saved locally by ingest.py, and uploaded to
# this bucket (under images/) so Cloud Run can serve them. Defaults to the chats bucket.
IMAGES_DIR = DATA_DIR / "images"
IMAGES_BUCKET = os.getenv("IMAGES_BUCKET", CHATS_BUCKET)
# Bump when the way images are cut out changes, so they are made again.
IMAGES_VERSION = 1

# When set, the app asks for this password before showing the chat.
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Class video lectures (Deepgram transcripts + OCR of the shared screen).
VIDEO_FOLDER_ID = os.getenv("VIDEO_FOLDER_ID", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
VIDEO_DIR = DATA_DIR / "videos"            # per-video transcript, frames, OCR (kept)
VIDEO_TMP = DATA_DIR / "video_tmp"         # the downloaded .mp4 (deleted after processing)
VIDEO_MANIFEST = DATA_DIR / "video_manifest.json"
