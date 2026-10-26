"""All Vertex AI calls: embeddings, slide transcription, answer generation."""
import threading

from google import genai
from google.genai import errors, types
from google.oauth2 import service_account
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from . import config


_local = threading.local()


def client(location: str = config.VERTEX_LOCATION) -> genai.Client:
    # One client per thread: a shared client fails with "client has been closed"
    # when several threads make their first request at the same time.
    clients = _local.__dict__.setdefault("clients", {})
    if location not in clients:
        clients[location] = genai.Client(vertexai=True, project=config.GCP_PROJECT, location=location)
    return clients[location]


def gen_client() -> genai.Client:
    if "gen" not in _local.__dict__:
        creds = None  # same project as embeddings: use the default credentials
        if config.GEN_PROJECT != config.GCP_PROJECT:
            creds = service_account.Credentials.from_service_account_file(
                config.GEN_CREDENTIALS, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        _local.gen = genai.Client(vertexai=True, project=config.GEN_PROJECT,
                                  location=config.GEN_LOCATION, credentials=creds)
    return _local.gen


def _retryable(e: Exception) -> bool:
    return isinstance(e, errors.APIError) and e.code in (429, 500, 503, 504)


_retry = retry(retry=retry_if_exception(_retryable), wait=wait_exponential(min=2, max=60),
               stop=stop_after_attempt(8), reraise=True)


@_retry
def _embed_batch(texts: list[str], task: str, usage=None) -> list[list[float]]:
    resp = client().models.embed_content(
        model=config.EMBED_MODEL, contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=config.EMBED_DIM),
    )
    if usage is not None:
        usage.add_embedding(sum(int(e.statistics.token_count or 0) if e.statistics else 0
                                for e in resp.embeddings))
    return [e.values for e in resp.embeddings]


def embed_documents(texts: list[str], batch: int = 20) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), batch):
        out += _embed_batch(texts[i:i + batch], "RETRIEVAL_DOCUMENT")
    return out


def embed_query(text: str, usage=None) -> list[float]:
    return _embed_batch([text], "RETRIEVAL_QUERY", usage)[0]


VISUAL_PROMPT = """This is one page (or image) from a product management class.
The text on it has already been extracted (below), so do NOT repeat it.
Describe only what the visuals convey: diagrams, frameworks, flows, charts, tables,
app/UI screenshots, photos. Name the framework or structure, its parts and how they
relate (axes, quadrants, steps, arrows), and what a screenshot shows (which app/tool,
which screen, what is highlighted). 2-6 short bullets. If the visuals are only
decoration (logos, stock photos, backgrounds), reply exactly: NONE

Extracted text:
{text}"""


@_retry
def describe_visual(image: bytes, known_text: str, mime: str = "image/png") -> str:
    """Gemini's explanation of the diagrams/screenshots on a page; '' if purely decorative."""
    resp = gen_client().models.generate_content(
        model=config.VISION_MODEL,
        contents=[types.Part.from_bytes(data=image, mime_type=mime),
                  VISUAL_PROMPT.format(text=known_text[:4000] or "(none)")],
        config=types.GenerateContentConfig(temperature=config.TEMPERATURE, max_output_tokens=2048,
                                           thinking_config=types.ThinkingConfig(thinking_level="low")),
    )
    text = (resp.text or "").strip()
    return "" if text.upper().startswith("NONE") else text


TABLES_PROMPT = """This is one page from a product management class's notes. Copy every
real table on it (a grid of rows and columns holding data) exactly as printed.
- Copy each cell's text word for word, with the same spelling and numbers. Don't fix,
  summarise, translate or fill in anything. A blank cell stays "".
- "header" is the table's top row; "rows" are the other rows, each with one entry per column.
- A cell spanning several columns: put its text in the first of them, "" in the rest.
- Slide layouts, boxes, bullet lists and diagrams are not tables. If the page has no real
  table, return [].

The page's text (for exact spelling):
{text}"""

_TABLES_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "title": {"type": "STRING"}, "header": {"type": "ARRAY", "items": {"type": "STRING"}},
    "rows": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "STRING"}}}},
    "required": ["header", "rows"]}}


@_retry
def read_tables(image: bytes, page_text: str) -> list[dict]:
    """Tables on a page image as [{"title", "header": [...], "rows": [[...]]}]."""
    import json
    resp = gen_client().models.generate_content(
        model=config.VISION_MODEL,
        contents=[types.Part.from_bytes(data=image, mime_type="image/jpeg"),
                  TABLES_PROMPT.format(text=page_text[:6000] or "(none)")],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=8192,
                                           response_mime_type="application/json",
                                           response_schema=_TABLES_SCHEMA,
                                           thinking_config=types.ThinkingConfig(thinking_level="low")),
    )
    return json.loads(resp.text or "[]")


@retry(retry=retry_if_exception(_retryable), wait=wait_exponential(min=2, max=10),
       stop=stop_after_attempt(3), reraise=True)
def _open_stream(model: str, system: str, prompt: str):
    # Gemini 3.x takes thinking_level; 2.5 only understands thinking_budget.
    thinking = (types.ThinkingConfig(thinking_level="low") if model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=1024))
    stream = gen_client().models.generate_content_stream(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=8192,
                                           temperature=config.TEMPERATURE, seed=42,
                                           thinking_config=thinking),
    )
    # The request only goes out on the first read, so pull one chunk here to let a
    # 429 surface inside the retry instead of halfway through the answer.
    return next(stream, None), stream


def answer_stream(system: str, prompt: str, usage=None, model: str | None = None):
    """Yields answer text. When the stream ends, the tokens it used are added to usage."""
    model = model or config.ANSWER_MODEL
    try:
        first, stream = _open_stream(model, system, prompt)
    except errors.APIError as e:
        # Gemini 3.x runs on shared capacity and 429s in bursts; fall back rather than fail.
        if e.code != 429 or not config.FALLBACK_MODEL:
            raise
        model = config.FALLBACK_MODEL
        first, stream = _open_stream(model, system, prompt)
    meta = None  # token counts arrive on the last chunk
    if first:
        meta = first.usage_metadata or meta
        if first.text:
            yield first.text
    for chunk in stream:
        meta = chunk.usage_metadata or meta
        if chunk.text:
            yield chunk.text
    if usage is not None:
        usage.add_generation(model, meta)


TRANSCRIBE_PROMPT = """Transcribe this voice question from a student of a product management
class. It may be in English, Hindi or Hinglish. Write Hindi words in Roman script (Hinglish),
not Devanagari. Return only the question text, nothing else. If there is no speech, return
nothing."""


def _transcribe(model: str, audio: bytes, mime: str):
    return gen_client().models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=audio, mime_type=mime), TRANSCRIBE_PROMPT],
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=1024,
                                           thinking_config=types.ThinkingConfig(
                                               thinking_level="minimal") if model.startswith("gemini-3")
                                           else types.ThinkingConfig(thinking_budget=0)),
    )


def transcribe(audio: bytes, mime: str = "audio/wav", usage=None) -> str:
    """Speech to text for dictated questions."""
    model = config.ANSWER_MODEL
    try:
        resp = _retry(_transcribe)(model, audio, mime)
    except errors.APIError as e:
        if e.code != 429 or not config.FALLBACK_MODEL:
            raise
        model = config.FALLBACK_MODEL
        resp = _retry(_transcribe)(model, audio, mime)
    if usage is not None:
        usage.add_generation(model, resp.usage_metadata)
    return (resp.text or "").strip()


PICTURE_PROMPT = """A student of a product management course asked a question. Below are
slides, diagrams, screenshots and sheets from the class material (their text was read off
the picture, plus a description of what the picture shows).

Score each one 0-10 for how much SEEING it would help this student with THIS question:
- 9-10: it is exactly what was asked about (e.g. the framework, process, template or
  example the question names), or the question asks to see it.
- 7-8: it directly explains a main part of what was asked.
- 4-6: same broad area, but about a side topic or a different step.
- 0-3: different topic, only shares a word, a title/agenda slide with no content, or the
  question isn't about course content at all (greetings, thanks, small talk).
Judge by the question, not by how good the slide is. A broad question (an overview, "the
basics of X") is helped by overview slides, not by slides on one specific tool or an
advanced topic within X.

Question:
{question}

Pictures:
{pictures}"""

_PICTURE_SCHEMA = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
    "id": {"type": "INTEGER"}, "score": {"type": "INTEGER"}}, "required": ["id", "score"]}}


@_retry
def _score_pictures(prompt: str):
    model = config.PICTURE_MODEL
    return gen_client().models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0, max_output_tokens=1024, response_mime_type="application/json",
            response_schema=_PICTURE_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level="minimal") if model.startswith("gemini-3")
            else types.ThinkingConfig(thinking_budget=0)),
    )


def score_pictures(question: str, pictures: list[str], usage=None) -> dict[int, int]:
    """{picture number (1-based): 0-10 fit to the question}."""
    import json
    listing = "\n\n".join(f"<picture id={i}>\n{p}\n</picture>" for i, p in enumerate(pictures, 1))
    resp = _score_pictures(PICTURE_PROMPT.format(question=question, pictures=listing))
    if usage is not None:
        usage.add_generation(config.PICTURE_MODEL, resp.usage_metadata)
    return {int(r["id"]): int(r["score"]) for r in json.loads(resp.text or "[]")}


REWRITE_PROMPT = """A student of a product management course asked the question below (it may
be informal, misspelled or in Hinglish). Rewrite it as ONE short search query in plain
English, using the words the course's slides and notes would use for this (for example
"graded / marks" -> "evaluation criteria weightage", "bad things to avoid" -> "mistakes
pitfalls"). Keep every product, company and framework name. Reply with the query only.

Question: {question}"""


def rewrite_query(question: str, usage=None) -> str:
    """The question in the course's own words, for a second search. '' if it fails."""
    model = config.PICTURE_MODEL  # Flash-Lite: fast and cheap
    try:
        resp = _retry(gen_client().models.generate_content)(
            model=model, contents=REWRITE_PROMPT.format(question=question[:1500]),
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=200,
                thinking_config=types.ThinkingConfig(thinking_level="minimal") if model.startswith("gemini-3")
                else types.ThinkingConfig(thinking_budget=0)))
    except Exception:
        return ""
    if usage is not None:
        usage.add_generation(model, resp.usage_metadata)
    return (resp.text or "").strip().splitlines()[0] if resp.text else ""


ROUTE_PROMPT = """A student of a product management course asked the question below. It may
be voice-typed, so a word can be mis-heard (e.g. "matrix" for "metrics", "cohort" for
"cohot"). Search found the classes below; for each, a few passages that matched best.

Decide where the question's topic is actually taught:
- "main": the classes whose passages really teach what the student is asking about (most
  relevant first, at most 3). A class that only uses the word in passing, or matched on a
  shared word with a different meaning, is not main. Leave it empty if no class teaches it.
- "also": other listed classes that add something useful on the same topic (at most 3).
- "kind":
  "focused" — one class is clearly where it's taught;
  "same_topic" — two or more main classes teach the same idea (e.g. introduced in one,
  covered in depth in another);
  "different_meanings" — the question's words fit different things in different classes
  (e.g. "matrix" = a 2x2 stakeholder matrix in one class, product metrics in another).
  Then list one main class per meaning, the meaning the student most likely wants first;
  "not_taught" — none of the classes teaches it (main is empty).
- "meanings": for each main class, 2-5 words on what the question means there (e.g.
  "product metrics", "2x2 stakeholder matrix").
- "corrected": if a word in the question looks like a voice-typing slip for a product
  management term (e.g. "matrix" for "metrics", "cano model" for "Kano model"), the
  question with that word swapped for the similar-sounding term (a different word, not
  a plural or grammar fix); give it even when the question also makes sense as
  written, if the fixed one is at least as likely what a PM student means. Otherwise "".
  Never change a term that is clearly meant (e.g. "BCG matrix", "RICE score").
- "corrected_meaning": 2-5 words for what the corrected question asks about, or "".
Use the class labels exactly as written.

Question: {question}

Classes found:
{classes}"""

_ROUTE_SCHEMA = {"type": "OBJECT", "properties": {
    "main": {"type": "ARRAY", "items": {"type": "STRING"}},
    "also": {"type": "ARRAY", "items": {"type": "STRING"}},
    "kind": {"type": "STRING", "enum": ["focused", "same_topic", "different_meanings", "not_taught"]},
    "meanings": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
        "class": {"type": "STRING"}, "meaning": {"type": "STRING"}}, "required": ["class", "meaning"]}},
    "corrected": {"type": "STRING"}, "corrected_meaning": {"type": "STRING"},
}, "required": ["main", "also", "kind", "meanings", "corrected", "corrected_meaning"]}


# The answer waits on this call, so give up quickly (the answer then works without it).
@retry(retry=retry_if_exception(_retryable), wait=wait_exponential(min=1, max=3),
       stop=stop_after_attempt(2), reraise=True)
def _route(prompt: str):
    model = config.PICTURE_MODEL  # Flash-Lite: fast and cheap
    return gen_client().models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0, max_output_tokens=600, response_mime_type="application/json",
            response_schema=_ROUTE_SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level="minimal") if model.startswith("gemini-3")
            else types.ThinkingConfig(thinking_budget=0)),
    )


def route_topic(question: str, classes: str, usage=None) -> dict:
    """Which of the found classes teach the question's topic (see ROUTE_PROMPT)."""
    import json
    resp = _route(ROUTE_PROMPT.format(question=question[:1500], classes=classes))
    if usage is not None:
        usage.add_generation(config.PICTURE_MODEL, resp.usage_metadata)
    return json.loads(resp.text or "{}")
