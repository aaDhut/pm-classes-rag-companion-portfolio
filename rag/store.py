"""Qdrant collection: one dense (Gemini) + one sparse (BM25) vector per chunk, hybrid search with RRF."""
import re
import uuid
from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from . import config, llm

DENSE, SPARSE = "dense", "bm25"


@lru_cache
def qdrant() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, port=6333, timeout=60)


@lru_cache
def bm25() -> SparseTextEmbedding:
    return SparseTextEmbedding(config.SPARSE_MODEL, specific_model_path=config.BM25_MODEL_PATH)


def ensure_collection():
    q = qdrant()
    if not q.collection_exists(config.COLLECTION):
        q.create_collection(
            config.COLLECTION,
            vectors_config={DENSE: models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    # Re-creating an existing index is a no-op, so new indexes reach old collections too.
    for field, schema in [("class_no", models.PayloadSchemaType.INTEGER),
                          ("class_label", models.PayloadSchemaType.KEYWORD),
                          ("file_id", models.PayloadSchemaType.KEYWORD),
                          ("teacher", models.PayloadSchemaType.KEYWORD),
                          ("has_images", models.PayloadSchemaType.BOOL),
                          ("source", models.PayloadSchemaType.KEYWORD)]:
        q.create_payload_index(config.COLLECTION, field, field_schema=schema)


def delete_file(file_id: str):
    qdrant().delete(config.COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(
        must=[models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id))])))


def delete_summary(file_id: str):
    """Only the class-summary points of one video."""
    qdrant().delete(config.COLLECTION, points_selector=models.FilterSelector(filter=models.Filter(must=[
        models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
        models.FieldCondition(key="source", match=models.MatchValue(value="summary"))])))


def upsert_chunks(chunks: list[dict], batch: int = 64):
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        texts = [c["embed_text"] for c in part]
        dense = llm.embed_documents(texts)
        sparse = list(bm25().embed(texts))
        qdrant().upsert(config.COLLECTION, points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c["key"])),
                vector={DENSE: d, SPARSE: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())},
                payload=c["payload"],
            ) for c, d, s in zip(part, dense, sparse)
        ])


_HAS_IMAGES = models.FieldCondition(key="has_images", match=models.MatchValue(value=True))


_LECTURE = models.FieldCondition(key="source", match=models.MatchAny(any=["lecture", "slide", "summary"]))


def _request(dense: list[float], sparse, flt, limit: int) -> models.QueryRequest:
    return models.QueryRequest(
        prefetch=[
            models.Prefetch(query=dense, using=DENSE, limit=30, filter=flt),
            models.Prefetch(query=sparse, using=SPARSE, limit=30, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit, with_payload=True,
    )


def _query(*requests: models.QueryRequest) -> list[list[dict]]:
    """Runs the searches in one round trip to Qdrant."""
    res = qdrant().query_batch_points(config.COLLECTION, requests=list(requests))
    return [[{"score": p.score, **p.payload} for p in r.points] for r in res]


def search(query: str, class_labels: list[str] | None = None, limit: int = 8,
           usage=None, pictures_only: bool = False, sources: str = "all") -> list[dict]:
    """pictures_only: search just the passages that come with a slide/picture.
    sources: "all", "lectures" (video transcripts, their slides, class summaries) or "notes" (Drive files).

    With "all", notes and lectures are searched separately and take turns, so the many
    short transcript pieces can't crowd the notes out (or the other way round)."""
    must = []
    if class_labels:
        must.append(models.FieldCondition(key="class_label", match=models.MatchAny(any=class_labels)))
    if pictures_only:
        must.append(_HAS_IMAGES)
    s = next(iter(bm25().query_embed(query)))
    sparse = models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())
    dense = llm.embed_query(query, usage)
    notes_flt = models.Filter(must=must, must_not=[_LECTURE])
    lecture_flt = models.Filter(must=must + [_LECTURE])
    if sources == "notes":
        return _query(_request(dense, sparse, notes_flt, limit))[0]
    if sources == "lectures":
        return _query(_request(dense, sparse, lecture_flt, limit))[0]
    notes, lectures = _query(_request(dense, sparse, notes_flt, limit), _request(dense, sparse, lecture_flt, limit))
    out = []
    for pair in zip(notes, lectures):
        out += pair
    longer = notes if len(notes) > len(lectures) else lectures
    out += longer[min(len(notes), len(lectures)):]
    return out[:limit]


_CLASS_NO = re.compile(r"^Class (\d+)\b")
_DAY_MONTH = re.compile(r"(\d{1,2})(?:st|nd|rd|th)?[-\s]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.I)
_MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()


def _class_order(label: str) -> tuple:
    """Classes by number, then the Sunday workshops by the date in their name."""
    if m := _CLASS_NO.match(label):
        return (0, int(m.group(1)), label)
    if m := _DAY_MONTH.search(label):
        return (1, _MONTHS.index(m.group(2).lower()) * 100 + int(m.group(1)), label)
    return (2, 0, label)


def indexed_classes() -> list[str]:
    """Label of every class and workshop in the collection. One facet call on the
    class_label index (about 1 s) instead of reading every point (about 8 s)."""
    res = qdrant().facet(config.COLLECTION, key="class_label", limit=1000)
    return sorted((h.value for h in res.hits), key=_class_order)


def picture_passages(file_ids: list[str]) -> list[dict]:
    """Every passage with a picture in these files (to read a slide's full text when
    only one of its passages was found)."""
    flt = models.Filter(must=[models.FieldCondition(key="file_id", match=models.MatchAny(any=file_ids)),
                              _HAS_IMAGES])
    out, offset = [], None
    while True:
        points, offset = qdrant().scroll(config.COLLECTION, scroll_filter=flt, limit=256, offset=offset,
                                         with_payload=True, with_vectors=False)
        out += [p.payload for p in points]
        if offset is None:
            return out


def set_images(chunks: list[dict], batch: int = 256):
    """Write each chunk's "images" fields onto its existing point, without re-embedding."""
    ops = [models.SetPayloadOperation(set_payload=models.SetPayload(
        payload={"images": c["payload"]["images"], "has_images": c["payload"]["has_images"]},
        points=[str(uuid.uuid5(uuid.NAMESPACE_URL, c["key"]))]))
        for c in chunks]
    for i in range(0, len(ops), batch):
        qdrant().batch_update_points(config.COLLECTION, update_operations=ops[i:i + batch])


def summary_payload(class_label: str) -> dict | None:
    """The stored summary point of one class's lecture video (full_text, link, …); None if none."""
    flt = models.Filter(must=[models.FieldCondition(key="class_label", match=models.MatchValue(value=class_label)),
                              models.FieldCondition(key="source", match=models.MatchValue(value="summary"))])
    points, _ = qdrant().scroll(config.COLLECTION, scroll_filter=flt, limit=64, with_payload=True,
                                with_vectors=False)
    return next((p.payload for p in points if p.payload.get("full_text")), None)


_SUMMARIES: dict = {}


def all_summaries() -> list[dict]:
    """Every class's stored summary point, in class order. One scroll, kept for an hour
    (summaries change only when new classes are added)."""
    import time
    if _SUMMARIES.get("at", 0) > time.time() - 3600:
        return _SUMMARIES["list"]
    flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value="summary"))])
    found, offset = {}, None
    while True:
        points, offset = qdrant().scroll(config.COLLECTION, scroll_filter=flt, limit=256, offset=offset,
                                         with_payload=True, with_vectors=False)
        for pt in points:
            if pt.payload.get("full_text"):
                found.setdefault(pt.payload["class_label"], pt.payload)
        if offset is None:
            break
    _SUMMARIES.update(at=time.time(), list=[found[k] for k in sorted(found, key=_class_order)])
    return _SUMMARIES["list"]


def class_summary(class_label: str) -> str | None:
    p = summary_payload(class_label)
    return p["full_text"] if p else None
