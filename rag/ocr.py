"""Google Cloud Vision OCR (DOCUMENT_TEXT_DETECTION) for slide images and screenshots."""
from functools import lru_cache

from google.api_core import exceptions as gexc
from google.cloud import vision
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import config

BATCH = 8  # Vision allows 16 images per call; 8 keeps request size well under the limit


@lru_cache
def client() -> vision.ImageAnnotatorClient:
    return vision.ImageAnnotatorClient(client_options={"quota_project_id": config.GCP_PROJECT})


@retry(retry=retry_if_exception_type((gexc.ResourceExhausted, gexc.ServiceUnavailable, gexc.DeadlineExceeded)),
       wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6), reraise=True)
def _annotate(images: list[bytes]) -> list[str]:
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    resp = client().batch_annotate_images(requests=[
        vision.AnnotateImageRequest(image=vision.Image(content=img), features=[feature],
                                    image_context=vision.ImageContext(language_hints=["en", "hi"]))
        for img in images
    ])
    out = []
    for r in resp.responses:
        if r.error.message:
            print(f"    ! Vision OCR error: {r.error.message}")
            out.append("")
        else:
            out.append(r.full_text_annotation.text.strip())
    return out


def ocr_images(images: list[bytes]) -> list[str]:
    """OCR text for each image (same order). Empty string where nothing was found."""
    out = []
    for i in range(0, len(images), BATCH):
        out += _annotate(images[i:i + BATCH])
    return out
