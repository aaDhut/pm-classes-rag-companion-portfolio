"""Entry point: the Streamlit app plus the /voice audio route.

Run:  uv run streamlit run server.py   (app.py on its own still works, without Gemini voice)
"""
import os
import queue

import streamlit as st
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from rag import images, netfix, voice

if not os.getenv("K_SERVICE"):  # local runs only: this Mac's DNS sometimes fails the Qdrant lookup
    netfix.install()


async def voice_route(request):
    chat_id = request.path_params["chat_id"]
    idx = request.path_params["idx"]
    if not voice.verify(chat_id, idx, request.query_params.get("sig", "")):
        return Response("Forbidden", status_code=403)
    found = await run_in_threadpool(voice.audio_for, chat_id, idx)
    if found is None:
        return Response("Not found", status_code=404)
    kind, data = found
    if kind == "mp3":
        return Response(data, media_type="audio/mpeg",
                        headers={"Cache-Control": "private, max-age=86400"})
    # Raw 16-bit mono PCM at 24 kHz, played by the page as it arrives.
    job, q = data

    def next_chunk():
        try:
            return q.get(timeout=1)
        except queue.Empty:
            return b""

    async def stream():
        # When the page stops or closes, Starlette cancels this; tell the job so it stops
        # making (and paying for) parts nobody will hear.
        try:
            while (c := await run_in_threadpool(next_chunk)) is not None:
                if c:
                    yield c
        finally:
            job.unsubscribe(q)

    return StreamingResponse(stream(), media_type="application/octet-stream",
                             headers={"X-Sample-Rate": str(voice.RATE), "Cache-Control": "no-store"})


async def image_route(request):
    name = request.path_params["name"]
    if not images.verify(name, request.query_params.get("sig", "")):
        return Response("Forbidden", status_code=403)
    data = await run_in_threadpool(images.read, name)
    if data is None:
        return Response("Not found", status_code=404)
    # Names carry a hash of the picture, so a name never changes content.
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=31536000, immutable"})


app = st.App("app.py", routes=[Route("/voice/{chat_id:str}/{idx:int}", voice_route),
                               Route("/img/{name:str}", image_route)])
