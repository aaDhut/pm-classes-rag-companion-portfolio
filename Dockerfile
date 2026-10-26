FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
# Fail the build (not the live app) if the committed BM25 model can't be loaded offline.
RUN python -c "from rag import store; store.bm25()"

CMD streamlit run server.py --server.port=${PORT:-8080} --server.address=0.0.0.0
