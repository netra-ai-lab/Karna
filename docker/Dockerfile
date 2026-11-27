# syntax=docker/dockerfile:1.7
# Karna: the web app + REST API with the model weights baked in (runs offline).
# After Netra-OCR's docker/Dockerfile (Netra Lab kit).
#
#   docker build -f docker/Dockerfile -t karna .
#   docker build -f docker/Dockerfile --build-arg MODELS=karna-base -t karna:base .   # smaller
#   docker run -p 8000:8000 -v karna-data:/data karna
#   then open http://localhost:8000
#
# CPU only, and no PyTorch: the models run on ONNX Runtime (karna/inference). The language model
# for beam search (Netra-Lab/karna-lm, CC BY-SA 4.0) is baked in next to them, in /models/karna-lm.

ARG PYTHON_VERSION=3.12

# ── base: shared by the build and runtime stages ─────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

# ── build: virtualenv with netra-karna[server,cpu], pruned ───────────────────
FROM base AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY karna ./karna
RUN pip install ".[server,cpu]"

# Prune what the app never needs (checked by docker/smoke_test.py below).
RUN find /opt/venv -depth -name __pycache__ -type d -exec rm -rf {} + && \
    pip uninstall -y pip

# ── weights: the model folders, fetched once at build time ───────────────────
# Then run every model and route once against the pruned environment, offline:
# a prune that removed something needed fails here, not on the first upload.
FROM build AS weights
ARG MODELS="karna-base karna-large"
COPY docker/fetch_weights.py docker/smoke_test.py /tmp/
COPY tests/golden /tmp/golden
RUN HF_HOME=/tmp/hf python /tmp/fetch_weights.py /models $MODELS && rm -rf /tmp/hf && \
    cd /tmp && KARNA_BUNDLES=/models HF_HUB_OFFLINE=1 python /tmp/smoke_test.py /tmp/golden

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
LABEL org.opencontainers.image.title="Karna" \
      org.opencontainers.image.description="Khmer and English speech recognition: web app and REST API" \
      org.opencontainers.image.source="https://github.com/netra-ai-lab/Karna" \
      org.opencontainers.image.licenses="Apache-2.0 AND OFL-1.1 AND CC-BY-SA-4.0"

RUN useradd --create-home --uid 10001 netra \
 && mkdir -p /data \
 && chown -R netra /data
COPY --from=build /opt/venv /opt/venv
COPY --from=weights /models /models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KARNA_BUNDLES=/models \
    HF_HUB_OFFLINE=1 \
    KARNA_DATA_DIR=/data \
    KARNA_CACHE_DIR=/data/cache \
    KARNA_HOST=0.0.0.0 \
    KARNA_PORT=8000 \
    KARNA_DEVICE=cpu

USER netra
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('KARNA_PORT','8000'), timeout=4)"
CMD ["karna", "serve"]
