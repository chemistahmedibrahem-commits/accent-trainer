# GPU image by default; build with --build-arg BASE=python:3.12-slim for CPU-only.
ARG BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY accent_engine ./accent_engine
COPY app ./app
COPY trainer ./trainer
RUN pip install --no-cache-dir ".[ml,speech,postgres,worker]"
# bake the default models into the image so the first request isn't a multi-GB download
ARG AT_ACOUSTIC_MODEL_ID=slplab/wav2vec2-large-robust-L2-english-phoneme-recognition
ENV AT_ACOUSTIC_MODEL_ID=${AT_ACOUSTIC_MODEL_ID}
RUN python -c "from huggingface_hub import snapshot_download as s; import os; \
s(os.environ['AT_ACOUSTIC_MODEL_ID']); s('openai/whisper-base.en'); \
s('rhasspy/piper-voices', allow_patterns=['en/en_US/lessac/medium/*'])"
COPY scripts ./scripts
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
