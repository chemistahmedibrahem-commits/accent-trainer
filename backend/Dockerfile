# GPU image by default; build with --build-arg BASE=python:3.12-slim for CPU-only.
ARG BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY accent_engine ./accent_engine
COPY app ./app
RUN pip install --no-cache-dir ".[ml,postgres,worker]"
COPY scripts ./scripts
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
