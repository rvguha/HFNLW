# Hugging Face Spaces (Docker SDK) serves on port 7860 and runs as uid 1000.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASKHUB_HOST=0.0.0.0 \
    ASKHUB_PORT=7860 \
    ASKHUB_DATA_DIR=/data

RUN useradd -m -u 1000 app
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY sources.yaml ./

# Writable for the embedding cache and any refreshed corpus.
RUN mkdir -p /data && chown -R app:app /data /app
USER app

EXPOSE 7860
CMD ["ask-hf-hub"]
