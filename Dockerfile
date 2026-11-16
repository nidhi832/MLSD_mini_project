FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (~200 MB instead of several GB of CUDA libraries).
# The GPU build exhausted Docker Desktop's memory and crashed it (see docs/INCIDENT_LOG.md, INC-2026-11-10-01).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so containers never download it at runtime.
# If the download fails (no network), the built-in embedder is used and recorded in the manifest.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    || echo "WARNING: embedding model download failed; built-in embedder will be used"

# Copy project files
COPY . .

# Build the vector index (idempotent: only new or changed chunks are embedded)
RUN python -m src.ingestion.pipeline

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
