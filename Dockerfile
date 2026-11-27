# Multi-stage production Dockerfile for End-to-End Advanced RAG with LLMOps
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment settings
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

# Install Python dependencies first for optimal Docker layer caching
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Pre-download and cache FastEmbed model weights into the container image
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Copy application source code and web assets
COPY multi_doc_chat ./multi_doc_chat
COPY static ./static
COPY templates ./templates
COPY main.py ./
COPY data/agentic_ai.txt data/NIPS-2017-attention-is-all-you-need-Paper.pdf ./data/

# Ensure runtime directories exist
RUN mkdir -p /app/data /app/logs

# Expose FastAPI service port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Start Uvicorn application server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
