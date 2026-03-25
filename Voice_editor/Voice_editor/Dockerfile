# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Use the full PyTorch image so heavy packages (torch, torchaudio, etc.) install
# quickly from pre-built wheels without any compilation.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS builder

# Build-time system deps needed by some Python packages (e.g. soundfile, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1-dev \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Install all Python dependencies into an isolated venv so we can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Slim base keeps the final image small; we bring in only what we need at runtime.
FROM python:3.11-slim AS runtime

# Runtime-only system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Pull the fully-installed venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p uploads outputs

EXPOSE 8002

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8002"]
