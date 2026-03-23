# ═══════════════════════════════════════════════════════════════════════════════
# Multi-stage build — produces a fully self-contained image
# No pre-built assets needed on the host machine
# ═══════════════════════════════════════════════════════════════════════════════

# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ .
RUN npm run build


# ── Stage 2: Build embeddable widget ──────────────────────────────────────────
FROM node:20-alpine AS widget-builder
WORKDIR /widget
COPY widget/package*.json ./
RUN npm ci --silent
COPY widget/ .
RUN npm run build


# ── Stage 3: Build Python dependencies ────────────────────────────────────────
FROM python:3.11-slim AS py-builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 4: Runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime
WORKDIR /app

# Copy installed Python packages
COPY --from=py-builder /install /usr/local

# Copy built frontend and widget
COPY --from=frontend-builder /frontend/dist ./frontend/dist/
COPY --from=widget-builder   /widget/dist   ./widget/dist/

# Copy backend application code
COPY backend/app/ ./app/

# Non-root user for security
RUN useradd -m -u 1000 appuser \
    && mkdir -p data \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
