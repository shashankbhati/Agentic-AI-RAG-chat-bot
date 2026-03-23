#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh  —  One-command deploy for RAG Chatbot on a fresh Ubuntu server
# Usage:  bash deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Check tools ────────────────────────────────────────────────────────────
info "Checking required tools..."
command -v docker  >/dev/null || error "Docker not found. Install: curl -fsSL https://get.docker.com | sh"
command -v node    >/dev/null || error "Node.js not found. Install: https://nodejs.org"
command -v npm     >/dev/null || error "npm not found"
command -v python3 >/dev/null || error "Python3 not found"

info "Docker: $(docker --version)"
info "Node:   $(node --version)"
info "npm:    $(npm --version)"

# ── 2. Create .env if missing ─────────────────────────────────────────────────
ENV_FILE="$REPO_DIR/backend/.env"
if [ ! -f "$ENV_FILE" ]; then
  warn ".env not found — creating from .env.example"
  cp "$REPO_DIR/backend/.env.example" "$ENV_FILE"

  # Auto-generate a strong JWT secret
  JWT=$(python3 -c "import secrets; print(secrets.token_hex(32))")
  sed -i "s/change-me-in-production-please/$JWT/" "$ENV_FILE"
  warn "Generated JWT_SECRET. Edit $ENV_FILE to set ALLOWED_ORIGINS and other values."
else
  info ".env found"
fi

# ── 3. Build frontend ─────────────────────────────────────────────────────────
info "Building frontend..."
cd "$REPO_DIR/frontend"
npm install --silent
npm run build
info "Frontend built → frontend/dist/"

# ── 4. Build widget ───────────────────────────────────────────────────────────
info "Building widget..."
cd "$REPO_DIR/widget"
npm install --silent
npm run build
info "Widget built → widget/dist/rag-widget.iife.js"

# ── 5. Start Docker stack ─────────────────────────────────────────────────────
info "Starting Docker Compose stack..."
cd "$REPO_DIR"
docker compose pull --quiet
docker compose up -d --build

# ── 6. Wait for backend health ────────────────────────────────────────────────
info "Waiting for backend to be healthy..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/api/v1/health >/dev/null 2>&1; then
    info "Backend is healthy!"
    break
  fi
  if [ "$i" -eq 30 ]; then
    error "Backend did not become healthy in time. Run: docker compose logs backend"
  fi
  sleep 3
done

# ── 7. Pull Ollama model if not present ───────────────────────────────────────
OLLAMA_MODEL="${OLLAMA_MODEL:-phi3:mini}"
info "Checking Ollama model ($OLLAMA_MODEL)..."
if ! docker exec rag_ollama ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL"; then
  info "Pulling $OLLAMA_MODEL (this may take a few minutes)..."
  docker exec rag_ollama ollama pull "$OLLAMA_MODEL"
else
  info "Model $OLLAMA_MODEL already present"
fi

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Deployment complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  App:        http://localhost:8000"
echo "  API docs:   http://localhost:8000/docs"
echo "  Widget:     http://localhost:8000/rag-widget.js"
echo "  Health:     http://localhost:8000/api/v1/health"
echo ""
echo "  Next steps:"
echo "  1. Open http://localhost:8000/register — create your account"
echo "  2. Upload PDFs in the dashboard"
echo "  3. Copy embed code from Dashboard → Embed Code tab"
echo "  4. Paste the <script> tag on any website"
echo ""
echo "  To view logs:  docker compose logs -f"
echo "  To stop:       docker compose down"
echo ""
