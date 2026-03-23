#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# RAG Chatbot Agent — One-Click Installer (Linux / macOS)
# ═══════════════════════════════════════════════════════════════════════════════
set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

print_step() { echo -e "\n${BOLD}${GREEN}▶ $1${RESET}"; }
print_warn() { echo -e "${YELLOW}⚠ $1${RESET}"; }
print_err()  { echo -e "${RED}✖ $1${RESET}"; exit 1; }

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║        RAG Chatbot Agent — Installer              ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# ── Check Docker ──────────────────────────────────────────────────────────────
print_step "Checking Docker"
command -v docker &>/dev/null || print_err "Docker is not installed. Install it from https://docker.com and re-run this script."
docker info &>/dev/null        || print_err "Docker is not running. Start Docker Desktop (or the Docker daemon) and try again."
command -v docker &>/dev/null && docker compose version &>/dev/null || \
    print_err "Docker Compose plugin not found. Update Docker Desktop or run: apt-get install docker-compose-plugin"
echo "Docker OK"

# ── Create .env if missing ─────────────────────────────────────────────────────
print_step "Setting up configuration"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        # Generate a random JWT secret
        JWT=$(openssl rand -hex 32 2>/dev/null || head -c 32 /dev/urandom | xxd -p | tr -d '\n')
        sed -i.bak "s|change-me-to-a-long-random-string|${JWT}|g" .env
        rm -f .env.bak
        echo ".env created with a random JWT secret."
        echo ""
        print_warn "Edit .env now if you want to switch to OpenAI instead of Ollama."
        read -rp "Press ENTER to continue with defaults, or Ctrl-C to edit .env first… "
    else
        print_warn ".env.example not found. Creating minimal .env"
        JWT=$(openssl rand -hex 32 2>/dev/null || echo "change-me-please")
        cat > .env <<EOF
LLM_PROVIDER=ollama
OLLAMA_MODEL=phi3:mini
JWT_SECRET=${JWT}
ALLOWED_ORIGINS=http://localhost:8000
EOF
    fi
else
    echo ".env already exists — using existing config."
fi

# ── Pull images and start ─────────────────────────────────────────────────────
print_step "Pulling Docker images (this may take a few minutes on first run)"
docker compose pull

print_step "Starting containers"
docker compose up -d

# ── Wait for health check ─────────────────────────────────────────────────────
print_step "Waiting for the app to be ready"
MAX=30
COUNT=0
until curl -sf http://localhost:8000/api/v1/health &>/dev/null; do
    COUNT=$((COUNT + 1))
    if [ "$COUNT" -ge "$MAX" ]; then
        print_warn "App did not become healthy in time. Check logs: docker compose logs rag-chatbot"
        break
    fi
    printf "."
    sleep 2
done
echo ""

# ── Pull Ollama model (skip if using OpenAI) ──────────────────────────────────
LLM_PROVIDER=$(grep "^LLM_PROVIDER" .env 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo "ollama")
if [ "$LLM_PROVIDER" = "ollama" ]; then
    print_step "Downloading Ollama model (phi3:mini, ~2 GB — runs once)"
    OLLAMA_MODEL=$(grep "^OLLAMA_MODEL" .env 2>/dev/null | cut -d= -f2 | tr -d ' "' || echo "phi3:mini")
    docker exec rag_ollama ollama pull "$OLLAMA_MODEL" || \
        print_warn "Model pull failed. Run manually: docker exec rag_ollama ollama pull ${OLLAMA_MODEL}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  ✓ RAG Chatbot is running!                        ║"
echo "║                                                   ║"
echo "║  Open:  http://localhost:8000                     ║"
echo "║  API:   http://localhost:8000/docs                ║"
echo "║                                                   ║"
echo "║  Stop:  docker compose down                       ║"
echo "║  Logs:  docker compose logs -f                    ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
