#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# scripts/build-and-push.sh
# Build the Docker image and push to Docker Hub, then package the release zip.
#
# Usage:
#   ./scripts/build-and-push.sh <dockerhub-username> [version]
#
# Example:
#   ./scripts/build-and-push.sh shashankbhati 1.0.0
#
# What it does:
#   1. Builds the multi-stage Docker image (frontend + widget + Python)
#   2. Tags it as :latest and :<version>
#   3. Pushes both tags to Docker Hub
#   4. Updates dist/docker-compose.yml with the real image name
#   5. Creates a release zip: rag-chatbot-agent-<version>.zip
# ═══════════════════════════════════════════════════════════════════════════════
set -e

BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

print_step() { echo -e "\n${BOLD}${GREEN}▶ $1${RESET}"; }
print_warn() { echo -e "${YELLOW}⚠ $1${RESET}"; }

# ── Args ──────────────────────────────────────────────────────────────────────
DOCKER_USER="${1:-}"
VERSION="${2:-latest}"

if [ -z "$DOCKER_USER" ]; then
    echo -e "${RED}Usage: $0 <dockerhub-username> [version]${RESET}"
    echo "  Example: $0 shashankbhati 1.0.0"
    exit 1
fi

IMAGE="${DOCKER_USER}/rag-chatbot"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║   RAG Chatbot — Build & Publish Script            ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
echo "  Docker Hub user : ${DOCKER_USER}"
echo "  Image           : ${IMAGE}"
echo "  Version         : ${VERSION}"
echo ""

# ── Check Docker login ────────────────────────────────────────────────────────
print_step "Checking Docker Hub login"
if ! docker info 2>/dev/null | grep -q "Username"; then
    print_warn "Not logged in to Docker Hub. Running: docker login"
    docker login
fi

# ── Build ─────────────────────────────────────────────────────────────────────
print_step "Building Docker image (this takes ~5 minutes first time)"
cd "$REPO_ROOT"
docker build \
    -f backend/Dockerfile \
    -t "${IMAGE}:latest" \
    -t "${IMAGE}:${VERSION}" \
    .

echo "Build complete: ${IMAGE}:${VERSION}"

# ── Push ──────────────────────────────────────────────────────────────────────
print_step "Pushing to Docker Hub"
docker push "${IMAGE}:latest"
docker push "${IMAGE}:${VERSION}"
echo "Pushed: ${IMAGE}:latest and ${IMAGE}:${VERSION}"

# ── Update dist/docker-compose.yml ───────────────────────────────────────────
print_step "Updating dist/docker-compose.yml"
DIST_COMPOSE="${REPO_ROOT}/dist/docker-compose.yml"
if [ -f "$DIST_COMPOSE" ]; then
    sed -i.bak \
        "s|image: DOCKER_USERNAME/rag-chatbot:latest|image: ${IMAGE}:${VERSION}|g" \
        "$DIST_COMPOSE"
    rm -f "${DIST_COMPOSE}.bak"
    echo "Updated image reference to ${IMAGE}:${VERSION}"
fi

# ── Package release zip ───────────────────────────────────────────────────────
print_step "Creating release package"
ZIP_NAME="rag-chatbot-agent-${VERSION}.zip"
cd "${REPO_ROOT}/dist"
zip -j "${REPO_ROOT}/${ZIP_NAME}" \
    docker-compose.yml \
    .env.example \
    install.sh \
    install.ps1 \
    README.txt

echo "Release zip created: ${REPO_ROOT}/${ZIP_NAME}"

# ── Restore docker-compose.yml placeholder ────────────────────────────────────
sed -i.bak \
    "s|image: ${IMAGE}:${VERSION}|image: DOCKER_USERNAME/rag-chatbot:latest|g" \
    "$DIST_COMPOSE"
rm -f "${DIST_COMPOSE}.bak"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Done!                                            ║"
echo "║                                                   ║"
echo "  Upload to your website: ${ZIP_NAME}"
echo "  Docker Hub image ready: ${IMAGE}:${VERSION}"
echo "║                                                   ║"
echo "║  Users can now run:                               ║"
echo "║    Linux/Mac:  ./install.sh                       ║"
echo "║    Windows:    Right-click install.ps1 → Run      ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""
