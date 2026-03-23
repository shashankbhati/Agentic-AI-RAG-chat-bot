# ═══════════════════════════════════════════════════════════════════════════════
# RAG Chatbot Agent — One-Click Installer (Windows PowerShell)
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1
# ═══════════════════════════════════════════════════════════════════════════════

$ErrorActionPreference = "Stop"

function Write-Step($msg)  { Write-Host "`n>> $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "!! $msg"  -ForegroundColor Yellow }
function Fail($msg)        { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "     RAG Chatbot Agent -- Windows Installer"       -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# ── Check Docker ──────────────────────────────────────────────────────────────
Write-Step "Checking Docker"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail "Docker not found. Install Docker Desktop from https://docker.com and re-run."
}
docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Fail "Docker is not running. Start Docker Desktop and try again."
}
Write-Host "Docker OK" -ForegroundColor Green

# ── Create .env if missing ────────────────────────────────────────────────────
Write-Step "Setting up configuration"
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        # Generate random JWT secret using .NET
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $jwt = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
        (Get-Content ".env") -replace "change-me-to-a-long-random-string", $jwt | Set-Content ".env"
        Write-Host ".env created with a random JWT secret."
        Write-Warn "Edit .env if you want to use OpenAI instead of Ollama, then press ENTER."
        Read-Host "Press ENTER to continue with defaults"
    } else {
        Write-Warn ".env.example not found. Creating minimal .env"
        $bytes = New-Object byte[] 32
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $jwt = ($bytes | ForEach-Object { $_.ToString("x2") }) -join ""
        @"
LLM_PROVIDER=ollama
OLLAMA_MODEL=phi3:mini
JWT_SECRET=$jwt
ALLOWED_ORIGINS=http://localhost:8000
"@ | Set-Content ".env"
    }
} else {
    Write-Host ".env already exists - using existing config."
}

# ── Pull images and start ─────────────────────────────────────────────────────
Write-Step "Pulling Docker images (this may take a few minutes on first run)"
docker compose pull
if ($LASTEXITCODE -ne 0) { Fail "docker compose pull failed." }

Write-Step "Starting containers"
docker compose up -d
if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed." }

# ── Wait for health check ─────────────────────────────────────────────────────
Write-Step "Waiting for the app to be ready"
$max = 30; $count = 0
do {
    Start-Sleep -Seconds 2
    $count++
    Write-Host -NoNewline "."
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { break }
    } catch {}
} while ($count -lt $max)
Write-Host ""

# ── Pull Ollama model ─────────────────────────────────────────────────────────
$envContent = Get-Content ".env" -ErrorAction SilentlyContinue
$providerLine = $envContent | Where-Object { $_ -match "^LLM_PROVIDER" } | Select-Object -First 1
if ($providerLine -notmatch "openai") {
    Write-Step "Downloading Ollama model (phi3:mini, ~2 GB - runs once)"
    $modelLine = $envContent | Where-Object { $_ -match "^OLLAMA_MODEL" } | Select-Object -First 1
    $model = if ($modelLine) { ($modelLine -split "=")[1].Trim() } else { "phi3:mini" }
    docker exec rag_ollama ollama pull $model
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Model pull failed. Run manually: docker exec rag_ollama ollama pull $model"
    }
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=================================================" -ForegroundColor Green
Write-Host "  RAG Chatbot is running!"                        -ForegroundColor Green
Write-Host ""
Write-Host "  Open:  http://localhost:8000"
Write-Host "  API:   http://localhost:8000/docs"
Write-Host ""
Write-Host "  Stop:  docker compose down"
Write-Host "  Logs:  docker compose logs -f"
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

# Auto-open browser
Start-Process "http://localhost:8000"
