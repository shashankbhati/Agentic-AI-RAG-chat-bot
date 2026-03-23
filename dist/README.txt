╔══════════════════════════════════════════════════════════════╗
║              RAG Chatbot Agent — Quick Start                 ║
╚══════════════════════════════════════════════════════════════╝

A self-hosted AI chatbot that answers questions from your PDFs.
No cloud. No subscriptions. Runs on your own machine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIREMENTS
  - Docker Desktop (free): https://www.docker.com/products/docker-desktop
  - 4 GB RAM minimum (8 GB recommended)
  - ~5 GB disk space (for the AI model)
  - Windows 10/11, macOS 12+, or Ubuntu 20.04+

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICK INSTALL

  Windows (PowerShell — run as normal user):
    Right-click install.ps1 → "Run with PowerShell"
    OR open PowerShell and run:
      powershell -ExecutionPolicy Bypass -File install.ps1

  Linux / macOS (Terminal):
    chmod +x install.sh
    ./install.sh

  The installer will:
    1. Check that Docker is running
    2. Create a .env config file with a secure random secret
    3. Pull the Docker images (~500 MB download)
    4. Start all services
    5. Download the AI model (~2 GB, first run only)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AFTER INSTALL

  Open your browser: http://localhost:8000

  1. Register an account (email + password)
  2. Go to Dashboard → upload your PDF files
  3. Start chatting with your documents!

  Embed on your website:
    Copy the snippet from Dashboard → Embed tab.
    Paste it into any webpage's <body> tag.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONFIGURATION (optional)

  Edit the .env file (created by installer) to customise:

  LLM_PROVIDER=ollama        # "ollama" (free, local) or "openai"
  OLLAMA_MODEL=phi3:mini     # any model from ollama.com/library
  OPENAI_API_KEY=            # only if using openai
  JWT_SECRET=...             # auto-generated — do not share
  ALLOWED_ORIGINS=http://localhost:8000

  After editing .env, restart:
    docker compose down && docker compose up -d

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DAILY USAGE

  Start:   docker compose up -d
  Stop:    docker compose down
  Logs:    docker compose logs -f
  Update:  docker compose pull && docker compose up -d

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USING A DIFFERENT AI MODEL

  By default the chatbot uses phi3:mini (fast, ~2 GB).
  To use a larger, smarter model, edit .env:
    OLLAMA_MODEL=llama3.2
  Then restart and the new model downloads automatically.

  For OpenAI (gpt-4o-mini):
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA & PRIVACY

  All data stays on your machine.
  Documents are stored in Docker volumes (not sent anywhere).
  The AI model runs locally via Ollama (unless you choose OpenAI).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TROUBLESHOOTING

  "Docker not running"
    → Start Docker Desktop and wait for the whale icon to be steady.

  "Port 8000 already in use"
    → Change the port in docker-compose.yml: "8080:8000" then restart.

  "Model download stuck"
    → Run: docker exec rag_ollama ollama pull phi3:mini

  Chat gives wrong answers
    → Upload more relevant documents via Dashboard → Documents.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
