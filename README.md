# RAG Chatbot

Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot. Upload PDFs and ask questions against them using local or cloud LLMs.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────────┐
│  React UI   │────▶│  FastAPI Backend │────▶│  Qdrant (VDB)  │
│  (Vite)     │◀────│  /api/v1         │     └────────────────┘
└─────────────┘     │                  │     ┌────────────────┐
                    │                  │────▶│  Ollama / OAI  │
                    └──────────────────┘     └────────────────┘
```

**Stack:** FastAPI · Qdrant · Sentence-Transformers (BGE) · Ollama (phi3:mini) · React · Vite

## Quick Start

### Option 1 — Docker Compose (recommended)

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

Then open **http://localhost:5173**.

Pull the LLM model once Ollama is running:
```bash
docker exec -it rag_ollama ollama pull phi3:mini
```

### Option 2 — Local development

**Prerequisites:** Python 3.11+, Node 20+, [Qdrant](https://qdrant.tech/documentation/quick-start/), [Ollama](https://ollama.com/)

```bash
# 1. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 2. Start Ollama and pull model
ollama serve
ollama pull phi3:mini

# 3. Backend
cd backend
cp .env.example .env        # edit as needed
pip install -r requirements.txt
uvicorn app.main:app --reload

# 4. Frontend
cd ../frontend
npm install
npm run dev
```

## API Reference

Base URL: `http://localhost:8000/api/v1`
Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + status |
| `POST` | `/chat` | Chat (non-streaming) |
| `POST` | `/chat/stream` | Chat with SSE streaming |
| `GET` | `/chat/{session_id}/history` | Get conversation history |
| `DELETE` | `/chat/{session_id}` | Clear conversation |
| `POST` | `/search` | Semantic search only |
| `POST` | `/documents/upload` | Upload & index a PDF |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{filename}` | Delete a document |

### Chat example

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the policy number?"}'
```

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@document.pdf"
```

## Configuration

All settings are controlled via environment variables (see `backend/.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `COLLECTION_NAME` | `rag_collection` | Qdrant collection name |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Sentence-Transformers model |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `phi3:mini` | Ollama model name |
| `OPENAI_API_KEY` | _(empty)_ | OpenAI key (if using OpenAI) |
| `TOP_K` | `3` | Chunks retrieved per query |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

## Switching to OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Running Tests

```bash
cd backend
pytest tests/ -v
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # App factory, CORS, rate limiting
│   │   ├── config.py            # Settings from env vars
│   │   ├── models/schemas.py    # Pydantic request/response models
│   │   ├── api/routes/          # Route handlers
│   │   │   ├── chat.py          # Chat + streaming + history
│   │   │   ├── documents.py     # Upload / list / delete
│   │   │   ├── search.py        # Semantic search
│   │   │   └── health.py        # Health check
│   │   └── services/            # Business logic
│   │       ├── embedder.py      # Singleton embedding model
│   │       ├── retriever.py     # Qdrant search
│   │       ├── llm_service.py   # Ollama + OpenAI abstraction
│   │       ├── ingest_service.py # PDF → chunks → Qdrant
│   │       └── session_store.py # In-memory conversation history
│   ├── tests/                   # pytest suite
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Root component
│   │   ├── components/          # Header, ChatWindow, DocumentPanel, MessageBubble
│   │   └── services/api.js      # Typed API client
│   ├── Dockerfile
│   └── nginx.conf               # Nginx with SSE + API proxy
├── docker-compose.yml
├── .github/workflows/ci.yml     # Lint + test + Docker build CI
└── README.md
```
