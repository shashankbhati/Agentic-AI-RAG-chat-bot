# AI Chatbot (RAG) – FastAPI + Qdrant + Ollama

This project implements a Retrieval-Augmented Generation (RAG) chatbot that:

* Processes PDF documents
* Stores embeddings in Qdrant
* Uses Ollama (local LLM) for response generation
* Exposes a FastAPI backend
* Connects to a WordPress frontend chatbot widget
* Is publicly accessible via ngrok (for development)

---

# Architecture Overview

Frontend (WordPress Chat Widget)
→ FastAPI Backend (`/chat`)
→ Qdrant (vector search)
→ Ollama (LLM response generation)

---

# Core Technologies

* FastAPI – Backend API
* Qdrant – Vector database for embeddings
* Ollama – Local LLM runtime
* ngrok – Public tunnel for local development
* WordPress – Frontend integration

---

# Project Structure

```
project-root/
│
├── app/
│   ├── api/
│   │   └── main.py
│   ├── services/
│   │   ├── rag_service.py
│   │   └── embedding_service.py
│
├── data/
│   ├── raw/          # Original PDFs
│   └── processed/    # Extracted JSON data
│
├── qdrant_load.py
├── requirements.txt
└── README.md
```

---

# Python Files Explanation

## `main.py`

**Location:** `app/api/main.py`

Responsible for:

* Initializing FastAPI
* Configuring CORS middleware
* Defining `/chat` endpoint
* Handling incoming user queries from frontend
* Returning generated responses

Key features:

* POST `/chat`
* Accepts JSON:

  ```json
  { "query": "User question" }
  ```
* Returns:

  ```json
  { "answer": "Generated response" }
  ```

---

## `rag_service.py`

Handles Retrieval-Augmented Generation logic:

1. Receives user query
2. Generates embedding
3. Searches Qdrant for relevant documents
4. Builds context prompt
5. Sends prompt to Ollama
6. Returns final LLM response

This is the core intelligence layer.

---

## `embedding_service.py`

Responsible for:

* Creating embeddings for text
* Communicating with embedding model
* Formatting vectors for storage

Used during:

* Document ingestion
* Query embedding

---

## `qdrant_load.py`

One-time script used to:

* Read processed documents
* Generate embeddings
* Store vectors inside Qdrant collection

Run after:

* Adding new PDFs
* Rebuilding vector database

---

# Data Pipeline

### 1. Add PDFs

Place PDF files in:

```
data/raw/
```

### 2. Process PDFs

Extract text → Save structured JSON in:

```
data/processed/
```

### 3. Load into Qdrant

Run:

```bash
python qdrant_load.py
```

This stores embeddings into Qdrant.

---

# Running the Backend

### Start Qdrant

Docker or local instance.

### Start Ollama

Ensure LLM model is available.

### Start FastAPI

```bash
uvicorn app.api.main:app --reload --port 8000
```

Swagger available at:

```
http://localhost:8000/docs
```

---

# Public Access (Development)

Expose backend with ngrok:

```bash
ngrok http 8000
```

Use generated URL in frontend JavaScript:

```
https://random-id.ngrok-free.dev/chat
```

Note: Free ngrok URLs change after restart.

---

# WordPress Integration

A custom HTML block is used to:

* Render chatbot UI
* Send POST requests to `/chat`
* Display LLM responses

Important:

* CORS enabled in FastAPI
* `ngrok-skip-browser-warning` header added to fetch

---

# Features

* Retrieval-Augmented Generation
* Local LLM (no OpenAI dependency)
* Vector similarity search
* WordPress-ready frontend
* Swagger API testing
* Fully local backend with optional public tunneling

---

# Development Notes

* If ngrok restarts → update frontend URL
* If Qdrant is not persistent → reload embeddings
* CORS must be enabled for website communication

---

# Future Improvements

* Deploy backend to VPS (remove ngrok dependency)
* Add conversation memory
* Add authentication
* Add streaming responses
* Improve prompt engineering

---


