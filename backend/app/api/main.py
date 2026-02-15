from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np
from app.services.llm import generate_answer  # <-- import your LLM helper

# ---------------- Config ----------------
COLLECTION_NAME = "dj_rag"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # best model
# ----------------------------------------

# Initialize FastAPI
app = FastAPI(title="DJ RAG Assistant")

# Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Embedding model
model = SentenceTransformer(EMBED_MODEL_NAME)

# ------------------ Request model ------------------
class QueryRequest(BaseModel):
    query: str

# ------------------ Endpoint ------------------
@app.post("/search")
def search(query_request: QueryRequest):
    query_text = query_request.query

    # 1️⃣ Embed query
    query_emb = model.encode(query_text).tolist()

    # 2️⃣ Search Qdrant
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=TOP_K
    )

    # 3️⃣ Prepare results (keep old format)
    results = []
    context_texts = []  # for LLM context
    for point in search_result.points:
        results.append({
            "score": point.score,
            "text": point.payload["text"],
            "filename": point.payload["filename"],
            "chunk_index": point.payload["chunk_index"]
        })
        context_texts.append(point.payload["text"])

    # 4️⃣ Generate final answer via LLM
    context_combined = "\n\n".join(context_texts)
    answer = generate_answer(query_text, context_combined)

    # 5️⃣ Return both LLM answer + original retrieved chunks
    return {
        "query": query_text,
        "answer": answer,
        "results": results
    }
