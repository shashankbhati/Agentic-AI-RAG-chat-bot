from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

# ---------------- CONFIG ----------------
COLLECTION_NAME = "dj_rag"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "phi3:mini"
# ----------------------------------------

# Initialize FastAPI
app = FastAPI(title="DJ AI Assistant")

# Qdrant client
client = QdrantClient(url="http://localhost:6333")

# Embedding model
model = SentenceTransformer(EMBED_MODEL_NAME)

# Request schema
class QueryRequest(BaseModel):
    query: str


# ---------------- ROOT ----------------
@app.get("/")
def read_root():
    return {"message": "DJ AI Assistant backend running"}


# ---------------- SEARCH (Retrieval Only) ----------------
@app.post("/search")
def search(query_request: QueryRequest):
    query_text = query_request.query

    # Embed query
    query_emb = model.encode(query_text).tolist()

    # Query Qdrant
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=TOP_K
    )

    results = []
    for point in search_result.points:
        results.append({
            "score": point.score,
            "text": point.payload["text"],
            "filename": point.payload["filename"],
            "chunk_index": point.payload["chunk_index"]
        })

    return {
        "query": query_text,
        "results": results
    }


# ---------------- CHAT (Full RAG) ----------------
@app.post("/chat")
def chat(query_request: QueryRequest):
    query_text = query_request.query

    # 1️⃣ Embed query
    query_emb = model.encode(query_text).tolist()

    # 2️⃣ Retrieve top-k chunks
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_emb,
        limit=TOP_K
    )

    contexts = []
    for point in search_result.points:
        contexts.append(point.payload["text"])

    context_text = "\n\n".join(contexts)

    # 3️⃣ Build RAG prompt
    prompt = f"""
You are a document assistant.
Answer the question strictly using the provided context.
If the answer is not found in the context, say:
"Not found in documents."

Context:
{context_text}

Question:
{query_text}

Answer:
"""

    # 4️⃣ Call Ollama (phi3:mini)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You answer strictly from provided context."},
            {"role": "user", "content": prompt}
        ],
        options={
            "temperature": 0
        }
    )

    answer = response["message"]["content"]

    return {
        "query": query_text,
        "answer": answer,
        "retrieved_chunks": contexts
    }