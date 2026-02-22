from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "dj_rag"
TOP_K = 3
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

app = FastAPI(title="DJ AI Assistant")

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer(EMBED_MODEL_NAME)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "DJ AI Assistant backend running"}

@app.post("/search")
def search(query_request: QueryRequest):
    query_text = query_request.query

    query_emb = model.encode(query_text).tolist()

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

    return {"query": query_text, "results": results}
