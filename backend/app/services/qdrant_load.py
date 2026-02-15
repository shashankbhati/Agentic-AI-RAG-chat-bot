from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
from qdrant_client.http.models import VectorParams, Distance


# ---- PATHS ----
CHUNK_FILE = Path("../../../data/chunks/all_chunks.json")
EMBED_FILE = Path("../../../data/embeddings/bge_all_embeddings.npy")  # best model
COLLECTION_NAME = "dj_rag"

# ---- Connect to local Qdrant ----
client = QdrantClient(url="http://localhost:6333")

# ---- Load chunks and embeddings ----
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(EMBED_FILE)

print(f"Loaded {len(chunks)} chunks and embeddings of shape {embeddings.shape}")

# ---- Create collection if not exists ----
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created")
    

# ---- Upload in batches ----
BATCH_SIZE = 64

for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading to Qdrant"):
    batch_chunks = chunks[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            {
                "id": int(i+j),
                "vector": batch_embeddings[j].tolist(),
                "payload": {
                    "text": batch_chunks[j]["text"],
                    "filename": batch_chunks[j]["source"],
                    "chunk_index": batch_chunks[j]["chunk_id"]
                }
            }
            for j in range(len(batch_chunks))
        ]
    )

print("\n✅ All chunks uploaded to Qdrant")
