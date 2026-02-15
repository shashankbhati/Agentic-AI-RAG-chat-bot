from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

PROCESSED_FOLDER = Path("../../../data/processed")
CHUNK_FOLDER = Path("../../../data/chunks")
EMBED_FOLDER = Path("../../../data/embeddings")

CHUNK_FOLDER.mkdir(exist_ok=True)
EMBED_FOLDER.mkdir(exist_ok=True)

# -------- CONFIG --------
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # overlap between chunks
EMBED_MODEL_NAME = "all-mpnet-base-v2"  # strong local model
# ------------------------


def semantic_chunk(text: str):
    """
    Simple sliding-window chunking with overlap.
    Later we can upgrade to true semantic splitting.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def process_document(file_path: Path, model):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = data["text"]
    metadata = data["metadata"]

    # ---- Chunking ----
    chunks = semantic_chunk(text)

    chunk_records = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        record = {
            "chunk_id": i,
            "text": chunk,
            "source": metadata["filename"],
            "version": metadata["version"],
        }
        chunk_records.append(record)

    # ---- Embeddings ----
    emb = model.encode([c["text"] for c in chunk_records])
    embeddings = emb.tolist()

    return chunk_records, embeddings


def run_pipeline():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    all_chunks = []
    all_embeddings = []

    files = list(PROCESSED_FOLDER.glob("*.json"))

    if not files:
        print("⚠️ No processed files found.")
        return

    for file in tqdm(files, desc="Processing docs"):
        chunks, embeds = process_document(file, model)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeds)

        # Save per-document chunks
        with open(CHUNK_FOLDER / f"{file.stem}_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # Save per-document embeddings
        np.save(EMBED_FOLDER / f"{file.stem}_embeddings.npy", np.array(embeds))

    # ---- Save global merged files (useful for vector DB) ----
    with open(CHUNK_FOLDER / "all_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    np.save(EMBED_FOLDER / "all_embeddings.npy", np.array(all_embeddings))

    print("\n✅ Chunking + embedding complete.")
    print(f"Total chunks created: {len(all_chunks)}")


if __name__ == "__main__":
    run_pipeline()
