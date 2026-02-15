from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import tiktoken

PROCESSED_FOLDER = Path("../../../data/processed")
EMBEDDINGS_FOLDER = Path("../../../data/embeddings")
EMBEDDINGS_FOLDER.mkdir(exist_ok=True)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast, free

# Tokenizer for chunking
tokenizer = tiktoken.get_encoding("cl100k_base")  # same as OpenAI tokenizer

# Chunk size & overlap (tokens)
CHUNK_SIZE = 300
OVERLAP = 50

def chunk_text(text):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += CHUNK_SIZE - OVERLAP  # overlap
    return chunks

def embed_document(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        doc = json.load(f)

    chunks = chunk_text(doc["text"])
    embeddings = model.encode(chunks)

    # Prepare data for vector DB
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "chunk_id": f"{doc['metadata']['filename']}_chunk{i}",
            "text": chunk,
            "metadata": {
                "filename": doc["metadata"]["filename"],
                "title": doc["title"],
                "headings": doc["headings"],
                "page_numbers": doc["pages"],
                "chunk_index": i
            },
            "embedding": embeddings[i].tolist()
        })

    # Save chunk embeddings
    output_path = EMBEDDINGS_FOLDER / (json_file.stem + "_embeddings.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)
    print(f"✅ Embedded {json_file.name} → {output_path.name}")

def embed_all_documents():
    json_files = list(PROCESSED_FOLDER.glob("*.json"))
    if not json_files:
        print("⚠️ No processed JSON files found")
        return

    for jf in json_files:
        embed_document(jf)

if __name__ == "__main__":
    embed_all_documents()
