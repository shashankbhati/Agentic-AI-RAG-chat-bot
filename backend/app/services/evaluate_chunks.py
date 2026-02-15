from pathlib import Path
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -------- PATHS --------
CHUNK_FILE = Path("../../../data/chunks/all_chunks.json")
EMBED_FILE_TEMPLATE = "../../../data/embeddings/{model}_all_embeddings.npy"
EVAL_FOLDER = Path("../../../data/evaluation")
EVAL_FOLDER.mkdir(exist_ok=True)

# -------- MODELS TO COMPARE --------
MODELS = {
    "miniLM": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}

TOP_K = 3  # retrieval depth

# -------- TEST QUERIES --------
TEST_QUERIES = [
    {"query": "What is the policy number?", "expected": "MAW76630012"},
    {"query": "which university is shashank in?", "expected": "tu freiberg"},
    {"query": "what is the monthly premium for insurance?", "expected": "25,00"},
    {"query": "what is the date of application?", "expected": "02.05.2022"},
    {"query": "who is elected chair of the student group by EURECA-PRO", "expected" : "Shashank Bhati"},
]
# -----------------------------------


def load_chunks():
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks_if_needed(model_name, model_path, chunks):
    """
    Create embeddings per model (only once).
    """
    embed_path = Path(EMBED_FILE_TEMPLATE.format(model=model_name))

    if embed_path.exists():
        return np.load(embed_path)

    print(f"Creating embeddings for {model_name}...")
    model = SentenceTransformer(model_path)

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts)

    np.save(embed_path, embeddings)
    return embeddings


def evaluate_model(model_name, model_path, chunks):
    model = SentenceTransformer(model_path)
    embeddings = embed_chunks_if_needed(model_name, model_path, chunks)

    results = []

    for test in TEST_QUERIES:
        query_emb = model.encode([test["query"]])
        sims = cosine_similarity(query_emb, embeddings)[0]

        # top-k indices
        top_k_idx = sims.argsort()[-TOP_K:][::-1]

        # check hit in top-k
        hit = any(
            test["expected"].lower() in chunks[i]["text"].lower()
            for i in top_k_idx
        )

        results.append({
            "model": model_name,
            "query": test["query"],
            "top_k_hit": hit,
            "best_score": float(sims[top_k_idx[0]])
        })

    return results


def run_evaluation():
    print("Loading chunks...")
    chunks = load_chunks()

    all_results = []

    for name, path in MODELS.items():
        print(f"\nEvaluating {name}...")
        res = evaluate_model(name, path, chunks)
        all_results.extend(res)

    df = pd.DataFrame(all_results)

    # ---- summary metrics ----
    summary = (
        df.groupby("model")["top_k_hit"]
        .mean()
        .reset_index()
        .rename(columns={"top_k_hit": "recall_at_k"})
    )

    # ---- save reports ----
    df.to_csv(EVAL_FOLDER / "chunk_detailed_results.csv", index=False)
    summary.to_csv(EVAL_FOLDER / "chunk_summary.csv", index=False)

    print("\n=== Chunk Retrieval Summary ===")
    print(summary)


if __name__ == "__main__":
    run_evaluation()
