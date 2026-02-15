from pathlib import Path
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

PROCESSED_FOLDER = Path("../../../data/processed")
EVAL_FOLDER = Path("../../../data/evaluation")
EVAL_FOLDER.mkdir(exist_ok=True)

# Embedding models to compare
MODELS = {
    "miniLM": "all-MiniLM-L6-v2",
    "mpnet": "all-mpnet-base-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}

# Test queries for retrieval evaluation
TEST_QUERIES = [
    {"query": "What is the policy number?", "expected": "MAW76630012"},
    {"query": "which university is shashank in?", "expected": "tu freiberg"},
    {"query": "what is the monthly premium for insurance?", "expected": "25,00"},
    {"query": "what is the date of application?", "expected": "02.05.2022"},
    {"query": "who is elected chair of the student group by EURECA-PRO", "expected" : "Shashank Bhati"},
]


def load_documents():
    docs = []
    for file in PROCESSED_FOLDER.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs.append(data["text"])
    return docs


def evaluate_model(model_name, model_path, docs):
    model = SentenceTransformer(model_path)

    # Embed documents
    doc_embeddings = model.encode(docs)

    results = []

    for test in TEST_QUERIES:
        query_emb = model.encode([test["query"]])
        sims = cosine_similarity(query_emb, doc_embeddings)[0]

        best_idx = sims.argmax()
        retrieved_text = docs[best_idx]

        hit = test["expected"].lower() in retrieved_text.lower()

        results.append({
            "model": model_name,
            "query": test["query"],
            "score": float(sims[best_idx]),
            "hit": hit
        })

    return results


def run_evaluation():
    docs = load_documents()
    all_results = []

    for name, path in MODELS.items():
        print(f"Evaluating {name}...")
        res = evaluate_model(name, path, docs)
        all_results.extend(res)

    df = pd.DataFrame(all_results)

    # Accuracy per model
    summary = df.groupby("model")["hit"].mean().reset_index()
    summary.rename(columns={"hit": "accuracy"}, inplace=True)

    # Save files
    df.to_csv(EVAL_FOLDER / "detailed_results.csv", index=False)
    summary.to_csv(EVAL_FOLDER / "summary.csv", index=False)

    print("\n=== Evaluation Summary ===")
    print(summary)


if __name__ == "__main__":
    run_evaluation()