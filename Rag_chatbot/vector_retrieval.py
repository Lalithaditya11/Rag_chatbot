import json
import re
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ==========================================
# CONFIG
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
COLLECTION_NAME = "pdf_collection"
QDRANT_PATH = BASE_DIR / "data" / "vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINAL_RESULTS = 5

# ==========================================
# HELPERS
# ==========================================


def normalize_text(text):
    return re.sub(r"\W+", " ", text.lower()).split()


def load_embedder():
    try:
        return SentenceTransformer(EMBED_MODEL, device="cpu")
    except Exception as exc:
        print(f"Embedding model unavailable: {exc}")
        return None


def load_client():
    try:
        return QdrantClient(path=str(QDRANT_PATH))
    except Exception as exc:
        print(f"Qdrant unavailable: {exc}")
        return None


def fallback_search(question):
    print("\nFalling back to processed text search...")
    query_terms = normalize_text(question)
    results = []

    for json_path in sorted(INPUT_FOLDER.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as handle:
            pages = json.load(handle)

        for page in pages:
            text = page.get("text", "")
            text_terms = normalize_text(text)
            score = sum(1 for term in query_terms if term in text_terms)
            if score > 0:
                results.append((page, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:FINAL_RESULTS]


def search(question):
    embedder = load_embedder()
    client = load_client()

    if embedder is None or client is None:
        return fallback_search(question)

    print("\nSearching Qdrant...")
    try:
        query_vector = embedder.encode(question, normalize_embeddings=True).tolist()
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=FINAL_RESULTS,
        ).points
        if not hits:
            return fallback_search(question)
        return [(hit.payload, hit.score) for hit in hits]
    except Exception as exc:
        print(f"Vector search failed: {exc}")
        return fallback_search(question)


# ==========================================
# MAIN
# ==========================================

def main():
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = []
        while True:
            try:
                question = input("\nAsk Question : ")
            except EOFError:
                break
            if question.lower() == "exit":
                break
            questions.append(question)

    for question in questions:
        results = search(question)
        print("\n========== TOP RESULTS ==========")
        for i, result in enumerate(results):
            if isinstance(result[0], dict):
                page, score = result
                print(f"Rank : {i + 1}")
                print(f"Match Score : {score}")
                print(f"PDF : {page['filename']}")
                print(f"Page : {page['page_number']}")
                print()
                print(page["text"][:500])
                print("\n" + "=" * 70 + "\n")
            else:
                page, score = result
                print(f"Rank : {i + 1}")
                print(f"Match Score : {score}")
                print(f"PDF : {page['filename']}")
                print(f"Page : {page['page_number']}")
                print()
                print(page["text"][:500])
                print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()