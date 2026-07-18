import json
import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
QDRANT_PATH = BASE_DIR / "data" / "vector_db"
COLLECTION = "pdf_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

try:
    embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
except Exception as exc:
    print(f"Embedding model unavailable: {exc}")
    embedder = None

try:
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
except Exception as exc:
    print(f"Local language model unavailable: {exc}")
    generator = None

try:
    vector_db = QdrantClient(path=str(QDRANT_PATH))
except Exception as exc:
    print(f"Qdrant unavailable: {exc}")
    vector_db = None

# ------------------------------------

def retrieve(question):
    if embedder is None or vector_db is None:
        return fallback_retrieve(question)

    try:
        vector = embedder.encode(question, normalize_embeddings=True).tolist()
        hits = vector_db.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=10,
        ).points
        if not hits:
            return fallback_retrieve(question)
        return [(hit.payload, hit.score) for hit in hits[:5]]
    except Exception as exc:
        print(f"Vector search failed: {exc}")
        return fallback_retrieve(question)


def fallback_retrieve(question):
    print("Using local processed text search...")
    results = []
    query_terms = [term.lower() for term in question.split() if term]

    for json_path in sorted(INPUT_FOLDER.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as handle:
            pages = json.load(handle)
        for page in pages:
            text = page.get("text", "")
            text_terms = [term.lower() for term in text.split() if term]
            score = sum(1 for term in query_terms if term in text_terms)
            if score > 0:
                results.append((page, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results[:5]

# ------------------------------------

def build_context(results):
    context = ""
    citations = []
    for result in results:
        if isinstance(result[0], dict):
            payload, score = result
        else:
            payload, score = result[0].payload, result[1]
        context += payload["text"]
        context += "\n\n"
        citations.append(f"{payload['filename']} (Page {payload['page_number']})")
    return context, citations


def answer_with_local_model(question, context):
    if generator is None:
        return (
            "I couldn't generate a local reply because the language model could not be loaded."
        )

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is missing, say: I couldn't find this information in the provided PDFs.

Context:
{context}

Question:
{question}
"""

    response = generator(
        prompt,
        max_new_tokens=220,
        do_sample=False,
        temperature=0.0,
    )[0]["generated_text"]
    return response.split("Question:\n", 1)[-1].strip() if "Question:" in response else response.strip()

# ------------------------------------

SYSTEM_PROMPT = """
You are an expert assistant.
Answer ONLY from the provided context.
If the answer does not exist,
say
"I couldn't find this information in the provided PDFs."
Never hallucinate.
Always cite the source.
"""

# ------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        questions = [" ".join(argv)]
    else:
        questions = []
        while True:
            try:
                question = input("\nQuestion : ")
            except EOFError:
                break
            if question.lower() == "exit":
                break
            questions.append(question)

    for question in questions:
        retrieved = retrieve(question)
        context, citations = build_context(retrieved)
        answer = answer_with_local_model(question, context)
        print("\n======================")
        print(answer)
        print("\nSources")
        for c in sorted(set(citations)):
            print("-", c)


if __name__ == "__main__":
    main()