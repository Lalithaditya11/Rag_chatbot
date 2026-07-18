import json
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# ====================================================
# CONFIG
# ====================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "data" / "processed"
COLLECTION_NAME = "pdf_collection"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
QDRANT_PATH = BASE_DIR / "data" / "vector_db"

# ====================================================
# EMBEDDING MODEL
# ====================================================
print("Loading Embedding Model...")
embedding_model = SentenceTransformer(EMBED_MODEL, device="cpu")

# ====================================================
# QDRANT
# ====================================================
client = QdrantClient(path=str(QDRANT_PATH))
embedding_dimension = embedding_model.get_sentence_embedding_dimension()
collections = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in collections:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
    )

# ====================================================
# TEXT SPLITTER
# ====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ====================================================
# PROCESS
# ====================================================

point_id = 0
json_files = [
    os.path.join(INPUT_FOLDER, file)
    for file in os.listdir(INPUT_FOLDER)
    if file.endswith(".json")
]
print(f"\nFound {len(json_files)} processed files.\n")
for json_file in json_files:
    print(f"Processing : {Path(json_file).name}")
    with open(json_file, "r", encoding="utf-8") as f:
        pages = json.load(f)
    points = []
    for page in tqdm(pages):
        chunks = splitter.split_text(page["text"])
        for chunk_number, chunk in enumerate(chunks):
            vector = embedding_model.encode(
                chunk,
                normalize_embeddings=True
            )
            payload = {
                "pdf_id": page["pdf_id"],
                "filename": page["filename"],
                "page_number": page["page_number"],
                "language": page["language"],
                "chunk_number": chunk_number,
                "text": chunk
            }
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
            )
            point_id += 1
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
print("\nEmbedding Completed Successfully.")