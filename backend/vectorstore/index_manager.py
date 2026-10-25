
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.embeddings.embedding_service import create_embeddings
from backend.vectorstore.faiss_store import build_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_project_index() -> None:
    """Embed the processed chunks and build the project's FAISS index."""

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    index_path = PROJECT_ROOT / "storage" / "faiss" / "spotify.index"
    metadata_path = PROJECT_ROOT / "storage" / "metadata" / "chunks.json"

    if not chunks_path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("The chunks file must contain a non-empty JSON list.")

    texts = [chunk.get("text", "") for chunk in chunks]

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every chunk must contain non-empty text.")

    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = create_embeddings(texts)

    print("Building FAISS index...")
    build_faiss_index(chunks, embeddings, index_path, metadata_path)

    print(f"Index saved: {index_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Vectors indexed: {len(chunks)}")


if __name__ == "__main__":
    build_project_index()
