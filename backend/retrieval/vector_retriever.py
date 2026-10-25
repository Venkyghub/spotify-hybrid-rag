
import os
from pathlib import Path

from dotenv import load_dotenv

from backend.embeddings.embedding_service import create_embeddings
from backend.vectorstore.faiss_store import (
    load_faiss_index,
    search_faiss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def retrieve_vector_context(
    question: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve relevant document chunks from the existing FAISS index.

    Returns chunk metadata, text, and cosine similarity scores.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from .env")

    index_path = PROJECT_ROOT / "storage" / "faiss" / "spotify.index"
    metadata_path = PROJECT_ROOT / "storage" / "metadata" / "chunks.json"

    # Load the persisted index; do not regenerate embeddings for the corpus.
    index, chunks = load_faiss_index(index_path, metadata_path)

    # Generate an embedding only for the user's question.
    query_embeddings = create_embeddings([question])

    if not query_embeddings:
        return []

    results = search_faiss(
        index=index,
        chunks=chunks,
        query_embedding=query_embeddings[0],
        top_k=top_k,
    )

    # Add a source type so hybrid retrieval can identify each result.
    for result in results:
        result["retrieval_source"] = "vector"

    return results
