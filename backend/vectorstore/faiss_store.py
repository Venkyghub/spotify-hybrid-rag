
import json
from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(
    chunks: list[dict],
    embeddings: list[list[float]],
    index_path: str | Path,
    metadata_path: str | Path,
) -> None:
    """Build and save a FAISS index with its corresponding chunk metadata."""

    if not chunks:
        raise ValueError("Cannot build an index without chunks.")

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count must match embedding count.")

    vectors = np.asarray(embeddings, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
        raise ValueError("Embeddings must form a non-empty 2D matrix.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values.")

    if not np.all(np.linalg.norm(vectors, axis=1) > 0):
        raise ValueError("Embeddings must not contain zero vectors.")

    # Normalize vectors so inner product measures cosine similarity.
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Write metadata first; replace it only after JSON serialization succeeds.
    temp_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with temp_metadata.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
    temp_metadata.replace(metadata_path)

    faiss.write_index(index, str(index_path))


def load_faiss_index(
    index_path: str | Path,
    metadata_path: str | Path,
):
    """Load a persisted FAISS index and its chunk metadata."""

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("FAISS index or metadata file was not found.")

    index = faiss.read_index(str(index_path))

    with metadata_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata counts do not match.")

    return index, chunks


def search_faiss(
    index,
    chunks: list[dict],
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Return the most similar chunks for a query embedding."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if index.ntotal != len(chunks):
        raise ValueError("FAISS index and metadata counts do not match.")

    if index.ntotal == 0:
        return []

    query = np.asarray([query_embedding], dtype=np.float32)

    if query.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match the index.")

    if not np.isfinite(query).all():
        raise ValueError("Query embedding contains non-finite values.")

    if np.linalg.norm(query) == 0:
        raise ValueError("Query embedding must not be a zero vector.")

    faiss.normalize_L2(query)
    scores, indices = index.search(query, min(top_k, index.ntotal))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            **chunks[int(idx)],
            "similarity": float(score),
        })

    return results
