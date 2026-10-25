
import json

import numpy as np
import pytest

from backend.vectorstore.faiss_store import (
    build_faiss_index,
    load_faiss_index,
    search_faiss,
)


@pytest.fixture
def sample_data():
    chunks = [
        {
            "source": "spotify_web_app_architecture.pdf",
            "page": 1,
            "chunk_id": 0,
            "text": "Spotify frontend uses Next.js and React.",
        },
        {
            "source": "spotify_web_app_architecture.pdf",
            "page": 2,
            "chunk_id": 1,
            "text": "Authentication uses OAuth2 and OIDC.",
        },
        {
            "source": "spotify_web_app_architecture.pdf",
            "page": 3,
            "chunk_id": 2,
            "text": "Audio playback uses HLS.js and HTML5 Audio.",
        },
    ]

    # Small synthetic vectors for unit testing; no API call required.
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    return chunks, embeddings


def test_build_and_load_faiss_index(tmp_path, sample_data):
    chunks, embeddings = sample_data
    index_path = tmp_path / "test.index"
    metadata_path = tmp_path / "metadata.json"

    build_faiss_index(chunks, embeddings, index_path, metadata_path)
    index, loaded_chunks = load_faiss_index(index_path, metadata_path)

    assert index.ntotal == 3
    assert len(loaded_chunks) == 3
    assert loaded_chunks[0]["source"] == chunks[0]["source"]


def test_search_returns_most_similar_chunk(tmp_path, sample_data):
    chunks, embeddings = sample_data
    index_path = tmp_path / "test.index"
    metadata_path = tmp_path / "metadata.json"

    build_faiss_index(chunks, embeddings, index_path, metadata_path)
    index, loaded_chunks = load_faiss_index(index_path, metadata_path)

    results = search_faiss(index, loaded_chunks, [0.99, 0.01, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == 0
    assert results[0]["similarity"] > results[1]["similarity"]


def test_rejects_mismatched_counts(tmp_path, sample_data):
    chunks, embeddings = sample_data

    with pytest.raises(ValueError):
        build_faiss_index(
            chunks,
            embeddings[:2],
            tmp_path / "test.index",
            tmp_path / "metadata.json",
        )


def test_rejects_invalid_top_k(tmp_path, sample_data):
    chunks, embeddings = sample_data
    index_path = tmp_path / "test.index"
    metadata_path = tmp_path / "metadata.json"

    build_faiss_index(chunks, embeddings, index_path, metadata_path)
    index, loaded_chunks = load_faiss_index(index_path, metadata_path)

    with pytest.raises(ValueError):
        search_faiss(index, loaded_chunks, [1.0, 0.0, 0.0], top_k=0)
