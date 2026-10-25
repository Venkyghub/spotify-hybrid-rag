
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from backend.embeddings.embedding_service import create_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_create_embeddings():
    """Test real embeddings using the configured OpenAI API key."""
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured.")

    texts = [
        "Spotify uses a frontend application.",
        "Spotify architecture includes backend services.",
    ]

    embeddings = create_embeddings(texts)

    assert len(embeddings) == len(texts)
    assert all(len(vector) == 1536 for vector in embeddings)
    assert all(
        all(isinstance(value, (int, float)) for value in vector)
        for vector in embeddings
    )


def test_create_embeddings_empty_input():
    """Empty input should return an empty list without an API call."""
    assert create_embeddings([]) == []


def test_create_embeddings_rejects_empty_text():
    """Blank text should be rejected before an API call."""
    with pytest.raises(ValueError):
        create_embeddings(["Valid text", "   "])


def test_create_embeddings_rejects_invalid_batch_size():
    """A batch size of zero should be rejected."""
    with pytest.raises(ValueError):
        create_embeddings(["Some text"], batch_size=0)
