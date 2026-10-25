
import os
from openai import OpenAI


DEFAULT_MODEL = "text-embedding-3-small"
MAX_BATCH_SIZE = 100


def create_embeddings(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    batch_size: int = MAX_BATCH_SIZE,
) -> list[list[float]]:
    """Generate embeddings for text using the OpenAI API."""

    if not texts:
        return []

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("All input texts must be non-empty strings.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI()
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float",
        )

        # Keep vectors in the same order as their input texts.
        ordered_data = sorted(response.data, key=lambda item: item.index)
        embeddings.extend(item.embedding for item in ordered_data)

    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding count does not match input text count.")

    return embeddings
