
import os

import requests


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000",
)


def ask_question(question: str) -> dict:
    """Send a question to the FastAPI Hybrid RAG backend."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    response = requests.post(
        f"{BACKEND_URL.rstrip('/')}/api/chat",
        json={"question": question.strip()},
        timeout=120,
    )

    response.raise_for_status()
    return response.json()
