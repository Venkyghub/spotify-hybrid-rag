
import os

from dotenv import load_dotenv
from openai import OpenAI

from backend.prompts.answer_prompt import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_prompt,
)


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence in the available documentation "
    "to answer this question reliably."
)


def generate_answer(
    question: str,
    evidence: list[dict],
    model: str | None = None,
) -> dict:
    """
    Generate an answer using retrieved evidence.

    Each evidence item may contain:
      - text
      - evidence
      - source
      - page
      - chunk_id
      - graph_relationships

    Returns the answer and source metadata.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(evidence, list):
        raise ValueError("Evidence must be a list.")

    # Load environment configuration without overriding existing variables.
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    selected_model = model or os.getenv(
        "OPENAI_MODEL",
        "gpt-4.1-mini",
    )

    # Build a bounded evidence context and retain citation metadata.
    evidence_sections = []
    sources = []
    seen_sources = set()

    for item in evidence:
        if not isinstance(item, dict):
            continue

        text = item.get("text") or item.get("evidence")

        if not isinstance(text, str) or not text.strip():
            relationships = item.get("graph_relationships", [])
            if isinstance(relationships, list):
                text = "\n".join(
                    rel.get("evidence", "")
                    for rel in relationships
                    if isinstance(rel, dict)
                    and isinstance(rel.get("evidence"), str)
                )

        if not isinstance(text, str) or not text.strip():
            continue

        source = item.get("source")
        page = item.get("page")
        chunk_id = item.get("chunk_id")

        # Fall back to metadata from graph relationship evidence.
        if source is None or page is None:
            relationships = item.get("graph_relationships", [])
            if isinstance(relationships, list):
                for rel in relationships:
                    if not isinstance(rel, dict):
                        continue
                    source = source or rel.get("source")
                    page = page if page is not None else rel.get("page")
                    if source is not None and page is not None:
                        break

        citation = (
            f"[Source: {source or 'unknown'}, "
            f"Page: {page if page is not None else 'unknown'}, "
            f"Chunk: {chunk_id if chunk_id is not None else 'unknown'}]"
        )

        evidence_sections.append(f"{citation}\n{text.strip()}")

        source_key = (source, page, chunk_id)
        if source_key not in seen_sources:
            sources.append({
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
            })
            seen_sources.add(source_key)

    if not evidence_sections:
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "model": None,
        }

    # Keep the context size bounded.
    evidence_context = "\n\n---\n\n".join(evidence_sections)[:30000]

    user_prompt = build_answer_prompt(
        question=question,
        evidence=evidence_context,
    )

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=selected_model,
            instructions=ANSWER_SYSTEM_PROMPT,
            input=user_prompt,
            max_output_tokens=1000,
        )
        answer = (response.output_text or "").strip()
    except Exception as exc:
        # Avoid returning credentials or provider internals to API clients.
        raise RuntimeError(
            "Answer generation failed. Check server logs and API configuration."
        ) from exc

    if not answer:
        answer = INSUFFICIENT_EVIDENCE_MESSAGE

    return {
        "answer": answer,
        "sources": sources,
        "model": selected_model,
    }
