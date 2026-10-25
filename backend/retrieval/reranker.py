
import re
from collections.abc import Mapping, Sequence


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "could", "do", "does", "for", "from", "how", "i", "in",
    "is", "it", "of", "on", "or", "please", "that", "the",
    "their", "this", "to", "use", "used", "using", "what",
    "which", "with", "would",
}


def _tokenize(text: str) -> set[str]:
    """Extract normalized words from text."""

    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def _get_evidence_text(result: Mapping) -> str:
    """Collect text available for evaluating result relevance."""

    parts = [
        result.get("text", ""),
        result.get("evidence", ""),
        result.get("entity", ""),
        result.get("related_entity", ""),
    ]

    relationships = result.get("graph_relationships", [])

    if isinstance(relationships, list):
        for relationship in relationships:
            if not isinstance(relationship, Mapping):
                continue

            parts.extend([
                relationship.get("entity", ""),
                relationship.get("relationship", ""),
                relationship.get("related_entity", ""),
                relationship.get("evidence", ""),
            ])

    return " ".join(
        part for part in parts if isinstance(part, str)
    )


def _lexical_relevance(question: str, evidence: str) -> float:
    """Calculate the proportion of question terms found in evidence."""

    question_terms = _tokenize(question)
    evidence_terms = _tokenize(evidence)

    if not question_terms or not evidence_terms:
        return 0.0

    return len(question_terms & evidence_terms) / len(question_terms)


def _graph_relevance(result: Mapping) -> float:
    """Score graph metadata without assuming it proves a claim."""

    relationships = result.get("graph_relationships", [])

    if not isinstance(relationships, list) or not relationships:
        # Standalone graph records may contain top-level relationship data.
        if result.get("entity") and result.get("relationship"):
            return 1.0
        return 0.0

    return 1.0


def _similarity_score(result: Mapping) -> float:
    """Normalize a FAISS cosine similarity score to the 0-1 range."""

    value = result.get("similarity")

    if not isinstance(value, (int, float)):
        return 0.0

    # Cosine similarity is in [-1, 1].
    return max(0.0, min(1.0, (float(value) + 1.0) / 2.0))


def rerank_results(
    question: str,
    results: Sequence[Mapping],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank hybrid retrieval results using lexical relevance,
    vector similarity, and the presence of graph metadata.

    This is a lightweight heuristic reranker, not a trained
    cross-encoder and not a factuality or evidence verifier.

    The output preserves the original result fields and adds:
      - rerank_score
      - lexical_relevance
      - graph_relevance
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise ValueError("results must be a sequence of result dictionaries.")

    scored_results = []

    for original in results:
        if not isinstance(original, Mapping):
            continue

        result = dict(original)

        evidence = _get_evidence_text(result)

        lexical_score = _lexical_relevance(
            question.strip(),
            evidence,
        )

        vector_score = _similarity_score(result)
        graph_score = _graph_relevance(result)

        # Give evidence matching the question the largest weight.
        # Vector similarity helps retain semantically relevant chunks.
        # Graph metadata is a small bonus, not proof of correctness.
        final_score = (
            0.60 * lexical_score
            + 0.30 * vector_score
            + 0.10 * graph_score
        )

        result["lexical_relevance"] = round(lexical_score, 4)
        result["graph_relevance"] = round(graph_score, 4)
        result["rerank_score"] = round(final_score, 4)

        scored_results.append(result)

    scored_results.sort(
        key=lambda item: (
            item["rerank_score"],
            item.get("similarity", 0.0)
                if isinstance(item.get("similarity"), (int, float))
                else 0.0,
        ),
        reverse=True,
    )

    return scored_results[:top_k]
