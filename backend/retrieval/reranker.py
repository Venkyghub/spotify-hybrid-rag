import re
from collections.abc import Mapping, Sequence


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "please",
    "that", "the", "their", "this", "to", "use", "used",
    "using", "what", "which", "with", "would", "each",
    "other", "main", "between",
}

SYNONYM_GROUPS = [
    {"authentication", "auth", "identity", "oauth2", "oidc", "jwt"},
    {"database", "databases", "postgresql", "redis", "clickhouse"},
    {"communication", "communicate", "communicating"},
    {"messaging", "message", "messages"},
    {"event", "events"},
    {"websocket", "websockets"},
    {"queue", "queues"},
    {"microservice", "microservices"},
    {"service", "services"},
    {"dependency", "dependencies", "depends", "dependent"},
]


def _normalize_token(token: str) -> str:
    """Normalize common plural and verb forms without external libraries."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return token


def _tokenize(text: str) -> set[str]:
    """Extract normalized words from text."""
    return {
        _normalize_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def _expand_terms(terms: set[str]) -> set[str]:
    """Expand architecture terms to related documented terminology."""
    expanded = set(terms)

    for group in SYNONYM_GROUPS:
        normalized_group = {_normalize_token(term) for term in group}
        if expanded & normalized_group:
            expanded.update(normalized_group)

    return expanded


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
    """Calculate question-term overlap, including related terminology."""
    question_terms = _tokenize(question)
    evidence_terms = _tokenize(evidence)

    if not question_terms or not evidence_terms:
        return 0.0

    expanded_question_terms = _expand_terms(question_terms)

    # Normalize for expansion size so scores remain between 0 and 1.
    matched = len(expanded_question_terms & evidence_terms)
    return min(1.0, matched / max(1, len(question_terms)))


def _graph_relevance(result: Mapping) -> float:
    """Score graph metadata without treating it as proof."""
    relationships = result.get("graph_relationships", [])

    if not isinstance(relationships, list) or not relationships:
        if result.get("entity") and result.get("relationship"):
            return 1.0
        return 0.0

    return 1.0


def _similarity_score(result: Mapping) -> float:
    """Normalize a FAISS cosine similarity score to the 0-1 range."""
    value = result.get("similarity")

    if not isinstance(value, (int, float)):
        return 0.0

    return max(0.0, min(1.0, (float(value) + 1.0) / 2.0))


def rerank_results(
    question: str,
    results: Sequence[Mapping],
    top_k: int = 5,
) -> list[dict]:
    """Rerank results using lexical relevance, vector similarity, and graph metadata."""

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

        lexical_score = _lexical_relevance(question.strip(), evidence)
        vector_score = _similarity_score(result)
        graph_score = _graph_relevance(result)

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
