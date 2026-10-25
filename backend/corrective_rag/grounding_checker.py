
import re
from collections.abc import Mapping, Sequence


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "can", "could", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "of", "on", "or", "please",
    "that", "the", "their", "this", "to", "use", "used",
    "using", "what", "which", "with", "would",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def check_grounding(
    answer: str,
    evidence: Sequence[Mapping],
    min_support: float = 0.50,
) -> dict:
    """
    Check answer-to-evidence term overlap.

    Returns:
        grounded: whether the answer meets the configured threshold
        support_score: proportion of answer terms found in evidence
        label: grounded or insufficient_evidence

    This lexical check does not establish semantic entailment or
    guarantee that a claim is factually correct.
    """

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if (
        isinstance(min_support, bool)
        or not isinstance(min_support, (int, float))
        or not 0.0 <= min_support <= 1.0
    ):
        raise ValueError("min_support must be between 0 and 1.")

    if isinstance(evidence, (str, bytes)) or not isinstance(
        evidence, Sequence
    ):
        raise ValueError("evidence must be a sequence of result dictionaries.")

    evidence_parts = []

    for item in evidence:
        if not isinstance(item, Mapping):
            continue

        for key in ("text", "evidence"):
            value = item.get(key)
            if isinstance(value, str):
                evidence_parts.append(value)

        relationships = item.get("graph_relationships", [])
        if isinstance(relationships, list):
            for relationship in relationships:
                if not isinstance(relationship, Mapping):
                    continue

                value = relationship.get("evidence")
                if isinstance(value, str):
                    evidence_parts.append(value)

    answer_terms = _tokenize(answer)
    evidence_terms = _tokenize(" ".join(evidence_parts))

    if not answer_terms:
        score = 0.0
    else:
        score = len(answer_terms & evidence_terms) / len(answer_terms)

    grounded = bool(answer_terms) and score >= min_support

    return {
        "grounded": grounded,
        "support_score": round(score, 4),
        "label": "grounded" if grounded else "insufficient_evidence",
    }
