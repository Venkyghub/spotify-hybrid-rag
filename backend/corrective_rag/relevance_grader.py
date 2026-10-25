
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
    """Convert text into normalized terms."""
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


def grade_relevance(
    question: str,
    results: Sequence[Mapping],
    min_relevance: float = 0.30,
) -> list[dict]:
    """
    Grade retrieved results using lexical evidence overlap.

    Adds:
      - relevance_score: proportion of question terms found
      - relevance_label: relevant or irrelevant

    This is a deterministic baseline, not an LLM grader
    or a guarantee of factual correctness.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if (
        isinstance(min_relevance, bool)
        or not isinstance(min_relevance, (int, float))
        or not 0.0 <= min_relevance <= 1.0
    ):
        raise ValueError("min_relevance must be between 0 and 1.")

    if isinstance(results, (str, bytes)) or not isinstance(
        results, Sequence
    ):
        raise ValueError("results must be a sequence of result dictionaries.")

    question_terms = _tokenize(question)

    graded_results = []

    for original in results:
        if not isinstance(original, Mapping):
            continue

        result = dict(original)

        # Include graph evidence as well as vector chunk text.
        evidence_parts = [
            result.get("text", ""),
            result.get("evidence", ""),
            result.get("entity", ""),
            result.get("related_entity", ""),
        ]

        relationships = result.get("graph_relationships", [])

        if isinstance(relationships, list):
            for relationship in relationships:
                if isinstance(relationship, Mapping):
                    evidence_parts.extend([
                        relationship.get("entity", ""),
                        relationship.get("relationship", ""),
                        relationship.get("related_entity", ""),
                        relationship.get("evidence", ""),
                    ])

        evidence_text = " ".join(
            part for part in evidence_parts
            if isinstance(part, str)
        )

        evidence_terms = _tokenize(evidence_text)

        if not question_terms:
            score = 0.0
        else:
            score = (
                len(question_terms & evidence_terms)
                / len(question_terms)
            )

        result["relevance_score"] = round(score, 4)
        result["relevance_label"] = (
            "relevant"
            if score >= min_relevance and score > 0
            else "irrelevant"
        )

        graded_results.append(result)

    return graded_results
