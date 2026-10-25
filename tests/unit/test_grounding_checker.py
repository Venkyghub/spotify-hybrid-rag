
import pytest

from backend.corrective_rag.grounding_checker import check_grounding


def test_supported_answer_is_grounded():
    answer = "Elasticsearch supports full-text indexing and autocomplete."

    evidence = [
        {
            "text": (
                "The search service uses Elasticsearch for full-text "
                "indexing and autocomplete suggestions."
            )
        }
    ]

    result = check_grounding(answer, evidence)

    assert result["grounded"] is True
    assert result["label"] == "grounded"


def test_unsupported_answer_is_rejected():
    answer = "The system uses MongoDB for facial recognition."

    evidence = [
        {"text": "Elasticsearch powers full-text search and autocomplete."}
    ]

    result = check_grounding(answer, evidence)

    assert result["grounded"] is False
    assert result["label"] == "insufficient_evidence"


def test_empty_answer_is_rejected():
    with pytest.raises(ValueError, match="Answer cannot be empty"):
        check_grounding("", [])


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="min_support"):
        check_grounding("Elasticsearch supports search.", [], min_support=1.5)
