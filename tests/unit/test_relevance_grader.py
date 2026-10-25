
import pytest

from backend.corrective_rag.relevance_grader import grade_relevance


def test_relevant_evidence_is_identified():
    question = (
        "Which search technologies support full-text indexing "
        "and autocomplete?"
    )

    results = [
        {
            "text": (
                "Elasticsearch and OpenSearch provide full-text "
                "indexing and autocomplete suggestions."
            )
        },
        {
            "text": (
                "The notification service sends emails and "
                "push notifications."
            )
        },
    ]

    graded = grade_relevance(question, results)

    assert graded[0]["relevance_label"] == "relevant"
    assert graded[1]["relevance_label"] == "irrelevant"
    assert graded[0]["relevance_score"] > graded[1]["relevance_score"]


def test_graph_evidence_is_included():
    question = "Which technologies support autocomplete?"

    results = [
        {
            "text": "Search service details.",
            "graph_relationships": [
                {
                    "entity": "OpenSearch",
                    "relationship": "USES",
                    "related_entity": "Search Service",
                    "evidence": "OpenSearch powers autocomplete suggestions.",
                }
            ],
        }
    ]

    graded = grade_relevance(question, results)

    assert graded[0]["relevance_label"] == "relevant"


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        grade_relevance("", [])


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="min_relevance"):
        grade_relevance("search technologies", [], min_relevance=1.5)
