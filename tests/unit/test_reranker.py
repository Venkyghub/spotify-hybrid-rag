
from backend.retrieval.reranker import rerank_results


def test_reranker_prioritizes_relevant_evidence():
    question = (
        "Which search technologies support full-text indexing "
        "and autocomplete?"
    )

    results = [
        {
            "retrieval_source": "vector",
            "similarity": 0.38,
            "text": "The notification service sends emails and push notifications.",
        },
        {
            "retrieval_source": "hybrid",
            "similarity": 0.37,
            "text": (
                "Elasticsearch and OpenSearch provide full-text indexing "
                "and autocomplete suggestions."
            ),
            "graph_relationships": [
                {
                    "entity": "OpenSearch",
                    "relationship": "USES",
                    "related_entity": "Search & Discovery Index Service",
                }
            ],
        },
    ]

    ranked = rerank_results(question, results, top_k=2)

    assert len(ranked) == 2
    assert "Elasticsearch" in ranked[0]["text"]
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_reranker_rejects_empty_question():
    try:
        rerank_results("", [], top_k=5)
    except ValueError as exc:
        assert "Question cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_reranker_rejects_invalid_top_k():
    try:
        rerank_results("search technologies", [], top_k=0)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
