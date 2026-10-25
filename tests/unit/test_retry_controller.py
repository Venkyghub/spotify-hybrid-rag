
import pytest

from backend.corrective_rag.retry_controller import decide_retry


def test_retries_when_no_relevant_evidence():
    result = decide_retry(
        relevance_results=[
            {"relevance_label": "irrelevant"}
        ],
        attempt=1,
        max_retries=2,
    )

    assert result["action"] == "retry_retrieval"


def test_accepts_relevant_evidence():
    result = decide_retry(
        relevance_results=[
            {"relevance_label": "relevant"}
        ],
        attempt=1,
        max_retries=2,
    )

    assert result["action"] == "accept_evidence"


def test_stops_after_retry_limit():
    result = decide_retry(
        relevance_results=[],
        attempt=3,
        max_retries=2,
    )

    assert result["action"] == "insufficient_evidence"
    assert result["retries_used"] == 2


def test_invalid_attempt_is_rejected():
    with pytest.raises(ValueError, match="attempt"):
        decide_retry([], attempt=0)


def test_negative_retry_limit_is_rejected():
    with pytest.raises(ValueError, match="max_retries"):
        decide_retry([], max_retries=-1)
