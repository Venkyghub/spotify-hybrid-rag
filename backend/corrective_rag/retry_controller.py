
from collections.abc import Mapping


def decide_retry(
    relevance_results: list[Mapping],
    attempt: int = 1,
    max_retries: int = 2,
) -> dict:
    """
    Decide whether retrieval should be retried.

    attempt:
        Current retrieval attempt, starting at 1.

    max_retries:
        Maximum additional retries allowed after the first attempt.

    Returns a decision with:
        action: accept_evidence, retry_retrieval, or insufficient_evidence
        attempt: current attempt
        retries_used: number of retries already used
        reason: explanation of the decision
    """

    if not isinstance(relevance_results, list):
        raise ValueError("relevance_results must be a list.")

    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise ValueError("attempt must be an integer greater than or equal to 1.")

    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or max_retries < 0
    ):
        raise ValueError("max_retries must be a non-negative integer.")

    relevant_found = any(
        isinstance(result, Mapping)
        and result.get("relevance_label") == "relevant"
        for result in relevance_results
    )

    retries_used = attempt - 1

    if relevant_found:
        return {
            "action": "accept_evidence",
            "attempt": attempt,
            "retries_used": retries_used,
            "reason": "At least one relevant result was found.",
        }

    if retries_used < max_retries:
        return {
            "action": "retry_retrieval",
            "attempt": attempt,
            "retries_used": retries_used,
            "reason": "No relevant evidence found; another retrieval attempt is allowed.",
        }

    return {
        "action": "insufficient_evidence",
        "attempt": attempt,
        "retries_used": retries_used,
        "reason": "No relevant evidence found and the retry limit has been reached.",
    }
