
from backend.corrective_rag.grounding_checker import check_grounding
from backend.corrective_rag.relevance_grader import grade_relevance
from backend.corrective_rag.retry_controller import decide_retry
from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.retrieval.hybrid_retriever import retrieve_hybrid_context
from backend.services.answer_service import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    generate_answer,
)


def _build_evidence_context(results: list[dict]) -> str:
    """Format retrieved evidence with source metadata for the answer model."""

    sections = []

    for item in results:
        text = item.get("text") or item.get("evidence")

        # Graph-only results may carry their evidence in relationships.
        if not isinstance(text, str) or not text.strip():
            relationships = item.get("graph_relationships", [])
            if isinstance(relationships, list):
                text = "\n".join(
                    relationship.get("evidence", "")
                    for relationship in relationships
                    if isinstance(relationship, dict)
                    and isinstance(relationship.get("evidence"), str)
                )

        if not isinstance(text, str) or not text.strip():
            continue

        source = item.get("source", "unknown")
        page = item.get("page", "unknown")
        chunk_id = item.get("chunk_id", "unknown")

        sections.append(
            f"[Source: {source}, Page: {page}, Chunk: {chunk_id}]\n"
            f"{text.strip()}"
        )

    return "\n\n---\n\n".join(sections)


def answer_question(
    question: str,
    graph_client: Neo4jClient,
    top_k: int = 5,
    graph_limit: int = 10,
    max_retries: int = 2,
) -> dict:
    """
    Run hybrid retrieval and a bounded Corrective RAG workflow.

    This service:
      1. Retrieves and reranks FAISS + Neo4j results.
      2. Grades evidence relevance.
      3. Retries with a larger top_k if no relevant evidence is found.
      4. Generates an answer from accepted evidence.
      5. Checks answer-to-evidence lexical grounding.
      6. Returns an insufficient-evidence fallback when checks fail.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    if (
        isinstance(graph_limit, bool)
        or not isinstance(graph_limit, int)
        or not 1 <= graph_limit <= 50
    ):
        raise ValueError("graph_limit must be between 1 and 50.")

    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or not 0 <= max_retries <= 3
    ):
        raise ValueError("max_retries must be between 0 and 3.")

    current_results = []
    graded_results = []
    attempt = 1
    decision = {}

    # Initial retrieval plus at most max_retries additional attempts.
    while True:
        current_top_k = min(top_k * attempt, 50)

        current_results = retrieve_hybrid_context(
            question=question.strip(),
            graph_client=graph_client,
            top_k=current_top_k,
            graph_limit=graph_limit,
        )

        graded_results = grade_relevance(
            question=question.strip(),
            results=current_results,
        )

        decision = decide_retry(
            relevance_results=graded_results,
            attempt=attempt,
            max_retries=max_retries,
        )

        if decision["action"] != "retry_retrieval":
            break

        attempt += 1

    accepted_evidence = [
        item
        for item in graded_results
        if item.get("relevance_label") == "relevant"
    ]

    if decision["action"] == "insufficient_evidence" or not accepted_evidence:
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_count": len(current_results),
            "retries_used": decision.get("retries_used", 0),
            "grounding": {
                "grounded": False,
                "support_score": 0.0,
                "label": "insufficient_evidence",
            },
            "status": "insufficient_evidence",
        }

    evidence_context = _build_evidence_context(accepted_evidence)

    if not evidence_context:
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_count": len(current_results),
            "retries_used": decision.get("retries_used", 0),
            "grounding": {
                "grounded": False,
                "support_score": 0.0,
                "label": "insufficient_evidence",
            },
            "status": "insufficient_evidence",
        }

    answer_result = generate_answer(
        question=question.strip(),
        evidence=accepted_evidence,
    )

    answer_text = answer_result["answer"]

    grounding = check_grounding(
        answer=answer_text,
        evidence=accepted_evidence,
    )

    if not grounding["grounded"]:
        return {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "retrieval_count": len(current_results),
            "retries_used": decision.get("retries_used", 0),
            "grounding": grounding,
            "status": "insufficient_evidence",
        }

    return {
        "answer": answer_text,
        "sources": answer_result.get("sources", []),
        "retrieval_count": len(current_results),
        "retries_used": decision.get("retries_used", 0),
        "grounding": grounding,
        "status": "answered",
    }
