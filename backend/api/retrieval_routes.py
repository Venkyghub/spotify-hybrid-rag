from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.retrieval.hybrid_retriever import retrieve_hybrid_context

router = APIRouter(prefix="/api/retrieval", tags=["Retrieval Inspector"])


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    graph_limit: int = Field(default=10, ge=1, le=20)


@router.post("/inspect")
def inspect_retrieval(request: RetrievalRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        with Neo4jClient() as graph_client:
            results = retrieve_hybrid_context(
                question=question,
                graph_client=graph_client,
                top_k=request.top_k,
                graph_limit=request.graph_limit,
            )

        return {
            "question": question,
            "result_count": len(results),
            "results": results,
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Retrieval inspection failed. Check backend logs.",
        )
