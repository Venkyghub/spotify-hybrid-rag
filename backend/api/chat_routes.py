
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.services.rag_service import answer_question

# Load environment variables from the project root .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

router = APIRouter(prefix="/api", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question about the Spotify Web App Architecture",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list = Field(default_factory=list)
    retrieval_count: int = 0
    retries_used: int = 0
    grounding: dict = Field(default_factory=dict)
    status: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        with Neo4jClient() as graph_client:
            result = answer_question(
                question=question,
                graph_client=graph_client,
            )
        return result

    except Exception:
        # Keep internal errors and credentials out of API responses.
        raise HTTPException(
            status_code=500,
            detail="Chat request failed. Check the backend logs.",
        )
