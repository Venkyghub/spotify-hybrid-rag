from fastapi import FastAPI
from backend.api.chat_routes import router as chat_router


app = FastAPI(
    title="Spotify Hybrid RAG API",
    description="Hybrid retrieval using FAISS and Neo4j",
    version="0.1.0",
)

app.include_router(chat_router)

@app.get("/")
def root():
    return {
        "message": "Spotify Hybrid RAG API is running",
        "status": "healthy",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "backend",
    }