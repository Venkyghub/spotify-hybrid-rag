import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.ingestion.ingestion_service import ingest_pdf
from backend.embeddings.embedding_service import create_embeddings
from backend.vectorstore.faiss_store import append_faiss_index
from backend.knowledge_graph.neo4j_client import Neo4jClient
from backend.knowledge_graph.graph_builder import build_graph

router = APIRouter(prefix="/api/documents", tags=["Documents"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
PREVIEW_DIR = PROJECT_ROOT / "data" / "processed" / "upload_preview"

FAISS_INDEX = PROJECT_ROOT / "storage" / "faiss" / "spotify.index"
FAISS_METADATA = PROJECT_ROOT / "storage" / "metadata" / "chunks.json"

MAX_PDF_BYTES = 20 * 1024 * 1024


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.pdf"
    total_bytes = 0

    try:
        with saved_path.open("wb") as output:
            while True:
                block = file.file.read(1024 * 1024)
                if not block:
                    break

                total_bytes += len(block)

                if total_bytes > MAX_PDF_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 20 MB upload limit.",
                    )

                output.write(block)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        chunks = ingest_pdf(
            saved_path,
            output_path=PREVIEW_DIR / f"{saved_path.stem}.json",
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No text chunks could be extracted from the PDF.",
            )

        # Give this upload a unique source to prevent collisions.
        source = f"{saved_path.stem}_{filename}"
        for chunk in chunks:
            chunk["source"] = source

        # Generate and validate embeddings before changing either store.
        embeddings = create_embeddings(
            [chunk["text"] for chunk in chunks]
        )

        if len(embeddings) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        # Store graph evidence and relationships.
        with Neo4jClient() as client:
            build_graph(client, chunks)

        # Append to the existing index; do not rebuild it.
        append_faiss_index(
            chunks,
            embeddings,
            FAISS_INDEX,
            FAISS_METADATA,
        )

        return {
            "status": "indexed",
            "original_filename": filename,
            "bytes_received": total_bytes,
            "chunks_generated": len(chunks),
            "faiss_updated": True,
            "neo4j_updated": True,
            "message": (
                "PDF processed and added to FAISS and Neo4j. "
                "Existing indexed chunks were preserved."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        # The two stores cannot share one atomic transaction.
        raise HTTPException(
            status_code=500,
            detail=(
                "Upload integration failed. FAISS and Neo4j may need "
                "consistency verification. Check backend logs."
            ),
        ) from exc
    finally:
        file.file.close()
