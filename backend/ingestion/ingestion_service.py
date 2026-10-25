
import json
from pathlib import Path

from backend.ingestion.pdf_loader import load_pdf
from backend.ingestion.text_cleaner import clean_text
from backend.ingestion.chunker import chunk_pages


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ingest_pdf(
    pdf_path: str | Path,
    output_path: str | Path | None = None,
) -> list[dict]:
    """Extract, clean, chunk, and save PDF content as JSON."""

    pdf_path = Path(pdf_path)

    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path

    pages = load_pdf(pdf_path)

    cleaned_pages = [
        {**page, "text": clean_text(page["text"])}
        for page in pages
    ]

    chunks = chunk_pages(cleaned_pages)

    if not chunks:
        raise ValueError("No text chunks were generated from the PDF.")

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"PDF: {pdf_path.name}")
    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks generated: {len(chunks)}")
    print(f"Saved to: {output_path}")

    return chunks
