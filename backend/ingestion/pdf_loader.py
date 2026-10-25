
from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path: str | Path) -> list[dict]:
    """Extract PDF text while preserving source filename and page number."""
    path = Path(pdf_path)

    # Validate the file type before checking whether it exists.
    if path.suffix.lower() != ".pdf":
        raise ValueError("The input must be a PDF file.")

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError("The input must be a PDF file.")

    reader = PdfReader(str(path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if text:
            pages.append({
                "source": path.name,
                "page": page_number,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF. The document may require OCR."
        )

    return pages
