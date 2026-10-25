
from pathlib import Path

import pytest
from pypdf import PdfWriter

from backend.ingestion.pdf_loader import load_pdf


def create_test_pdf(path: Path) -> None:
    """Create a minimal PDF containing extractable text."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    # A blank PDF page has no extractable text.
    # Use the existing project PDF for the actual extraction test.
    writer.write(path.open("wb"))


def test_load_pdf_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("this_file_does_not_exist.pdf")


def test_load_pdf_rejects_non_pdf():
    with pytest.raises(ValueError):
        load_pdf("document.txt")
