"""Read and normalize source data."""

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving useful structure."""
    if not text:
        return ""

    # Normalize Unicode representation.
    text = unicodedata.normalize("NFC", text)

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse spaces/tabs inside each line.
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    # Keep paragraph boundaries but avoid excessive blank lines.
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical PDF page while preserving traceability."""
    reader = PdfReader(path)

    pages: list[DocumentPage] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        raw_text = pdf_page.extract_text() or ""
        text = normalize_text(raw_text)

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={
                    "empty_page": not bool(text),
                    "file_name": path.name,
                },
            )
        )

    return pages