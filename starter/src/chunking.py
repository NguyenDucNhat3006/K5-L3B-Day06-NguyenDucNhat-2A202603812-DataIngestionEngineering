"""Implement fixed-size and structure-aware chunking."""

import re

from .models import Chunk, DocumentPage


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split page text into overlapping character windows.

    Each chunk preserves source/page metadata so retrieval results
    can always be traced back to the original PDF page.
    """
    if size <= 0:
        raise ValueError("size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = page.text.strip()

    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}-p{page.page}-fixed-{index}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    metadata={
                        **page.metadata,
                        "strategy": "fixed_size",
                        "chunk_index": index,
                        "start_char": start,
                        "end_char": end,
                        "size": size,
                        "overlap": overlap,
                    },
                )
            )
            index += 1

        if end >= len(text):
            break

        start += step

    return chunks


def _split_oversized_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph into smaller coherent pieces."""
    if len(paragraph) <= max_size:
        return [paragraph]

    # Prefer sentence-like boundaries where possible.
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)

    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # A single sentence may itself be larger than max_size.
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""

            for start in range(0, len(sentence), max_size):
                piece = sentence[start:start + max_size].strip()
                if piece:
                    pieces.append(piece)

            continue

        candidate = sentence if not current else f"{current} {sentence}"

        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)

    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group paragraphs while respecting max_size.

    Paragraph boundaries are preferred over arbitrary character cuts.
    Oversized paragraphs are split separately.
    """
    if max_size <= 0:
        raise ValueError("max_size must be > 0")

    text = page.text.strip()

    if not text:
        return []

    # Primary structure signal: blank lines.
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    # Some PDF extractors do not preserve blank lines well.
    # In that case, treat individual non-empty lines as structural units.
    if len(paragraphs) == 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) > 1:
            paragraphs = lines

    units: list[str] = []

    for paragraph in paragraphs:
        units.extend(_split_oversized_paragraph(paragraph, max_size))

    grouped: list[str] = []
    current = ""

    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"

        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                grouped.append(current)
            current = unit

    if current:
        grouped.append(current)

    chunks: list[Chunk] = []

    for index, chunk_text in enumerate(grouped):
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}-p{page.page}-paragraph-{index}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata={
                    **page.metadata,
                    "strategy": "paragraph",
                    "chunk_index": index,
                    "max_size": max_size,
                },
            )
        )

    return chunks