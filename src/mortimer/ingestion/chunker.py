"""Document chunking with deterministic IDs and section detection."""
import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from mortimer.models.schemas import DocumentChunk, DocumentMetadata, DocumentPage

_SECTION_PATTERN = re.compile(
    r"^(\d+[\.\d]*\.?\s+[A-Z][^\n]{0,60}"
    r"|Abstract|Introduction|Conclusion|Related Work|References)",
    re.MULTILINE,
)


def chunk_document(
    pages: list[DocumentPage],
    title: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    url: str = "",
) -> list[DocumentChunk]:
    """Split document pages into overlapping text chunks.

    Args:
        pages: List of DocumentPage objects to chunk.
        title: Document title to embed in metadata.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of DocumentChunk objects with deterministic IDs.
    """
    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks: list[DocumentChunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        splits = splitter.split_text(page.text)
        for chunk_index, text in enumerate(splits):
            if not text.strip():
                continue
            chunk_id = _make_chunk_id(page.source, page.page_number, chunk_index)
            section = _detect_section(text)
            metadata = DocumentMetadata(
                source=page.source,
                title=title,
                page_number=page.page_number,
                section=section,
                url=url,
            )
            chunks.append(
                DocumentChunk(
                    content=text,
                    metadata=metadata,
                    chunk_id=chunk_id,
                )
            )
    return chunks


def _detect_section(text: str) -> str:
    """Detect section header from the beginning of a text chunk.

    Args:
        text: Chunk text to inspect.

    Returns:
        Detected section header string, or empty string if none found.
    """
    if not text:
        return ""
    match = _SECTION_PATTERN.search(text[:300])
    if match:
        return match.group(0).strip()
    return ""


def _make_chunk_id(source: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from its coordinates.

    Args:
        source: Source filename.
        page_number: Zero-based page index.
        chunk_index: Index of this chunk within the page.

    Returns:
        Hex digest string (SHA-256 truncated to 16 chars).
    """
    raw = f"{source}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
