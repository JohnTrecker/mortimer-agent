"""PDF loading utilities: download, page extraction, title extraction."""
from pathlib import Path
from urllib.parse import urlparse

import fitz  # PyMuPDF
import httpx

from mortimer.models.schemas import DocumentPage

# Security: enforce an upper bound on downloaded PDF size (50 MB) to prevent
# memory exhaustion from maliciously large remote files.
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB

# Security: enforce a read timeout so a slow/hung server cannot stall the
# process indefinitely.
_DOWNLOAD_TIMEOUT_SECONDS = 30


def download_pdf(url: str, dest_dir: Path) -> Path:
    """Download a PDF from url to dest_dir, skipping if already present.

    Only HTTPS URLs are accepted to prevent plaintext-HTTP MITM attacks and
    to block SSRF attempts that target non-HTTP schemes (file://, ftp://, etc.).

    Args:
        url: HTTPS URL of the PDF to download.
        dest_dir: Directory where the PDF will be saved.

    Returns:
        Path to the downloaded (or already existing) PDF file.

    Raises:
        ValueError: If the URL scheme is not HTTPS.
        httpx.RequestError: If the download fails.
        ValueError: If the response body exceeds _MAX_PDF_BYTES.
    """
    # Security: reject non-HTTPS URLs to prevent HTTP downgrade attacks and
    # block SSRF via file://, ftp://, gopher://, or other schemes.
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(
            f"Only HTTPS URLs are accepted for PDF download; got scheme '{parsed.scheme}'"
        )

    filename = _url_to_filename(url)
    dest_path = dest_dir / filename

    if dest_path.exists():
        return dest_path

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Security: set an explicit timeout to prevent hanging on slow servers,
    # and cap response size to prevent memory exhaustion.
    with httpx.stream(
        "GET",
        url,
        follow_redirects=True,
        timeout=_DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_bytes(chunk_size=65536):
            total += len(chunk)
            if total > _MAX_PDF_BYTES:
                raise ValueError(
                    f"Remote PDF exceeds maximum allowed size of {_MAX_PDF_BYTES} bytes"
                )
            chunks.append(chunk)

    dest_path.write_bytes(b"".join(chunks))
    return dest_path


def extract_pages(pdf_path: Path) -> list[DocumentPage]:
    """Extract text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of DocumentPage objects, one per page.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append(
            DocumentPage(
                source=pdf_path.name,
                page_number=page_num,
                text=text,
            )
        )
    doc.close()
    return pages


def extract_title(pdf_path: Path) -> str:
    """Extract the document title from PDF metadata or first line fallback.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Title string. Falls back to first non-empty text line if metadata is absent.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    metadata = doc.metadata
    doc_title = metadata.get("title", "").strip()

    if doc_title:
        doc.close()
        return doc_title

    first_text = _extract_first_line(doc)
    doc.close()
    return first_text or pdf_path.stem


def _url_to_filename(url: str) -> str:
    """Derive a .pdf filename from a URL."""
    parsed = urlparse(url)
    path_part = parsed.path.rstrip("/")
    name = Path(path_part).name
    if not name.endswith(".pdf"):
        name = name + ".pdf"
    return name


def _extract_first_line(doc: fitz.Document) -> str:
    """Return the first non-empty line from the first page."""
    if len(doc) == 0:
        return ""
    text = doc[0].get_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""
