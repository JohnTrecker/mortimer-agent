"""End-to-end RAG pipeline: ingest and query."""
from pathlib import Path

from mortimer.config import Settings
from mortimer.generation.llm_client import LLMClient
from mortimer.generation.prompt import build_messages
from mortimer.ingestion.chunker import chunk_document
from mortimer.ingestion.loader import download_pdf, extract_pages, extract_title
from mortimer.models.schemas import IngestionResult, RAGResponse
from mortimer.retrieval.embedder import Embedder
from mortimer.retrieval.vector_store import VectorStore


class RAGPipeline:
    """Orchestrates ingestion and query flows for the RAG assistant."""

    def __init__(self) -> None:
        """Initialize the pipeline using application settings."""
        settings = Settings()
        self._settings = settings
        self._embedder = Embedder(settings.embedding_model)
        self._vector_store = VectorStore(persist_dir=settings.chroma_persist_dir)
        self._llm = LLMClient(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
        )

    def ingest(self, pdf_urls: list[str]) -> list[IngestionResult]:
        """Download, extract, chunk, embed, and store PDFs.

        Already-indexed documents (identified by source filename) are skipped.

        Args:
            pdf_urls: List of PDF URLs or local file paths to ingest.

        Returns:
            List of IngestionResult objects. Skipped documents have total_chunks=0.
        """
        results: list[IngestionResult] = []
        for url in pdf_urls:
            result = self._ingest_one(url)
            results.append(result)
        return results

    # Security: cap question length to prevent oversized inputs from exhausting
    # LLM token budgets or enabling prompt-injection via extremely long payloads.
    _MAX_QUESTION_LENGTH = 2000

    def query(self, question: str) -> RAGResponse:
        """Answer a question using the RAG pipeline.

        Args:
            question: User question string (max 2000 characters).

        Returns:
            RAGResponse with answer and source citations.

        Raises:
            ValueError: If question is empty or exceeds the maximum length.
        """
        # Security: validate question is non-empty and within safe bounds.
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        if len(question) > self._MAX_QUESTION_LENGTH:
            raise ValueError(
                f"Question exceeds maximum length of {self._MAX_QUESTION_LENGTH} characters"
            )

        query_embedding = self._embedder.embed_query(question)
        retrieved = self._vector_store.query(query_embedding, top_k=self._settings.retrieval_top_k)
        messages = build_messages(question, retrieved)
        return self._llm.generate(messages)

    def reset(self) -> None:
        """Clear all stored documents from the vector store."""
        self._vector_store.reset()

    def _ingest_one(self, url: str) -> IngestionResult:
        """Download and index a single PDF.

        Args:
            url: URL or local path string.

        Returns:
            IngestionResult. Returns total_chunks=0 if already indexed.
        """
        pdf_path = _resolve_path(url, self._settings.pdf_download_dir)
        source_name = pdf_path.name

        if self._vector_store.has_document(source_name):
            title = extract_title(pdf_path)
            return IngestionResult(
                document_path=str(pdf_path),
                total_chunks=0,
                title=title,
            )

        title = extract_title(pdf_path)
        pages = extract_pages(pdf_path)
        source_url = url if url.startswith("https://") else ""
        chunks = chunk_document(
            pages,
            title=title,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            url=source_url,
        )

        if chunks:
            texts = [c.content for c in chunks]
            embeddings = self._embedder.embed_texts(texts)
            self._vector_store.add_chunks(chunks, embeddings)

        return IngestionResult(
            document_path=str(pdf_path),
            total_chunks=len(chunks),
            title=title,
        )


def _resolve_path(url: str, pdf_dir: Path) -> Path:
    """Return a local Path, downloading if url is a remote HTTPS URL.

    Security rules enforced here:
    - Plain HTTP URLs are rejected; only HTTPS is permitted (MITM prevention).
    - For local paths the resolved path must stay within pdf_dir to prevent
      directory traversal (e.g. "../../etc/passwd").

    Args:
        url: HTTPS URL or local file path string confined to pdf_dir.
        pdf_dir: Directory for downloaded PDFs and the allowed root for local paths.

    Returns:
        Local Path to the PDF.

    Raises:
        ValueError: If a plain HTTP URL is supplied.
        ValueError: If a local path resolves outside of pdf_dir.
    """
    if url.startswith("https://"):
        return download_pdf(url, pdf_dir)

    # Security: explicitly reject plain HTTP to prevent MITM attacks.
    if url.startswith("http://"):
        raise ValueError(
            "Plain HTTP URLs are not accepted; use HTTPS instead"
        )

    # Security: treat remaining strings as local file paths and enforce that
    # the resolved path stays within pdf_dir to block directory traversal.
    candidate = Path(url)
    if not candidate.is_absolute():
        candidate = pdf_dir / candidate

    # Resolve symlinks and ".." components before comparing.
    try:
        resolved = candidate.resolve(strict=False)
        allowed_root = pdf_dir.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"Unable to resolve path '{url}': {exc}") from exc

    if not str(resolved).startswith(str(allowed_root)):
        raise ValueError(
            f"Path '{url}' resolves to '{resolved}', which is outside the "
            f"allowed directory '{allowed_root}'"
        )

    return resolved
