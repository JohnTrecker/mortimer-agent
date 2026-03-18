"""Unit tests for Pydantic v2 schema models."""
import pytest
from pydantic import ValidationError


class TestDocumentMetadata:
    def test_valid_metadata_with_required_fields(self):
        from mortimer.models.schemas import DocumentMetadata

        meta = DocumentMetadata(source="paper.pdf", title="Test Paper", page_number=1)
        assert meta.source == "paper.pdf"
        assert meta.title == "Test Paper"
        assert meta.page_number == 1
        assert meta.section == ""

    def test_valid_metadata_with_section(self):
        from mortimer.models.schemas import DocumentMetadata

        meta = DocumentMetadata(
            source="paper.pdf", title="Test Paper", page_number=3, section="2. Methods"
        )
        assert meta.section == "2. Methods"

    def test_metadata_missing_source_raises(self):
        from mortimer.models.schemas import DocumentMetadata

        with pytest.raises(ValidationError):
            DocumentMetadata(title="Test", page_number=1)

    def test_metadata_missing_title_raises(self):
        from mortimer.models.schemas import DocumentMetadata

        with pytest.raises(ValidationError):
            DocumentMetadata(source="x.pdf", page_number=1)

    def test_metadata_missing_page_number_raises(self):
        from mortimer.models.schemas import DocumentMetadata

        with pytest.raises(ValidationError):
            DocumentMetadata(source="x.pdf", title="Test")

    def test_metadata_is_immutable(self):
        from mortimer.models.schemas import DocumentMetadata

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        with pytest.raises((ValidationError, TypeError)):
            meta.source = "other.pdf"


class TestDocumentChunk:
    def test_valid_chunk(self):
        from mortimer.models.schemas import DocumentChunk, DocumentMetadata

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        chunk = DocumentChunk(content="Hello world", metadata=meta, chunk_id="abc123")
        assert chunk.content == "Hello world"
        assert chunk.chunk_id == "abc123"

    def test_chunk_missing_content_raises(self):
        from mortimer.models.schemas import DocumentChunk, DocumentMetadata

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        with pytest.raises(ValidationError):
            DocumentChunk(metadata=meta, chunk_id="abc")

    def test_chunk_missing_chunk_id_raises(self):
        from mortimer.models.schemas import DocumentChunk, DocumentMetadata

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        with pytest.raises(ValidationError):
            DocumentChunk(content="text", metadata=meta)

    def test_chunk_is_immutable(self):
        from mortimer.models.schemas import DocumentChunk, DocumentMetadata

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        chunk = DocumentChunk(content="Hello", metadata=meta, chunk_id="x")
        with pytest.raises((ValidationError, TypeError)):
            chunk.content = "changed"


class TestRetrievedChunk:
    def test_valid_retrieved_chunk(self):
        from mortimer.models.schemas import DocumentChunk, DocumentMetadata, RetrievedChunk

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        chunk = DocumentChunk(content="text", metadata=meta, chunk_id="id1")
        retrieved = RetrievedChunk(chunk=chunk, score=0.95)
        assert retrieved.score == 0.95
        assert retrieved.chunk.content == "text"

    def test_retrieved_chunk_score_boundary(self):
        from mortimer.models.schemas import DocumentChunk, DocumentMetadata, RetrievedChunk

        meta = DocumentMetadata(source="paper.pdf", title="Test", page_number=1)
        chunk = DocumentChunk(content="text", metadata=meta, chunk_id="id1")
        r1 = RetrievedChunk(chunk=chunk, score=0.0)
        r2 = RetrievedChunk(chunk=chunk, score=1.0)
        assert r1.score == 0.0
        assert r2.score == 1.0


class TestRAGResponse:
    def test_valid_rag_response(self):
        from mortimer.models.schemas import RAGResponse

        resp = RAGResponse(
            question="What is X?",
            answer="X is Y.",
            sources=["paper.pdf section 1", "paper.pdf section 2"],
        )
        assert resp.question == "What is X?"
        assert resp.answer == "X is Y."
        assert len(resp.sources) == 2

    def test_rag_response_empty_sources(self):
        from mortimer.models.schemas import RAGResponse

        resp = RAGResponse(question="What?", answer="Nothing.", sources=[])
        assert resp.sources == []

    def test_rag_response_serializes_to_json(self):
        from mortimer.models.schemas import RAGResponse

        resp = RAGResponse(question="Q?", answer="A.", sources=["src1"])
        data = resp.model_dump()
        assert set(data.keys()) == {"question", "answer", "sources"}

    def test_rag_response_missing_question_raises(self):
        from mortimer.models.schemas import RAGResponse

        with pytest.raises(ValidationError):
            RAGResponse(answer="A.", sources=[])

    def test_rag_response_missing_answer_raises(self):
        from mortimer.models.schemas import RAGResponse

        with pytest.raises(ValidationError):
            RAGResponse(question="Q?", sources=[])


class TestIngestionResult:
    def test_valid_ingestion_result(self):
        from mortimer.models.schemas import IngestionResult

        result = IngestionResult(
            document_path="/data/paper.pdf", total_chunks=42, title="Test Paper"
        )
        assert result.document_path == "/data/paper.pdf"
        assert result.total_chunks == 42
        assert result.title == "Test Paper"

    def test_ingestion_result_zero_chunks(self):
        from mortimer.models.schemas import IngestionResult

        result = IngestionResult(document_path="x.pdf", total_chunks=0, title="T")
        assert result.total_chunks == 0

    def test_ingestion_result_missing_fields_raises(self):
        from mortimer.models.schemas import IngestionResult

        with pytest.raises(ValidationError):
            IngestionResult(total_chunks=1)


class TestDocumentPage:
    def test_valid_document_page(self):
        from mortimer.models.schemas import DocumentPage

        page = DocumentPage(source="x.pdf", page_number=0, text="Page text here")
        assert page.source == "x.pdf"
        assert page.page_number == 0
        assert page.text == "Page text here"

    def test_document_page_missing_text_raises(self):
        from mortimer.models.schemas import DocumentPage

        with pytest.raises(ValidationError):
            DocumentPage(source="x.pdf", page_number=0)
