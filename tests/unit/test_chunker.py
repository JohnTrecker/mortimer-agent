"""Unit tests for document chunker module."""


class TestChunkDocument:
    def test_chunk_document_returns_list(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title="Test Paper")
        assert isinstance(chunks, list)

    def test_chunk_document_produces_document_chunks(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages
        from mortimer.models.schemas import DocumentChunk

        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title="Test Paper")
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_chunk_document_chunk_ids_are_unique(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title="Test Paper")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_document_respects_chunk_size(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunk_size = 200
        chunks = chunk_document(pages, title="Test Paper", chunk_size=chunk_size, chunk_overlap=20)
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size * 1.2

    def test_chunk_document_metadata_has_title(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title="My Test Paper")
        for chunk in chunks:
            assert chunk.metadata.title == "My Test Paper"

    def test_chunk_document_metadata_has_source(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title="T")
        for chunk in chunks:
            assert len(chunk.metadata.source) > 0

    def test_chunk_document_chunk_ids_are_deterministic(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunks1 = chunk_document(pages, title="Test Paper")
        chunks2 = chunk_document(pages, title="Test Paper")
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]
        assert ids1 == ids2

    def test_chunk_document_empty_pages_returns_empty(self):
        from mortimer.ingestion.chunker import chunk_document

        chunks = chunk_document([], title="Empty")
        assert chunks == []

    def test_chunk_document_content_is_nonempty(self, fixture_pdf_path):
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title="T")
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0


class TestDetectSection:
    def test_detect_section_finds_numbered_section(self):
        from mortimer.ingestion.chunker import _detect_section

        text = "1. Introduction\nSome content"
        assert _detect_section(text) == "1. Introduction"

    def test_detect_section_finds_abstract(self):
        from mortimer.ingestion.chunker import _detect_section

        text = "Abstract\nThis paper presents..."
        assert "Abstract" in _detect_section(text)

    def test_detect_section_returns_empty_for_no_header(self):
        from mortimer.ingestion.chunker import _detect_section

        text = "Some random text without a header."
        result = _detect_section(text)
        assert isinstance(result, str)

    def test_detect_section_handles_empty_string(self):
        from mortimer.ingestion.chunker import _detect_section

        result = _detect_section("")
        assert isinstance(result, str)
        assert result == ""
