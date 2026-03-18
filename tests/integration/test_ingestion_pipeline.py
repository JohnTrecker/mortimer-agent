"""Integration tests for the full ingestion pipeline."""

import pytest


@pytest.fixture
def embedder():
    from mortimer.retrieval.embedder import Embedder
    return Embedder("all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(tmp_path):
    from mortimer.retrieval.vector_store import VectorStore
    return VectorStore(persist_dir=tmp_path / "chroma", collection_name="integration_test")


class TestIngestionIntegration:
    def test_full_ingestion_pipeline_from_pdf(self, fixture_pdf_path, embedder, vector_store):
        """Test loader -> chunker -> embedder -> vector_store chain."""
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages, extract_title

        title = extract_title(fixture_pdf_path)
        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title=title)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)

        vector_store.add_chunks(chunks, embeddings)

        assert vector_store.count() == len(chunks)
        assert len(chunks) > 0

    def test_ingestion_enables_retrieval(self, fixture_pdf_path, embedder, vector_store):
        """Ingested chunks can be retrieved by semantic query."""
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages, extract_title

        title = extract_title(fixture_pdf_path)
        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title=title)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(chunks, embeddings)

        query_embedding = embedder.embed_query("introduction testing")
        results = vector_store.query(query_embedding, top_k=3)

        assert len(results) > 0
        assert all(r.score >= 0.0 for r in results)

    def test_retrieved_chunks_contain_valid_metadata(
        self, fixture_pdf_path, embedder, vector_store
    ):
        """Retrieved chunks preserve source and page metadata."""
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages, extract_title

        title = extract_title(fixture_pdf_path)
        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title=title)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(chunks, embeddings)

        query_embedding = embedder.embed_query("methods")
        results = vector_store.query(query_embedding, top_k=2)

        for result in results:
            assert result.chunk.metadata.source == fixture_pdf_path.name
            assert result.chunk.metadata.title == title
            assert result.chunk.metadata.page_number >= 0

    def test_ingestion_idempotent_via_upsert(self, fixture_pdf_path, embedder, vector_store):
        """Re-ingesting the same document does not duplicate chunks."""
        from mortimer.ingestion.chunker import chunk_document
        from mortimer.ingestion.loader import extract_pages, extract_title

        title = extract_title(fixture_pdf_path)
        pages = extract_pages(fixture_pdf_path)
        chunks = chunk_document(pages, title=title)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)

        vector_store.add_chunks(chunks, embeddings)
        count_after_first = vector_store.count()

        vector_store.add_chunks(chunks, embeddings)
        count_after_second = vector_store.count()

        assert count_after_first == count_after_second
