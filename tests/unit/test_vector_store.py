"""Unit tests for VectorStore class."""
import pytest


@pytest.fixture
def vector_store(tmp_path):
    """Return a VectorStore backed by a temp ChromaDB directory."""
    from mortimer.retrieval.vector_store import VectorStore

    return VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_collection")


@pytest.fixture
def embedder():
    from mortimer.retrieval.embedder import Embedder

    return Embedder("all-MiniLM-L6-v2")


class TestVectorStoreAdd:
    def test_add_chunks_increases_count(self, vector_store, sample_chunks, embedder):
        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)
        assert vector_store.count() == len(sample_chunks)

    def test_add_chunks_idempotent_on_same_ids(self, vector_store, sample_chunks, embedder):
        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)
        vector_store.add_chunks(sample_chunks, embeddings)
        assert vector_store.count() == len(sample_chunks)

    def test_add_empty_chunks_does_not_raise(self, vector_store):
        vector_store.add_chunks([], [])
        assert vector_store.count() == 0


class TestVectorStoreQuery:
    def test_query_returns_retrieved_chunks(self, vector_store, sample_chunks, embedder):
        from mortimer.models.schemas import RetrievedChunk

        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)

        query_embedding = embedder.embed_query("introduction concepts")
        results = vector_store.query(query_embedding, top_k=2)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_query_respects_top_k(self, vector_store, sample_chunks, embedder):
        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)

        query_embedding = embedder.embed_query("test")
        results = vector_store.query(query_embedding, top_k=1)
        assert len(results) <= 1

    def test_query_scores_are_floats(self, vector_store, sample_chunks, embedder):
        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)

        query_embedding = embedder.embed_query("methods approach")
        results = vector_store.query(query_embedding, top_k=2)
        for r in results:
            assert isinstance(r.score, float)

    def test_query_empty_store_returns_empty(self, vector_store, embedder):
        query_embedding = embedder.embed_query("anything")
        results = vector_store.query(query_embedding, top_k=5)
        assert results == []


class TestVectorStoreHasDocument:
    def test_has_document_true_after_adding(self, vector_store, sample_chunks, embedder):
        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)
        assert vector_store.has_document("paper.pdf") is True

    def test_has_document_false_when_absent(self, vector_store):
        assert vector_store.has_document("nonexistent.pdf") is False


class TestVectorStoreReset:
    def test_reset_empties_store(self, vector_store, sample_chunks, embedder):
        texts = [c.content for c in sample_chunks]
        embeddings = embedder.embed_texts(texts)
        vector_store.add_chunks(sample_chunks, embeddings)
        vector_store.reset()
        assert vector_store.count() == 0

    def test_reset_on_empty_store_does_not_raise(self, vector_store):
        vector_store.reset()
        assert vector_store.count() == 0
