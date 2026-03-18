"""Tests demonstrating retrieval design goals.

These tests use real embeddings (all-MiniLM-L6-v2) and real ChromaDB to verify
that the vector store exhibits meaningful semantic retrieval properties.
No mocks — the tests prove the system's actual embedding quality.
"""


class TestSemanticSimilarity:
    def test_semantically_similar_query_ranks_relevant_chunk_first(
        self, populated_vector_store, embedder
    ):
        """Query about neural network learning should rank the ML chunk first.

        Demonstrates that embeddings capture semantic meaning beyond keywords.
        """
        query_embedding = embedder.embed_query("How do neural networks learn?")
        results = populated_vector_store.query(query_embedding, top_k=6)

        assert len(results) > 0
        top_chunk = results[0].chunk
        assert top_chunk.chunk_id == "chunk-ml"

    def test_semantically_distant_query_scores_lower(
        self, populated_vector_store, embedder
    ):
        """A cooking query should score cooking chunk higher than quantum computing.

        Demonstrates that cosine distance reflects topical distance.
        """
        query_embedding = embedder.embed_query("cooking recipes and ingredients")
        results = populated_vector_store.query(query_embedding, top_k=6)

        scores_by_id = {r.chunk.chunk_id: r.score for r in results}
        assert scores_by_id["chunk-cook"] > scores_by_id["chunk-quantum"]

    def test_synonym_query_matches_without_keyword_overlap(
        self, populated_vector_store, embedder
    ):
        """Paraphrase with no word overlap should still retrieve the photosynthesis chunk.

        The query 'plant energy conversion from sunlight' shares no tokens with
        'Photosynthesis is the process by which plants convert sunlight...' — wait,
        'sunlight' and 'plants/plant' do appear — so we use a stricter paraphrase.

        Query: 'how green organisms produce food from light'
        Expected: photosynthesis chunk in top 3.
        Demonstrates: embeddings capture synonymy and paraphrase.
        """
        query_embedding = embedder.embed_query("how green organisms produce food from light")
        results = populated_vector_store.query(query_embedding, top_k=3)

        chunk_ids = [r.chunk.chunk_id for r in results]
        assert "chunk-bio" in chunk_ids


class TestTopKRetrieval:
    def test_top_k_limits_result_count(self, populated_vector_store, embedder):
        """6 chunks indexed, top_k=3 should return exactly 3 results."""
        query_embedding = embedder.embed_query("science")
        results = populated_vector_store.query(query_embedding, top_k=3)
        assert len(results) == 3

    def test_results_ordered_by_descending_score(self, populated_vector_store, embedder):
        """Scores in the result list must be non-increasing."""
        query_embedding = embedder.embed_query("history revolution")
        results = populated_vector_store.query(query_embedding, top_k=6)

        assert len(results) > 1
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score, (
                f"Score at index {i} ({results[i].score}) < score at index {i + 1} "
                f"({results[i + 1].score})"
            )

    def test_top_k_exceeding_total_returns_all(self, populated_vector_store, embedder):
        """top_k=20 with only 6 indexed chunks should return all 6 without error."""
        query_embedding = embedder.embed_query("anything at all")
        results = populated_vector_store.query(query_embedding, top_k=20)
        assert len(results) == 6


class TestRelevanceScoring:
    def test_exact_content_query_has_high_score(self, populated_vector_store, embedder):
        """Querying with verbatim chunk text should yield a very high similarity score.

        Self-similarity in embedding space should produce score > 0.85.
        """
        verbatim = "The French Revolution began in 1789 with the storming of the Bastille prison in Paris."
        query_embedding = embedder.embed_query(verbatim)
        results = populated_vector_store.query(query_embedding, top_k=1)

        assert len(results) == 1
        assert results[0].score > 0.85, (
            f"Expected score > 0.85 for verbatim query, got {results[0].score}"
        )

    def test_unrelated_query_has_low_score(self, populated_vector_store, embedder):
        """Nonsense query should produce uniformly low scores across all chunks.

        Demonstrates that low scores signal irrelevance.
        """
        query_embedding = embedder.embed_query("xkzwq nonsense gibberish zzzyy")
        results = populated_vector_store.query(query_embedding, top_k=6)

        for r in results:
            assert r.score < 0.7, (
                f"Expected score < 0.7 for nonsense query, got {r.score} "
                f"for chunk {r.chunk.chunk_id}"
            )


class TestChunkIntegrity:
    def test_retrieved_chunk_preserves_metadata(self, populated_vector_store, embedder):
        """Metadata fields (source, title, page_number, section) survive a ChromaDB round-trip."""
        query_embedding = embedder.embed_query("quantum superposition qubits")
        results = populated_vector_store.query(query_embedding, top_k=1)

        assert len(results) == 1
        meta = results[0].chunk.metadata

        assert meta.source == "physics.pdf"
        assert meta.title == "Quantum Physics"
        assert meta.page_number == 4
        assert meta.section == "Quantum Computing"

    def test_retrieved_chunk_content_matches_indexed_content(
        self, populated_vector_store, embedder
    ):
        """Content field must be verbatim after a ChromaDB round-trip."""
        expected_content = (
            "Shakespeare wrote 37 plays including Hamlet, Macbeth, and A Midsummer Night's Dream."
        )
        query_embedding = embedder.embed_query("Shakespeare plays Hamlet")
        results = populated_vector_store.query(query_embedding, top_k=1)

        assert len(results) == 1
        assert results[0].chunk.content == expected_content
