"""Chunking strategy tests for arxiv PDF ingestion.

These tests act as a living specification for what "good chunking" means in
the context of the Mortimer RAG system.  They verify that the default settings
of chunk_size=1000 and chunk_overlap=200 produce chunks that are:

  * within the embedding model's token budget
  * long enough to carry semantic meaning
  * properly overlapping so boundary context is not lost
  * coherent at sentence / section boundaries
  * equipped with complete, valid metadata
  * assigned unique, deterministic IDs
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(text: str, source: str = "arxiv.pdf", page_number: int = 1):
    """Return a DocumentPage with the given text."""
    from mortimer.models.schemas import DocumentPage

    return DocumentPage(source=source, page_number=page_number, text=text)


def _lorem(n_chars: int) -> str:
    """Return a realistic prose string of at least n_chars characters.

    The text is built from repeating readable sentences so that
    RecursiveCharacterTextSplitter can find natural split points.
    """
    sentence = (
        "The transformer architecture relies on self-attention mechanisms "
        "to model long-range dependencies in sequential data. "
    )
    repeats = (n_chars // len(sentence)) + 2
    return (sentence * repeats)[:n_chars]


# ---------------------------------------------------------------------------
# Class: TestChunkSizeBounds
# ---------------------------------------------------------------------------


class TestChunkSizeBounds:
    """Verify that chunk_size=1000 keeps every chunk within the embedding
    model's token budget and above the minimum meaningful length.
    """

    def test_no_chunk_exceeds_character_limit(self):
        """Embedding model all-MiniLM-L6-v2 has 512-token hard limit; 1000
        chars ≈ 250 tokens, well within the cap.  Any chunk longer than 1000
        chars would risk truncation.
        """
        from mortimer.ingestion.chunker import chunk_document

        page = _make_page(_lorem(5000))
        chunks = chunk_document([page], title="Test", chunk_size=1000, chunk_overlap=200)

        assert chunks, "Expected at least one chunk from a 5000-char page"
        for chunk in chunks:
            assert len(chunk.content) <= 1000, (
                f"Chunk exceeds 1000-char limit: {len(chunk.content)} chars — "
                f"first 80: {chunk.content[:80]!r}"
            )

    def test_no_chunk_is_trivially_short(self):
        """Chunks shorter than 50 chars are likely headers, page numbers, or
        splitter artefacts — too short to carry semantic meaning.
        """
        from mortimer.ingestion.chunker import chunk_document

        # 400+ chars of realistic prose — comfortably above the 50-char floor
        page = _make_page(_lorem(400))
        chunks = chunk_document([page], title="Test", chunk_size=1000, chunk_overlap=200)

        assert chunks, "Expected at least one chunk from a 400-char page"
        for chunk in chunks:
            assert len(chunk.content) >= 50, (
                f"Chunk is trivially short: {len(chunk.content)} chars — "
                f"content: {chunk.content!r}"
            )

    def test_chunk_size_respected_across_page_boundaries(self):
        """The splitter is applied per-page; this verifies size limits hold
        even when a page has multiple paragraphs.
        """
        from mortimer.ingestion.chunker import chunk_document

        pages = [
            _make_page(_lorem(800), page_number=i + 1)
            for i in range(3)
        ]
        chunks = chunk_document(pages, title="Test", chunk_size=500, chunk_overlap=50)

        assert chunks, "Expected chunks from 3 pages of 800 chars each"
        for chunk in chunks:
            assert len(chunk.content) <= 500, (
                f"Chunk exceeds 500-char limit on multi-page input: "
                f"{len(chunk.content)} chars"
            )


# ---------------------------------------------------------------------------
# Class: TestChunkOverlapContinuity
# ---------------------------------------------------------------------------


class TestChunkOverlapContinuity:
    """Verify that the 200-char overlap prevents context from being lost at
    chunk boundaries.
    """

    def test_consecutive_chunks_share_overlap_content(self):
        """Overlap ensures a sentence straddling a boundary appears in at
        least one chunk intact.  Without overlap, retrieval could miss the
        answer entirely.

        Strategy: check that the last 100 chars of chunk N appear (partially)
        in the first 200 chars of chunk N+1.  We use a 40-char sliding window
        to account for word-boundary adjustments by the splitter.
        """
        from mortimer.ingestion.chunker import chunk_document

        page = _make_page(_lorem(2500))
        chunks = chunk_document([page], title="Test", chunk_size=1000, chunk_overlap=200)

        # Need at least 2 chunks from this page to test overlap
        same_page_chunks = [c for c in chunks if c.metadata.page_number == 1]
        if len(same_page_chunks) < 2:
            # If all text fits in one chunk, the test is vacuously satisfied
            return

        overlap_found_count = 0
        for i in range(len(same_page_chunks) - 1):
            tail = same_page_chunks[i].content[-100:]
            head = same_page_chunks[i + 1].content[:200]

            # Look for any 40-char window from the tail inside the head
            window_size = 40
            found = any(
                tail[j : j + window_size] in head
                for j in range(0, len(tail) - window_size + 1, 10)
            )
            if found:
                overlap_found_count += 1

        pairs = len(same_page_chunks) - 1
        assert overlap_found_count >= max(1, pairs // 2), (
            f"Overlap content not detected in consecutive chunks: "
            f"{overlap_found_count}/{pairs} pairs showed overlap"
        )

    def test_overlap_ratio_is_reasonable(self):
        """20% overlap is the standard starting point for academic PDFs.
        Below 10% risks boundary gaps; above 40% wastes context window with
        redundant content.
        """
        from mortimer.config import Settings

        # Patch required env vars so Settings can be instantiated without a
        # real .env file present in the test environment.
        import os
        env_backup = os.environ.copy()
        os.environ.setdefault("OPENAI_API_KEY", "test-key-placeholder")
        try:
            settings = Settings()
        finally:
            # Restore only the keys we may have added
            for key in list(os.environ):
                if key not in env_backup:
                    del os.environ[key]

        ratio = settings.chunk_overlap / settings.chunk_size
        assert ratio == 0.2, (
            f"Overlap ratio is {ratio:.2%}; expected 20% "
            f"(chunk_overlap={settings.chunk_overlap}, "
            f"chunk_size={settings.chunk_size})"
        )

    def test_no_duplicate_only_chunks(self):
        """Overlap copies content, it does not duplicate entire chunks.  Two
        identical chunks would mean chunk_overlap >= chunk_size, which is a
        misconfiguration.
        """
        from mortimer.ingestion.chunker import chunk_document

        page = _make_page(_lorem(3000))
        chunks = chunk_document([page], title="Test", chunk_size=1000, chunk_overlap=200)

        contents = [c.content for c in chunks]
        assert len(contents) == len(set(contents)), (
            "Duplicate chunks detected — chunk_overlap may be >= chunk_size"
        )


# ---------------------------------------------------------------------------
# Class: TestChunkCoherence
# ---------------------------------------------------------------------------


class TestChunkCoherence:
    """Verify that chunks are semantically meaningful units rather than
    arbitrary character slices.
    """

    def test_chunk_does_not_split_mid_sentence(self):
        """RecursiveCharacterTextSplitter tries sentence and paragraph
        boundaries before hard-cutting.  A chunk containing complete sentences
        is more semantically coherent than one ending mid-sentence.
        """
        from mortimer.ingestion.chunker import chunk_document

        sentences = [
            "The model learns representations through contrastive pre-training on image-text pairs.",
            "Fine-tuning on downstream tasks requires only a small labelled dataset.",
            "Attention heads in the encoder capture both local and global dependencies.",
            "Positional encodings allow the model to distinguish token order without recurrence.",
            "Layer normalisation is applied before each sub-layer rather than after.",
        ]
        text = "  ".join(sentences)  # 5 sentences, well under 1000 chars

        page = _make_page(text)
        chunks = chunk_document([page], title="Test", chunk_size=1000, chunk_overlap=200)

        assert len(chunks) == 1, (
            f"Expected a single chunk for {len(text)}-char text with chunk_size=1000; "
            f"got {len(chunks)}"
        )

        single_chunk = chunks[0].content
        for sentence in sentences:
            # Each sentence should be fully contained in the single chunk
            assert sentence in single_chunk, (
                f"Sentence missing from chunk — possible mid-sentence split: "
                f"{sentence!r}"
            )

    def test_section_header_detected_in_chunk(self):
        """Section metadata enables the LLM to cite specific sections in the
        source list.  If headers aren't detected, all sources show empty
        section.
        """
        from mortimer.ingestion.chunker import chunk_document

        intro_text = (
            "1. Introduction\n"
            "This paper presents a novel approach to document retrieval using "
            "dense vector representations.  Prior work relied on sparse TF-IDF "
            "methods that fail to capture semantic similarity between queries "
            "and passages.  Our method addresses this limitation by training "
            "a bi-encoder on contrastive pairs sampled from academic corpora."
        )
        page = _make_page(intro_text)
        chunks = chunk_document([page], title="Test", chunk_size=1000, chunk_overlap=200)

        assert chunks, "Expected at least one chunk"
        first_chunk = chunks[0]
        assert first_chunk.metadata.section != "", (
            f"Section not detected for chunk starting with '1. Introduction'; "
            f"got section={first_chunk.metadata.section!r}"
        )

    def test_empty_pages_produce_no_chunks(self):
        """PDF pages sometimes extract as whitespace (scanned images, blank
        pages).  Empty pages should produce no chunks to avoid polluting the
        vector store.
        """
        from mortimer.ingestion.chunker import chunk_document

        page = _make_page("   \n  ")
        chunks = chunk_document([page], title="Test")

        assert chunks == [], (
            f"Expected no chunks for whitespace-only page; got {len(chunks)}"
        )


# ---------------------------------------------------------------------------
# Class: TestChunkMetadataIntegrity
# ---------------------------------------------------------------------------


class TestChunkMetadataIntegrity:
    """Verify that metadata is threaded correctly into every chunk produced
    by chunk_document.
    """

    def test_every_chunk_has_source_title_page(self):
        """The LLM uses source/title/page_number to populate the sources field
        of RAGResponse.  Missing metadata means citations are broken.
        """
        from mortimer.ingestion.chunker import chunk_document

        pages = [
            _make_page(_lorem(600), source="paper.pdf", page_number=i + 1)
            for i in range(3)
        ]
        chunks = chunk_document(pages, title="Attention Is All You Need")

        assert chunks, "Expected chunks from 3-page document"
        for chunk in chunks:
            assert chunk.metadata.source != "", (
                f"Empty source on chunk_id={chunk.chunk_id!r}"
            )
            assert chunk.metadata.title == "Attention Is All You Need", (
                f"Wrong title on chunk_id={chunk.chunk_id!r}: "
                f"{chunk.metadata.title!r}"
            )
            assert chunk.metadata.page_number >= 0, (
                f"Negative page_number on chunk_id={chunk.chunk_id!r}: "
                f"{chunk.metadata.page_number}"
            )

    def test_chunk_ids_are_unique_within_document(self):
        """ChromaDB uses chunk_id as the upsert key.  Duplicate IDs would
        silently overwrite earlier chunks, causing data loss.
        """
        from mortimer.ingestion.chunker import chunk_document

        # 5 pages of 600 chars each should comfortably produce 5+ chunks
        pages = [
            _make_page(_lorem(600), source="paper.pdf", page_number=i + 1)
            for i in range(5)
        ]
        chunks = chunk_document(pages, title="Test")

        assert len(chunks) >= 5, (
            f"Expected at least 5 chunks from a 5-page document; got {len(chunks)}"
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"Duplicate chunk IDs detected in {len(chunks)}-chunk document"
        )

    def test_chunk_ids_are_deterministic(self):
        """Deterministic IDs enable idempotent ingestion: re-ingesting the
        same PDF does not duplicate chunks in the vector store.
        """
        from mortimer.ingestion.chunker import chunk_document

        pages = [
            _make_page(_lorem(600), source="paper.pdf", page_number=i + 1)
            for i in range(3)
        ]
        chunks_run1 = chunk_document(pages, title="Test")
        chunks_run2 = chunk_document(pages, title="Test")

        ids_run1 = {c.chunk_id for c in chunks_run1}
        ids_run2 = {c.chunk_id for c in chunks_run2}

        assert ids_run1 == ids_run2, (
            f"Chunk IDs differ between runs — ingestion is not idempotent.\n"
            f"Run 1 only: {ids_run1 - ids_run2}\n"
            f"Run 2 only: {ids_run2 - ids_run1}"
        )
