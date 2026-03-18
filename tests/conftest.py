"""Shared test fixtures for Mortimer Agent tests."""
from pathlib import Path

import fitz  # PyMuPDF
import pytest


# ---------------------------------------------------------------------------
# Retrieval fixtures (real embedder + real ChromaDB)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def embedder():
    """Load the real sentence-transformer model once per test session."""
    from mortimer.retrieval.embedder import Embedder

    return Embedder("all-MiniLM-L6-v2")


@pytest.fixture(scope="session")
def populated_vector_store(tmp_path_factory, embedder):
    """Create a temporary ChromaDB VectorStore pre-loaded with 6 topically distinct chunks."""
    from mortimer.models.schemas import DocumentChunk, DocumentMetadata
    from mortimer.retrieval.vector_store import VectorStore

    tmp_dir = tmp_path_factory.mktemp("chroma")
    store = VectorStore(
        persist_dir=tmp_dir / "chroma",
        collection_name="test_retrieval",
    )

    topic_chunks = [
        DocumentChunk(
            content="Neural networks learn by adjusting weights through backpropagation using gradient descent.",
            metadata=DocumentMetadata(
                source="ml.pdf",
                title="Machine Learning",
                page_number=1,
                section="Neural Networks",
            ),
            chunk_id="chunk-ml",
        ),
        DocumentChunk(
            content="Photosynthesis is the process by which plants convert sunlight into chemical energy using chlorophyll.",
            metadata=DocumentMetadata(
                source="biology.pdf",
                title="Plant Biology",
                page_number=2,
                section="Photosynthesis",
            ),
            chunk_id="chunk-bio",
        ),
        DocumentChunk(
            content="The French Revolution began in 1789 with the storming of the Bastille prison in Paris.",
            metadata=DocumentMetadata(
                source="history.pdf",
                title="European History",
                page_number=3,
                section="French Revolution",
            ),
            chunk_id="chunk-hist",
        ),
        DocumentChunk(
            content="Quantum computing uses qubits that can exist in superposition to perform parallel computations.",
            metadata=DocumentMetadata(
                source="physics.pdf",
                title="Quantum Physics",
                page_number=4,
                section="Quantum Computing",
            ),
            chunk_id="chunk-quantum",
        ),
        DocumentChunk(
            content="Shakespeare wrote 37 plays including Hamlet, Macbeth, and A Midsummer Night's Dream.",
            metadata=DocumentMetadata(
                source="literature.pdf",
                title="English Literature",
                page_number=5,
                section="Shakespeare",
            ),
            chunk_id="chunk-lit",
        ),
        DocumentChunk(
            content="The Maillard reaction occurs when proteins and sugars react at high temperatures during cooking.",
            metadata=DocumentMetadata(
                source="cooking.pdf",
                title="Culinary Science",
                page_number=6,
                section="Cooking Chemistry",
            ),
            chunk_id="chunk-cook",
        ),
    ]

    texts = [c.content for c in topic_chunks]
    embeddings = embedder.embed_texts(texts)
    store.add_chunks(topic_chunks, embeddings)
    return store


@pytest.fixture(scope="session")
def fixture_pdf_path(tmp_path_factory) -> Path:
    """Create a minimal test PDF using PyMuPDF."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = tmp_dir / "test_paper.pdf"

    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Test Paper: A Study of Testing\n\nAbstract\nThis is the abstract text for testing.",
        fontsize=12,
    )

    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "1. Introduction\nThis section introduces the topic. Testing is important.",
        fontsize=12,
    )

    page3 = doc.new_page()
    page3.insert_text(
        (50, 72),
        "2. Methods\nThis section describes the methodology used in the study.",
        fontsize=12,
    )

    doc.set_metadata({"title": "Test Paper: A Study of Testing", "author": "Test Author"})
    doc.save(str(pdf_path))
    doc.close()

    return pdf_path


@pytest.fixture
def sample_chunks():
    """Return a list of sample DocumentChunk objects."""
    from mortimer.models.schemas import DocumentChunk, DocumentMetadata

    meta1 = DocumentMetadata(
        source="paper.pdf",
        title="Test Paper",
        page_number=1,
        section="1. Introduction",
    )
    meta2 = DocumentMetadata(
        source="paper.pdf",
        title="Test Paper",
        page_number=2,
        section="2. Methods",
    )
    return [
        DocumentChunk(
            content="Introduction text about important concepts.",
            metadata=meta1,
            chunk_id="chunk-001",
        ),
        DocumentChunk(
            content="Methods text describing the approach used.",
            metadata=meta2,
            chunk_id="chunk-002",
        ),
    ]
