"""Shared test fixtures for Mortimer Agent tests."""
from pathlib import Path

import fitz  # PyMuPDF
import pytest


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
