"""Unit tests for PDF loader module."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestExtractTitle:
    def test_extract_title_from_metadata(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_title

        title = extract_title(fixture_pdf_path)
        assert isinstance(title, str)
        assert len(title) > 0

    def test_extract_title_returns_string_for_empty_metadata(self, tmp_path):
        import fitz

        from mortimer.ingestion.loader import extract_title

        pdf_path = tmp_path / "no_meta.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "First line is the title fallback", fontsize=12)
        doc.save(str(pdf_path))
        doc.close()

        title = extract_title(pdf_path)
        assert isinstance(title, str)
        assert len(title) > 0

    def test_extract_title_nonexistent_file_raises(self):
        from mortimer.ingestion.loader import extract_title

        with pytest.raises((FileNotFoundError, Exception)):
            extract_title(Path("/nonexistent/path/file.pdf"))

    def test_extract_title_fallback_uses_first_line(self, tmp_path):
        import fitz

        from mortimer.ingestion.loader import extract_title

        pdf_path = tmp_path / "fallback.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 72), "My Document Title\nSome content below", fontsize=12)
        doc.set_metadata({"title": ""})
        doc.save(str(pdf_path))
        doc.close()

        title = extract_title(pdf_path)
        assert isinstance(title, str)
        assert len(title) > 0


class TestExtractPages:
    def test_extract_pages_returns_list(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        assert isinstance(pages, list)
        assert len(pages) > 0

    def test_extract_pages_count_matches_pdf(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        assert len(pages) == 3

    def test_extract_pages_have_correct_structure(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_pages
        from mortimer.models.schemas import DocumentPage

        pages = extract_pages(fixture_pdf_path)
        for page in pages:
            assert isinstance(page, DocumentPage)
            assert isinstance(page.page_number, int)
            assert isinstance(page.text, str)
            assert isinstance(page.source, str)

    def test_extract_pages_source_is_filename(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        for page in pages:
            assert page.source == fixture_pdf_path.name

    def test_extract_pages_page_numbers_are_sequential(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        for i, page in enumerate(pages):
            assert page.page_number == i

    def test_extract_pages_text_is_nonempty(self, fixture_pdf_path):
        from mortimer.ingestion.loader import extract_pages

        pages = extract_pages(fixture_pdf_path)
        for page in pages:
            assert len(page.text.strip()) > 0

    def test_extract_pages_nonexistent_file_raises(self):
        from mortimer.ingestion.loader import extract_pages

        with pytest.raises((FileNotFoundError, Exception)):
            extract_pages(Path("/nonexistent/file.pdf"))


def _make_stream_mock(content: bytes = b"%PDF-1.4 fake content"):
    """Return a context-manager mock that simulates httpx.stream()."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_bytes.return_value = iter([content])

    @contextmanager
    def _stream_ctx(*args, **kwargs):
        yield mock_response

    return _stream_ctx, mock_response


class TestDownloadPdf:
    def test_download_pdf_saves_to_destination(self, tmp_path, mocker):
        from mortimer.ingestion.loader import download_pdf

        stream_ctx, _ = _make_stream_mock()
        mocker.patch("httpx.stream", new=stream_ctx)

        url = "https://arxiv.org/pdf/2401.00001"
        result = download_pdf(url, tmp_path)

        assert result.parent == tmp_path
        assert result.suffix == ".pdf"

    def test_download_pdf_skips_if_exists(self, tmp_path, mocker):
        from mortimer.ingestion.loader import download_pdf

        existing = tmp_path / "2401.00001.pdf"
        existing.write_bytes(b"%PDF-1.4 existing")

        mock_stream = mocker.patch("httpx.stream")
        url = "https://arxiv.org/pdf/2401.00001"
        result = download_pdf(url, tmp_path)

        mock_stream.assert_not_called()
        assert result == existing

    def test_download_pdf_returns_path_object(self, tmp_path, mocker):
        from mortimer.ingestion.loader import download_pdf

        stream_ctx, _ = _make_stream_mock(b"%PDF-1.4 content")
        mocker.patch("httpx.stream", new=stream_ctx)

        result = download_pdf("https://arxiv.org/pdf/2401.00002", tmp_path)
        assert isinstance(result, Path)

    def test_download_pdf_invalid_url_raises(self, tmp_path, mocker):
        import httpx

        from mortimer.ingestion.loader import download_pdf

        @contextmanager
        def _failing_stream(*args, **kwargs):
            raise httpx.RequestError("connection failed")
            yield  # pragma: no cover

        mocker.patch("httpx.stream", new=_failing_stream)

        with pytest.raises(httpx.RequestError):
            download_pdf("https://invalid.url/bad", tmp_path)

    def test_download_pdf_rejects_http_url(self, tmp_path):
        """Security: plain HTTP URLs must be rejected."""
        from mortimer.ingestion.loader import download_pdf

        with pytest.raises(ValueError, match="Only HTTPS URLs are accepted"):
            download_pdf("http://arxiv.org/pdf/2401.00001", tmp_path)

    def test_download_pdf_rejects_file_scheme(self, tmp_path):
        """Security: file:// scheme must be rejected (SSRF vector)."""
        from mortimer.ingestion.loader import download_pdf

        with pytest.raises(ValueError, match="Only HTTPS URLs are accepted"):
            download_pdf("file:///etc/passwd", tmp_path)

    def test_download_pdf_enforces_size_limit(self, tmp_path, mocker):
        """Security: downloads exceeding _MAX_PDF_BYTES must raise ValueError."""
        from mortimer.ingestion import loader
        from mortimer.ingestion.loader import download_pdf

        # Produce two chunks whose combined size exceeds the cap.
        oversized_chunk = b"X" * (loader._MAX_PDF_BYTES // 2 + 1)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_bytes.return_value = iter([oversized_chunk, oversized_chunk])

        @contextmanager
        def _stream_ctx(*args, **kwargs):
            yield mock_response

        mocker.patch("httpx.stream", new=_stream_ctx)

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            download_pdf("https://arxiv.org/pdf/big.pdf", tmp_path)
