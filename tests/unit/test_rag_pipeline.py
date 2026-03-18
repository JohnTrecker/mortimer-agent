"""Unit tests for RAGPipeline -- external calls are mocked."""
import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
    """Patch Settings so no env var is needed."""
    from pydantic import SecretStr

    from mortimer.config import Settings

    settings = Settings.model_construct(
        openai_api_key=SecretStr("sk-fake"),
        openai_model="gpt-4o-mini",
        embedding_model="all-MiniLM-L6-v2",
        chroma_persist_dir=tmp_path / "chroma",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_top_k=3,
        pdf_download_dir=tmp_path / "pdfs",
    )
    monkeypatch.setattr("mortimer.pipeline.rag.Settings", lambda: settings)
    return settings


@pytest.fixture
def mock_llm_response():
    """Build a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(
        {
            "question": "What is attention?",
            "answer": "Attention is a mechanism...",
            "sources": ["paper.pdf | 1. Introduction | page 1"],
        }
    )
    response = MagicMock()
    response.choices = [choice]
    return response


class TestRAGPipelineIngest:
    def test_ingest_returns_ingestion_results(
        self, mock_settings, fixture_pdf_path, mocker
    ):
        from mortimer.models.schemas import IngestionResult
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )

        pipeline = RAGPipeline()
        results = pipeline.ingest(["https://fake.url/test_paper.pdf"])

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], IngestionResult)

    def test_ingest_result_has_chunk_count(self, mock_settings, fixture_pdf_path, mocker):
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )

        pipeline = RAGPipeline()
        results = pipeline.ingest(["https://fake.url/test_paper.pdf"])
        assert results[0].total_chunks > 0

    def test_ingest_skips_already_ingested(self, mock_settings, fixture_pdf_path, mocker):
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        results2 = pipeline.ingest(["https://fake.url/test_paper.pdf"])
        assert results2[0].total_chunks == 0

    def test_ingest_empty_list_returns_empty(self, mock_settings):
        from mortimer.pipeline.rag import RAGPipeline

        pipeline = RAGPipeline()
        results = pipeline.ingest([])
        assert results == []


class TestRAGPipelineQuery:
    def test_query_returns_rag_response(
        self, mock_settings, fixture_pdf_path, mocker, mock_llm_response
    ):
        from mortimer.models.schemas import RAGResponse
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_openai.return_value.chat.completions.create.return_value = mock_llm_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        result = pipeline.query("What is attention?")

        assert isinstance(result, RAGResponse)

    def test_query_response_has_required_fields(
        self, mock_settings, fixture_pdf_path, mocker, mock_llm_response
    ):
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_openai.return_value.chat.completions.create.return_value = mock_llm_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        result = pipeline.query("What is attention?")

        assert hasattr(result, "question")
        assert hasattr(result, "answer")
        assert hasattr(result, "sources")
        assert isinstance(result.sources, list)

    def test_query_serializes_to_correct_json_structure(
        self, mock_settings, fixture_pdf_path, mocker, mock_llm_response
    ):
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_openai.return_value.chat.completions.create.return_value = mock_llm_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        result = pipeline.query("What is attention?")
        data = result.model_dump()

        assert set(data.keys()) == {"question", "answer", "sources"}

    def test_query_calls_llm(
        self, mock_settings, fixture_pdf_path, mocker, mock_llm_response
    ):
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.return_value = mock_llm_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        pipeline.query("What is testing?")

        mock_create.assert_called_once()


class TestRAGPipelineReset:
    def test_reset_clears_vector_store(self, mock_settings, fixture_pdf_path, mocker):
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        pipeline.reset()
        assert pipeline._vector_store.count() == 0


class TestResolvePath:
    """Security tests for the _resolve_path helper."""

    def test_https_url_calls_download(self, tmp_path, mocker):
        from mortimer.pipeline.rag import _resolve_path

        mock_dl = mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=tmp_path / "paper.pdf",
        )
        _resolve_path("https://arxiv.org/pdf/1234.5678", tmp_path)
        mock_dl.assert_called_once()

    def test_http_url_raises(self, tmp_path):
        """Security: plain HTTP must be rejected."""
        from mortimer.pipeline.rag import _resolve_path

        with pytest.raises(ValueError, match="Plain HTTP URLs are not accepted"):
            _resolve_path("http://arxiv.org/pdf/1234.5678", tmp_path)

    def test_local_path_within_pdf_dir_is_allowed(self, tmp_path):
        """A relative path inside pdf_dir must resolve successfully."""
        from mortimer.pipeline.rag import _resolve_path

        pdf_file = tmp_path / "paper.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")
        result = _resolve_path(str(pdf_file), tmp_path)
        assert result == pdf_file.resolve()

    def test_path_traversal_outside_pdf_dir_raises(self, tmp_path):
        """Security: a path that escapes pdf_dir via '..' must be rejected."""
        from mortimer.pipeline.rag import _resolve_path

        traversal = str(tmp_path / ".." / "etc" / "passwd")
        with pytest.raises(ValueError, match="outside the allowed directory"):
            _resolve_path(traversal, tmp_path)


class TestQueryInputValidation:
    """Security tests for question input validation in RAGPipeline.query."""

    def test_empty_question_raises(self, mock_settings):
        from mortimer.pipeline.rag import RAGPipeline

        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="must not be empty"):
            pipeline.query("")

    def test_whitespace_only_question_raises(self, mock_settings):
        from mortimer.pipeline.rag import RAGPipeline

        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="must not be empty"):
            pipeline.query("   ")

    def test_oversized_question_raises(self, mock_settings):
        from mortimer.pipeline.rag import RAGPipeline

        pipeline = RAGPipeline()
        long_question = "a" * 2001
        with pytest.raises(ValueError, match="exceeds maximum length"):
            pipeline.query(long_question)

    def test_max_length_question_accepted(self, mock_settings, mocker):
        import json

        from mortimer.pipeline.rag import RAGPipeline

        choice = MagicMock()
        choice.message.content = json.dumps(
            {"question": "q", "answer": "a", "sources": []}
        )
        llm_response = MagicMock()
        llm_response.choices = [choice]

        mocker.patch("openai.OpenAI").return_value.chat.completions.create.return_value = (
            llm_response
        )

        pipeline = RAGPipeline()
        # Exactly at the limit should not raise
        result = pipeline.query("a" * 2000)
        assert result is not None
