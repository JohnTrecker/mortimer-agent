"""Integration tests for the full RAG query pipeline (LLM is mocked)."""
import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
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
def llm_mock_response():
    choice = MagicMock()
    choice.message.content = json.dumps(
        {
            "question": "What topics are covered?",
            "answer": "The document covers testing methodologies.",
            "sources": ["test_paper.pdf | Abstract | page 1"],
        }
    )
    response = MagicMock()
    response.choices = [choice]
    return response


class TestRAGQueryIntegration:
    def test_full_query_pipeline_returns_rag_response(
        self, mock_settings, fixture_pdf_path, mocker, llm_mock_response
    ):
        """Test: ingest real PDF -> query -> get RAGResponse with correct structure."""
        from mortimer.models.schemas import RAGResponse
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_openai.return_value.chat.completions.create.return_value = llm_mock_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        result = pipeline.query("What topics are covered?")

        assert isinstance(result, RAGResponse)
        assert result.question == "What topics are covered?"
        assert isinstance(result.answer, str)
        assert isinstance(result.sources, list)

    def test_query_response_json_matches_required_schema(
        self, mock_settings, fixture_pdf_path, mocker, llm_mock_response
    ):
        """The serialized JSON output must have exactly question, answer, sources."""
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_openai.return_value.chat.completions.create.return_value = llm_mock_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        result = pipeline.query("What topics are covered?")
        data = result.model_dump()

        assert set(data.keys()) == {"question", "answer", "sources"}
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert all(isinstance(s, dict) for s in data["sources"])
        assert all({"title", "page", "url"} <= s.keys() for s in data["sources"])

    def test_ingest_then_reset_empties_store(
        self, mock_settings, fixture_pdf_path, mocker
    ):
        """Ingest then reset should leave vector store empty."""
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )

        pipeline = RAGPipeline()
        results = pipeline.ingest(["https://fake.url/test_paper.pdf"])
        assert results[0].total_chunks > 0

        pipeline.reset()
        assert pipeline._vector_store.count() == 0

    def test_ingest_populates_vector_store(
        self, mock_settings, fixture_pdf_path, mocker
    ):
        """After ingestion, vector store should contain chunks."""
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )

        pipeline = RAGPipeline()
        results = pipeline.ingest(["https://fake.url/test_paper.pdf"])

        assert pipeline._vector_store.count() == results[0].total_chunks

    def test_llm_receives_context_in_messages(
        self, mock_settings, fixture_pdf_path, mocker, llm_mock_response
    ):
        """Messages passed to LLM must contain retrieved context."""
        from mortimer.pipeline.rag import RAGPipeline

        mocker.patch(
            "mortimer.pipeline.rag.download_pdf",
            return_value=fixture_pdf_path,
        )
        mock_openai = mocker.patch("openai.OpenAI")
        mock_create = mock_openai.return_value.chat.completions.create
        mock_create.return_value = llm_mock_response

        pipeline = RAGPipeline()
        pipeline.ingest(["https://fake.url/test_paper.pdf"])
        pipeline.query("What is this document about?")

        call_kwargs = mock_create.call_args.kwargs
        messages = call_kwargs["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")

        assert "What is this document about?" in user_msg["content"]
