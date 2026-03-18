"""Unit tests for the Click CLI -- pipeline is mocked."""
import json

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_pipeline(mocker):
    """Mock RAGPipeline to avoid real embedding/LLM calls."""
    mock = mocker.MagicMock()
    mocker.patch("mortimer.cli.RAGPipeline", return_value=mock)
    return mock


class TestIngestCommand:
    def test_ingest_with_url_succeeds(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import IngestionResult

        mock_pipeline.ingest.return_value = [
            IngestionResult(
                document_path="/tmp/paper.pdf",
                total_chunks=42,
                title="Test Paper",
            )
        ]

        result = runner.invoke(cli, ["ingest", "https://arxiv.org/pdf/2401.00001"])
        assert result.exit_code == 0
        assert "42" in result.output

    def test_ingest_multiple_urls(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import IngestionResult

        mock_pipeline.ingest.return_value = [
            IngestionResult(document_path="/tmp/a.pdf", total_chunks=10, title="A"),
            IngestionResult(document_path="/tmp/b.pdf", total_chunks=20, title="B"),
        ]

        result = runner.invoke(
            cli,
            [
                "ingest",
                "https://arxiv.org/pdf/2401.00001",
                "https://arxiv.org/pdf/2401.00002",
            ],
        )
        assert result.exit_code == 0
        mock_pipeline.ingest.assert_called_once()

    def test_ingest_default_urls_called_when_no_args(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import IngestionResult

        mock_pipeline.ingest.return_value = [
            IngestionResult(document_path="/tmp/a.pdf", total_chunks=5, title="Default")
        ]

        result = runner.invoke(cli, ["ingest"])
        assert result.exit_code == 0
        mock_pipeline.ingest.assert_called_once()
        urls_passed = mock_pipeline.ingest.call_args[0][0]
        assert len(urls_passed) >= 1

    def test_ingest_shows_skipped_message(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import IngestionResult

        mock_pipeline.ingest.return_value = [
            IngestionResult(document_path="/tmp/a.pdf", total_chunks=0, title="Already")
        ]

        result = runner.invoke(cli, ["ingest", "https://arxiv.org/pdf/2401.00001"])
        assert result.exit_code == 0
        output = result.output.lower()
        assert "skip" in output or "already" in output or "0" in output


class TestAskCommand:
    def test_ask_outputs_json(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import RAGResponse

        mock_pipeline.query.return_value = RAGResponse(
            question="What is attention?",
            answer="Attention is a mechanism.",
            sources=["paper.pdf section 1"],
        )

        result = runner.invoke(cli, ["ask", "What is attention?"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["question"] == "What is attention?"
        assert data["answer"] == "Attention is a mechanism."
        assert isinstance(data["sources"], list)

    def test_ask_response_json_structure(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import RAGResponse

        mock_pipeline.query.return_value = RAGResponse(
            question="Q?",
            answer="A.",
            sources=[],
        )

        result = runner.invoke(cli, ["ask", "Q?"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert set(data.keys()) == {"question", "answer", "sources"}

    def test_ask_calls_pipeline_query(self, runner, mock_pipeline):
        from mortimer.cli import cli
        from mortimer.models.schemas import RAGResponse

        mock_pipeline.query.return_value = RAGResponse(
            question="Test question?", answer="Test answer.", sources=[]
        )

        runner.invoke(cli, ["ask", "Test question?"])
        mock_pipeline.query.assert_called_once_with("Test question?")

    def test_ask_missing_question_shows_error(self, runner, mock_pipeline):
        from mortimer.cli import cli

        result = runner.invoke(cli, ["ask"])
        assert result.exit_code != 0


class TestResetCommand:
    def test_reset_calls_pipeline_reset(self, runner, mock_pipeline):
        from mortimer.cli import cli

        result = runner.invoke(cli, ["reset"])
        assert result.exit_code == 0
        mock_pipeline.reset.assert_called_once()

    def test_reset_outputs_confirmation(self, runner, mock_pipeline):
        from mortimer.cli import cli

        result = runner.invoke(cli, ["reset"])
        assert result.exit_code == 0
        assert len(result.output.strip()) > 0
