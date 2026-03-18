"""Unit tests for LLMClient -- all OpenAI calls are mocked."""
import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_openai_client(mocker):
    """Patch openai.OpenAI to prevent real API calls."""
    mock_client = MagicMock()
    mocker.patch("openai.OpenAI", return_value=mock_client)
    return mock_client


def _make_chat_response(content: str):
    """Build a minimal mock ChatCompletion response object."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestLLMClient:
    def test_generate_returns_rag_response(self, mock_openai_client):
        from mortimer.generation.llm_client import LLMClient
        from mortimer.models.schemas import RAGResponse

        payload = json.dumps(
            {"question": "What is X?", "answer": "X is Y.", "sources": ["source1"]}
        )
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(payload)

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is X?"},
        ]
        result = client.generate(messages)
        assert isinstance(result, RAGResponse)

    def test_generate_maps_fields_correctly(self, mock_openai_client):
        from mortimer.generation.llm_client import LLMClient

        payload = json.dumps(
            {
                "question": "What is transfer learning?",
                "answer": "Transfer learning is...",
                "sources": ["paper.pdf section 1", "paper.pdf section 2"],
            }
        )
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(payload)

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        result = client.generate([{"role": "user", "content": "What is transfer learning?"}])

        assert result.question == "What is transfer learning?"
        assert result.answer == "Transfer learning is..."
        assert len(result.sources) == 2

    def test_generate_with_empty_sources(self, mock_openai_client):
        from mortimer.generation.llm_client import LLMClient

        payload = json.dumps({"question": "Q?", "answer": "A.", "sources": []})
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(payload)

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        result = client.generate([{"role": "user", "content": "Q?"}])
        assert result.sources == []

    def test_generate_calls_openai_with_json_mode(self, mock_openai_client):
        from mortimer.generation.llm_client import LLMClient

        payload = json.dumps({"question": "Q?", "answer": "A.", "sources": []})
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(payload)

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        client.generate([{"role": "user", "content": "Q?"}])

        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs is not None
        kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        response_format = kwargs.get("response_format", {})
        assert response_format.get("type") == "json_object"

    def test_generate_invalid_json_raises(self, mock_openai_client):
        from mortimer.generation.llm_client import LLMClient

        mock_openai_client.chat.completions.create.return_value = _make_chat_response(
            "not valid json {"
        )

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        with pytest.raises((ValueError, Exception)):
            client.generate([{"role": "user", "content": "Q?"}])

    def test_generate_coerces_dict_answer_to_string(self, mock_openai_client):
        """LLM sometimes returns answer as a nested dict; it must be coerced to str."""
        from mortimer.generation.llm_client import LLMClient

        payload = json.dumps(
            {
                "question": "Compare human vs LLM thinking",
                "answer": {"Human Thinking": {"Natural": "..."}, "LLM Thinking": "..."},
                "sources": ["doc.pdf section 1"],
            }
        )
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(payload)

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        result = client.generate([{"role": "user", "content": "Compare human vs LLM thinking"}])

        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_generate_coerces_list_answer_to_string(self, mock_openai_client):
        """LLM sometimes returns answer as a list; it must be coerced to str."""
        from mortimer.generation.llm_client import LLMClient

        payload = json.dumps(
            {
                "question": "List the steps",
                "answer": ["Step 1", "Step 2", "Step 3"],
                "sources": [],
            }
        )
        mock_openai_client.chat.completions.create.return_value = _make_chat_response(payload)

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        result = client.generate([{"role": "user", "content": "List the steps"}])

        assert isinstance(result.answer, str)

    def test_generate_openai_error_propagates(self, mock_openai_client):
        from openai import OpenAIError

        from mortimer.generation.llm_client import LLMClient

        mock_openai_client.chat.completions.create.side_effect = OpenAIError("rate limit")

        client = LLMClient(api_key="sk-fake", model="gpt-4o-mini")
        with pytest.raises(OpenAIError):
            client.generate([{"role": "user", "content": "Q?"}])
