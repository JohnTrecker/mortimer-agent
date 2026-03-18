"""Tests demonstrating data structuring goals.

These tests verify that LLMClient correctly parses and coerces LLM JSON output
into typed RAGResponse / Source Pydantic models. OpenAI is mocked throughout;
the tests exercise real JSON parsing and Pydantic coercion logic.
"""
import json
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_mock_response(raw_json: str) -> MagicMock:
    """Wrap a JSON string into a minimal mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = raw_json
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_client(mocker):
    """Return (LLMClient, mock_openai_client) with openai.OpenAI patched out."""
    mock_openai_client = MagicMock()
    mocker.patch("openai.OpenAI", return_value=mock_openai_client)

    from mortimer.generation.llm_client import LLMClient

    client = LLMClient(api_key="test-api-key", model="test")
    return client, mock_openai_client


def _set_response(llm_client_tuple, raw_json: str) -> None:
    """Configure the mock to return raw_json as the next completion."""
    _client, mock_openai_client = llm_client_tuple
    mock_openai_client.chat.completions.create.return_value = _make_mock_response(raw_json)


# ---------------------------------------------------------------------------
# TestWellFormedLLMOutput
# ---------------------------------------------------------------------------


class TestWellFormedLLMOutput:
    def test_valid_json_with_source_dicts_parses_to_rag_response(self, llm_client):
        """Properly structured JSON with Source dicts should produce a RAGResponse."""
        from mortimer.models.schemas import RAGResponse

        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "What is ML?",
                "answer": "Machine learning is a subfield of AI.",
                "sources": [{"title": "ml.pdf", "page": "1", "url": ""}],
            }
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "What is ML?"}])

        assert isinstance(result, RAGResponse)
        assert result.question == "What is ML?"
        assert result.answer == "Machine learning is a subfield of AI."
        assert len(result.sources) == 1

    def test_empty_sources_list_accepted(self, llm_client):
        """sources=[] should produce a RAGResponse with an empty sources list."""
        client, _ = llm_client
        payload = json.dumps(
            {"question": "Empty?", "answer": "Yes.", "sources": []}
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "Empty?"}])

        assert result.sources == []

    def test_source_fields_are_typed_strings(self, llm_client):
        """Each Source should have str title, page, and url fields."""
        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "Q?",
                "answer": "A.",
                "sources": [
                    {"title": "paper.pdf", "page": "42", "url": "https://example.com"}
                ],
            }
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "Q?"}])

        source = result.sources[0]
        assert isinstance(source.title, str)
        assert isinstance(source.page, str)
        assert isinstance(source.url, str)


# ---------------------------------------------------------------------------
# TestAnswerCoercion
# ---------------------------------------------------------------------------


class TestAnswerCoercion:
    def test_dict_answer_coerced_to_json_string(self, llm_client):
        """When the LLM returns answer as a dict it must be coerced to a JSON string."""
        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "Compare A vs B",
                "answer": {"A": "fast", "B": "slow"},
                "sources": [],
            }
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "Compare A vs B"}])

        assert isinstance(result.answer, str)
        assert json.loads(result.answer) == {"A": "fast", "B": "slow"}

    def test_list_answer_coerced_to_json_string(self, llm_client):
        """When the LLM returns answer as a list it must be coerced to a JSON string."""
        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "List items",
                "answer": ["item1", "item2"],
                "sources": [],
            }
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "List items"}])

        assert isinstance(result.answer, str)
        assert result.answer == '["item1", "item2"]'

    def test_string_answer_unchanged(self, llm_client):
        """A string answer must pass through without modification."""
        client, _ = llm_client
        payload = json.dumps(
            {"question": "Plain?", "answer": "plain text", "sources": []}
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "Plain?"}])

        assert result.answer == "plain text"


# ---------------------------------------------------------------------------
# TestSourceCoercion
# ---------------------------------------------------------------------------


class TestSourceCoercion:
    def test_string_sources_coerced_to_source_dicts(self, llm_client):
        """Plain string sources should be wrapped into Source(title=s, page='', url='')."""
        from mortimer.models.schemas import Source

        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "Q?",
                "answer": "A.",
                "sources": ["paper.pdf | Abstract | page 1"],
            }
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "Q?"}])

        assert len(result.sources) == 1
        src = result.sources[0]
        assert isinstance(src, Source)
        assert src.title == "paper.pdf | Abstract | page 1"
        assert src.page == ""
        assert src.url == ""

    def test_mixed_source_formats_first_element_determines_path(self, llm_client):
        """Document the behavior when sources[0] is a string but sources[1] is a dict.

        The LLMClient coercion logic checks sources[0] to decide the branch. When the
        first element is a plain string the branch wraps every element as
        Source(title=s, page='', url='') — including the second element, which is a raw
        dict. Passing a dict as the 'title' string field violates the Source schema, so
        Pydantic raises a ValidationError.

        This test documents that mixed-format source lists are an unsupported input and
        that the coercion branch does NOT silently corrupt data — it fails loudly.
        """
        from pydantic import ValidationError

        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "Q?",
                "answer": "A.",
                "sources": ["plain string", {"title": "T", "page": "2", "url": ""}],
            }
        )
        _set_response(llm_client, payload)

        with pytest.raises(ValidationError):
            client.generate([{"role": "user", "content": "Q?"}])


# ---------------------------------------------------------------------------
# TestSerializationFidelity
# ---------------------------------------------------------------------------


class TestSerializationFidelity:
    def test_rag_response_json_round_trip(self, llm_client):
        """model_dump_json() then model_validate_json() must produce an equal object."""
        from mortimer.models.schemas import RAGResponse

        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "Round trip?",
                "answer": "Yes it works.",
                "sources": [{"title": "doc.pdf", "page": "7", "url": "https://x.com"}],
            }
        )
        _set_response(llm_client, payload)

        original = client.generate([{"role": "user", "content": "Round trip?"}])
        json_str = original.model_dump_json()
        restored = RAGResponse.model_validate_json(json_str)

        assert restored == original

    def test_model_dump_keys_match_api_contract(self, llm_client):
        """model_dump() must contain exactly {question, answer, sources}.
        Each source must have exactly {title, page, url}.
        """
        client, _ = llm_client
        payload = json.dumps(
            {
                "question": "Contract?",
                "answer": "Yes.",
                "sources": [{"title": "f.pdf", "page": "1", "url": ""}],
            }
        )
        _set_response(llm_client, payload)

        result = client.generate([{"role": "user", "content": "Contract?"}])
        dumped = result.model_dump()

        assert set(dumped.keys()) == {"question", "answer", "sources"}
        assert len(dumped["sources"]) == 1
        assert set(dumped["sources"][0].keys()) == {"title", "page", "url"}


# ---------------------------------------------------------------------------
# TestMalformedLLMOutput
# ---------------------------------------------------------------------------


class TestMalformedLLMOutput:
    def test_missing_question_field_raises_validation_error(self, llm_client):
        """JSON missing the 'question' field should raise pydantic ValidationError."""
        from pydantic import ValidationError

        client, _ = llm_client
        payload = json.dumps({"answer": "A.", "sources": []})
        _set_response(llm_client, payload)

        with pytest.raises(ValidationError):
            client.generate([{"role": "user", "content": "anything"}])

    def test_invalid_json_raises_json_decode_error(self, llm_client):
        """Non-JSON text returned by the LLM should raise json.JSONDecodeError."""
        client, _ = llm_client
        _set_response(llm_client, "this is not json {{{")

        with pytest.raises(json.JSONDecodeError):
            client.generate([{"role": "user", "content": "anything"}])
