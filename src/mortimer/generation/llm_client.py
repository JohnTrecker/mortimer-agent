"""OpenAI LLM client for generating RAG responses."""
import json

import openai

from mortimer.models.schemas import RAGResponse


class LLMClient:
    """Wraps OpenAI chat completions with JSON mode output."""

    def __init__(self, api_key: str, model: str) -> None:
        """Initialize the LLM client.

        Args:
            api_key: OpenAI API key string.
            model: Model identifier, e.g. 'gpt-4o-mini'.
        """
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate(self, messages: list[dict]) -> RAGResponse:
        """Call the OpenAI chat completions API and parse the JSON response.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            RAGResponse parsed from the model's JSON output.

        Raises:
            openai.OpenAIError: If the API call fails.
            json.JSONDecodeError: If the response is not valid JSON.
            pydantic.ValidationError: If the JSON does not match RAGResponse schema.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)
        # Coerce answer to str: the LLM occasionally returns a nested dict or list
        # despite prompt instructions to return a flat string.
        if not isinstance(data.get("answer"), str):
            data["answer"] = json.dumps(data["answer"])
        return RAGResponse(**data)
