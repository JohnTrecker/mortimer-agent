"""Unit tests for prompt building utilities."""
import pytest


class TestLoadTemplate:
    def test_load_system_template(self):
        from mortimer.generation.prompt import load_template

        text = load_template("rag_system")
        assert isinstance(text, str)
        assert len(text) > 0

    def test_load_user_template(self):
        from mortimer.generation.prompt import load_template

        text = load_template("rag_user")
        assert isinstance(text, str)
        assert "{context}" in text
        assert "{question}" in text

    def test_load_template_nonexistent_raises(self):
        from mortimer.generation.prompt import load_template

        with pytest.raises(FileNotFoundError):
            load_template("nonexistent_template")


class TestFormatContext:
    def test_format_context_returns_string(self, sample_chunks):
        from mortimer.generation.prompt import format_context
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        result = format_context(retrieved)
        assert isinstance(result, str)

    def test_format_context_numbers_chunks(self, sample_chunks):
        from mortimer.generation.prompt import format_context
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        result = format_context(retrieved)
        assert "[1]" in result
        assert "[2]" in result

    def test_format_context_includes_source(self, sample_chunks):
        from mortimer.generation.prompt import format_context
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        result = format_context(retrieved)
        assert "paper.pdf" in result

    def test_format_context_includes_content(self, sample_chunks):
        from mortimer.generation.prompt import format_context
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        result = format_context(retrieved)
        assert "Introduction text" in result

    def test_format_context_empty_list_returns_string(self):
        from mortimer.generation.prompt import format_context

        result = format_context([])
        assert isinstance(result, str)
        assert result == ""


class TestBuildMessages:
    def test_build_messages_returns_list(self, sample_chunks):
        from mortimer.generation.prompt import build_messages
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        messages = build_messages("What is this about?", retrieved)
        assert isinstance(messages, list)

    def test_build_messages_has_system_and_user(self, sample_chunks):
        from mortimer.generation.prompt import build_messages
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        messages = build_messages("What is X?", retrieved)
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_build_messages_user_contains_question(self, sample_chunks):
        from mortimer.generation.prompt import build_messages
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        question = "What is the main finding?"
        messages = build_messages(question, retrieved)
        user_msg = next(m for m in messages if m["role"] == "user")
        assert question in user_msg["content"]

    def test_build_messages_does_not_mutate_chunks(self, sample_chunks):
        from mortimer.generation.prompt import build_messages
        from mortimer.models.schemas import RetrievedChunk

        retrieved = [RetrievedChunk(chunk=c, score=0.9) for c in sample_chunks]
        original_ids = [r.chunk.chunk_id for r in retrieved]
        build_messages("Test?", retrieved)
        assert [r.chunk.chunk_id for r in retrieved] == original_ids
