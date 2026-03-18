"""Unit tests for Settings config."""
from pathlib import Path

import pytest


class TestSettings:
    def test_settings_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from importlib import reload

        import mortimer.config as config_mod

        reload(config_mod)
        from mortimer.config import Settings

        s = Settings()
        assert s.openai_api_key.get_secret_value() == "sk-test-key"

    def test_settings_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from mortimer.config import Settings

        s = Settings()
        assert s.openai_model == "gpt-4o-mini"
        assert s.embedding_model == "all-MiniLM-L6-v2"
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.retrieval_top_k == 5

    def test_settings_chroma_persist_dir_is_path(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from mortimer.config import Settings

        s = Settings()
        assert isinstance(s.chroma_persist_dir, Path)

    def test_settings_pdf_download_dir_is_path(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from mortimer.config import Settings

        s = Settings()
        assert isinstance(s.pdf_download_dir, Path)

    def test_settings_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from pydantic import ValidationError

        from mortimer.config import Settings

        with pytest.raises((ValidationError, Exception)):
            Settings()

    def test_settings_custom_chunk_size(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
        monkeypatch.setenv("CHUNK_SIZE", "500")
        from mortimer.config import Settings

        s = Settings()
        assert s.chunk_size == 500

    def test_settings_custom_top_k(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-key")
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        from mortimer.config import Settings

        s = Settings()
        assert s.retrieval_top_k == 10
