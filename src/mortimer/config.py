"""Application configuration via pydantic-settings."""
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    openai_api_key: SecretStr
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: Path = Path("data/chroma_db")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    pdf_download_dir: Path = Path("data/pdfs")
