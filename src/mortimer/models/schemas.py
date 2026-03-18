"""Pydantic v2 schema models for Mortimer Agent."""
from pydantic import BaseModel, ConfigDict


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str
    page: str
    url: str


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    title: str
    page_number: int
    section: str = ""
    url: str = ""


class DocumentPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    metadata: DocumentMetadata
    chunk_id: str


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk
    score: float


class RAGResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    answer: str
    sources: list[Source]


class IngestionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_path: str
    total_chunks: int
    title: str
