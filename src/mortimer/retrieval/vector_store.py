"""ChromaDB-backed vector store for document chunks."""
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from mortimer.models.schemas import DocumentChunk, RetrievedChunk


class VectorStore:
    """Persistent ChromaDB collection for storing and querying document chunks."""

    def __init__(self, persist_dir: Path, collection_name: str = "mortimer_docs") -> None:
        """Initialize the vector store.

        Args:
            persist_dir: Directory where ChromaDB persists data.
            collection_name: Name of the ChromaDB collection.
        """
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert document chunks with precomputed embeddings.

        Args:
            chunks: DocumentChunk objects to store.
            embeddings: Corresponding embedding vectors (must match length of chunks).
        """
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "source": c.metadata.source,
                "title": c.metadata.title,
                "page_number": c.metadata.page_number,
                "section": c.metadata.section,
                "url": c.metadata.url,
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a query embedding.

        Args:
            query_embedding: Float vector of the query.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by descending similarity.
        """
        total = self._collection.count()
        if total == 0:
            return []

        effective_k = min(top_k, total)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        return _parse_query_results(results)

    def has_document(self, source: str) -> bool:
        """Check whether any chunks from a given source are stored.

        Args:
            source: Source filename to check.

        Returns:
            True if at least one chunk from source exists.
        """
        results = self._collection.get(where={"source": source}, limit=1)
        return len(results["ids"]) > 0

    def reset(self) -> None:
        """Delete and recreate the collection, removing all stored data."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()


def _parse_query_results(results: dict) -> list[RetrievedChunk]:
    """Parse raw ChromaDB query results into RetrievedChunk objects.

    Args:
        results: Raw dict returned by chromadb collection.query().

    Returns:
        List of RetrievedChunk objects.
    """
    from mortimer.models.schemas import DocumentMetadata

    retrieved: list[RetrievedChunk] = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, content, meta, distance in zip(
        ids, documents, metadatas, distances, strict=True
    ):
        doc_meta = DocumentMetadata(
            source=meta["source"],
            title=meta["title"],
            page_number=int(meta["page_number"]),
            section=meta.get("section", ""),
            url=meta.get("url", ""),
        )
        from mortimer.models.schemas import DocumentChunk

        chunk = DocumentChunk(content=content, metadata=doc_meta, chunk_id=chunk_id)
        score = 1.0 - float(distance)
        retrieved.append(RetrievedChunk(chunk=chunk, score=score))

    return retrieved
