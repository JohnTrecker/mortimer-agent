"""Sentence-transformer based text embedder."""
import logging

from sentence_transformers import SentenceTransformer

# Suppress "BertModel LOAD REPORT" key-mismatch warnings from transformers
# and unauthenticated HF Hub request warnings. Progress bars are suppressed
# via HF_HUB_DISABLE_PROGRESS_BARS set in cli.py before library imports.
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


class Embedder:
    """Wraps SentenceTransformer to produce fixed-length float embeddings."""

    def __init__(self, model_name: str) -> None:
        """Initialize the embedder with a named sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier, e.g. 'all-MiniLM-L6-v2'.
        """
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into float vectors.

        Args:
            texts: Strings to embed. Empty list returns empty list.

        Returns:
            List of embedding vectors, one per input text.
        """
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: Query text.

        Returns:
            Embedding vector as list of floats.
        """
        vector = self._model.encode([query], convert_to_numpy=True)
        return vector[0].tolist()
