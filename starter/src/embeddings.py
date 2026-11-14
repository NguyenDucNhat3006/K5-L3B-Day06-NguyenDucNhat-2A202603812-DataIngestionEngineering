"""Embedding adapter using a real local SentenceTransformer model."""

from typing import Protocol
import math

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Local multilingual embedding provider."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("Could not determine embedding dimensions.")

        self.dimensions = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Document texts must be non-empty strings")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = vectors.tolist()
        if any(len(vector) != self.dimensions or not all(math.isfinite(float(v)) for v in vector) for vector in result):
            raise ValueError("Embedding model returned an invalid document vector")
        return result

    def embed_query(self, question: str) -> list[float]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question must be a non-empty string")
        vector = self.model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        result = vector.tolist()
        if len(result) != self.dimensions or not all(math.isfinite(float(v)) for v in result):
            raise ValueError("Embedding model returned an invalid query vector")
        return result
