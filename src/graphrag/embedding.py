"""Embedding + tokenization boundary (same model for both, on purpose).

Chunk sizes are counted with the embedding model's own tokenizer, so "200 tokens"
means 200 tokens to the model that actually has the 256-token input limit.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Character (start, end) offsets of each token in `text`."""
        ...


class Embedder(Protocol):
    max_tokens: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HFTokenizer:
    """Fast HuggingFace tokenizer. Offsets let us slice the ORIGINAL text, so chunks
    keep their case and formatting (MiniLM's tokenizer lowercases; decode() would too)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return [tuple(span) for span in enc["offset_mapping"]]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self.max_tokens = int(self._model.max_seq_length)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True: unit vectors, so L2 distance ranks exactly like
        # cosine similarity (||a-b||^2 = 2 - 2cos). No vector-store metric config needed.
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
