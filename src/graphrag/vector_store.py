"""Vector store boundary. ChromaDB in production, an in-memory fake in tests."""

from __future__ import annotations

import contextlib
from typing import Any, Protocol


class VectorStore(Protocol):
    def reset(self) -> None: ...

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def count(self) -> int: ...

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float, str]]:
        """Top-k (chunk_id, distance, text), nearest first."""
        ...

    def get(self, ids: list[str]) -> dict[str, tuple[str, list[float]]]:
        """chunk_id -> (text, embedding) for the ids that exist."""
        ...


class ChromaVectorStore:
    def __init__(self, path: str, collection: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        self._col = self._client.get_or_create_collection(collection)

    def reset(self) -> None:
        """Drop and recreate, so chunks from deleted or edited docs never linger."""
        # The "collection not found" exception type differs across Chroma versions.
        with contextlib.suppress(Exception):
            self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(self._name)

    def add(self, ids, embeddings, documents, metadatas) -> None:
        # Embeddings are always passed in, so Chroma's default embedding fn never runs.
        self._col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def count(self) -> int:
        return self._col.count()

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float, str]]:
        n = min(k, self.count())
        if n == 0:
            return []
        res = self._col.query(
            query_embeddings=[embedding], n_results=n, include=["documents", "distances"]
        )
        return list(zip(res["ids"][0], res["distances"][0], res["documents"][0], strict=True))

    def get(self, ids: list[str]) -> dict[str, tuple[str, list[float]]]:
        if not ids:
            return {}
        res = self._col.get(ids=ids, include=["documents", "embeddings"])
        # Embeddings may come back as a numpy array; never test it for truthiness.
        return {
            cid: (doc, [float(x) for x in emb])
            for cid, doc, emb in zip(res["ids"], res["documents"], res["embeddings"], strict=True)
        }
