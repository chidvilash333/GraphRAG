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
