"""On-disk cache of extraction results.

Key = extractor fingerprint (model + prompt + settings) + chunk content hash. So
re-ingesting unchanged docs costs zero Groq calls, while editing a doc or the prompt
automatically re-extracts only what changed.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from graphrag.extraction.parse import Triple


class ExtractionCache:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, list[dict[str, str]]] = {}
        if self._path.is_file():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    @staticmethod
    def key(fingerprint: str, content_hash: str) -> str:
        return f"{fingerprint}:{content_hash}"

    def get(self, key: str) -> list[Triple] | None:
        items = self._data.get(key)
        return None if items is None else [Triple(**item) for item in items]

    def put(self, key: str, triples: list[Triple]) -> None:
        self._data[key] = [asdict(t) for t in triples]

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=1), encoding="utf-8")
        tmp.replace(self._path)  # atomic: a crash mid-write can't corrupt the cache

    def __len__(self) -> int:
        return len(self._data)
