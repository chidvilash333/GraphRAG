"""Load .txt/.md documents from a directory, with stable slug IDs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: str
    text: str


def slugify(name: str) -> str:
    return _NON_ALNUM.sub("-", name.casefold()).strip("-") or "doc"


def load_documents(docs_dir: str | Path) -> list[Document]:
    """Recursively load supported files in sorted order (deterministic chunk IDs)."""
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {root}")

    docs: list[Document] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        rel = path.relative_to(root)
        doc_id = slugify(str(rel.with_suffix("")))
        if doc_id in seen:
            raise ValueError(f"Doc ID collision '{doc_id}': {seen[doc_id]} and {path}")
        seen[doc_id] = path
        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(Document(doc_id=doc_id, path=str(rel), text=text))
    return docs
