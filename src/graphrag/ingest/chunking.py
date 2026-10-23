"""Token-aware sliding-window chunking with stable IDs.

IDs are `<doc_id>:<index>` (e.g. `zoraxin:000`): deterministic for the same input,
short enough for an LLM to cite reliably in Phase 4. A content hash is stored
alongside for change detection and as the extraction-cache key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from graphrag.embedding import Tokenizer
from graphrag.ingest.documents import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    start_char: int
    end_char: int
    n_tokens: int
    content_hash: str


def _at_word_start(text: str, char_pos: int) -> bool:
    return char_pos == 0 or text[char_pos - 1].isspace()


def chunk_document(
    doc: Document, tokenizer: Tokenizer, *, max_tokens: int, overlap_tokens: int
) -> list[Chunk]:
    spans = tokenizer.token_spans(doc.text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    step = max_tokens - overlap_tokens
    start = 0
    while True:
        # Don't begin a chunk mid-word (e.g. "##rax" of "Zoraxin"). Slide forward a few
        # sub-word tokens; the overlap means the whole word is in the previous chunk.
        adj = start
        while (
            adj > 0 and adj < start + overlap_tokens and not _at_word_start(doc.text, spans[adj][0])
        ):
            adj += 1
        if adj >= len(spans):
            break
        end = min(adj + max_tokens, len(spans))
        s_char, e_char = spans[adj][0], spans[end - 1][1]
        text = doc.text[s_char:e_char]
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{len(chunks):03d}",
                doc_id=doc.doc_id,
                index=len(chunks),
                text=text,
                start_char=s_char,
                end_char=e_char,
                n_tokens=end - adj,
                content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            )
        )
        if end >= len(spans):
            break
        start += step
    return chunks
