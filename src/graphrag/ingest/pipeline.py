"""Ingest: documents -> chunks -> (embeddings -> ChromaDB) + (triples -> graph).

Every ingest is a full rebuild of the vector index and graph. That's cheap here:
embedding is local and LLM extraction is served from the cache for unchanged
chunks. It also guarantees the two stores always describe the same chunk set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.config import AppConfig
from graphrag.embedding import Embedder, Tokenizer
from graphrag.extraction import TripleExtractor
from graphrag.graph import KnowledgeGraph
from graphrag.ingest.cache import ExtractionCache
from graphrag.ingest.chunking import Chunk, chunk_document
from graphrag.ingest.documents import load_documents
from graphrag.ratelimit import RateLimiter
from graphrag.vector_store import VectorStore

log = logging.getLogger(__name__)
_SPECIAL_TOKENS = 2  # [CLS] and [SEP], added by the embedder on top of chunk tokens


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    failed_chunks: list[str] = field(default_factory=list)
    triples: int = 0
    dropped_triples: int = 0
    graph: dict[str, int] = field(default_factory=dict)


def _extract_all(
    chunks: list[Chunk],
    extractor: TripleExtractor,
    cache: ExtractionCache,
    limiter: RateLimiter,
    kg: KnowledgeGraph,
    stats: IngestStats,
) -> None:
    for i, chunk in enumerate(chunks, 1):
        key = ExtractionCache.key(extractor.fingerprint, chunk.content_hash)
        triples = cache.get(key)
        if triples is not None:
            stats.cache_hits += 1
        else:
            limiter.wait()
            result = extractor.extract(chunk.text)
            stats.llm_calls += 1
            stats.dropped_triples += len(result.dropped)
            if result.error:
                # Not cached, so the next ingest retries this chunk.
                stats.failed_chunks.append(chunk.chunk_id)
                continue
            triples = result.triples
            cache.put(key, triples)
            cache.save()  # save as we go: an interrupted ingest keeps its progress
            log.info("[%d/%d] %s: %d triples", i, len(chunks), chunk.chunk_id, len(triples))
        kg.add_triples(triples, chunk.chunk_id)
        stats.triples += len(triples)


def run_ingest(
    cfg: AppConfig,
    *,
    tokenizer: Tokenizer,
    embedder: Embedder,
    store: VectorStore,
    extractor: TripleExtractor,
    limiter: RateLimiter | None = None,
) -> tuple[KnowledgeGraph, IngestStats]:
    if cfg.chunking.max_tokens + _SPECIAL_TOKENS > embedder.max_tokens:
        raise ValueError(
            f"chunking.max_tokens={cfg.chunking.max_tokens} exceeds the embedding model's "
            f"limit ({embedder.max_tokens} incl. {_SPECIAL_TOKENS} special tokens)"
        )

    stats = IngestStats()
    docs = load_documents(cfg.paths.docs_dir)
    stats.documents = len(docs)
    chunks = [
        c
        for d in docs
        for c in chunk_document(
            d,
            tokenizer,
            max_tokens=cfg.chunking.max_tokens,
            overlap_tokens=cfg.chunking.overlap_tokens,
        )
    ]
    stats.chunks = len(chunks)
    log.info("Loaded %d docs -> %d chunks", len(docs), len(chunks))

    # 1) Vector side
    store.reset()
    if chunks:
        store.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embedder.embed([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "index": c.index,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "content_hash": c.content_hash,
                }
                for c in chunks
            ],
        )

    # 2) Graph side
    kg = KnowledgeGraph()
    cache = ExtractionCache(cfg.paths.cache_path)
    limiter = limiter or RateLimiter(cfg.llm.requests_per_minute)
    _extract_all(chunks, extractor, cache, limiter, kg, stats)

    # Invariant that links the two stores: every chunk ID in the graph is in the index.
    known = {c.chunk_id for c in chunks}
    missing = kg.all_chunk_ids() - known
    if missing:
        raise RuntimeError(f"Graph references unknown chunk IDs: {sorted(missing)[:5]}")

    Path(cfg.paths.graph_path).parent.mkdir(parents=True, exist_ok=True)
    kg.save(cfg.paths.graph_path)
    stats.graph = kg.stats()
    return kg, stats


def build_default(cfg: AppConfig) -> dict:
    """Real components (downloads MiniLM on first run; needs GROQ_API_KEY)."""
    from graphrag.embedding import HFTokenizer, SentenceTransformerEmbedder
    from graphrag.llm import GroqChatClient
    from graphrag.vector_store import ChromaVectorStore

    return {
        "tokenizer": HFTokenizer(cfg.embedding.model),
        "embedder": SentenceTransformerEmbedder(cfg.embedding.model, cfg.embedding.batch_size),
        "store": ChromaVectorStore(cfg.paths.index_dir, cfg.vector_store.collection),
        "extractor": TripleExtractor(GroqChatClient(cfg.llm), cfg.llm, cfg.extraction),
    }
