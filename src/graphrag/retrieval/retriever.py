"""Vector-only and hybrid (graph + vector) retrieval behind one interface.

Both modes return at most `final_k` chunks, so Phase 5 compares them on an equal
context budget: GraphRAG must win by choosing better chunks, not by seeing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.config import AppConfig, RetrievalConfig
from graphrag.embedding import Embedder
from graphrag.graph import KnowledgeGraph
from graphrag.retrieval.entities import QueryEntityFinder, SeedMatch
from graphrag.retrieval.fusion import rrf
from graphrag.retrieval.rerank import PathReranker
from graphrag.retrieval.traversal import traverse
from graphrag.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    vector_rank: int | None = None
    graph_rank: int | None = None
    graph_hop: int | None = None
    graph_path: tuple[str, ...] = ()

    @property
    def sources(self) -> str:
        if self.vector_rank and self.graph_rank:
            return "both"
        return "vector" if self.vector_rank else "graph"


@dataclass
class RetrievalResult:
    question: str
    chunks: list[RetrievedChunk]
    seeds: list[SeedMatch] = field(default_factory=list)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class Retriever:
    def __init__(
        self,
        cfg: RetrievalConfig,
        *,
        embedder: Embedder,
        store: VectorStore,
        kg: KnowledgeGraph,
        entity_finder: QueryEntityFinder,
        reranker: PathReranker | None = None,
    ) -> None:
        self._cfg, self._embedder, self._store = cfg, embedder, store
        self._kg, self._finder, self._reranker = kg, entity_finder, reranker
        # Graph-side work (entity LLM call, traversal, rerank LLM call) depends only on
        # the question, so it is memoized: evaluating several budgets costs no extra calls.
        self._graph_memo: dict[str, tuple] = {}

    def vector_only(self, question: str, final_k: int | None = None) -> RetrievalResult:
        """Plain RAG baseline: top final_k by embedding similarity."""
        q = self._embedder.embed([question])[0]
        hits = self._store.query(q, final_k or self._cfg.final_k)
        chunks = [
            RetrievedChunk(cid, text, score=-dist, vector_rank=i)
            for i, (cid, dist, text) in enumerate(hits, 1)
        ]
        return RetrievalResult(question, chunks)

    def hybrid(self, question: str, final_k: int | None = None) -> RetrievalResult:
        cfg = self._cfg
        final_k = final_k or cfg.final_k
        q = self._embedder.embed([question])[0]

        # (a) vector candidates
        vec_hits = self._store.query(q, cfg.vector_k)
        texts = {cid: text for cid, _, text in vec_hits}
        vector_ranking = [cid for cid, _, _ in vec_hits]
        # Unit vectors: squared L2 distance d = 2 - 2cos, so cos = 1 - d/2.
        sims = {cid: 1.0 - dist / 2.0 for cid, dist, _ in vec_hits}

        # (b) graph candidates: seeds -> bounded traversal -> ranked chunk IDs (memoized)
        if question not in self._graph_memo:
            self._graph_memo[question] = self._graph_side(question, q)
        seeds, graph_hits, graph_texts, graph_sims = self._graph_memo[question]
        texts = {**texts, **graph_texts}
        sims.update(graph_sims)
        graph_ranking = [h.chunk_id for h in graph_hits]
        by_id = {h.chunk_id: h for h in graph_hits}

        # (c) fuse, dedupe (RRF keys on chunk ID), keep the top final_k
        scores = rrf({"vector": vector_ranking, "graph": graph_ranking}, cfg.rrf_k)
        v_rank = {cid: i for i, cid in enumerate(vector_ranking, 1)}
        g_rank = {cid: i for i, cid in enumerate(graph_ranking, 1)}

        # Equal RRF scores (e.g. vector rank 2 vs graph rank 2) are broken by similarity
        # to the question: a neutral rule that doesn't favour either retriever.
        ordered = sorted(scores, key=lambda cid: (-scores[cid], -sims.get(cid, 0.0), cid))
        chunks = []
        for cid in ordered[:final_k]:
            hit = by_id.get(cid)
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=texts[cid],
                    score=round(scores[cid], 5),
                    vector_rank=v_rank.get(cid),
                    graph_rank=g_rank.get(cid),
                    graph_hop=hit.hop if hit else None,
                    graph_path=hit.path if hit else (),
                )
            )
        return RetrievalResult(question, chunks, seeds)

    def _graph_side(self, question: str, q: list[float]) -> tuple:
        cfg = self._cfg
        seeds = self._finder.find(question)
        hits = traverse(
            self._kg,
            [s.node for s in seeds],
            max_hops=cfg.max_hops,
            max_neighbors=cfg.max_neighbors_per_node,
        )
        stored = self._store.get([h.chunk_id for h in hits])
        texts = {cid: t for cid, (t, _) in stored.items()}
        sim = {cid: _dot(q, emb) for cid, (_, emb) in stored.items()}
        live = [h for h in hits if h.chunk_id in stored]  # skip IDs missing from the index
        # Base order: hop distance, then similarity to the question.
        live.sort(key=lambda h: (h.hop, -sim[h.chunk_id], h.chunk_id))
        if self._reranker is not None and cfg.path_rerank:
            head = self._reranker.rerank(question, live[: cfg.rerank_candidates])
            live = head + live[cfg.rerank_candidates :]
        return seeds, live[: cfg.max_graph_chunks], texts, sim


def build_default_retriever(cfg: AppConfig) -> Retriever:
    """Real components. Requires a prior ingest (graph file + Chroma index)."""
    from pathlib import Path

    from graphrag.embedding import SentenceTransformerEmbedder
    from graphrag.llm import GroqChatClient
    from graphrag.vector_store import ChromaVectorStore

    if not Path(cfg.paths.graph_path).is_file():
        raise FileNotFoundError(f"{cfg.paths.graph_path} not found; run scripts/ingest.py first")
    kg = KnowledgeGraph.load(cfg.paths.graph_path)
    embedder = SentenceTransformerEmbedder(cfg.embedding.model, cfg.embedding.batch_size)
    client = GroqChatClient(cfg.llm) if cfg.retrieval.entity_mode != "lexical" else None
    finder = QueryEntityFinder(
        kg,
        mode=cfg.retrieval.entity_mode,
        client=client,
        model=cfg.llm.light_model,
        embedder=embedder,
        threshold=cfg.retrieval.entity_match_threshold,
        min_lexical_chars=cfg.retrieval.min_lexical_chars,
    )
    store = ChromaVectorStore(cfg.paths.index_dir, cfg.vector_store.collection)
    reranker = None
    if cfg.retrieval.path_rerank:
        reranker = PathReranker(client or GroqChatClient(cfg.llm), cfg.llm.light_model)
    return Retriever(
        cfg.retrieval,
        embedder=embedder,
        store=store,
        kg=kg,
        entity_finder=finder,
        reranker=reranker,
    )
