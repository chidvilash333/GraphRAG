from graphrag.config import RetrievalConfig
from graphrag.extraction.parse import Triple
from graphrag.graph import KnowledgeGraph
from graphrag.retrieval.entities import QueryEntityFinder
from graphrag.retrieval.retriever import Retriever
from helpers import InMemoryVectorStore, KeywordEmbedder

CHUNKS = {
    "zoraxin:000": "Zoraxin is an anticoagulant. Zoraxin is cleared by Velkase-3.",
    "tremidol:000": "Tremidol is an antifungal. Tremidol inhibits Velkase-3.",
    "brimlase:000": "Brimlase is an anticoagulant that raises bleeding risk in patients.",
    "kaldmark:000": "Kaldmark is a city with a large harbor and old bridges.",
}
QUESTION = "Which drug raises Zoraxin levels in patients?"


def T(s, r, o):
    return Triple(s, r, o, s.casefold(), o.casefold())


def _retriever(**overrides):
    emb = KeywordEmbedder()
    store = InMemoryVectorStore()
    ids = list(CHUNKS)
    store.add(ids, emb.embed([CHUNKS[i] for i in ids]), [CHUNKS[i] for i in ids], [{}] * 4)
    kg = KnowledgeGraph()
    kg.add_triples([T("Zoraxin", "cleared by", "Velkase-3")], "zoraxin:000")
    kg.add_triples([T("Tremidol", "inhibits", "Velkase-3")], "tremidol:000")
    cfg = RetrievalConfig(**{"vector_k": 3, "final_k": 3, **overrides})
    finder = QueryEntityFinder(kg, mode="lexical")
    return Retriever(cfg, embedder=emb, store=store, kg=kg, entity_finder=finder)


def test_vector_only_misses_the_second_hop():
    ids = [c.chunk_id for c in _retriever().vector_only(QUESTION).chunks]
    assert "tremidol:000" not in ids  # no word overlap with the question
    assert len(ids) == 3  # same budget as hybrid


def test_hybrid_reaches_second_hop_via_graph():
    res = _retriever().hybrid(QUESTION)
    by_id = {c.chunk_id: c for c in res.chunks}
    assert [s.node for s in res.seeds] == ["zoraxin"]
    assert set(by_id) == {"zoraxin:000", "brimlase:000", "tremidol:000"}
    assert by_id["zoraxin:000"].sources == "both"  # vector and graph agree -> ranked first
    assert res.chunks[0].chunk_id == "zoraxin:000"
    t = by_id["tremidol:000"]
    assert (t.sources, t.graph_hop) == ("graph", 2)
    assert t.graph_path[-1] == "Tremidol -[inhibits]-> Velkase-3"
    assert "Tremidol inhibits" in t.text  # text fetched from the store for graph-only hits


def test_hybrid_respects_budget_and_dedupes():
    res = _retriever(final_k=2).hybrid(QUESTION)
    ids = [c.chunk_id for c in res.chunks]
    assert len(ids) == len(set(ids)) == 2


def test_tight_budget_limitation_is_documented():
    # With only 2 slots, the vector distractor (vector rank 1) outscores the 2-hop
    # chunk (graph rank 2) under RRF. Multi-hop needs room in the context budget.
    ids = [c.chunk_id for c in _retriever(final_k=2).hybrid(QUESTION).chunks]
    assert ids == ["zoraxin:000", "brimlase:000"]


def test_no_seeds_falls_back_to_vector_ranking():
    res = _retriever().hybrid("Tell me about harbors and bridges")
    assert res.seeds == []
    assert res.chunks[0].chunk_id == "kaldmark:000"
    assert all(c.sources == "vector" for c in res.chunks)
