from graphrag.extraction.parse import Triple
from graphrag.graph import KnowledgeGraph


def T(s, r, o):
    return Triple(s, r, o, s.casefold(), o.casefold())


def test_case_variants_merge_across_chunks_and_keep_provenance():
    kg = KnowledgeGraph()
    kg.add_triples([T("Zoraxin", "is cleared by", "Velkase-3")], "zoraxin:000")
    kg.add_triples([T("Tremidol", "inhibits", "velkase-3")], "tremidol:000")
    assert kg.stats() == {"nodes": 3, "edges": 2, "components": 1}
    assert kg.chunk_ids("velkase-3") == {"zoraxin:000", "tremidol:000"}


def test_display_name_is_most_common_surface():
    kg = KnowledgeGraph()
    kg.add_triples([T("velkase-3", "x", "A")], "d:000")
    kg.add_triples([T("Velkase-3", "x", "B"), T("Velkase-3", "y", "C")], "d:001")
    assert kg.name("velkase-3") == "Velkase-3"


def test_typed_parallel_edges_and_edge_dedup():
    kg = KnowledgeGraph()
    kg.add_triples([T("A", "inhibits", "B"), T("A", "binds", "B")], "d:000")
    kg.add_triples([T("A", "inhibits", "B")], "d:001")
    assert kg.g.number_of_edges() == 2
    assert kg.g.edges["a", "b", "inhibits"]["chunk_ids"] == {"d:000", "d:001"}


def test_save_load_round_trip(tmp_path):
    kg = KnowledgeGraph()
    kg.add_triples([T("A", "inhibits", "B")], "d:000")
    path = tmp_path / "g" / "graph.json"
    kg.save(path)
    loaded = KnowledgeGraph.load(path)
    assert loaded.stats() == kg.stats()
    assert loaded.name("a") == "A"
    assert loaded.chunk_ids("b") == {"d:000"}
    assert loaded.g.edges["a", "b", "inhibits"]["chunk_ids"] == {"d:000"}
