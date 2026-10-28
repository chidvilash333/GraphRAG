from graphrag.extraction.parse import Triple
from graphrag.graph import KnowledgeGraph
from graphrag.retrieval.traversal import traverse


def T(s, r, o):
    return Triple(s, r, o, s.casefold(), o.casefold())


def _kg():
    kg = KnowledgeGraph()
    kg.add_triples([T("Zoraxin", "cleared by", "Velkase-3")], "zoraxin:000")
    kg.add_triples([T("Tremidol", "inhibits", "Velkase-3")], "tremidol:000")
    kg.add_triples([T("Tremidol", "approved by", "Board")], "board:000")
    return kg


def test_two_hops_follow_edges_backwards():
    hits = {h.chunk_id: h for h in traverse(_kg(), ["zoraxin"], max_hops=2, max_neighbors=8)}
    assert hits["zoraxin:000"].hop == 1
    assert hits["tremidol:000"].hop == 2  # reached by walking inhibits -> backwards
    assert hits["tremidol:000"].path == (
        "Zoraxin -[cleared by]-> Velkase-3",
        "Tremidol -[inhibits]-> Velkase-3",
    )
    assert "board:000" not in hits  # 3 hops away


def test_hop_limit():
    hits = traverse(_kg(), ["zoraxin"], max_hops=1, max_neighbors=8)
    assert [h.chunk_id for h in hits] == ["zoraxin:000"]


def test_neighbor_cap_prefers_supported_then_low_degree():
    kg = KnowledgeGraph()
    kg.add_triples([T("Seed", "r", "Specific")], "a:000")
    kg.add_triples([T("Seed", "r", "Specific")], "b:000")  # support 2
    kg.add_triples([T("Seed", "r", "Hub")], "c:000")
    for i in range(5):
        kg.add_triples([T("Hub", "r", f"X{i}")], f"h{i}:000")  # Hub has degree 6
    kg.add_triples([T("Seed", "r", "Leaf")], "d:000")
    hits = traverse(kg, ["seed"], max_hops=1, max_neighbors=2)
    assert {h.chunk_id for h in hits} == {"a:000", "b:000", "d:000"}  # Hub cut


def test_unknown_seeds_and_no_seeds():
    assert traverse(_kg(), ["nope"], max_hops=2, max_neighbors=8) == []
    assert traverse(_kg(), [], max_hops=2, max_neighbors=8) == []


def test_all_shortest_paths_kept_per_chunk():
    kg = KnowledgeGraph()
    kg.add_triples([T("Zoraxin", "developed by", "Halvern")], "zoraxin:000")
    kg.add_triples(
        [T("Halvern", "focuses on", "drugs"), T("Halvern", "headquartered in", "Brevik")],
        "halvern:000",
    )
    hits = {h.chunk_id: h for h in traverse(kg, ["zoraxin"], max_hops=2, max_neighbors=8)}
    last_edges = {p[-1] for p in hits["halvern:000"].paths}
    assert last_edges == {"Halvern -[focuses on]-> drugs", "Halvern -[headquartered in]-> Brevik"}
