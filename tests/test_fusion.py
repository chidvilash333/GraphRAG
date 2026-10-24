from graphrag.retrieval.fusion import rrf


def test_agreement_beats_single_top_rank():
    s = rrf({"vector": ["a", "b"], "graph": ["b", "c"]}, k=60)
    assert s["b"] > s["a"] > s["c"]
    assert abs(s["b"] - (1 / 62 + 1 / 61)) < 1e-12


def test_empty_rankings():
    assert rrf({"vector": [], "graph": []}) == {}
