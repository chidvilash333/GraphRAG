import json

from graphrag.llm import LLMOutputError
from graphrag.retrieval.rerank import PathReranker, format_candidates
from graphrag.retrieval.traversal import GraphHit
from helpers import FakeClient

HITS = [
    GraphHit("halvern:000", 2, (("Zoraxin -[developed]-> Halvern", "Halvern -[x]-> Y"),)),
    GraphHit("tremidol:000", 2, (("Zoraxin -[cleared by]-> V3", "Tremidol -[inhibits]-> V3"),)),
    GraphHit("ostrand:000", 2, (("Zoraxin -[cleared by]-> V3", "Ostrand -[discovered]-> V3"),)),
]


def _ids(hits):
    return [h.chunk_id for h in hits]


def test_ranked_first_then_rest_in_original_order():
    rr = PathReranker(FakeClient(json.dumps({"ranking": [2]})), "light")
    assert _ids(rr.rerank("q", HITS)) == ["tremidol:000", "halvern:000", "ostrand:000"]


def test_invalid_entries_ignored():
    reply = json.dumps({"ranking": [9, 0, "2", True, 3, 3, 2]})
    rr = PathReranker(FakeClient(reply), "light")
    assert _ids(rr.rerank("q", HITS)) == ["ostrand:000", "tremidol:000", "halvern:000"]


def test_failure_keeps_order():
    for client in (FakeClient("not json"), FakeClient(exc=LLMOutputError("x"))):
        assert _ids(PathReranker(client, "light").rerank("q", HITS)) == _ids(HITS)


def test_prompt_shows_chains_and_uses_light_model():
    client = FakeClient(json.dumps({"ranking": []}))
    PathReranker(client, "light").rerank("Which drug?", HITS)
    assert client.calls[0]["model"] == "light"
    assert "2. Zoraxin -[cleared by]-> V3 ; Tremidol -[inhibits]-> V3" in client.calls[0]["user"]
    assert format_candidates(HITS[:1]).startswith("1. ")


def test_single_candidate_skips_llm():
    client = FakeClient()
    PathReranker(client, "light").rerank("q", HITS[:1])
    assert client.calls == []
