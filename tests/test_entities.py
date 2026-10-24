import json

from graphrag.extraction.parse import Triple
from graphrag.graph import KnowledgeGraph
from graphrag.llm import LLMOutputError
from graphrag.retrieval.entities import QueryEntityFinder, lexical_matches
from helpers import FakeClient, KeywordEmbedder


def T(s, r, o):
    return Triple(s, r, o, s.casefold(), o.casefold())


def _kg():
    kg = KnowledgeGraph()
    kg.add_triples(
        [
            T("Zoraxin", "cleared by", "Velkase-3"),
            T("Ostrand Labs", "discovered", "Velkase-3"),
            T("Labs", "x", "ACE"),
        ],
        "d:000",
    )
    return kg


def test_lexical_whole_token_and_longest_match():
    q = "Did Ostrand Labs study Velkase-3 or acetaminophen?"
    nodes = [m.node for m in lexical_matches(q, _kg(), min_chars=3)]
    assert nodes == ["ostrand labs", "velkase-3"]  # not "labs"; "ace" not in "acetaminophen"


def test_lexical_case_insensitive():
    assert [m.node for m in lexical_matches("what clears ZORAXIN?", _kg(), 3)] == ["zoraxin"]


def test_llm_exact_and_embedding_fallback():
    kg = _kg()
    reply = json.dumps({"entities": ["Zoraxin", "Ostrand Labs institute", "Unrelated"]})
    finder = QueryEntityFinder(
        kg, mode="llm", client=FakeClient(reply), embedder=KeywordEmbedder(), threshold=0.6
    )
    seeds = {m.node: m.method for m in finder.find("anything")}
    assert seeds == {"zoraxin": "llm-exact", "ostrand labs": "llm-embedding"}


def test_llm_failure_degrades_to_lexical():
    finder = QueryEntityFinder(_kg(), mode="both", client=FakeClient(exc=LLMOutputError("x")))
    assert [m.node for m in finder.find("Is Zoraxin safe?")] == ["zoraxin"]


def test_lexical_mode_never_calls_llm():
    client = FakeClient('{"entities": []}')
    QueryEntityFinder(_kg(), mode="lexical", client=client).find("Zoraxin?")
    assert client.calls == []
