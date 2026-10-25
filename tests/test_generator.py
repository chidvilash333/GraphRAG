import json

from graphrag.config import GenerationConfig
from graphrag.generation.generator import AnswerGenerator, select_context
from graphrag.llm import LLMOutputError
from graphrag.retrieval.retriever import RetrievalResult, RetrievedChunk
from helpers import FakeClient

Q = "Which drug could raise Zoraxin levels?"


def _retrieval(*items):
    return RetrievalResult(Q, [RetrievedChunk(cid, text, 1.0) for cid, text in items])


R = _retrieval(
    ("zoraxin:000", "Zoraxin is cleared by Velkase-3."),
    ("tremidol:000", "Tremidol inhibits Velkase-3."),
)


def _gen(reply="", exc=None, **cfg):
    client = FakeClient(reply, exc=exc)
    return AnswerGenerator(client, "heavy", GenerationConfig(**cfg)), client


def test_multi_hop_answer_with_valid_citations():
    reply = json.dumps(
        {
            "answer": "Tremidol, because it inhibits Velkase-3 (tremidol:000), "
            "which clears Zoraxin [zoraxin:000]."
        }
    )
    gen, client = _gen(reply)
    ans = gen.answer(R)
    assert ans.status == "answered"
    assert ans.citations == ["tremidol:000", "zoraxin:000"]
    assert "[tremidol:000]" in ans.text  # canonicalized from (tremidol:000)
    call = client.calls[0]
    assert call["model"] == "heavy"
    assert call["user"].index("[zoraxin:000]") < call["user"].index("[tremidol:000]")
    assert "Question: " + Q in call["user"]


def test_hallucinated_citation_rejected():
    gen, _ = _gen(json.dumps({"answer": "Brimlase does [brimlase:000]."}))
    ans = gen.answer(R)
    assert ans.status == "rejected" and ans.invalid_citations == ["brimlase:000"]


def test_refusal_passes_through():
    gen, _ = _gen(json.dumps({"answer": "INSUFFICIENT_EVIDENCE"}))
    assert gen.answer(R).status == "insufficient_evidence"


def test_empty_retrieval_refuses_without_llm_call():
    gen, client = _gen()
    ans = gen.answer(RetrievalResult(Q, []))
    assert ans.status == "insufficient_evidence" and client.calls == []


def test_bad_json_and_llm_errors_are_rejected_not_raised():
    gen, _ = _gen("not json")
    assert gen.answer(R).status == "rejected"
    gen, _ = _gen(json.dumps({"answer": 42}))
    assert gen.answer(R).status == "rejected"
    gen, _ = _gen(exc=LLMOutputError("bad"))
    assert gen.answer(R).status == "rejected"


def test_context_budget_drops_whole_tail_chunks():
    r = _retrieval(("a:000", "x" * 50), ("b:000", "y" * 50), ("c:000", "z" * 50))
    assert [cid for cid, _ in select_context(r, 120)] == ["a:000", "b:000"]
    assert [cid for cid, _ in select_context(r, 10)] == ["a:000"]  # always keep the top one
