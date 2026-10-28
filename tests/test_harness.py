import pytest

from graphrag.evaluation.harness import (
    EvalQuestion,
    evaluate,
    is_correct,
    load_questions,
    retrieval_recall,
    summarize,
)
from graphrag.generation.generator import Answer
from graphrag.retrieval.retriever import RetrievalResult, RetrievedChunk

MH = EvalQuestion("q1", "multi_hop", "Which drug?", ("Tremidol",), ("zoraxin", "tremidol"))
UN = EvalQuestion("q7", "unanswerable", "Dose for children?", (), ())


def _r(*ids):
    return RetrievalResult("q", [RetrievedChunk(i, "t", 1.0) for i in ids])


def test_recall_is_doc_level():
    assert retrieval_recall(_r("zoraxin:000", "brimlase:000"), MH.required_docs) == 0.5
    assert retrieval_recall(_r("zoraxin:001", "tremidol:000"), MH.required_docs) == 1.0
    assert retrieval_recall(_r(), UN.required_docs) == 1.0


def test_correctness_rules():
    assert is_correct(MH, Answer("q", "answered", "TREMIDOL does [tremidol:000]."))
    assert not is_correct(MH, Answer("q", "rejected", "Tremidol [x:000]"))  # guard rejected
    assert not is_correct(MH, Answer("q", "insufficient_evidence", "INSUFFICIENT_EVIDENCE"))
    assert is_correct(UN, Answer("q", "insufficient_evidence", "INSUFFICIENT_EVIDENCE"))
    assert not is_correct(UN, Answer("q", "answered", "5 mg [zoraxin:000]"))


def test_load_questions_validates(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text("- {id: a, type: multi_hop, question: x, expected_any: [y], required_docs: [d]}\n")
    assert load_questions(p)[0].required_docs == ("d",)
    p.write_text("- {id: a, type: multi_hop, question: x}\n")
    with pytest.raises(ValueError):
        load_questions(p)
    p.write_text("- {id: a, type: weird, question: x}\n")
    with pytest.raises(ValueError):
        load_questions(p)


def test_repo_question_set_is_valid():
    from pathlib import Path

    qs = load_questions(Path(__file__).resolve().parents[1] / "data/eval/questions.yaml")
    assert {q.type for q in qs} == {"multi_hop", "single_hop", "unanswerable"}


class StubRetriever:
    """vector misses tremidol; graphrag finds it."""

    def vector_only(self, question, final_k=None):
        return _r(*["zoraxin:000", "brimlase:000", "fendrel:000"][: final_k or 2])

    def hybrid(self, question, final_k=None):
        return _r(*["zoraxin:000", "tremidol:000", "brimlase:000"][: final_k or 2])


class StubGenerator:
    def answer(self, retrieval):
        ids = [c.chunk_id for c in retrieval.chunks]
        if "tremidol:000" in ids:
            return Answer(retrieval.question, "answered", "Tremidol [tremidol:000].")
        return Answer(retrieval.question, "insufficient_evidence", "INSUFFICIENT_EVIDENCE")


def test_evaluate_and_summarize_end_to_end():
    results = evaluate([MH, UN], StubRetriever(), StubGenerator(), final_k=2, recall_ks=(1, 2, 3))
    v, g = results[0].runs["vector"], results[0].runs["graphrag"]
    assert (v.correct, v.recall) == (False, 0.5)
    assert (g.correct, g.recall) == (True, 1.0)
    assert g.recall_at_k == {1: 0.5, 2: 1.0, 3: 1.0}
    s = summarize(results, (1, 2, 3))
    assert s["graphrag"]["accuracy_multi_hop"] == 1.0
    assert s["vector"]["accuracy_multi_hop"] == 0.0
    assert s["vector"]["recall_at_k"][2] == 0.5
