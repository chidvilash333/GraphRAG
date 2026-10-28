"""Side-by-side evaluation: vector-only RAG vs GraphRAG on the same questions.

Two metrics, deliberately separated:
- retrieval recall: fraction of the question's required documents present in the
  context. Deterministic, needs no LLM judge, and measured at several budgets (k).
- answer correctness: did the final answer contain an expected string (or, for an
  unanswerable question, correctly refuse)? Keyword matching is crude but cheap and
  reproducible; it is the second metric, not the only one.

Both systems get the same context budget (final_k) and the same answer generator, so
any difference comes from retrieval alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from graphrag.generation.generator import Answer, AnswerGenerator
from graphrag.retrieval.retriever import RetrievalResult, Retriever

TYPES = {"multi_hop", "single_hop", "unanswerable"}
SYSTEMS = ("vector", "graphrag")


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    type: str
    question: str
    expected_any: tuple[str, ...]
    required_docs: tuple[str, ...]
    note: str = ""


@dataclass
class SystemRun:
    status: str
    answer: str
    correct: bool
    recall: float  # at the configured final_k
    context_docs: list[str]
    citations: list[str]
    recall_at_k: dict[int, float] = field(default_factory=dict)


@dataclass
class QuestionResult:
    question: EvalQuestion
    runs: dict[str, SystemRun]


def load_questions(path: str | Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    questions = []
    for item in raw:
        q = EvalQuestion(
            id=str(item["id"]),
            type=item["type"],
            question=item["question"],
            expected_any=tuple(item.get("expected_any") or ()),
            required_docs=tuple(item.get("required_docs") or ()),
            note=item.get("note", ""),
        )
        if q.type not in TYPES:
            raise ValueError(f"{q.id}: unknown type {q.type!r}")
        if q.type != "unanswerable" and not (q.expected_any and q.required_docs):
            raise ValueError(f"{q.id}: answerable questions need expected_any and required_docs")
        questions.append(q)
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("duplicate question ids")
    return questions


def doc_of(chunk_id: str) -> str:
    return chunk_id.split(":", 1)[0]


def retrieval_recall(retrieval: RetrievalResult, required_docs: tuple[str, ...]) -> float:
    if not required_docs:
        return 1.0  # nothing required (unanswerable): recall is vacuously complete
    docs = {doc_of(c.chunk_id) for c in retrieval.chunks}
    return sum(d in docs for d in required_docs) / len(required_docs)


def is_correct(q: EvalQuestion, ans: Answer) -> bool:
    if q.type == "unanswerable":
        return ans.status == "insufficient_evidence"
    text = ans.text.casefold()
    return ans.status == "answered" and any(e.casefold() in text for e in q.expected_any)


def _retrieve(retriever: Retriever, system: str, question: str, k: int) -> RetrievalResult:
    if system == "vector":
        return retriever.vector_only(question, final_k=k)
    return retriever.hybrid(question, final_k=k)


def evaluate(
    questions: list[EvalQuestion],
    retriever: Retriever,
    generator: AnswerGenerator,
    *,
    final_k: int,
    recall_ks: tuple[int, ...],
    before_llm=lambda: None,  # hook for rate limiting between LLM-heavy steps
) -> list[QuestionResult]:
    results = []
    for q in questions:
        runs = {}
        for system in SYSTEMS:
            before_llm()
            retrieval = _retrieve(retriever, system, q.question, final_k)
            ans = generator.answer(retrieval)
            runs[system] = SystemRun(
                status=ans.status,
                answer=ans.text,
                correct=is_correct(q, ans),
                recall=retrieval_recall(retrieval, q.required_docs),
                context_docs=[doc_of(c.chunk_id) for c in retrieval.chunks],
                citations=ans.citations,
                # Extra budgets reuse the memoized graph work: no extra LLM calls.
                recall_at_k={
                    k: retrieval_recall(
                        _retrieve(retriever, system, q.question, k), q.required_docs
                    )
                    for k in recall_ks
                },
            )
        results.append(QuestionResult(q, runs))
    return results


def summarize(results: list[QuestionResult], recall_ks: tuple[int, ...]) -> dict:
    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    answerable = [r for r in results if r.question.type != "unanswerable"]
    out: dict = {}
    for system in SYSTEMS:
        out[system] = {
            "accuracy_all": mean([float(r.runs[system].correct) for r in results]),
            "accuracy_multi_hop": mean(
                [float(r.runs[system].correct) for r in results if r.question.type == "multi_hop"]
            ),
            "recall_answerable": mean([r.runs[system].recall for r in answerable]),
            "recall_at_k": {
                k: mean([r.runs[system].recall_at_k[k] for r in answerable]) for k in recall_ks
            },
        }
    return out


def save_results(results: list[QuestionResult], summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "questions": [
            {"question": asdict(r.question), "runs": {s: asdict(run) for s, run in r.runs.items()}}
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")


def format_report(results: list[QuestionResult], summary: dict, final_k: int) -> str:
    def mark(ok: bool) -> str:
        return "PASS" if ok else "fail"

    lines = [f"Per question (context budget k={final_k}, same for both systems)", ""]
    header = f"{'id':<4} {'type':<13} {'vector':<16} {'graphrag':<16}"
    lines += [header, "-" * len(header)]
    for r in results:
        v, g = r.runs["vector"], r.runs["graphrag"]
        lines.append(
            f"{r.question.id:<4} {r.question.type:<13} "
            f"{mark(v.correct)} rec={v.recall:<5.2f}  {mark(g.correct)} rec={g.recall:<5.2f}"
        )
    lines += ["", "Summary"]
    for system in SYSTEMS:
        s = summary[system]
        lines.append(
            f"  {system:<9} accuracy={s['accuracy_all']:.2f}  "
            f"multi-hop accuracy={s['accuracy_multi_hop']:.2f}  "
            f"recall={s['recall_answerable']:.2f}"
        )
    lines += ["", "Retrieval recall by budget k (answerable questions)"]
    ks = list(summary["vector"]["recall_at_k"])
    lines.append("  k        " + "  ".join(f"{k:>5}" for k in ks))
    for system in SYSTEMS:
        vals = summary[system]["recall_at_k"]
        lines.append(f"  {system:<9}" + "  ".join(f"{vals[k]:>5.2f}" for k in ks))
    return "\n".join(lines)
