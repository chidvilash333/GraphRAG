"""The before/after demo: vector-only RAG vs GraphRAG on the evaluation set.

Usage:
  uv run python scripts/compare.py              # full evaluation (about 4 Groq calls/question)
  uv run python scripts/compare.py --only q1    # one question, with full answers printed

Run scripts/ingest.py first.
"""

import argparse
import logging

from graphrag.config import load_config
from graphrag.evaluation.harness import (
    evaluate,
    format_report,
    load_questions,
    save_results,
    summarize,
)
from graphrag.generation.generator import build_default_generator
from graphrag.ratelimit import RateLimiter
from graphrag.retrieval.retriever import build_default_retriever

logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("--only", help="run a single question id, e.g. q1")
args = parser.parse_args()

cfg = load_config()
questions = load_questions(cfg.evaluation.questions_path)
if args.only:
    questions = [q for q in questions if q.id == args.only]
    if not questions:
        raise SystemExit(f"no question with id {args.only!r}")

retriever = build_default_retriever(cfg)
generator = build_default_generator(cfg)
# Each system run makes up to 3 Groq calls; spacing them out avoids free-tier 429s.
limiter = RateLimiter(max(1, cfg.llm.requests_per_minute // 3))

results = evaluate(
    questions,
    retriever,
    generator,
    final_k=cfg.retrieval.final_k,
    recall_ks=cfg.evaluation.recall_at_k,
    before_llm=limiter.wait,
)

for r in results:
    print(f"\n{r.question.id} [{r.question.type}] {r.question.question}")
    print(f"   expected: {list(r.question.expected_any) or 'INSUFFICIENT_EVIDENCE'}")
    for system, run in r.runs.items():
        verdict = "PASS" if run.correct else "FAIL"
        print(f"   {system:<9} {verdict}  context={run.context_docs}")
        print(f"             {run.answer}")

summary = summarize(results, cfg.evaluation.recall_at_k)
print("\n" + format_report(results, summary, cfg.retrieval.final_k))
if not args.only:
    save_results(results, summary, cfg.evaluation.results_path)
    print(f"\nSaved {cfg.evaluation.results_path}")
