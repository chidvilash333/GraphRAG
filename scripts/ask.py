"""Answer a question with citations.

Usage:
  uv run python scripts/ask.py "Which drug could raise Zoraxin levels?"
  uv run python scripts/ask.py --mode vector "Which drug could raise Zoraxin levels?"
"""

import argparse
import logging

from graphrag.config import load_config
from graphrag.generation.generator import build_default_generator
from graphrag.retrieval.retriever import build_default_retriever

logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="+")
parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
args = parser.parse_args()
question = " ".join(args.question)

cfg = load_config()
retriever = build_default_retriever(cfg)
generator = build_default_generator(cfg)

retrieval = retriever.hybrid(question) if args.mode == "hybrid" else retriever.vector_only(question)

ans = generator.answer(retrieval)
texts = {c.chunk_id: c.text for c in retrieval.chunks}

print(f"\nQ: {question}   (mode={args.mode})")
print(f"context: {ans.context_ids}")
print(f"\nstatus: {ans.status}")
if ans.reason:
    print(f"reason: {ans.reason}")
print(f"\n{ans.text}\n")
for cid in ans.citations:
    print(f"[{cid}] {texts[cid].strip()[:160]}...")
