"""Compare vector-only vs hybrid retrieval for one question (no answer generation yet).

Usage: uv run python scripts/retrieve.py "Which drug could raise Zoraxin levels?"
"""

import logging
import sys

from graphrag.config import load_config
from graphrag.retrieval.retriever import build_default_retriever

logging.basicConfig(level=logging.WARNING)

question = " ".join(sys.argv[1:]) or "Which drug could raise Zoraxin levels?"
cfg = load_config()
retriever = build_default_retriever(cfg)

print(f"\nQ: {question}")
vec = retriever.vector_only(question)
print(f"\n=== Vector-only (top {cfg.retrieval.final_k}) ===")
for c in vec.chunks:
    print(f"  #{c.vector_rank} {c.chunk_id}")

hyb = retriever.hybrid(question)
print("\n=== Seed entities ===")
for s in hyb.seeds:
    print(f"  {s.node}  via {s.method} (mention={s.mention!r}, score={s.score})")
print(f"\n=== Hybrid (top {cfg.retrieval.final_k}) ===")
for c in hyb.chunks:
    print(
        f"  {c.chunk_id}  score={c.score}  from={c.sources}"
        f"  vec_rank={c.vector_rank}  graph_rank={c.graph_rank}  hop={c.graph_hop}"
    )
    for step in c.graph_path:
        print(f"      via {step}")
