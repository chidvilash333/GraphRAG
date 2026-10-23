"""Run the full ingest and print a summary of the resulting graph.

Usage: uv run python scripts/ingest.py
(A proper Typer CLI arrives in Phase 6.)
"""

import logging

from graphrag.config import load_config
from graphrag.ingest.pipeline import build_default, run_ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

cfg = load_config()
kg, stats = run_ingest(cfg, **build_default(cfg))

print("\n=== Ingest summary ===")
print(
    f"docs={stats.documents} chunks={stats.chunks} llm_calls={stats.llm_calls} "
    f"cache_hits={stats.cache_hits} triples={stats.triples} dropped={stats.dropped_triples}"
)
if stats.failed_chunks:
    print("failed chunks (will retry next run):", stats.failed_chunks)
print("graph:", stats.graph)

print("\n=== Edges ===")
for u, v, rel, d in sorted(kg.g.edges(keys=True, data=True)):
    print(f"  ({kg.name(u)}) -[{rel}]-> ({kg.name(v)})   from {sorted(d['chunk_ids'])}")

print("\n=== Entities appearing in more than one document (cross-doc bridges) ===")
for key in sorted(kg.g.nodes):
    docs = sorted({cid.split(":")[0] for cid in kg.chunk_ids(key)})
    if len(docs) > 1:
        print(f"  {kg.name(key)}: {docs}")
