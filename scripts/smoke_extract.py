"""Live check against Groq (uses your free-tier quota: one call).

Usage: uv run python scripts/smoke_extract.py
"""

import logging

from graphrag.config import load_config
from graphrag.extraction import TripleExtractor
from graphrag.llm import GroqChatClient

SAMPLE = (
    "Warfarin is an anticoagulant metabolized primarily by the liver enzyme CYP2C9. "
    "Fluconazole, an antifungal drug, inhibits CYP2C9."
)

logging.basicConfig(level=logging.INFO)
cfg = load_config()
extractor = TripleExtractor(GroqChatClient(cfg.llm), cfg.llm, cfg.extraction)
result = extractor.extract(SAMPLE)

print("\nKept:")
for t in result.triples:
    print(f"  ({t.subject}) -[{t.relation}]-> ({t.object})")
print("Dropped:")
for item, reason in result.dropped:
    print(f"  {reason}: {item}")
if result.error:
    print("Error:", result.error)
