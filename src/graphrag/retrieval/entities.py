"""Find the graph nodes a question is about ("seed entities").

Two complementary matchers:
- lexical: node names that appear verbatim in the question. Free, deterministic,
  exact, but blind to paraphrase.
- llm: the light model lists the question's entities; each is matched to a node by
  exact normalized key, else by embedding similarity above a strict threshold.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from graphrag.embedding import Embedder
from graphrag.extraction.normalize import entity_key, match_text
from graphrag.graph import KnowledgeGraph
from graphrag.llm import ChatClient, LLMOutputError

log = logging.getLogger(__name__)

QUERY_ENTITY_PROMPT = """List the named entities mentioned in the question (drugs, genes,
proteins, people, organizations, places, products, specific concepts).
Return ONLY a JSON object: {"entities": ["...", "..."]}
Use the shortest name that identifies each entity, copied from the question
(write "COX-1", not "the enzyme COX-1"). At most 5. If none, return {"entities": []}."""


@dataclass(frozen=True)
class SeedMatch:
    node: str  # graph node key
    mention: str  # what matched in the question
    method: str  # "lexical" | "llm-exact" | "llm-embedding"
    score: float


def _contains(haystack: str, needle: str) -> bool:
    # Whole-token match: "ace" must not match inside "acetaminophen".
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def lexical_matches(question: str, kg: KnowledgeGraph, min_chars: int) -> list[SeedMatch]:
    q = match_text(question)
    found = [k for k in kg.g.nodes if len(k) >= min_chars and _contains(q, k)]
    # Longest first, and drop a match that sits inside a longer one
    # ("labs" inside "ostrand labs") so we seed the specific entity, not the generic one.
    found.sort(key=lambda k: (-len(k), k))
    kept: list[str] = []
    for key in found:
        if not any(_contains(longer, key) for longer in kept):
            kept.append(key)
    return [SeedMatch(k, kg.name(k), "lexical", 1.0) for k in kept]


class QueryEntityFinder:
    def __init__(
        self,
        kg: KnowledgeGraph,
        *,
        mode: str = "both",
        client: ChatClient | None = None,
        model: str = "",
        embedder: Embedder | None = None,
        threshold: float = 0.75,
        min_lexical_chars: int = 3,
    ) -> None:
        self._kg, self._mode = kg, mode
        self._client, self._model, self._embedder = client, model, embedder
        self._threshold, self._min_chars = threshold, min_lexical_chars
        self._node_keys: list[str] | None = None
        self._node_vecs: list[list[float]] = []

    def find(self, question: str) -> list[SeedMatch]:
        seeds: dict[str, SeedMatch] = {}
        if self._mode in ("lexical", "both"):
            for m in lexical_matches(question, self._kg, self._min_chars):
                seeds.setdefault(m.node, m)
        if self._mode in ("llm", "both") and self._client is not None:
            for mention in self._llm_mentions(question):
                m = self._match(mention)
                if m is not None:
                    seeds.setdefault(m.node, m)
        return list(seeds.values())

    def _llm_mentions(self, question: str) -> list[str]:
        try:
            raw = self._client.complete_json(
                system=QUERY_ENTITY_PROMPT, user=f"Question: {question}", model=self._model
            )
            data = json.loads(raw)
        except (LLMOutputError, json.JSONDecodeError) as exc:
            # Degrade to lexical-only rather than failing the whole query.
            log.warning("Query entity extraction failed: %s", exc)
            return []
        items = data.get("entities", []) if isinstance(data, dict) else []
        return [s for s in items if isinstance(s, str) and s.strip()][:5]

    def _match(self, mention: str) -> SeedMatch | None:
        key = entity_key(mention)
        if key in self._kg.g:
            return SeedMatch(key, mention, "llm-exact", 1.0)
        if self._embedder is None or self._kg.g.number_of_nodes() == 0:
            return None
        if self._node_keys is None:  # embed node names once, lazily
            self._node_keys = sorted(self._kg.g.nodes)
            self._node_vecs = self._embedder.embed([self._kg.name(k) for k in self._node_keys])
        q = self._embedder.embed([mention])[0]
        # Unit vectors, so the dot product is the cosine similarity.
        sims = [sum(a * b for a, b in zip(q, v, strict=True)) for v in self._node_vecs]
        best = max(range(len(sims)), key=sims.__getitem__)
        if sims[best] < self._threshold:
            return None
        return SeedMatch(self._node_keys[best], mention, "llm-embedding", round(sims[best], 3))
