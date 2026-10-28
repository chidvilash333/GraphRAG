"""LLM reranking of graph evidence paths.

Why: within the same hop distance, embedding similarity between the question and a
chunk is a weak signal. "Which drug could raise Zoraxin levels?" is answered by the
chain "Zoraxin -[cleared by]-> Velkase-3, Tremidol -[inhibits]-> Velkase-3", but no
single chunk says "raises Zoraxin levels". Judging whether a chain of relations answers
a question is multi-step reasoning, which an LLM does well and embeddings do not.

Cost is bounded: one light-model call per query, over short path strings (not chunk
text), capped at `rerank_candidates`. On any failure the original order is kept.
"""

from __future__ import annotations

import json
import logging

from graphrag.llm import ChatClient, LLMOutputError
from graphrag.retrieval.traversal import GraphHit

log = logging.getLogger(__name__)

RERANK_PROMPT = """You rank knowledge-graph evidence for answering a question.
Each numbered candidate is one or more chains of facts written as
subject -[relation]-> object.
Return ONLY a JSON object: {"ranking": [candidate numbers]}
List the candidates that help answer the question, most useful first, and omit the rest.
A chain can be useful even if no single fact repeats the question's wording: reason over
the whole chain."""


def format_candidates(hits: list[GraphHit]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        chains = " | ".join(" ; ".join(p) for p in h.paths)
        lines.append(f"{i}. {chains}")
    return "\n".join(lines)


class PathReranker:
    def __init__(self, client: ChatClient, model: str) -> None:
        self._client, self._model = client, model

    def rerank(self, question: str, hits: list[GraphHit]) -> list[GraphHit]:
        """Return `hits` reordered: LLM-ranked candidates first, then the rest unchanged."""
        if len(hits) < 2:
            return hits
        user = f"Question: {question}\n\nCandidates:\n{format_candidates(hits)}"
        try:
            raw = self._client.complete_json(system=RERANK_PROMPT, user=user, model=self._model)
            ranking = json.loads(raw).get("ranking", [])
        except (LLMOutputError, json.JSONDecodeError, AttributeError) as exc:
            log.warning("Path rerank failed, keeping similarity order: %s", exc)
            return hits
        picked: list[int] = []
        for n in ranking if isinstance(ranking, list) else []:
            # Validate: integers in range, no duplicates. Anything else is ignored.
            valid = isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(hits)
            if valid and n - 1 not in picked:
                picked.append(n - 1)
        rest = [i for i in range(len(hits)) if i not in picked]
        return [hits[i] for i in picked + rest]
