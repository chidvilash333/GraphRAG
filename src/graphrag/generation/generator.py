"""Retrieved chunks -> cited answer, passed through the citation guard."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from graphrag.config import AppConfig, GenerationConfig
from graphrag.generation.citations import GuardResult, check_answer
from graphrag.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    REFUSAL,
    build_answer_user_prompt,
    build_context,
)
from graphrag.llm import ChatClient, LLMOutputError
from graphrag.retrieval.retriever import RetrievalResult

log = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    status: str  # "answered" | "insufficient_evidence" | "rejected"
    text: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    reason: str | None = None
    context_ids: list[str] = field(default_factory=list)  # what the model was shown
    raw: str = ""  # unmodified model output, for debugging


def select_context(retrieval: RetrievalResult, max_chars: int) -> list[tuple[str, str]]:
    """Keep chunks in rank order while they fit. Whole chunks only: cutting a chunk
    mid-sentence could remove exactly the fact the question needs."""
    passages: list[tuple[str, str]] = []
    used = 0
    for c in retrieval.chunks:
        if passages and used + len(c.text) > max_chars:
            break
        passages.append((c.chunk_id, c.text))
        used += len(c.text)
    return passages


class AnswerGenerator:
    def __init__(self, client: ChatClient, model: str, cfg: GenerationConfig) -> None:
        self._client, self._model, self._cfg = client, model, cfg

    def answer(self, retrieval: RetrievalResult) -> Answer:
        question = retrieval.question
        passages = select_context(retrieval, self._cfg.max_context_chars)
        ids = [cid for cid, _ in passages]
        if not passages:
            # Nothing retrieved: refuse without spending an LLM call.
            return Answer(question, "insufficient_evidence", REFUSAL, reason="no context")

        user = build_answer_user_prompt(question, build_context(passages))
        try:
            raw = self._client.complete_json(
                system=ANSWER_SYSTEM_PROMPT, user=user, model=self._model
            )
        except LLMOutputError as exc:
            return Answer(question, "rejected", "", reason=str(exc), context_ids=ids)

        try:
            data = json.loads(raw)
            text = data["answer"]
            if not isinstance(text, str):
                raise TypeError("answer is not a string")
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            return Answer(
                question, "rejected", "", reason=f"unparseable: {exc}", context_ids=ids, raw=raw
            )

        g: GuardResult = check_answer(text, set(ids))
        if g.status == "rejected":
            log.warning("Citation guard rejected answer: %s", g.reason)
        return Answer(
            question,
            g.status,
            g.text,
            citations=g.citations,
            invalid_citations=g.invalid_citations,
            reason=g.reason,
            context_ids=ids,
            raw=raw,
        )


def build_default_generator(cfg: AppConfig) -> AnswerGenerator:
    from graphrag.llm import GroqChatClient

    heavy = cfg.generation.model_role == "heavy"
    model = cfg.llm.extraction_model if heavy else cfg.llm.light_model
    return AnswerGenerator(GroqChatClient(cfg.llm), model, cfg.generation)
