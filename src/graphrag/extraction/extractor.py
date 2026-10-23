"""Chunk text -> validated triples, via an injected ChatClient."""

from __future__ import annotations

import hashlib
import logging

from graphrag.config import ExtractionConfig, LLMConfig
from graphrag.extraction.parse import ParseResult, parse_triples
from graphrag.extraction.prompts import build_system_prompt, build_user_prompt
from graphrag.llm import ChatClient, LLMOutputError

log = logging.getLogger(__name__)


class TripleExtractor:
    def __init__(self, client: ChatClient, llm_cfg: LLMConfig, cfg: ExtractionConfig) -> None:
        self._client = client
        self._model = llm_cfg.extraction_model
        self._cfg = cfg
        self._system = build_system_prompt(cfg.max_triples_per_chunk)

    @property
    def fingerprint(self) -> str:
        """Identifies everything that affects output. Changing the model, prompt, or
        validation settings changes this, which invalidates cached extractions."""
        material = "\n".join([self._model, self._system, repr(self._cfg)])
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def extract(self, text: str) -> ParseResult:
        text = text.strip()
        if not text:
            return ParseResult()
        if len(text) > self._cfg.max_chunk_chars:
            log.warning("Chunk of %d chars truncated to %d", len(text), self._cfg.max_chunk_chars)
            text = text[: self._cfg.max_chunk_chars]

        try:
            raw = self._client.complete_json(
                system=self._system, user=build_user_prompt(text), model=self._model
            )
        except LLMOutputError as exc:
            log.warning("Skipping chunk: %s", exc)
            return ParseResult(error=str(exc))

        result = parse_triples(
            raw,
            max_triples=self._cfg.max_triples_per_chunk,
            max_entity_chars=self._cfg.max_entity_chars,
            max_relation_chars=self._cfg.max_relation_chars,
            source_text=text if self._cfg.require_grounded_entities else None,
        )
        if result.error:
            log.warning("Unparseable extraction output: %s", result.error)
        return result
