"""LLM client boundary.

Everything above this module depends on the `ChatClient` protocol, not on the
openai SDK. That lets tests inject a fake client and run fully offline.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol

from graphrag.config import LLMConfig

log = logging.getLogger(__name__)


class LLMOutputError(RuntimeError):
    """The model produced unusable output (recoverable: skip this chunk)."""


class ChatClient(Protocol):
    def complete_json(self, *, system: str, user: str, model: str) -> str:
        """Return the raw text of a JSON-mode completion."""
        ...


class GroqChatClient:
    """Groq via its OpenAI-compatible endpoint."""

    def __init__(self, cfg: LLMConfig, api_key: str | None = None) -> None:
        from dotenv import load_dotenv
        from openai import OpenAI  # imported lazily so tests never need the SDK

        load_dotenv()
        key = api_key or os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{cfg.api_key_env} is not set (copy .env.example to .env)")
        self._cfg = cfg
        self._client = OpenAI(
            api_key=key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    def complete_json(self, *, system: str, user: str, model: str) -> str:
        from openai import BadRequestError

        extra_body = {}
        if self._cfg.reasoning_effort:
            extra_body["reasoning_effort"] = self._cfg.reasoning_effort
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._cfg.temperature,
                max_completion_tokens=self._cfg.max_completion_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body or None,
            )
        except BadRequestError as exc:
            # Groq rejects JSON-mode output that fails to parse server-side.
            # Treat that as a bad chunk, not a fatal error. Auth/other 400s re-raise.
            if "json_validate_failed" in str(exc):
                raise LLMOutputError(f"Model returned invalid JSON: {exc}") from exc
            raise

        choice = resp.choices[0]
        if choice.finish_reason == "length":
            log.warning("Completion hit max_completion_tokens; output may be truncated")
        return choice.message.content or ""
