"""Typed config loaded from a single YAML file.

Unknown keys are rejected so a typo (e.g. `max_token`) fails loudly instead of
silently falling back to a default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

_REASONING_EFFORTS = {None, "low", "medium", "high"}


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed, or out of range."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    extraction_model: str
    light_model: str
    temperature: float = 0.0
    max_completion_tokens: int = 2048
    reasoning_effort: str | None = "low"
    timeout_s: float = 60.0
    max_retries: int = 3
    api_key_env: str = "GROQ_API_KEY"

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("llm.temperature must be in [0, 2]")
        if self.max_completion_tokens <= 0:
            raise ConfigError("llm.max_completion_tokens must be > 0")
        if self.reasoning_effort not in _REASONING_EFFORTS:
            raise ConfigError(f"llm.reasoning_effort must be one of {_REASONING_EFFORTS}")


@dataclass(frozen=True)
class ExtractionConfig:
    max_triples_per_chunk: int = 15
    max_chunk_chars: int = 4000
    max_entity_chars: int = 80
    max_relation_chars: int = 40
    require_grounded_entities: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_triples_per_chunk",
            "max_chunk_chars",
            "max_entity_chars",
            "max_relation_chars",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"extraction.{name} must be > 0")


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    extraction: ExtractionConfig


def _build(cls: type, data: Any, section: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"Section '{section}' must be a mapping")
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"Unknown keys in '{section}': {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:  # missing required keys
        raise ConfigError(f"Invalid '{section}' section: {exc}") from exc


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top level of config must be a mapping")
    if "llm" not in raw:
        raise ConfigError("Missing required section 'llm'")
    return AppConfig(
        llm=_build(LLMConfig, raw["llm"], "llm"),
        extraction=_build(ExtractionConfig, raw.get("extraction", {}), "extraction"),
    )
