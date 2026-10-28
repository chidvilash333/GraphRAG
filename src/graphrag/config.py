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
_ENTITY_MODES = {"lexical", "llm", "both"}
_MODEL_ROLES = {"heavy", "light"}


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
    requests_per_minute: int = 20

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ConfigError("llm.requests_per_minute must be > 0")
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
class PathsConfig:
    docs_dir: str = "data/docs"
    index_dir: str = "data/index"
    graph_path: str = "data/graph/graph.json"
    cache_path: str = "data/cache/extractions.json"


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError("embedding.batch_size must be > 0")


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 200
    overlap_tokens: int = 40

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ConfigError("chunking.max_tokens must be > 0")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ConfigError("chunking.overlap_tokens must be in [0, max_tokens)")


@dataclass(frozen=True)
class VectorStoreConfig:
    collection: str = "chunks"


@dataclass(frozen=True)
class RetrievalConfig:
    vector_k: int = 3
    max_hops: int = 2
    max_neighbors_per_node: int = 8
    max_graph_chunks: int = 8
    final_k: int = 4
    rrf_k: int = 60
    entity_mode: str = "both"
    entity_match_threshold: float = 0.75
    min_lexical_chars: int = 3
    path_rerank: bool = True
    rerank_candidates: int = 10

    def __post_init__(self) -> None:
        for name in (
            "vector_k",
            "max_neighbors_per_node",
            "max_graph_chunks",
            "final_k",
            "rrf_k",
            "min_lexical_chars",
            "rerank_candidates",
        ):
            if getattr(self, name) <= 0:
                raise ConfigError(f"retrieval.{name} must be > 0")
        if not 1 <= self.max_hops <= 3:
            raise ConfigError("retrieval.max_hops must be in [1, 3]")
        if self.entity_mode not in _ENTITY_MODES:
            raise ConfigError(f"retrieval.entity_mode must be one of {sorted(_ENTITY_MODES)}")
        if not 0.0 < self.entity_match_threshold <= 1.0:
            raise ConfigError("retrieval.entity_match_threshold must be in (0, 1]")


@dataclass(frozen=True)
class GenerationConfig:
    model_role: str = "heavy"  # heavy -> llm.extraction_model, light -> llm.light_model
    max_context_chars: int = 6000

    def __post_init__(self) -> None:
        if self.model_role not in _MODEL_ROLES:
            raise ConfigError(f"generation.model_role must be one of {sorted(_MODEL_ROLES)}")
        if self.max_context_chars <= 0:
            raise ConfigError("generation.max_context_chars must be > 0")


@dataclass(frozen=True)
class EvaluationConfig:
    questions_path: str = "data/eval/questions.yaml"
    results_path: str = "data/eval/results.json"
    recall_at_k: tuple[int, ...] = (2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        ks = tuple(self.recall_at_k)
        if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
            raise ConfigError("evaluation.recall_at_k must be a non-empty list of positive ints")
        object.__setattr__(self, "recall_at_k", ks)  # YAML gives a list; store a tuple


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    extraction: ExtractionConfig
    paths: PathsConfig = PathsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


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
    unknown = set(raw) - {f.name for f in fields(AppConfig)}
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {sorted(unknown)}")
    if "llm" not in raw:
        raise ConfigError("Missing required section 'llm'")
    return AppConfig(
        llm=_build(LLMConfig, raw["llm"], "llm"),
        extraction=_build(ExtractionConfig, raw.get("extraction", {}), "extraction"),
        paths=_build(PathsConfig, raw.get("paths", {}), "paths"),
        embedding=_build(EmbeddingConfig, raw.get("embedding", {}), "embedding"),
        chunking=_build(ChunkingConfig, raw.get("chunking", {}), "chunking"),
        vector_store=_build(VectorStoreConfig, raw.get("vector_store", {}), "vector_store"),
        retrieval=_build(RetrievalConfig, raw.get("retrieval", {}), "retrieval"),
        generation=_build(GenerationConfig, raw.get("generation", {}), "generation"),
        evaluation=_build(EvaluationConfig, raw.get("evaluation", {}), "evaluation"),
    )
