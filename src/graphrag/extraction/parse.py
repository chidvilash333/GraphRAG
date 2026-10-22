"""Parse and validate raw LLM output into triples.

Policy: lenient on transport (tolerate code fences or stray prose around the JSON),
strict on schema (any item that isn't exactly three non-empty strings is dropped).
Each drop is recorded with a reason so extraction quality is measurable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from graphrag.extraction.normalize import (
    PRONOUNS,
    clean_surface,
    entity_key,
    match_text,
    relation_key,
)

_FIELDS = ("subject", "relation", "object")


@dataclass(frozen=True)
class Triple:
    subject: str  # display name
    relation: str  # normalized relation
    object: str  # display name
    subject_key: str  # merge key
    object_key: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject_key, self.relation, self.object_key)


@dataclass
class ParseResult:
    triples: list[Triple] = field(default_factory=list)
    dropped: list[tuple[Any, str]] = field(default_factory=list)  # (raw item, reason)
    error: str | None = None  # set when the whole response was unusable


def _loads_lenient(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(raw[start : end + 1])


def _validate_item(
    item: Any,
    *,
    max_entity_chars: int,
    max_relation_chars: int,
    source: str | None,
) -> Triple | str:
    """Return a Triple, or a string describing why the item was dropped."""
    if not isinstance(item, dict):
        return "not an object"
    if any(not isinstance(item.get(f), str) for f in _FIELDS):
        return "missing or non-string field"

    subj, obj = clean_surface(item["subject"]), clean_surface(item["object"])
    rel = relation_key(item["relation"])
    s_key, o_key = entity_key(subj), entity_key(obj)

    if not (s_key and o_key and rel):
        return "empty after normalization"
    if len(subj) > max_entity_chars or len(obj) > max_entity_chars:
        return "entity too long"
    if len(rel) > max_relation_chars:
        return "relation too long"
    if s_key in PRONOUNS or o_key in PRONOUNS:
        return "pronoun entity"
    if s_key == o_key:
        return "self-loop"
    if source is not None and (s_key not in source or o_key not in source):
        return "entity not found in source text"
    return Triple(subj, rel, obj, s_key, o_key)


def parse_triples(
    raw: str,
    *,
    max_triples: int,
    max_entity_chars: int = 80,
    max_relation_chars: int = 40,
    source_text: str | None = None,
) -> ParseResult:
    """Parse raw model text. If source_text is given, enforce grounding."""
    result = ParseResult()
    try:
        data = _loads_lenient(raw)
    except json.JSONDecodeError as exc:
        result.error = f"invalid JSON: {exc}"
        return result
    if not isinstance(data, dict) or not isinstance(data.get("triples"), list):
        result.error = 'expected an object with a "triples" list'
        return result

    source = match_text(source_text) if source_text is not None else None
    seen: set[tuple[str, str, str]] = set()
    for item in data["triples"]:
        outcome = _validate_item(
            item,
            max_entity_chars=max_entity_chars,
            max_relation_chars=max_relation_chars,
            source=source,
        )
        if isinstance(outcome, str):
            result.dropped.append((item, outcome))
        elif outcome.key in seen:
            result.dropped.append((item, "duplicate"))
        elif len(result.triples) >= max_triples:
            result.dropped.append((item, "over max_triples"))
        else:
            seen.add(outcome.key)
            result.triples.append(outcome)
    return result
