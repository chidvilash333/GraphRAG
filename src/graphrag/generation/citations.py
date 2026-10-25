"""Citation guard: validate an answer's citations against what was actually retrieved.

Bracket-agnostic: models cite as [id], (id), 【id】, [id1, id2], or bare id. We find
chunk IDs by their SHAPE (`<doc-slug>:<3 digits>`) instead of by bracket style,
rewrite every citation to the canonical [id] form, then check each ID against the
retrieved set. An answer is accepted only if it cites at least one passage and every
cited ID was really in the context; citing an unseen chunk is treated as fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from graphrag.generation.prompts import REFUSAL

# doc slug (lowercase letters/digits/hyphens, at least one letter) + ":" + 3 digits.
_ID = r"(?=[a-z0-9-]*[a-z])[a-z0-9]+(?:-[a-z0-9]+)*\s*:\s*\d{3}"
_ID_RE = re.compile(rf"(?<![\w-])({_ID})(?!\d)", re.IGNORECASE)
_OPEN, _CLOSE = r"[\[\(【〔{<]", r"[\]\)】〕}>]"
_GROUP_RE = re.compile(rf"{_OPEN}\s*({_ID}(?:\s*[,;]\s*{_ID}|\s+{_ID})*)\s*{_CLOSE}", re.IGNORECASE)


def _norm(raw_id: str) -> str:
    return re.sub(r"\s+", "", raw_id).casefold()


def canonicalize(text: str) -> str:
    """Rewrite any bracket style (or none) to [id][id] form."""

    def group(m: re.Match[str]) -> str:
        return "".join(f"[{_norm(i)}]" for i in _ID_RE.findall(m.group(1)))

    text = _GROUP_RE.sub(group, text)
    # Remaining bare IDs (not already inside canonical brackets).
    return re.sub(
        rf"(?<!\[)(?<![\w-])({_ID})(?!\d)(?!\])",
        lambda m: f"[{_norm(m.group(1))}]",
        text,
        flags=re.IGNORECASE,
    )


def extract_citations(canonical_text: str) -> list[str]:
    """Cited IDs in first-appearance order, deduplicated."""
    seen: dict[str, None] = {}
    for m in re.finditer(r"\[([^\[\]]+)\]", canonical_text):
        if _ID_RE.fullmatch(m.group(1)):
            seen.setdefault(m.group(1), None)
    return list(seen)


def is_refusal(text: str) -> bool:
    # Any mention counts: a half-answer that also says "insufficient" is ambiguous,
    # and treating it as a refusal is the safe reading.
    return REFUSAL.casefold() in text.casefold()


@dataclass
class GuardResult:
    status: str  # "answered" | "insufficient_evidence" | "rejected"
    text: str  # canonicalized answer text
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    reason: str | None = None


def check_answer(answer: str, retrieved_ids: set[str]) -> GuardResult:
    if is_refusal(answer):
        return GuardResult("insufficient_evidence", REFUSAL)
    text = canonicalize(answer.strip())
    cited = extract_citations(text)
    if not cited:
        return GuardResult("rejected", text, reason="answer has no citations")
    invalid = [c for c in cited if c not in retrieved_ids]
    if invalid:
        return GuardResult(
            "rejected",
            text,
            citations=[c for c in cited if c in retrieved_ids],
            invalid_citations=invalid,
            reason=f"cited chunks that were not retrieved: {invalid}",
        )
    return GuardResult("answered", text, citations=cited)
