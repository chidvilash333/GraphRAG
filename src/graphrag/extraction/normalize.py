"""Entity/relation normalization.

Two strings per entity:
- display name: cleaned surface form shown to users ("Aspirin")
- key: canonical form used for merging ("aspirin"). Phase 2 merges graph nodes by key.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_DASHES = re.compile(r"[\u2010-\u2015\u2212]")  # unicode hyphens/dashes/minus -> "-"
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"
_EDGE_PUNCT = ".,;:!?"
_ARTICLES = ("the ", "a ", "an ")

PRONOUNS = frozenset(
    {
        "it",
        "its",
        "they",
        "them",
        "their",
        "he",
        "him",
        "his",
        "she",
        "her",
        "we",
        "us",
        "i",
        "you",
        "this",
        "that",
        "these",
        "those",
        "which",
        "who",
    }
)


def clean_surface(text: str) -> str:
    """Unicode-normalize, collapse whitespace, strip wrapping quotes. Case preserved."""
    text = unicodedata.normalize("NFKC", text)
    text = _DASHES.sub("-", text)
    text = _WS.sub(" ", text).strip()
    return text.strip(_QUOTES).strip()


def entity_key(text: str) -> str:
    """Canonical merge key: 'The  Aspirin.' -> 'aspirin'."""
    key = clean_surface(text).casefold().strip(_EDGE_PUNCT).strip()
    for article in _ARTICLES:
        if key.startswith(article) and len(key) > len(article):
            key = key[len(article) :]
            break
    return key


def relation_key(text: str) -> str:
    """'Is_Located  In' -> 'is located in'."""
    key = clean_surface(text).casefold().replace("_", " ")
    return _WS.sub(" ", key).strip(_EDGE_PUNCT).strip()


def match_text(text: str) -> str:
    """Form of the source chunk used for grounding checks."""
    return _WS.sub(" ", _DASHES.sub("-", unicodedata.normalize("NFKC", text))).casefold()
