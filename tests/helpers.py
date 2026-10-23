"""Offline fakes shared by tests. No network, no model downloads."""

from __future__ import annotations

import json
import re

_WORD = re.compile(r"\S+")


class FakeClient:
    """Stand-in for GroqChatClient. `responses` maps a substring of the chunk to a reply."""

    def __init__(self, response: str = "", exc: Exception | None = None, responses=None):
        self.response, self.exc, self.responses, self.calls = response, exc, responses, []

    def complete_json(self, *, system, user, model):
        self.calls.append({"system": system, "user": user, "model": model})
        if self.exc:
            raise self.exc
        if self.responses:
            for needle, reply in self.responses.items():
                if needle in user:
                    return reply
            return '{"triples": []}'
        return self.response


def triples_json(*triples):
    return json.dumps(
        {"triples": [{"subject": s, "relation": r, "object": o} for s, r, o in triples]}
    )


class WhitespaceTokenizer:
    """One token per whitespace-separated word, with real character offsets."""

    def token_spans(self, text):
        return [m.span() for m in _WORD.finditer(text)]


class SubwordTokenizer:
    """Splits words into 3-char pieces, to test mid-word chunk boundaries."""

    def token_spans(self, text):
        spans = []
        for m in _WORD.finditer(text):
            spans += [(i, min(i + 3, m.end())) for i in range(m.start(), m.end(), 3)]
        return spans


class FakeEmbedder:
    max_tokens = 256

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[float(len(t)), 1.0] for t in texts]


class InMemoryVectorStore:
    def __init__(self):
        self.rows = {}
        self.resets = 0

    def reset(self):
        self.rows, self.resets = {}, self.resets + 1

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas, strict=True):
            assert i not in self.rows, f"duplicate id {i}"
            self.rows[i] = {"embedding": e, "document": d, "metadata": m}

    def count(self):
        return len(self.rows)
