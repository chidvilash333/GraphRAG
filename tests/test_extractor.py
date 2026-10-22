import json

from graphrag.config import ExtractionConfig, LLMConfig
from graphrag.extraction import TripleExtractor
from graphrag.llm import LLMOutputError

LLM_CFG = LLMConfig(base_url="http://fake", extraction_model="heavy", light_model="light")


class FakeClient:
    """Offline stand-in for GroqChatClient. Records every call."""

    def __init__(self, response: str = "", exc: Exception | None = None):
        self.response, self.exc, self.calls = response, exc, []

    def complete_json(self, *, system, user, model):
        self.calls.append({"system": system, "user": user, "model": model})
        if self.exc:
            raise self.exc
        return self.response


def _resp(*triples):
    return json.dumps(
        {"triples": [{"subject": s, "relation": r, "object": o} for s, r, o in triples]}
    )


def test_extract_happy_path_uses_heavy_model_and_delimits_chunk():
    client = FakeClient(_resp(("Warfarin", "is metabolized by", "CYP2C9")))
    ex = TripleExtractor(client, LLM_CFG, ExtractionConfig())
    res = ex.extract("Warfarin is metabolized by CYP2C9.")
    assert [t.key for t in res.triples] == [("warfarin", "is metabolized by", "cyp2c9")]
    call = client.calls[0]
    assert call["model"] == "heavy"
    assert "<<<\nWarfarin is metabolized by CYP2C9.\n>>>" in call["user"]
    assert "at most 15 triples" in call["system"]


def test_empty_chunk_skips_llm_call():
    client = FakeClient()
    assert TripleExtractor(client, LLM_CFG, ExtractionConfig()).extract("   ").triples == []
    assert client.calls == []


def test_long_chunk_truncated():
    client = FakeClient(_resp())
    TripleExtractor(client, LLM_CFG, ExtractionConfig(max_chunk_chars=10)).extract("x" * 50)
    assert "x" * 11 not in client.calls[0]["user"]


def test_llm_output_error_is_recoverable():
    client = FakeClient(exc=LLMOutputError("bad json"))
    res = TripleExtractor(client, LLM_CFG, ExtractionConfig()).extract("Aspirin inhibits COX-1.")
    assert res.triples == [] and "bad json" in res.error


def test_grounding_can_be_disabled():
    client = FakeClient(_resp(("Aspirin", "treats", "Migraine")))
    cfg = ExtractionConfig(require_grounded_entities=False)
    assert len(TripleExtractor(client, LLM_CFG, cfg).extract("Aspirin helps.").triples) == 1
