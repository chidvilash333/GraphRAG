from graphrag.config import ExtractionConfig, LLMConfig
from graphrag.extraction import TripleExtractor
from graphrag.llm import LLMOutputError
from helpers import FakeClient, triples_json

LLM_CFG = LLMConfig(base_url="http://fake", extraction_model="heavy", light_model="light")


def test_extract_happy_path_uses_heavy_model_and_delimits_chunk():
    client = FakeClient(triples_json(("Warfarin", "is metabolized by", "CYP2C9")))
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
    client = FakeClient(triples_json())
    TripleExtractor(client, LLM_CFG, ExtractionConfig(max_chunk_chars=10)).extract("x" * 50)
    assert "x" * 11 not in client.calls[0]["user"]


def test_llm_output_error_is_recoverable():
    client = FakeClient(exc=LLMOutputError("bad json"))
    res = TripleExtractor(client, LLM_CFG, ExtractionConfig()).extract("Aspirin inhibits COX-1.")
    assert res.triples == [] and "bad json" in res.error


def test_grounding_can_be_disabled():
    client = FakeClient(triples_json(("Aspirin", "treats", "Migraine")))
    cfg = ExtractionConfig(require_grounded_entities=False)
    assert len(TripleExtractor(client, LLM_CFG, cfg).extract("Aspirin helps.").triples) == 1
