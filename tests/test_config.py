from pathlib import Path

import pytest

from graphrag.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_loads():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.llm.extraction_model == "openai/gpt-oss-120b"
    assert cfg.llm.light_model == "openai/gpt-oss-20b"
    assert cfg.extraction.require_grounded_entities is True
    assert cfg.chunking.max_tokens == 200
    assert cfg.embedding.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.retrieval.max_hops == 2
    assert cfg.retrieval.entity_mode == "both"
    assert cfg.generation.model_role == "heavy"
    assert cfg.retrieval.path_rerank is True
    assert cfg.evaluation.recall_at_k == (2, 3, 4, 5, 6)


def test_bad_retrieval_values_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    base = "llm:\n  base_url: x\n  extraction_model: a\n  light_model: b\n"
    for bad in ("max_hops: 5", "entity_mode: fuzzy", "entity_match_threshold: 0"):
        p.write_text(base + "retrieval:\n  " + bad + "\n")
        with pytest.raises(ConfigError):
            load_config(p)


def test_unknown_section_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  base_url: x\n  extraction_model: a\n  light_model: b\nchunkin: {}\n")
    with pytest.raises(ConfigError, match="chunkin"):
        load_config(p)


def test_overlap_must_be_smaller_than_window(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "llm:\n  base_url: x\n  extraction_model: a\n  light_model: b\n"
        "chunking:\n  max_tokens: 10\n  overlap_tokens: 10\n"
    )
    with pytest.raises(ConfigError):
        load_config(p)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  base_url: x\n  extraction_model: a\n  light_model: b\n  max_token: 5\n")
    with pytest.raises(ConfigError, match="max_token"):
        load_config(p)


def test_out_of_range_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  base_url: x\n  extraction_model: a\n  light_model: b\n  temperature: 5\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config("nope.yaml")
