from pathlib import Path

import pytest

from graphrag.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_loads():
    cfg = load_config(ROOT / "config.yaml")
    assert cfg.llm.extraction_model == "openai/gpt-oss-120b"
    assert cfg.llm.light_model == "openai/gpt-oss-20b"
    assert cfg.extraction.require_grounded_entities is True


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
