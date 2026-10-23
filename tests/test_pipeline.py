import dataclasses

import pytest

from graphrag.config import (
    AppConfig,
    ChunkingConfig,
    ExtractionConfig,
    LLMConfig,
    PathsConfig,
)
from graphrag.extraction import TripleExtractor
from graphrag.graph import KnowledgeGraph
from graphrag.ingest.pipeline import run_ingest
from graphrag.llm import LLMOutputError
from helpers import (
    FakeClient,
    FakeEmbedder,
    InMemoryVectorStore,
    WhitespaceTokenizer,
    triples_json,
)

DOCS = {
    "zoraxin.md": "Zoraxin is cleared by the liver enzyme Velkase-3.",
    "tremidol.md": "Tremidol strongly inhibits Velkase-3.",
}
REPLIES = {
    "Zoraxin": triples_json(("Zoraxin", "is cleared by", "Velkase-3")),
    "Tremidol": triples_json(("Tremidol", "inhibits", "velkase-3")),
}


class NoWait:
    def wait(self):
        pass


def _setup(tmp_path, max_tokens=50):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in DOCS.items():
        (docs / name).write_text(text)
    llm = LLMConfig(base_url="x", extraction_model="heavy", light_model="light")
    cfg = AppConfig(
        llm=llm,
        extraction=ExtractionConfig(),
        paths=PathsConfig(
            docs_dir=str(docs),
            index_dir=str(tmp_path / "index"),
            graph_path=str(tmp_path / "graph" / "graph.json"),
            cache_path=str(tmp_path / "cache" / "x.json"),
        ),
        chunking=ChunkingConfig(max_tokens=max_tokens, overlap_tokens=5),
    )
    return cfg


def _run(cfg, client, store=None):
    return run_ingest(
        cfg,
        tokenizer=WhitespaceTokenizer(),
        embedder=FakeEmbedder(),
        store=store or InMemoryVectorStore(),
        extractor=TripleExtractor(client, cfg.llm, cfg.extraction),
        limiter=NoWait(),
    )


def test_end_to_end_links_graph_to_vector_store(tmp_path):
    cfg = _setup(tmp_path)
    store = InMemoryVectorStore()
    kg, stats = _run(cfg, FakeClient(responses=REPLIES), store)

    assert (stats.documents, stats.chunks, stats.llm_calls) == (2, 2, 2)
    assert set(store.rows) == {"tremidol:000", "zoraxin:000"}
    # The bridge entity is one merged node pointing at chunks from BOTH docs...
    assert kg.chunk_ids("velkase-3") == {"zoraxin:000", "tremidol:000"}
    # ...and every graph chunk ID resolves to a real stored chunk.
    assert kg.all_chunk_ids() <= set(store.rows)
    assert store.rows["zoraxin:000"]["metadata"]["doc_id"] == "zoraxin"
    # Saved graph reloads identically.
    assert KnowledgeGraph.load(cfg.paths.graph_path).stats() == stats.graph


def test_second_ingest_uses_cache_and_rebuilds_store(tmp_path):
    cfg = _setup(tmp_path)
    store = InMemoryVectorStore()
    _run(cfg, FakeClient(responses=REPLIES), store)
    client = FakeClient(responses=REPLIES)
    kg, stats = _run(cfg, client, store)
    assert client.calls == [] and stats.cache_hits == 2
    assert store.resets == 2 and store.count() == 2
    assert kg.stats()["edges"] == 2


def test_changed_prompt_settings_invalidate_cache(tmp_path):
    cfg = _setup(tmp_path)
    _run(cfg, FakeClient(responses=REPLIES))
    cfg2 = dataclasses.replace(cfg, extraction=ExtractionConfig(max_triples_per_chunk=5))
    client = FakeClient(responses=REPLIES)
    _run(cfg2, client)
    assert len(client.calls) == 2


def test_failed_chunks_reported_and_not_cached(tmp_path):
    cfg = _setup(tmp_path)
    kg, stats = _run(cfg, FakeClient(exc=LLMOutputError("bad")))
    assert sorted(stats.failed_chunks) == ["tremidol:000", "zoraxin:000"]
    assert kg.stats()["nodes"] == 0
    client = FakeClient(responses=REPLIES)
    _run(cfg, client)
    assert len(client.calls) == 2  # retried


def test_chunk_size_over_embedder_limit_rejected(tmp_path):
    cfg = _setup(tmp_path, max_tokens=255)
    with pytest.raises(ValueError, match="exceeds"):
        _run(cfg, FakeClient(responses=REPLIES))
