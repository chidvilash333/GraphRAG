"""Knowledge graph on NetworkX.

- Node id = normalized entity key ("cyp2c9"), so case/format variants merge.
- Node attrs: display name, surface-form counts, and the chunk IDs it was extracted
  from. Those chunk IDs are the link back to source text in the vector store.
- Edges: MultiDiGraph keyed by relation, so A -inhibits-> B and A -binds-> B coexist
  as separate typed edges. Each edge also keeps its own source chunk IDs (provenance).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import networkx as nx

from graphrag.extraction.parse import Triple

FORMAT_VERSION = 1


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ---- building -------------------------------------------------------------
    def _touch_node(self, key: str, surface: str, chunk_id: str) -> None:
        if key not in self.g:
            self.g.add_node(key, surfaces=Counter(), chunk_ids=set())
        data = self.g.nodes[key]
        data["surfaces"][surface] += 1
        data["chunk_ids"].add(chunk_id)

    def add_triples(self, triples: list[Triple], chunk_id: str) -> None:
        for t in triples:
            self._touch_node(t.subject_key, t.subject, chunk_id)
            self._touch_node(t.object_key, t.object, chunk_id)
            if self.g.has_edge(t.subject_key, t.object_key, key=t.relation):
                self.g.edges[t.subject_key, t.object_key, t.relation]["chunk_ids"].add(chunk_id)
            else:
                self.g.add_edge(t.subject_key, t.object_key, key=t.relation, chunk_ids={chunk_id})

    # ---- reading --------------------------------------------------------------
    def name(self, key: str) -> str:
        """Most frequent surface form; ties go to the first one seen (deterministic)."""
        return self.g.nodes[key]["surfaces"].most_common(1)[0][0]

    def chunk_ids(self, key: str) -> set[str]:
        return set(self.g.nodes[key]["chunk_ids"])

    def all_chunk_ids(self) -> set[str]:
        return {cid for _, d in self.g.nodes(data=True) for cid in d["chunk_ids"]}

    def stats(self) -> dict[str, int]:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "components": nx.number_weakly_connected_components(self.g) if self.g else 0,
        }

    # ---- persistence ----------------------------------------------------------
    # Hand-rolled JSON instead of nx.node_link_data: sets/Counters need converting
    # anyway, and this avoids node_link_data's keyword changes across NetworkX versions.
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": FORMAT_VERSION,
            "nodes": [
                {
                    "id": key,
                    "surfaces": dict(d["surfaces"]),
                    "chunk_ids": sorted(d["chunk_ids"]),
                }
                for key, d in self.g.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "relation": rel, "chunk_ids": sorted(d["chunk_ids"])}
                for u, v, rel, d in self.g.edges(keys=True, data=True)
            ],
        }
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported graph format in {path}; re-run ingest")
        kg = cls()
        for n in payload["nodes"]:
            kg.g.add_node(n["id"], surfaces=Counter(n["surfaces"]), chunk_ids=set(n["chunk_ids"]))
        for e in payload["edges"]:
            kg.g.add_edge(
                e["source"], e["target"], key=e["relation"], chunk_ids=set(e["chunk_ids"])
            )
        return kg
