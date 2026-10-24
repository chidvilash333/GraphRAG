"""Bounded breadth-first traversal from seed entities to source chunks.

Bounds, so the search can't explode on hub nodes:
- max_hops: depth limit.
- max_neighbors: per-node cap. Neighbors are ranked by edge support (how many chunks
  state the connection), then by LOWER degree, so generic hubs ("patients", "liver")
  lose to specific entities when the cap bites.

Edges are followed in both directions: "Tremidol inhibits Velkase-3" must lead from
Velkase-3 back to Tremidol. Each edge carries its own chunk IDs, which are more precise
than a node's chunk IDs (a hub node is mentioned in many unrelated chunks).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from graphrag.graph import KnowledgeGraph


@dataclass(frozen=True)
class GraphHit:
    chunk_id: str
    hop: int  # 1 = an edge touching a seed entity
    path: tuple[str, ...]  # human-readable edges from a seed to the edge that found it


def _ranked_neighbors(
    kg: KnowledgeGraph, node: str, max_neighbors: int
) -> list[tuple[str, list[tuple[str, set[str]]]]]:
    g = kg.g
    grouped: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for _, v, rel, d in g.out_edges(node, keys=True, data=True):
        grouped[v].append((f"{kg.name(node)} -[{rel}]-> {kg.name(v)}", d["chunk_ids"]))
    for u, _, rel, d in g.in_edges(node, keys=True, data=True):
        grouped[u].append((f"{kg.name(u)} -[{rel}]-> {kg.name(node)}", d["chunk_ids"]))

    def rank(nb: str) -> tuple[int, int, str]:
        support = len(set().union(*(cids for _, cids in grouped[nb])))
        return (-support, g.degree(nb), nb)

    ordered = sorted(grouped, key=rank)[:max_neighbors]
    return [(nb, sorted(grouped[nb], key=lambda e: e[0])) for nb in ordered]


def traverse(
    kg: KnowledgeGraph, seeds: list[str], *, max_hops: int, max_neighbors: int
) -> list[GraphHit]:
    """Return one hit per reachable chunk, at its shortest hop distance."""
    frontier = sorted({s for s in seeds if s in kg.g})
    path_to: dict[str, tuple[str, ...]] = {s: () for s in frontier}
    visited = set(frontier)
    hits: dict[str, GraphHit] = {}

    for depth in range(1, max_hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            for nb, edges in _ranked_neighbors(kg, node, max_neighbors):
                for edge_str, chunk_ids in edges:
                    for cid in sorted(chunk_ids):
                        # BFS order means the first time we see a chunk is its shortest hop.
                        hits.setdefault(cid, GraphHit(cid, depth, path_to[node] + (edge_str,)))
                if nb not in visited:
                    visited.add(nb)
                    path_to[nb] = path_to[node] + (edges[0][0],)
                    next_frontier.append(nb)
        frontier = next_frontier
        if not frontier:
            break
    return list(hits.values())
