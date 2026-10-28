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

MAX_PATHS_PER_HIT = 4


@dataclass(frozen=True)
class GraphHit:
    chunk_id: str
    hop: int  # 1 = an edge touching a seed entity
    # Every shortest path (up to MAX_PATHS_PER_HIT) that reached this chunk. A chunk is
    # often reached by several edges; keeping them all lets the reranker see, e.g., the
    # "headquartered in" edge and not just whichever edge sorted first.
    paths: tuple[tuple[str, ...], ...]

    @property
    def path(self) -> tuple[str, ...]:
        return self.paths[0]


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
    found: dict[str, tuple[int, list[tuple[str, ...]]]] = {}

    for depth in range(1, max_hops + 1):
        next_frontier: list[str] = []
        for node in frontier:
            for nb, edges in _ranked_neighbors(kg, node, max_neighbors):
                for edge_str, chunk_ids in edges:
                    path = path_to[node] + (edge_str,)
                    for cid in sorted(chunk_ids):
                        # BFS order: the first depth we see a chunk at is its shortest hop.
                        if cid not in found:
                            found[cid] = (depth, [path])
                        elif (
                            found[cid][0] == depth
                            and path not in found[cid][1]
                            and len(found[cid][1]) < MAX_PATHS_PER_HIT
                        ):
                            found[cid][1].append(path)
                            found[cid][1].append(path)
                if nb not in visited:
                    visited.add(nb)
                    path_to[nb] = path_to[node] + (edges[0][0],)
                    next_frontier.append(nb)
        frontier = next_frontier
        if not frontier:
            break
    return [GraphHit(cid, hop, tuple(paths)) for cid, (hop, paths) in found.items()]
