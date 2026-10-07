"""Retrieval over the HierarGraph without LLM steps (Phase 4 baseline).

Ports of the original repo's search functions
(core/graph_construct/feature_graph.py), minus the LLM reranking:
- ``direct_cases``  ~ search_similar_nodes_direct: kNN over all Case nodes
- ``cluster_cases`` ~ search_similar_nodes_top: nearest Cluster nodes, then
  nearest Case nodes inside them. The original lets an LLM rerank the top
  5 clusters and keeps 2; here the 2 are taken by cosine.
- ``laws_from_cases``: follow RELATES_TO_LAW from the retrieved cases, in
  case rank order (as the original collects laws).
- ``direct_laws`` ~ query_similar_laws_naive: query vs Law node text.
- ``frequency_prior``: laws ordered by how many corpus cases cite them; a
  floor any method has to beat on this skewed data.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..graph.graph_db import HierarGraph

Ranked = List[Tuple[str, float]]


def direct_cases(g: HierarGraph, query: np.ndarray, top_k: int = 5) -> Ranked:
    return g.search(query, "Case", top_k=top_k)


def cluster_cases(
    g: HierarGraph, query: np.ndarray, n_clusters: int = 2, top_k: int = 5
) -> Ranked:
    """Top ``n_clusters`` clusters by cosine, ``top_k`` cases from each,
    merged by similarity."""
    merged = {}
    for cluster, _ in g.search(query, "Cluster", top_k=n_clusters):
        members = g.predecessors(cluster, "BELONGS_TO")
        for case, score in g.search(query, "Case", top_k=top_k, among=members):
            merged[case] = max(score, merged.get(case, -1.0))
    return sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))


def laws_from_cases(g: HierarGraph, ranked_cases: Ranked) -> List[int]:
    out: List[int] = []
    for case, _ in ranked_cases:
        for law in g.neighbors(case, "RELATES_TO_LAW"):
            entry = g.node(law)["entry"]
            if entry not in out:
                out.append(entry)
    return out


def blhs_laws(g: HierarGraph) -> List[str]:
    """Law nodes of the penal code. Other codes (BLTTHS, XLVPHC, ND282)
    reuse article numbers, and these baselines key results by number."""
    return [n for n in g.nodes_of("Law") if g.node(n).get("bo_luat", "BLHS") == "BLHS"]


def direct_laws(g: HierarGraph, query: np.ndarray, top_k: int = 26) -> List[int]:
    return [g.node(law)["entry"] for law, _ in g.search(query, "Law", top_k=top_k, among=blhs_laws(g))]


def frequency_prior(g: HierarGraph) -> List[int]:
    counts = {g.node(law)["entry"]: len(g.predecessors(law, "RELATES_TO_LAW")) for law in blhs_laws(g)}
    return [e for e, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
