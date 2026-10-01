"""In-memory property graph for the HierarGraph (Crime / Law / Case /
Cluster), equivalent to the original repo's core/graph_construct/graph_db.py.

Differences from the original, on purpose:
- Node ids carry their type ("law:249", "case:vicsr-12") instead of
  random UUIDs, so ids are stable across rebuilds.
- Embeddings live in one normalized matrix per node type, and search is a
  matrix product. The original rebuilt its vector index on every node
  insert and compared vectors one by one.
- Neighbor lookup by relation uses the graph's adjacency directly instead
  of scanning every node.
"""
from __future__ import annotations

import collections
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

NODE_TYPES = ("Crime", "Law", "Case", "Cluster")


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class HierarGraph:
    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._emb: Dict[str, Dict[str, np.ndarray]] = {t: {} for t in NODE_TYPES}
        self._index: Dict[str, Tuple[List[str], np.ndarray]] = {}

    # ---- nodes / edges -------------------------------------------------

    def add_node(self, node_id: str, node_type: str, embedding: Optional[np.ndarray] = None, **attrs) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type {node_type!r}")
        self.graph.add_node(node_id, node_type=node_type, **attrs)
        if embedding is not None:
            self._emb[node_type][node_id] = _normalize(embedding)
            self._index.pop(node_type, None)

    def add_edge(self, source: str, target: str, relation: str, **attrs) -> None:
        for node in (source, target):
            if node not in self.graph:
                raise KeyError(f"Edge {relation} refers to missing node {node!r}")
        self.graph.add_edge(source, target, relation=relation, **attrs)

    def node(self, node_id: str) -> Dict:
        return dict(self.graph.nodes[node_id])

    def nodes_of(self, node_type: str) -> List[str]:
        return [n for n, d in self.graph.nodes(data=True) if d["node_type"] == node_type]

    def neighbors(self, node_id: str, relation: str) -> List[str]:
        out = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data["relation"] == relation and target not in out:
                out.append(target)
        return out

    def predecessors(self, node_id: str, relation: str) -> List[str]:
        out = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if data["relation"] == relation and source not in out:
                out.append(source)
        return out

    # ---- vectors ---------------------------------------------------------

    def embedding(self, node_id: str) -> Optional[np.ndarray]:
        node_type = self.graph.nodes[node_id]["node_type"]
        return self._emb[node_type].get(node_id)

    def _matrix(self, node_type: str) -> Tuple[List[str], np.ndarray]:
        if node_type not in self._index:
            ids = sorted(self._emb[node_type])
            matrix = np.stack([self._emb[node_type][i] for i in ids]) if ids else np.zeros((0, 0), np.float32)
            self._index[node_type] = (ids, matrix)
        return self._index[node_type]

    def search(
        self, query: np.ndarray, node_type: str, top_k: int = 5, among: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, float]]:
        """Cosine-similarity search over nodes of one type, optionally
        restricted to ``among``."""
        ids, matrix = self._matrix(node_type)
        if not ids:
            return []
        scores = matrix @ _normalize(query)
        if among is not None:
            allowed = set(among)
            pairs = [(i, float(s)) for i, s in zip(ids, scores) if i in allowed]
        else:
            pairs = [(i, float(s)) for i, s in zip(ids, scores)]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs[:top_k]

    def similarity_matrix(self, node_type: str) -> Tuple[List[str], np.ndarray]:
        ids, matrix = self._matrix(node_type)
        return ids, (matrix @ matrix.T if ids else matrix)

    # ---- persistence / stats ---------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"graph": self.graph, "embeddings": self._emb}, f)

    @classmethod
    def load(cls, path: str) -> "HierarGraph":
        with open(path, "rb") as f:
            data = pickle.load(f)
        g = cls()
        g.graph = data["graph"]
        g._emb = data["embeddings"]
        return g

    def stats(self) -> Dict[str, Dict[str, int]]:
        nodes = collections.Counter(d["node_type"] for _, d in self.graph.nodes(data=True))
        edges = collections.Counter(d["relation"] for _, _, d in self.graph.edges(data=True))
        return {"nodes": dict(nodes), "edges": dict(edges)}
