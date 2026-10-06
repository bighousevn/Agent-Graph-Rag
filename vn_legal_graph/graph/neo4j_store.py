"""The HierarGraph in Neo4j: export from the in-memory graph, and a
read-only `Neo4jGraph` with the query methods the pipeline uses (node,
nodes_of, neighbors, predecessors, search, stats), so retrieval and Q&A run
against the database instead of outputs/hierargraph.pkl.

    docker compose up -d neo4j
    python scripts/export_neo4j.py              # pkl -> Neo4j
    python scripts/run_qa_pilot.py --backend neo4j ...

Model:
- one label per node type (:Law, :Crime, :Case, :Cluster), plus :Node on
  every node for the unique `id` constraint;
- properties as stored in the graph; values Neo4j cannot hold (lists of
  maps such as related_laws, nested dicts) are JSON strings in
  `<name>__json` and decoded back by `Neo4jGraph.node`;
- `embedding`: the L2-normalised vector, with a cosine vector index per
  label;
- relationships typed by relation (RELATED_CRIME, RELATES_TO_LAW,
  SIMILAR_TO, BELONGS_TO) with their attributes.

`search` is exact by default: vector.similarity.cosine over every node of
the label, like HierarGraph.search (1,117 Law nodes: a few ms). The HNSW
vector index (quantization off) is used with `exact=False`; on this graph
it missed a true top-5 node in 6 of 160 test queries.

Connection: NEO4J_URI (bolt://localhost:7687), NEO4J_USER (neo4j),
NEO4J_PASSWORD (legalgraph-local) from the environment.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

NODE_TYPES = ("Law", "Crime", "Case", "Cluster")
JSON_SUFFIX = "__json"


def connect(uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(user or os.environ.get("NEO4J_USER", "neo4j"), password or os.environ.get("NEO4J_PASSWORD", "legalgraph-local")),
    )


def _is_primitive(v: Any) -> bool:
    return v is None or isinstance(v, (str, bool, int, float))


def to_props(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Graph attributes -> Neo4j properties. Primitives and lists of
    primitives stay as they are; anything else becomes `<key>__json`."""
    props: Dict[str, Any] = {}
    for k, v in attrs.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        if v is None:
            continue
        if _is_primitive(v) or (isinstance(v, (list, tuple)) and all(_is_primitive(x) and x is not None for x in v)
                                and len({type(x) for x in v}) <= 1):
            props[k] = list(v) if isinstance(v, tuple) else v
        else:
            props[k + JSON_SUFFIX] = json.dumps(v, ensure_ascii=False)
    return props


def from_props(props: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in props.items():
        if k in ("embedding", "ord"):
            continue
        if k.endswith(JSON_SUFFIX):
            out[k[: -len(JSON_SUFFIX)]] = json.loads(v)
        else:
            out[k] = v
    return out


def _batches(rows: List[Dict], size: int = 500) -> Iterable[List[Dict]]:
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def export_graph(g, driver, database: Optional[str] = None, clear: bool = True) -> Dict[str, int]:
    """Write a HierarGraph into Neo4j (replacing what is there by default)."""
    dims = None
    with driver.session(database=database) as s:
        if clear:
            s.run("MATCH (n) DETACH DELETE n")
            for t in NODE_TYPES:
                s.run(f"DROP INDEX {t.lower()}_embedding IF EXISTS")
        s.run("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
        for t in NODE_TYPES:
            rows = []
            for ord_, (node_id, data) in enumerate(g.graph.nodes(data=True)):
                if data.get("node_type") != t:
                    continue
                attrs = {k: v for k, v in data.items() if k != "node_type"}
                attrs["ord"] = ord_  # insertion order: HierarGraph.nodes_of returns nodes in it
                emb = g._emb[t].get(node_id)
                props = to_props(attrs)
                if emb is not None:
                    props["embedding"] = [float(x) for x in emb]
                    dims = len(emb)
                rows.append({"id": node_id, "props": props})
            for batch in _batches(rows):
                s.run(f"UNWIND $rows AS r MERGE (n:Node:{t} {{id: r.id}}) SET n += r.props", rows=batch)
            if dims:
                s.run(
                    f"CREATE VECTOR INDEX {t.lower()}_embedding IF NOT EXISTS FOR (n:{t}) ON n.embedding "
                    "OPTIONS {indexConfig: {`vector.dimensions`: $d, `vector.similarity_function`: 'cosine', "
                    # quantization (on by default) shifts scores by ~0.002 and can swap close ranks
                    "`vector.quantization.enabled`: false}}",
                    d=dims,
                )
        # HierarGraph returns out-neighbours in out-edge order and
        # predecessors in in-edge order of each node; keep both.
        in_seq = {}
        for node in g.graph.nodes:
            for i, (src, dst, key) in enumerate(g.graph.in_edges(node, keys=True)):
                in_seq[(src, dst, key)] = i
        by_rel: Dict[str, List[Dict]] = {}
        for src, dst, key, data in g.graph.edges(keys=True, data=True):
            attrs = {k: v for k, v in data.items() if k != "relation"}
            attrs["out_seq"] = list(g.graph.out_edges(src, keys=True)).index((src, dst, key))
            attrs["in_seq"] = in_seq[(src, dst, key)]
            by_rel.setdefault(data["relation"], []).append({"src": src, "dst": dst, "props": to_props(attrs)})
        for rel, rows in by_rel.items():
            for batch in _batches(rows):
                s.run(
                    f"UNWIND $rows AS r MATCH (a:Node {{id: r.src}}), (b:Node {{id: r.dst}}) "
                    f"CREATE (a)-[e:{rel}]->(b) SET e += r.props",
                    rows=batch,
                )
        s.run("CALL db.awaitIndexes(300)")
    return {"nodes": g.graph.number_of_nodes(), "edges": g.graph.number_of_edges()}


class Neo4jGraph:
    """Read side, same query methods as HierarGraph."""

    def __init__(self, driver, database: Optional[str] = None, exact: bool = True) -> None:
        self.driver = driver
        self.database = database
        self.exact = exact
        self._node_cache: Dict[str, Dict] = {}

    def _run(self, query: str, **params) -> List[Any]:
        with self.driver.session(database=self.database) as s:
            return list(s.run(query, **params))

    def node(self, node_id: str) -> Dict:
        if node_id not in self._node_cache:
            rows = self._run("MATCH (n:Node {id: $id}) RETURN n, labels(n) AS labels", id=node_id)
            if not rows:
                raise KeyError(node_id)
            data = from_props(dict(rows[0]["n"]))
            data["node_type"] = next(l for l in rows[0]["labels"] if l in NODE_TYPES)
            self._node_cache[node_id] = data
        return dict(self._node_cache[node_id])

    def nodes_of(self, node_type: str) -> List[str]:
        return [r["id"] for r in self._run(f"MATCH (n:{node_type}) RETURN n.id AS id ORDER BY n.ord")]

    def neighbors(self, node_id: str, relation: str) -> List[str]:
        rows = self._run(f"MATCH (:Node {{id: $id}})-[e:{relation}]->(m) RETURN m.id AS id ORDER BY e.out_seq", id=node_id)
        return list(dict.fromkeys(r["id"] for r in rows))

    def predecessors(self, node_id: str, relation: str) -> List[str]:
        rows = self._run(f"MATCH (m)-[e:{relation}]->(:Node {{id: $id}}) RETURN m.id AS id ORDER BY e.in_seq", id=node_id)
        return list(dict.fromkeys(r["id"] for r in rows))

    def search(
        self, query: np.ndarray, node_type: str, top_k: int = 5, among: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, float]]:
        q = np.asarray(query, dtype=float)
        n = np.linalg.norm(q)
        q = (q / n if n else q).tolist()
        if among is not None or self.exact:
            where = "n.id IN $among AND " if among is not None else ""
            rows = self._run(
                f"MATCH (n:{node_type}) WHERE {where}n.embedding IS NOT NULL "
                "WITH n, vector.similarity.cosine(n.embedding, $q) AS s "
                "RETURN n.id AS id, s ORDER BY s DESC, id LIMIT $k",
                among=list(among) if among is not None else [], q=q, k=top_k + 5,
            )
        else:
            rows = self._run(
                "CALL db.index.vector.queryNodes($index, $k, $q) YIELD node, score RETURN node.id AS id, score AS s",
                index=f"{node_type.lower()}_embedding", k=top_k + 5, q=q,
            )
        # Neo4j cosine scores are (1 + cos) / 2; return the cosine itself
        pairs = [(r["id"], 2 * r["s"] - 1) for r in rows]
        pairs.sort(key=lambda p: (-round(p[1], 5), p[0]))  # same tie rule as HierarGraph.search
        return pairs[:top_k]

    def stats(self) -> Dict[str, Dict[str, int]]:
        nodes = {r["t"]: r["c"] for r in self._run(
            "MATCH (n:Node) UNWIND [l IN labels(n) WHERE l <> 'Node'] AS t RETURN t, count(*) AS c")}
        edges = {r["t"]: r["c"] for r in self._run("MATCH ()-[e]->() RETURN type(e) AS t, count(*) AS c")}
        return {"nodes": nodes, "edges": edges}
