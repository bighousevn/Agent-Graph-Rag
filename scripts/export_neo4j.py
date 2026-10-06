#!/usr/bin/env python3
"""Copy the HierarGraph (outputs/hierargraph.pkl from build_graph.py) into
Neo4j, then check that the database answers like the in-memory graph.

    docker compose up -d neo4j
    python scripts/export_neo4j.py

No LLM calls and no .env. Connection from NEO4J_URI / NEO4J_USER /
NEO4J_PASSWORD (default bolt://localhost:7687, neo4j, legalgraph-local).
Existing data in the database is replaced.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from vn_legal_graph.graph.graph_db import HierarGraph
from vn_legal_graph.graph.neo4j_store import Neo4jGraph, connect, export_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--check", type=int, default=50, help="Random queries to compare with the in-memory graph.")
    args = parser.parse_args()

    g = HierarGraph.load(args.graph)
    driver = connect()
    driver.verify_connectivity()
    t = time.time()
    print(f"Ghi {g.stats()} ...")
    export_graph(g, driver)
    db = Neo4jGraph(driver)
    print(f"Xong sau {time.time() - t:.0f}s. Neo4j: {db.stats()}")

    # Same answers? Queries = stored embeddings of random nodes.
    rng = np.random.default_rng(0)
    same_top, same_set, n = 0, 0, 0
    for node_type in ("Law", "Case", "Crime", "Cluster"):
        ids = g.nodes_of(node_type)
        for node_id in rng.choice(ids, size=min(args.check, len(ids)), replace=False):
            q = g._emb[node_type][node_id] + rng.normal(0, 0.02, size=len(g._emb[node_type][node_id]))
            a, b = g.search(q, node_type, top_k=5), db.search(q, node_type, top_k=5)
            n += 1
            same_top += a[0][0] == b[0][0]
            same_set += {x for x, _ in a} == {x for x, _ in b}
            assert abs(a[0][1] - b[0][1]) < 1e-3 or a[0][0] != b[0][0], (node_id, a[0], b[0])
    edges_ok = all(
        g.neighbors(c, "RELATES_TO_LAW") == db.neighbors(c, "RELATES_TO_LAW") for c in g.nodes_of("Case")[:100]
    ) and all(g.predecessors(c, "BELONGS_TO") == db.predecessors(c, "BELONGS_TO") for c in g.nodes_of("Cluster"))
    print(f"Kiểm tra {n} truy vấn: top-1 giống {same_top}/{n}, top-5 giống {same_set}/{n}; cạnh giống: {edges_ok}")
    driver.close()


if __name__ == "__main__":
    main()
