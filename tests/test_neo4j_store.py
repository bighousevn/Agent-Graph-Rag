"""Tests for vn_legal_graph.graph.neo4j_store.

Property conversion runs everywhere. The round trip through a real
database replaces its contents, so it only runs with NEO4J_TEST=1
(docker compose up -d neo4j).
"""
import os

import numpy as np
import pytest

from tests.test_graph import fake_embed
from vn_legal_graph.graph.build import build_base_graph
from vn_legal_graph.graph.neo4j_store import from_props, to_props


def test_props_round_trip_keeps_nested_values():
    attrs = {
        "entry": np.int64(173),
        "description": "Điều 173. Tội trộm cắp tài sản",
        "judge_dep": ["Có lén lút không?", "Có chiếm đoạt không?"],
        "related_laws": [{"loai": "van_ban_huong_dan", "id": "NQ 04/2025", "text": "..."}],
        "empty": None,
        "mixed": [1, "a"],
    }
    props = to_props(attrs)
    assert props["entry"] == 173 and isinstance(props["entry"], int)
    assert props["judge_dep"] == attrs["judge_dep"]
    assert "related_laws__json" in props and "related_laws" not in props
    assert "mixed__json" in props and "empty" not in props
    back = from_props({**props, "embedding": [0.1, 0.2]})
    assert back["related_laws"] == attrs["related_laws"] and back["mixed"] == [1, "a"]
    assert "embedding" not in back


@pytest.mark.skipif(os.environ.get("NEO4J_TEST") != "1", reason="needs a Neo4j it may wipe (NEO4J_TEST=1)")
def test_neo4j_answers_like_the_in_memory_graph():
    from vn_legal_graph.graph.neo4j_store import Neo4jGraph, connect, export_graph
    from tests.test_graph import LAWS

    g, _ = build_base_graph(LAWS, [], fake_embed)
    driver = connect()
    export_graph(g, driver)
    db = Neo4jGraph(driver)
    q = fake_embed("trộm cắp xe máy")
    assert [x for x, _ in db.search(q, "Law", 2)] == [x for x, _ in g.search(q, "Law", 2)]
    assert db.node("law:173")["entry"] == 173 and db.node("law:173")["node_type"] == "Law"
    assert db.predecessors("crime:173", "RELATED_CRIME") == g.predecessors("crime:173", "RELATED_CRIME")
    driver.close()
