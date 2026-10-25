"""Tests for vn_legal_graph.graph.csv_export."""
import csv

import numpy as np

from vn_legal_graph.graph.csv_export import cell, export_csv
from vn_legal_graph.graph.graph_db import HierarGraph


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def small_graph():
    g = HierarGraph()
    g.add_node("law:173", "Law", np.ones(3), entry=173, title="Tội trộm cắp tài sản",
               description="Điều 173. Tội trộm cắp tài sản\n1. Người nào…",
               judge_dep=["Có lén lút chiếm đoạt không?", "Trị giá từ 2 triệu?"],
               related_laws=[{"loai": "van_ban_huong_dan", "id": "Nghị quyết 02/2019, Điều 2", "text": "Hướng dẫn…"},
                             {"loai": "dan_chieu_dieu_luat", "id": "Điều 168", "text": "…"}])
    g.add_node("crime:173", "Crime", np.ones(3), entry=173, description="Tội trộm cắp tài sản")
    g.add_node("case:a1", "Case", np.ones(3), caseId="a1", description="Lấy trộm xe", law=[173])
    g.add_edge("law:173", "crime:173", "RELATED_CRIME", match_type="exact")
    g.add_edge("case:a1", "law:173", "RELATES_TO_LAW")
    return g


def test_export_writes_nodes_types_edges_and_guidance(tmp_path):
    counts = export_csv(small_graph(), str(tmp_path))
    assert counts == {"nodes.csv": 3, "law.csv": 1, "crime.csv": 1, "case.csv": 1, "cluster.csv": 0,
                      "edges.csv": 2, "guidance.csv": 1}
    nodes = {r["id"]: r for r in read(tmp_path / "nodes.csv")}
    assert nodes["law:173"]["type"] == "Law" and nodes["law:173"]["label"] == "Tội trộm cắp tài sản"
    assert nodes["case:a1"]["label"] == "Lấy trộm xe"
    law = read(tmp_path / "law.csv")[0]
    assert list(law)[:3] == ["id", "entry", "title"] and "embedding" not in law and "node_type" not in law
    assert law["judge_dep"] == "Có lén lút chiếm đoạt không? | Trị giá từ 2 triệu?"
    assert law["description"].startswith("Điều 173.") and "\n" in law["description"]
    edges = read(tmp_path / "edges.csv")
    assert {(e["source"], e["target"], e["relation"]) for e in edges} == {
        ("law:173", "crime:173", "RELATED_CRIME"), ("case:a1", "law:173", "RELATES_TO_LAW")}
    assert [e["match_type"] for e in edges if e["relation"] == "RELATED_CRIME"] == ["exact"]
    guidance = read(tmp_path / "guidance.csv")
    assert guidance == [{"law_id": "law:173", "law_label": "Tội trộm cắp tài sản",
                         "guidance_id": "Nghị quyết 02/2019, Điều 2", "guidance_text": "Hướng dẫn…"}]
    assert (tmp_path / "nodes.csv").read_bytes().startswith(b"\xef\xbb\xbf")  # BOM: Excel reads UTF-8


def test_cell_formats():
    assert cell(None) == "" and cell(5) == "5"
    assert cell(["a", "b"]) == "a | b"
    assert cell([173, 174]) == "[173, 174]"
    assert cell({"k": "ạ"}) == '{"k": "ạ"}'
