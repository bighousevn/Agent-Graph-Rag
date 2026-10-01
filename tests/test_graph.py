"""Tests for vn_legal_graph.graph (graph_db + build) with a fake embedder."""
import numpy as np
import pytest

from vn_legal_graph.graph.build import (
    add_knn_edges,
    attach_clusters,
    build_base_graph,
    cluster_inputs,
    cluster_prompt,
    detect_communities,
)
from vn_legal_graph.graph.graph_db import HierarGraph

TOPICS = ["ma túy", "trộm cắp", "cây"]


def fake_embed(text: str) -> np.ndarray:
    """One dimension per topic keyword, plus a small text-dependent term so
    no two vectors are identical."""
    v = np.array([1.0 if t in text else 0.0 for t in TOPICS] + [0.01 * (len(text) % 7)])
    return v if v[:3].any() else v + np.array([0.1, 0.1, 0.1, 0.0])


LAWS = [
    {"id": 173, "suffix": "", "items": [{"text": "Điều 173. Tội trộm cắp tài sản ...", "crime": ["Tội trộm cắp tài sản"],
                                         "judge_dep": ["Có lén lút chiếm đoạt tài sản không?"], "related_laws": []}]},
    {"id": 249, "suffix": "", "items": [{"text": "Điều 249. Tội tàng trữ trái phép chất ma túy ...",
                                         "crime": ["Tội tàng trữ trái phép chất ma túy"], "judge_dep": [], "related_laws": []}]},
]


def case(i, dieu, desc, acts=("x",), role="corpus", err=False):
    return {
        "id": f"c{i}", "vai_tro": role, "dac_trung_loi": err, "dieu": dieu, "toi_danh": [f"tội {dieu}"],
        "dac_trung": {"criminal_acts": list(acts)}, "mo_ta_dac_trung": desc, "dieu_khoan": [],
    }


CASES = (
    [case(i, [249], f"Hành vi phạm tội: tàng trữ ma túy {i}") for i in range(5)]
    + [case(10 + i, [173], f"Hành vi phạm tội: trộm cắp xe máy {i}") for i in range(5)]
    + [case(90, [999], "Hành vi phạm tội: trộm cắp điều không tồn tại")]
    + [case(91, [249], "không có hành vi", acts=())]
    + [case(92, [249], "lỗi", err=True)]
)


@pytest.fixture()
def built():
    g, report = build_base_graph(LAWS, CASES, fake_embed)
    return g, report


def test_base_graph_nodes_and_edges(built):
    g, report = built
    stats = g.stats()
    assert stats["nodes"] == {"Law": 2, "Crime": 2, "Case": 11}
    assert stats["edges"]["RELATED_CRIME"] == 2
    assert stats["edges"]["RELATES_TO_LAW"] == 10  # c90 points to a missing law
    assert g.neighbors("law:249", "RELATED_CRIME") == ["crime:249"]
    assert g.node("law:173")["judge_dep"] == ["Có lén lút chiếm đoạt tài sản không?"]
    assert report["thieu_law"] == [("c90", 999)]
    assert report["bo_qua_khong_co_hanh_vi"] == ["c91"]
    assert report["bo_qua_dac_trung_loi"] == ["c92"]


def test_test_cases_are_refused():
    with pytest.raises(ValueError):
        build_base_graph(LAWS, [case(1, [249], "ma túy", role="test")], fake_embed)


def test_search_finds_same_topic(built):
    g, _ = built
    hits = g.search(fake_embed("ma túy"), "Case", top_k=3)
    assert all(g.node(n)["law"] == [249] for n, _ in hits)
    restricted = g.search(fake_embed("ma túy"), "Case", top_k=3, among=["case:c10", "case:c11"])
    assert {n for n, _ in restricted} == {"case:c10", "case:c11"}


def test_knn_edges_stay_within_topic(built):
    g, _ = built
    assert add_knn_edges(g, k=3) == 11 * 3
    for target in g.neighbors("case:c0", "SIMILAR_TO"):
        assert g.node(target)["law"] == [249]
    assert "case:c0" not in g.neighbors("case:c0", "SIMILAR_TO")


def test_communities_and_cluster_inputs(built):
    g, _ = built
    add_knn_edges(g, k=3)
    comms = detect_communities(g, seed=42)
    assert sum(len(v) for v in comms.values()) == 11
    # drug and theft cases do not share a community
    for members in comms.values():
        laws = {tuple(g.node(m)["law"]) for m in members}
        assert not ({(249,), (173,)} <= laws)
    inputs = cluster_inputs(g, comms)
    drug = next(ci for ci in inputs if "Tội tàng trữ trái phép chất ma túy" in ci["top_crimes"])
    assert drug["top_crime_counts"][0] >= 1
    assert "Tội danh chủ yếu: Tội tàng trữ trái phép chất ma túy" in drug["input"]
    assert cluster_prompt(drug).endswith(drug["input"])


def test_attach_clusters_checks_input(built):
    g, _ = built
    add_knn_edges(g, k=3)
    inputs = cluster_inputs(g, detect_communities(g, seed=42))
    first, second = inputs[0], inputs[1]
    summaries = {
        str(first["cluster_id"]): {"input": first["input"], "summary": "Nhóm hành vi phạm tội: ma túy"},
        str(second["cluster_id"]): {"input": "đầu vào cũ", "summary": "x"},
    }
    report = attach_clusters(g, inputs, summaries, fake_embed)
    assert report["gan"] == [str(first["cluster_id"])]
    assert report["tom_tat_cu"] == [str(second["cluster_id"])]
    cid = f"cluster:{first['cluster_id']}"
    assert g.node(cid)["summary"] == "Nhóm hành vi phạm tội: ma túy"
    assert set(g.predecessors(cid, "BELONGS_TO")) == set(first["case_ids"])
    assert g.search(fake_embed("ma túy"), "Cluster", top_k=1)[0][0] == cid


def test_save_load_roundtrip(built, tmp_path):
    g, _ = built
    path = tmp_path / "g.pkl"
    g.save(str(path))
    g2 = HierarGraph.load(str(path))
    assert g2.stats() == g.stats()
    assert g2.search(fake_embed("trộm cắp"), "Case", 1) == g.search(fake_embed("trộm cắp"), "Case", 1)


def test_edge_to_missing_node_rejected():
    g = HierarGraph()
    g.add_node("law:1", "Law")
    with pytest.raises(KeyError):
        g.add_edge("law:1", "crime:1", "RELATED_CRIME")
