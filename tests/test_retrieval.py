"""Tests for vn_legal_graph.retrieval (search + metrics) on a small graph."""
import numpy as np
import pytest

from vn_legal_graph.graph.build import attach_clusters, build_base_graph, cluster_inputs, detect_communities, add_knn_edges
from vn_legal_graph.retrieval.metrics import evaluate, hit_at_k, recall_at_k
from vn_legal_graph.retrieval.search import (
    cluster_cases,
    direct_cases,
    direct_laws,
    frequency_prior,
    laws_from_cases,
)
from tests.test_graph import CASES, LAWS, fake_embed


@pytest.fixture()
def g():
    graph, _ = build_base_graph(LAWS, CASES, fake_embed)
    add_knn_edges(graph, 3)
    inputs = cluster_inputs(graph, detect_communities(graph, seed=42))
    summaries = {
        str(ci["cluster_id"]): {"input": ci["input"], "summary": " ".join(ci["top_crimes"]).lower()}
        for ci in inputs
    }
    attach_clusters(graph, inputs, summaries, fake_embed)
    return graph


def test_direct_route_finds_drug_law(g):
    cases = direct_cases(g, fake_embed("ma túy"), top_k=3)
    assert laws_from_cases(g, cases) == [249]


def test_cluster_route_finds_theft_law(g):
    cases = cluster_cases(g, fake_embed("trộm cắp"), n_clusters=1, top_k=3)
    assert cases and laws_from_cases(g, cases)[0] == 173


def test_laws_in_case_rank_order_without_duplicates(g):
    ranked = [("case:c10", 0.9), ("case:c0", 0.8), ("case:c11", 0.7)]
    assert laws_from_cases(g, ranked) == [173, 249]


def test_direct_laws_and_prior(g):
    assert direct_laws(g, fake_embed("ma túy"), top_k=1) == [249]
    # 173 and 249 both have 5 linked cases (c91/c92 are skipped); ties go by number.
    assert frequency_prior(g) == [173, 249]


def test_recall_and_hit():
    assert recall_at_k([249, 251], [249, 173, 251], 1) == 0.5
    assert recall_at_k([249, 251], [249, 173, 251], 3) == 1.0
    assert hit_at_k([251], [249, 173], 2) == 0.0
    assert hit_at_k([251], [249, 251], 2) == 1.0


def test_evaluate_aggregates():
    rows = [{"gold": [249], "pred": [249, 251]}, {"gold": [249, 251], "pred": [173, 251]}]
    res = evaluate(rows, ks=(1, 2))
    assert res["R@1"] == pytest.approx((1 + 0) / 2)
    assert res["R@2"] == pytest.approx((1 + 0.5) / 2)
    assert res["Hit@2"] == 1.0
    assert res["theo_dieu"][249]["R@1"] == pytest.approx(0.5)
    assert res["theo_dieu"][251] == {"R@1": 0.0, "R@2": 1.0, "n": 1}


def test_baselines_ignore_other_codes_with_the_same_numbers():
    from tests.test_graph import LAWS, fake_embed
    from vn_legal_graph.graph.build import build_base_graph
    from vn_legal_graph.retrieval.search import direct_laws, frequency_prior

    other = {"id": 249, "suffix": "", "bo_luat": "BLTTHS", "items": [{"text": "Điều 249. ma túy thủ tục", "crime": [], "judge_dep": [], "related_laws": []}]}
    g, _ = build_base_graph(LAWS + [other], [], fake_embed)
    assert len([e for e in frequency_prior(g) if e == 249]) == 1
    assert all(g.node(n)["bo_luat"] == "BLHS" for n in g.nodes_of("Law") if g.node(n)["entry"] in direct_laws(g, fake_embed("ma túy"), top_k=5) and n.count(":") == 1)
