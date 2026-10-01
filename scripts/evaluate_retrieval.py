#!/usr/bin/env python3
"""Phase 4, step 1: retrieval baseline without LLM steps.

For each held-out test case, retrieve candidate BLHS articles from the
graph and compare with the articles it was convicted under (Recall@k).
No LLM calls, no .env.

    python scripts/evaluate_retrieval.py

Queries use the test cases' LLM features (extracted WITHOUT the crime
hint, as the original's query path), embedded with the same model and
format as Case nodes. "truc_tiep_dien_bien_tho" embeds the raw facts
instead, to see what the feature step buys.

Writes outputs/retrieval_eval.json (metrics + per-case predictions).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.config import EmbeddingConfig
from vn_legal_graph.embedding import CachedEmbedder, embedder_from_config
from vn_legal_graph.graph.graph_db import HierarGraph
from vn_legal_graph.retrieval.metrics import evaluate
from vn_legal_graph.retrieval.search import (
    cluster_cases,
    direct_cases,
    direct_laws,
    frequency_prior,
    laws_from_cases,
)

KS = (1, 2, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--test", default="data/processed/cases_vn_test_features.json")
    parser.add_argument("--output", default="outputs/retrieval_eval.json")
    parser.add_argument("--top-k-cases", type=int, default=5)
    parser.add_argument("--n-clusters", type=int, default=2)
    args = parser.parse_args()

    g = HierarGraph.load(args.graph)
    with open(args.test, encoding="utf-8") as f:
        tests = json.load(f)
    in_graph = {g.node(n)["caseId"] for n in g.nodes_of("Case")}
    leaked = [t["id"] for t in tests if t["id"] in in_graph]
    if leaked:
        raise SystemExit(f"Test cases found in the graph: {leaked}")

    cfg = EmbeddingConfig()
    embedder = CachedEmbedder(embedder_from_config(cfg), cfg.model_name)
    prior = frequency_prior(g)

    methods = {
        "tan_suat": lambda t, q, raw: prior,
        "truc_tiep": lambda t, q, raw: laws_from_cases(g, direct_cases(g, q, args.top_k_cases)),
        "qua_cum": lambda t, q, raw: laws_from_cases(g, cluster_cases(g, q, args.n_clusters, args.top_k_cases)),
        "so_voi_dieu_luat": lambda t, q, raw: direct_laws(g, q),
        "truc_tiep_dien_bien_tho": lambda t, q, raw: laws_from_cases(g, direct_cases(g, raw, args.top_k_cases)),
    }
    rows = {m: [] for m in methods}
    for t in tests:
        q = embedder.encode_long_text(t["mo_ta_dac_trung"])
        raw = embedder.encode_long_text(t["dien_bien"])
        for name, fn in methods.items():
            rows[name].append({"id": t["id"], "gold": t["dieu"], "pred": fn(t, q, raw)})

    results = {name: evaluate(r, KS) for name, r in rows.items()}
    articles = sorted({a for t in tests for a in t["dieu"]})
    print(f"{len(tests)} án test | gold theo điều: "
          + ", ".join(f"{a}: {results['tan_suat']['theo_dieu'][a]['n']}" for a in articles))
    print(f"\n{'cách':26s}" + "".join(f"{'R@'+str(k):>7s}" for k in KS) + "".join(f"{'Hit@'+str(k):>7s}" for k in KS)
          + "   R@1 theo điều " + " ".join(f"{a:>5d}" for a in articles))
    for name, res in results.items():
        print(f"{name:26s}" + "".join(f"{res['R@'+str(k)]:7.2f}" for k in KS)
              + "".join(f"{res['Hit@'+str(k)]:7.2f}" for k in KS)
              + "                  " + " ".join(f"{res['theo_dieu'][a]['R@1']:5.2f}" for a in articles))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "predictions": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã ghi {args.output}")


if __name__ == "__main__":
    main()
