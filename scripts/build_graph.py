#!/usr/bin/env python3
"""Phase 3.2: build the HierarGraph.

No LLM calls and no .env: embeddings are computed locally (the default
PhoBERT-based model downloads once, ~540 MB, and runs on CPU).

    python scripts/build_graph.py

Inputs (data/processed/):
    law_to_crime_vn.json      from scripts/build_law_layer.py
    cases_vn_features.json    from scripts/extract_case_features.py (user, LLM)
    cluster_summaries.json    from scripts/summarize_clusters.py (user, LLM), optional

Outputs:
    outputs/hierargraph.pkl           the graph
    data/processed/cluster_inputs.json  what summarize_clusters.py sends to the LLM

Clusters need summaries, so a first run builds everything except the
Cluster nodes. After running scripts/summarize_clusters.py, run this
again: embeddings come from .cache/emb, so it is fast, and the summaries
are attached.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.config import EmbeddingConfig
from vn_legal_graph.embedding import CachedEmbedder, embedder_from_config
from vn_legal_graph.graph.build import (
    add_knn_edges,
    attach_clusters,
    build_base_graph,
    cluster_inputs,
    detect_communities,
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="outputs/hierargraph.pkl")
    parser.add_argument("--embedding-backend", default=EmbeddingConfig.backend)
    parser.add_argument("--embedding-model", default=EmbeddingConfig.model_name)
    parser.add_argument("--embedding-api-url", default=EmbeddingConfig.api_url)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    law_to_crime = load(os.path.join(args.data_dir, "law_to_crime_vn.json"))
    cases = load(os.path.join(args.data_dir, "cases_vn_features.json"))
    summaries_path = os.path.join(args.data_dir, "cluster_summaries.json")
    summaries = load(summaries_path) if os.path.exists(summaries_path) else {}

    config = EmbeddingConfig(
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        api_url=args.embedding_api_url,
        device=args.device,
    )
    model_key = args.embedding_model if args.embedding_backend != "bge-m3" else "bge-m3"
    embedder = CachedEmbedder(embedder_from_config(config), model_key)

    print(f"[1/4] Node Law/Crime ({len(law_to_crime)} điều) và Case ({len(cases)} án) ...")
    g, report = build_base_graph(law_to_crime, cases, embedder.encode_long_text)
    for key, items in report.items():
        if items:
            print(f"    {key}: {len(items)} {items[:5]}")
    missing_dep = [law["id"] for law in law_to_crime if not law["items"][0].get("judge_dep")]
    if missing_dep:
        print(f"    Lưu ý: {len(missing_dep)} điều chưa có judge_dep (chạy build_law_layer.py không có --skip-judge-dep).")

    print(f"[2/4] kNN top-{args.knn} ...")
    print(f"    {add_knn_edges(g, args.knn)} cạnh SIMILAR_TO")

    print("[3/4] Louvain + PageRank ...")
    communities = detect_communities(g, seed=args.seed)
    sizes = sorted((len(v) for v in communities.values()), reverse=True)
    print(f"    {len(communities)} cụm có án, kích thước: {sizes}")
    inputs = cluster_inputs(g, communities)
    inputs_path = os.path.join(args.data_dir, "cluster_inputs.json")
    with open(inputs_path, "w", encoding="utf-8") as f:
        json.dump(inputs, f, ensure_ascii=False, indent=2)
    print(f"    Đã ghi {inputs_path}")

    print("[4/4] Node Cluster ...")
    if summaries:
        rep = attach_clusters(g, inputs, summaries, embedder.encode_long_text)
        print(f"    gắn: {len(rep['gan'])}, thiếu tóm tắt: {rep['thieu_tom_tat']}, tóm tắt cũ (đầu vào đã đổi): {rep['tom_tat_cu']}")
        if rep["thieu_tom_tat"] or rep["tom_tat_cu"]:
            print("    -> chạy lại scripts/summarize_clusters.py rồi chạy lại lệnh này.")
    else:
        print("    Chưa có cluster_summaries.json: graph chưa có node Cluster.")
        print("    -> chạy scripts/summarize_clusters.py (cần LLM) rồi chạy lại lệnh này.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    g.save(args.output)
    print(f"\nEmbedding cache: {embedder.hits} dùng lại, {embedder.misses} tính mới")
    print(f"Graph: {g.stats()}")
    print(f"Đã ghi {args.output}")


if __name__ == "__main__":
    main()
