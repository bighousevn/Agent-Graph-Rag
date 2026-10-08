#!/usr/bin/env python3
"""The original judgment pipeline on the held-out test cases:
per defendant Researcher -> Auditor -> Adjudicator
(vn_legal_graph/judge/case_pipeline.py), scored like the paper
(evaluation/evaluate_results.py): charge and article accuracy / micro-F1,
per judgment (our labels are per judgment).

    python scripts/run_case_pipeline.py --dry-run
    python scripts/run_case_pipeline.py --per-crime 2 --tag thu
    python scripts/run_case_pipeline.py                      # all 55 test cases
    python scripts/run_case_pipeline.py --without-graph      # baseline: Adjudicator alone
    python scripts/run_case_pipeline.py --adjudicator-context an+huong-dan --tag ctx

Writes outputs/case_pipeline_<model>[_tag].json.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.graph.graph_db import HierarGraph
from vn_legal_graph.shard import shard_suffix, take_shard
from vn_legal_graph.judge.case_pipeline import DEFAULTS, adjudicate_without_graph, analyze_case, score_case, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="data/processed/cases_vn_test.json")
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--backend", default="pkl", choices=["pkl", "neo4j"])
    parser.add_argument("--per-crime", type=int, default=None, help="Only N test cases per crime (trial).")
    parser.add_argument("--without-graph", action="store_true", help="Baseline: Adjudicator without retrieval.")
    parser.add_argument("--segment", action="store_true", help="With --without-graph: per defendant, as the pipeline.")
    parser.add_argument("--auditor", default="loc", choices=["loc", "goi-y", "bo-qua"],
                        help="loc: original hard filter; goi-y: all candidates, Auditor verdict as a hint; bo-qua: all candidates, no verdicts.")
    parser.add_argument("--adjudicator-context", default="", choices=["", "an", "an+huong-dan"],
                        help="Extension: also show the Adjudicator the retrieved cases (an), and the guidance of the shown articles (an+huong-dan).")
    parser.add_argument("--max-defendants", type=int, default=DEFAULTS["max_defendants"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--shard", default="", help="K/N: only every N-th case from the K-th (several notebooks in parallel).")
    parser.add_argument("--merge", nargs="+", default=[], help="Join the result files of the shards into --merge-out, rescored.")
    parser.add_argument("--merge-out", default="")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.merge:
        return merge_files(args)
    cases = json.load(open(args.test, encoding="utf-8"))
    if args.per_crime:
        taken, kept = collections.Counter(), []
        for c in cases:
            key = str(c["dieu"][0])
            if taken[key] < args.per_crime:
                taken[key] += 1
                kept.append(c)
        cases = kept
    if args.shard:
        cases = take_shard(cases, args.shard)
    print(f"{len(cases)} án test: {dict(collections.Counter(str(c['dieu'][0]) for c in cases))}")

    if args.dry_run:
        if args.without_graph:
            print(f"1 lời gọi/án = {len(cases)} lời gọi")
            return
        per_def = 1 + 2 + 1 + 2 * 6 + 1  # features, 2 rerank, crimes, judge <=6 articles x2, adjudicate
        est = len(cases) * (1 + 1.6 * (1 + per_def))
        if args.adjudicator_context:
            print(f"Adjudicator thêm ngữ cảnh '{args.adjudicator_context}': chỉ ~{len(cases) * 1.6:.0f} lời gọi Adjudicator là mới "
                  f"(các bước trước dùng lại cache), mỗi lời gọi thêm ~{(2400 + (2000 if 'huong' in args.adjudicator_context else 0)) // 3:,} token")
        print(f"~{est:.0f} lời gọi LLM (giả sử 1,6 bị cáo/án, ≤6 điều ứng viên), ~{est * 3000:,.0f} token đầu vào")
        return

    from vn_legal_graph.config import AppConfig, EmbeddingConfig
    from vn_legal_graph.embedding import CachedEmbedder, embedder_from_config
    from vn_legal_graph.llm import LLMClient

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    print(f"LLM: {client.llm_config.provider} / {client.llm_config.model}")
    if args.backend == "neo4j":
        from vn_legal_graph.graph.neo4j_store import Neo4jGraph, connect

        g = Neo4jGraph(connect())
    else:
        g = HierarGraph.load(args.graph)
    cfg_e = EmbeddingConfig()
    embedder = CachedEmbedder(embedder_from_config(cfg_e), cfg_e.model_name)
    crime_articles = {f"{g.node(n)['entry']}{g.node(n).get('suffix', '') or ''}" for n in g.nodes_of("Law")
                      if g.node(n).get("bo_luat", "BLHS") == "BLHS" and g.neighbors(n, "RELATED_CRIME")}

    model_slug = re.sub(r"[^\w.-]", "_", client.llm_config.model)
    name = ("case_adjudicator_only" + ("_segment" if args.segment else "")) if args.without_graph else "case_pipeline"
    path = f"outputs/{name}_{model_slug}{'_' + args.tag if args.tag else ''}{shard_suffix(args.shard)}.json"
    results = []
    for i, c in enumerate(cases, 1):
        if args.without_graph:
            out = adjudicate_without_graph(client.generate, c["dien_bien"][: DEFAULTS["fact_chars"]], args.segment, args.max_defendants)
        else:
            out = analyze_case(g, c["dien_bien"], client.generate, embedder.encode_long_text, {"max_defendants": args.max_defendants, "auditor": args.auditor,
                                                                                         "adjudicator_context": args.adjudicator_context})
        s = score_case(out, c["dieu"], c["toi_danh"], crime_articles)
        results.append({"id": c["id"], "dieu": c["dieu"], "toi_danh": c["toi_danh"], "diem": s, **out})
        print(f"[{i}/{len(cases)}] {c['id']} gold {c['dieu']} -> {out['du_doan_dieu']} "
              f"{'✓' if s['dieu_dung_het'] else '✗'} | bị cáo: {out.get('bi_cao', '-')}", flush=True)
        write_json(path, {"metrics": summarize([r["diem"] for r in results]), "chua_xong": i < len(cases), "cases": results})

    write_report(path, results)
    print(f"Đã ghi {path}")


def write_report(path, results) -> None:
    metrics = summarize([r["diem"] for r in results])
    by_crime = {}
    for key in sorted({str(r["dieu"][0]) for r in results}, key=lambda x: int(re.match(r"\d+", x).group())):
        by_crime[key] = summarize([r["diem"] for r in results if str(r["dieu"][0]) == key])
    write_json(path, {"metrics": metrics, "theo_toi": by_crime, "cases": results})
    print(f"\nTổng {metrics['so_an']} án | tội danh: acc {metrics['toi_danh_accuracy']:.2f}, micro-F1 {metrics['toi_danh_micro_f1']:.2f} "
          f"| điều luật: acc {metrics['dieu_accuracy']:.2f}, micro-F1 {metrics['dieu_micro_f1']:.2f}")
    for k, m in by_crime.items():
        print(f"  {k:>4}: {m['so_an']} án | tội acc {m['toi_danh_accuracy']:.2f} | điều acc {m['dieu_accuracy']:.2f}")


def merge_files(args) -> None:
    if not args.merge_out:
        raise SystemExit("--merge cần --merge-out")
    results, seen = [], set()
    for p in args.merge:
        data = json.load(open(p, encoding="utf-8"))
        if data.get("chua_xong"):
            print(f"Cảnh báo: {p} chưa chạy xong")
        for r in data["cases"]:
            if r["id"] in seen:
                raise SystemExit(f"Án {r['id']} có ở hai file")
            seen.add(r["id"])
            results.append(r)
    write_report(args.merge_out, results)
    print(f"Gộp {len(args.merge)} file -> {args.merge_out}")


def write_json(path, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
