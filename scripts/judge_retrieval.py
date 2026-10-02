#!/usr/bin/env python3
"""Phase 4, step 2: judge_law on retrieved articles, then re-score.

Run by the user (needs the API key). Claude only runs --dry-run, which
never loads .env.

    python scripts/judge_retrieval.py --dry-run                 # calls/tokens for both modes
    python scripts/judge_retrieval.py --mode gop --per-crime 2  # trial
    python scripts/judge_retrieval.py --mode gop                # all 40 test cases

Takes the top --candidates articles of a retrieval method from
outputs/retrieval_eval.json (scripts/evaluate_retrieval.py), runs judge_law
on each against the test case's facts, puts accepted articles first and
reports Recall@k before and after. Needs judge_dep in
data/processed/law_to_crime_vn.json (scripts/build_law_layer.py without
--skip-judge-dep).
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.judge.judge_law import (
    MODES,
    batch_prompt,
    element_prompts,
    final_prompt,
    judge_law,
    rerank_by_judgment,
)
from vn_legal_graph.retrieval.metrics import evaluate

KS = (1, 2, 3)
CHARS_PER_TOKEN = 3.5


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="gop")
    parser.add_argument("--method", default="truc_tiep", help="Retrieval method whose ranking is judged.")
    parser.add_argument("--candidates", type=int, default=3, help="Top-N retrieved articles to judge per case.")
    parser.add_argument("--retrieval", default="outputs/retrieval_eval.json")
    parser.add_argument("--laws", default="data/processed/law_to_crime_vn.json")
    parser.add_argument("--test", default="data/processed/cases_vn_test_features.json")
    parser.add_argument("--per-crime", type=int, default=None)
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    laws = {
        l["id"]: {
            "entry": l["id"],
            "description": l["items"][0]["text"],
            "judge_dep": l["items"][0].get("judge_dep", []),
            "related_laws": l["items"][0].get("related_laws", []),
        }
        for l in load(args.laws)
    }
    tests = {t["id"]: t for t in load(args.test)}
    preds = [p for p in load(args.retrieval)["predictions"][args.method]]
    if args.per_crime:
        seen = collections.Counter()
        picked = []
        for p in preds:
            key = tuple(p["gold"])
            if seen[key] < args.per_crime:
                picked.append(p)
                seen[key] += 1
        preds = picked

    jobs = [(p, a) for p in preds for a in p["pred"][: args.candidates]]
    missing = sorted({a for _, a in jobs if not laws[a]["judge_dep"]})
    if missing:
        print(f"Cảnh báo: các điều chưa có judge_dep: {missing} (chạy build_law_layer.py không có --skip-judge-dep)")

    if args.dry_run:
        for mode in MODES:
            calls = chars = 0
            for p, a in jobs:
                text = tests[p["id"]]["dien_bien"]
                law = laws[a]
                if mode == "trung-thanh":
                    ps = element_prompts(text, law)
                else:
                    ps = [batch_prompt(text, law)] if law["judge_dep"] else []
                ps.append(final_prompt(text, law, law["judge_dep"][:5], []))
                calls += len(ps)
                chars += sum(len(x) for x in ps)
            print(f"{mode:12s}: {len(preds)} án × ≤{args.candidates} điều = {len(jobs)} lần xét, "
                  f"{calls:,} lần gọi LLM, ~{chars / CHARS_PER_TOKEN:,.0f} token đầu vào")
        return

    from vn_legal_graph.config import AppConfig
    from vn_legal_graph.llm import LLMClient

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    decisions = {}
    for i, (p, a) in enumerate(jobs, 1):
        result = judge_law(client.generate, tests[p["id"]]["dien_bien"], laws[a], args.mode)
        decisions.setdefault(p["id"], {})[a] = result
        print(f"\r{i}/{len(jobs)} lần xét", end="", flush=True)
    print()

    before = [{"id": p["id"], "gold": p["gold"], "pred": p["pred"]} for p in preds]
    after = [
        {
            "id": p["id"],
            "gold": p["gold"],
            "pred": rerank_by_judgment(p["pred"], {a: d["ap_dung"] for a, d in decisions[p["id"]].items()}),
        }
        for p in preds
    ]
    res_before, res_after = evaluate(before, KS), evaluate(after, KS)
    articles = sorted({a for p in preds for a in p["gold"]})
    print(f"\n{'':10s}" + "".join(f"{'R@'+str(k):>7s}" for k in KS) + f"{'Hit@1':>7s}   R@1 theo điều " + " ".join(f"{a:>5d}" for a in articles))
    for name, res in (("trước", res_before), ("sau", res_after)):
        print(f"{name:10s}" + "".join(f"{res['R@'+str(k)]:7.2f}" for k in KS) + f"{res['Hit@1']:7.2f}"
              + "                  " + " ".join(f"{res['theo_dieu'][a]['R@1']:5.2f}" for a in articles))

    # How judge_law decided on gold vs non-gold candidates.
    tally = collections.Counter()
    for p in preds:
        for a, d in decisions[p["id"]].items():
            tally[("đúng" if a in p["gold"] else "sai", "chấp nhận" if d["ap_dung"] else "bác")] += 1
    print("\nĐiều ứng viên (đúng/sai so với nhãn) × quyết định judge_law:", dict(tally))
    unreadable = sum(d["loi_doc_danh_sach"] for ds in decisions.values() for d in ds.values())
    unknown = sum(len(d["khong_ro"]) for ds in decisions.values() for d in ds.values())
    print(f"Không đọc được danh sách: {unreadable} | yếu tố không rõ: {unknown}")

    out = f"outputs/judge_law_{args.mode}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {"mode": args.mode, "method": args.method, "candidates": args.candidates,
             "metrics": {"truoc": res_before, "sau": res_after},
             "decisions": decisions, "predictions_sau": after},
            f, ensure_ascii=False, indent=2,
        )
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
