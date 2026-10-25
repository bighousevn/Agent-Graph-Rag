#!/usr/bin/env python3
"""Answer every question of a CSV/XLSX file with the Q&A pipeline, without
reference answers (no scoring): same graph, settings and Qwen as
run_qa_pilot.py. The result file is written after every question, so a run
cut off (Kaggle 12 h) keeps what it did and a rerun is answered from cache.

    python scripts/answer_batch.py --input data/raw/fb/cau_hoi.csv --tag fb --dry-run
    python scripts/answer_batch.py --input data/raw/fb/cau_hoi.csv --tag fb --shard 1/2
    python scripts/answer_batch.py --merge outputs/answers_qwen3_8b-q8_0_fb_phan1-2.json \
        outputs/answers_qwen3_8b-q8_0_fb_phan2-2.json --merge-out outputs/answers_qwen3_8b-q8_0_fb.json

--merge also writes the Excel file (same name, .xlsx) with empty columns for
judging. Input columns: --id-col (default id), --text-col (default cau_hoi),
--title-col (optional, put before the question as in run_qa_pilot.py).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.qa.batch import question_of, read_table, write_excel
from vn_legal_graph.shard import shard_suffix, take_shard


def write_json(path, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def merge_files(paths, out) -> list:
    results, seen = [], set()
    for path in paths:
        for r in json.load(open(path, encoding="utf-8")):
            if r["id"] in seen:
                raise SystemExit(f"Câu {r['id']} có ở hai file")
            seen.add(r["id"])
            results.append(r)
    write_json(out, results)
    write_excel(os.path.splitext(out)[0] + ".xlsx", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="cau_hoi")
    parser.add_argument("--title-col", default="")
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N questions (a trial).")
    parser.add_argument("--shard", default="", help="K/N: every N-th question from the K-th (one per GPU).")
    parser.add_argument("--merge", nargs="+", default=[], help="Join shard result files into --merge-out (+ .xlsx).")
    parser.add_argument("--merge-out", default="")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.merge:
        if not args.merge_out:
            raise SystemExit("--merge cần --merge-out")
        results = merge_files(args.merge, args.merge_out)
        print(f"Gộp {len(args.merge)} file, {len(results)} câu -> {args.merge_out} và {os.path.splitext(args.merge_out)[0]}.xlsx")
        return
    if not args.input:
        raise SystemExit("Cần --input (file câu hỏi .csv hoặc .xlsx)")

    rows = read_table(args.input)
    if rows and (args.id_col not in rows[0] or args.text_col not in rows[0]):
        raise SystemExit(f"Thiếu cột {args.id_col!r} hoặc {args.text_col!r}; file có: {list(rows[0])}")
    if args.limit:
        rows = rows[: args.limit]
    if args.shard:
        rows = take_shard(rows, args.shard)
    print(f"{len(rows)} câu từ {args.input}{' (phần ' + args.shard + ')' if args.shard else ''}")
    if args.dry_run:
        from vn_legal_graph.qa.pipeline import DEFAULTS

        per_q = 1 + 1 + 1 + 2 * (DEFAULTS["max_candidates"] + DEFAULTS["general_slots"]) + 2
        print(f"Tối đa {per_q} lời gọi LLM mỗi câu (diễn đạt lại, đoán tội, vấn đề pháp lý, thẩm định ≤"
              f"{DEFAULTS['max_candidates']}+{DEFAULTS['general_slots']} điều × 2, trả lời + hỏi lại nếu ngắn) "
              f"= tối đa {len(rows) * per_q} lời gọi; không chấm điểm")
        for r in rows[:5]:
            print(f"  {r[args.id_col]}: {question_of(r, args.text_col, args.title_col)[:100]}")
        return

    from vn_legal_graph.config import AppConfig, EmbeddingConfig
    from vn_legal_graph.embedding import CachedEmbedder, embedder_from_config
    from vn_legal_graph.graph.graph_db import HierarGraph
    from vn_legal_graph.llm import LLMClient
    from vn_legal_graph.qa.pipeline import answer_question

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    print(f"LLM: {client.llm_config.provider} / {client.llm_config.model}")
    g = HierarGraph.load(args.graph)
    cfg = EmbeddingConfig()
    embed = CachedEmbedder(embedder_from_config(cfg), cfg.model_name).encode_long_text
    model_slug = re.sub(r"[^\w.-]", "_", client.llm_config.model)
    out = f"outputs/answers_{model_slug}{'_' + args.tag if args.tag else ''}{shard_suffix(args.shard)}.json"
    results = []
    for i, r in enumerate(rows, 1):
        question = question_of(r, args.text_col, args.title_col)
        trace = answer_question(g, question, client.generate, embed)
        results.append({"id": str(r[args.id_col]), "cau_hoi": question, **trace})
        write_json(out, results)  # after every question
        answer = trace["tra_loi"]
        print(f"[{i}/{len(rows)}] {r[args.id_col]} -> {[c.get('dieu') for c in answer.get('dieu_luat', []) if isinstance(c, dict)]}"
              f" | {len(answer.get('cau_tra_loi', '').split())} từ{' | LỖI ĐỊNH DẠNG' if trace['tra_loi_loi'] else ''}", flush=True)
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
