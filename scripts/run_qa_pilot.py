#!/usr/bin/env python3
"""Legal Q&A pilot: answer the first N questions of data/raw/questions.xlsx
with the HierarGraph and score them against data/qa/pilot_gold.json.

    python scripts/run_qa_pilot.py --dry-run     # what will run, rough cost; no LLM, no .env
    python scripts/run_qa_pilot.py --n 5

Per question: features, candidate articles (4 routes), judge_law on each
candidate, a short answer with structured crimes/articles, then scoring:
articles (and khoản/điểm) and crimes against the gold labels, the
conclusion graded by the LLM against the lawyer's answer, and length.
The system only ever sees title + question; the lawyer's answer is used
for grading only.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.config import EmbeddingConfig
from vn_legal_graph.embedding import CachedEmbedder, embedder_from_config
from vn_legal_graph.graph.graph_db import HierarGraph
from vn_legal_graph.prompts.vi import QA_GRADE_PROMPT
from vn_legal_graph.qa.pipeline import DEFAULTS, answer_question
from vn_legal_graph.qa.scoring import load_questions, parse_grade, question_text, references_text, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="data/raw/questions.xlsx")
    parser.add_argument("--gold", default="data/qa/pilot_gold.json")
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--judge-mode", default=DEFAULTS["judge_mode"], choices=["gop", "trung-thanh"])
    parser.add_argument("--max-candidates", type=int, default=DEFAULTS["max_candidates"])
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--tag", default="", help="Suffix for the output file, e.g. lan2.")
    parser.add_argument("--loc-theo-judge", action="store_true", help="Original hard filter: answer only from accepted articles.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gold = {g["qa_number"]: g for g in json.load(open(args.gold, encoding="utf-8"))["cau_hoi"]}
    questions = load_questions(args.questions)[: args.n]
    missing = [q["qa_number"] for q in questions if str(q["qa_number"]) not in gold]
    if missing:
        raise SystemExit(f"Thiếu đáp án chuẩn cho: {missing}")

    g = HierarGraph.load(args.graph)
    stats = g.stats()
    print(f"Graph: {stats['nodes']}")
    if args.dry_run:
        per_q = 1 + 1 + 2 * args.max_candidates + 1 + 1
        print(f"{len(questions)} câu × tối đa {per_q} lần gọi LLM "
              f"(đặc trưng, đoán tội, judge ≤{args.max_candidates} điều × 2, trả lời, chấm) "
              f"= tối đa {len(questions) * per_q} lần gọi, ước ~{len(questions) * 60_000:,} token đầu vào")
        for q in questions:
            print(f"  #{q['qa_number']}: {q['title']}")
        return

    from vn_legal_graph.config import AppConfig
    from vn_legal_graph.llm import LLMClient

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    print(f"LLM: {client.llm_config.provider} / {client.llm_config.model}")
    cfg = EmbeddingConfig()
    embedder = CachedEmbedder(embedder_from_config(cfg), cfg.model_name)
    run_cfg = {"judge_mode": args.judge_mode, "max_candidates": args.max_candidates,
               "loc_theo_judge": args.loc_theo_judge}

    results = []
    for i, q in enumerate(questions, 1):
        qid = str(q["qa_number"])
        print(f"\n[{i}/{len(questions)}] #{qid} {q['title']}")
        trace = answer_question(g, question_text(q), client.generate, embedder.encode_long_text, run_cfg)
        pred = trace["tra_loi"]
        s = score(pred, gold[qid])
        grade = parse_grade(client.generate(
            QA_GRADE_PROMPT.format(question=question_text(q), reference=references_text(q["answer"]),
                                   answer=pred.get("cau_tra_loi", "")),
            max_tokens=256,
        ))
        results.append({"qa_number": qid, "title": q["title"], "diem_tu_dong": s, "cham_ket_luan": grade,
                        "dap_an_chuan": gold[qid], **trace})
        print(f"  ứng viên: {trace['truy_xuat']['ung_vien']}")
        print(f"  dùng để trả lời: {trace['dieu_dung_de_tra_loi']}")
        print(f"  trả lời: {pred.get('cau_tra_loi', '')}")
        print(f"  trích: {[(c.get('luat'), c.get('dieu'), c.get('khoan'), c.get('diem')) for c in pred.get('dieu_luat', [])]}")
        print(f"  điểm: điều R={s['dieu_recall']:.2f} P={s['dieu_precision']:.2f} | khoản/điểm={s['khoan_diem']} | "
              f"tội R={s['toi_danh_recall']:.2f} P={s['toi_danh_precision']:.2f} | kết luận={grade and grade['diem']} | {s['so_tu']} từ")

    model_slug = re.sub(r"[^\w.-]", "_", client.llm_config.model)
    out = f"outputs/qa_pilot_{model_slug}{'_' + args.tag if args.tag else ''}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    n = len(results)
    mean = lambda k: sum(r["diem_tu_dong"][k] for r in results) / n
    kd = [r["diem_tu_dong"]["khoan_diem"] for r in results if r["diem_tu_dong"]["khoan_diem"] is not None]
    grades = [r["cham_ket_luan"]["diem"] if r["cham_ket_luan"] else "khong_doc_duoc" for r in results]
    print(f"\n=== Tổng {n} câu ===")
    print(f"Điều: recall {mean('dieu_recall'):.2f}, precision {mean('dieu_precision'):.2f} | "
          f"khoản/điểm {sum(kd) / len(kd):.2f} ({len(kd)} câu có khoản) | "
          f"tội danh: recall {mean('toi_danh_recall'):.2f}, precision {mean('toi_danh_precision'):.2f} | "
          f"kết luận: {dict((x, grades.count(x)) for x in set(grades))} | độ dài TB {mean('so_tu'):.0f} từ")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
