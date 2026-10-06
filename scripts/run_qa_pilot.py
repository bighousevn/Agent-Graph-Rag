#!/usr/bin/env python3
"""Legal Q&A pilot: answer the first N questions of data/raw/questions.xlsx
with the HierarGraph and score them against data/qa/pilot_gold.json.

    python scripts/run_qa_pilot.py --dry-run     # what will run, rough cost; no LLM, no .env
    python scripts/run_qa_pilot.py --n 5
    python scripts/run_qa_pilot.py --auto-gold --n 40 --tag tudong40

--auto-gold: reference articles come from the lawyer's answer
(vn_legal_graph/qa/auto_gold.py), not data/qa/pilot_gold.json. Only
questions whose answer cites BLHS/BLTTHS are used: every one that repeats
a TANDTC letter in the graph ("co_cong_van"), then the others in file order
up to --n. Results are reported per group.

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
from vn_legal_graph.qa.auto_gold import build_gold, khoan_precision
from vn_legal_graph.qa.pipeline import DEFAULTS, answer_question
from vn_legal_graph.qa.scoring import load_questions, parse_grade, question_text, references_text, score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="data/raw/questions.xlsx")
    parser.add_argument("--gold", default="data/qa/pilot_gold.json")
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--backend", default="pkl", choices=["pkl", "neo4j"],
                        help="pkl: load --graph into memory; neo4j: query the database (scripts/export_neo4j.py).")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--judge-mode", default=DEFAULTS["judge_mode"], choices=["gop", "trung-thanh"])
    parser.add_argument("--max-candidates", type=int, default=DEFAULTS["max_candidates"])
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--tag", default="", help="Suffix for the output file, e.g. lan2.")
    parser.add_argument("--loc-theo-judge", action="store_true", help="Original hard filter: answer only from accepted articles.")
    parser.add_argument("--rerank", action="store_true", help="Original LLM rerank of clusters and cases (2 more calls per question).")
    parser.add_argument("--auto-gold", action="store_true", help="Reference articles from the lawyer's answer.")
    parser.add_argument("--guidance-links", default="data/raw/guidance/guidance_links.json")
    parser.add_argument("--cong-van-tay", default="data/qa/cong_van_tay.json")
    parser.add_argument("--ids-from", default="", help="Re-run the questions of an earlier result file, e.g. outputs/qa_pilot_deepseek-flash_tudong40.json.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.backend == "neo4j":
        from vn_legal_graph.graph.neo4j_store import Neo4jGraph, connect

        g = Neo4jGraph(connect())
    else:
        g = HierarGraph.load(args.graph)
    stats = g.stats()
    print(f"Graph: {stats['nodes']}")
    if args.auto_gold:
        questions, gold = auto_gold_questions(g, args)
    else:
        gold = {g_["qa_number"]: g_ for g_ in json.load(open(args.gold, encoding="utf-8"))["cau_hoi"]}
        questions = load_questions(args.questions)[: args.n]
    missing = [q["qa_number"] for q in questions if str(q["qa_number"]) not in gold]
    if missing:
        raise SystemExit(f"Thiếu đáp án chuẩn cho: {missing}")
    if args.dry_run:
        per_q = 1 + 1 + 2 * args.max_candidates + 1 + 1 + (2 if args.rerank else 0)
        print(f"{len(questions)} câu × tối đa {per_q} lần gọi LLM "
              f"(đặc trưng, đoán tội, judge ≤{args.max_candidates} điều × 2, trả lời, chấm) "
              f"= tối đa {len(questions) * per_q} lần gọi, ước ~{len(questions) * 60_000:,} token đầu vào")
        for q in questions:
            print(f"  #{q['qa_number']} [{gold[str(q['qa_number'])].get('nhom', '')}]: {q['title']}")
        return

    from vn_legal_graph.config import AppConfig
    from vn_legal_graph.llm import LLMClient

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    print(f"LLM: {client.llm_config.provider} / {client.llm_config.model}")
    cfg = EmbeddingConfig()
    embedder = CachedEmbedder(embedder_from_config(cfg), cfg.model_name)
    run_cfg = {"judge_mode": args.judge_mode, "max_candidates": args.max_candidates,
               "loc_theo_judge": args.loc_theo_judge, "rerank": args.rerank}

    results = []
    for i, q in enumerate(questions, 1):
        qid = str(q["qa_number"])
        print(f"\n[{i}/{len(questions)}] #{qid} {q['title']}")
        trace = answer_question(g, question_text(q), client.generate, embedder.encode_long_text, run_cfg)
        pred = trace["tra_loi"]
        s = score(pred, gold[qid])
        if args.auto_gold:
            s["khoan_diem"] = khoan_precision(pred, gold[qid])
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

    report(results, "Tổng", args.auto_gold)
    if args.auto_gold:
        for nhom in ("co_cong_van", "khong_cong_van"):
            report([r for r in results if r["dap_an_chuan"]["nhom"] == nhom], nhom, True)
        report([r for r in results if r["dap_an_chuan"].get("nghi_quyet_trich")], "co_nghi_quyet", True)
        if args.ids_from:
            compare(json.load(open(args.ids_from, encoding="utf-8")), results)
    print(f"Đã ghi {out}")


def report(results, label, auto) -> None:
    n = len(results)
    if not n:
        return
    mean = lambda k: sum(r["diem_tu_dong"][k] for r in results) / n
    kd = [r["diem_tu_dong"]["khoan_diem"] for r in results if r["diem_tu_dong"]["khoan_diem"] is not None]
    grades = [r["cham_ket_luan"]["diem"] if r["cham_ket_luan"] else "khong_doc_duoc" for r in results]
    crimes = (f"tội danh: precision {mean('toi_danh_precision'):.2f}" if auto else
              f"tội danh: recall {mean('toi_danh_recall'):.2f}, precision {mean('toi_danh_precision'):.2f}")
    print(f"\n=== {label}: {n} câu ===")
    print(f"Điều: recall {mean('dieu_recall'):.2f}, precision {mean('dieu_precision'):.2f} | "
          f"khoản/điểm {sum(kd) / len(kd) if kd else float('nan'):.2f} ({len(kd)} câu) | {crimes} | "
          f"kết luận: {dict((x, grades.count(x)) for x in sorted(set(grades)))} | độ dài TB {mean('so_tu'):.0f} từ")


def compare(before, after) -> None:
    """Per-question change of the graded conclusion against an earlier run."""
    grade = lambda r: r["cham_ket_luan"]["diem"] if r.get("cham_ket_luan") else "khong_doc_duoc"
    old = {r["qa_number"]: grade(r) for r in before}
    rank = {"sai": 0, "khong_doc_duoc": 0, "mot_phan": 1, "dung": 2}
    better = [(r["qa_number"], old[r["qa_number"]], grade(r)) for r in after if rank[grade(r)] > rank[old[r["qa_number"]]]]
    worse = [(r["qa_number"], old[r["qa_number"]], grade(r)) for r in after if rank[grade(r)] < rank[old[r["qa_number"]]]]
    print(f"\n=== So với lần trước: {len(better)} câu tốt hơn, {len(worse)} câu kém hơn ===")
    print(f"tốt hơn: {better}")
    print(f"kém hơn: {worse}")


def auto_gold_questions(g, args):
    links = json.load(open(args.guidance_links, encoding="utf-8"))
    items = [it for it in links if it["from"].startswith("Công văn")]  # group = Công văn only
    manual = {k: v for k, v in json.load(open(args.cong_van_tay, encoding="utf-8")).items() if not k.startswith("_")}
    letters = {re.match(r"Công văn (\d+)/", it["from"]).group(1) for it in items}
    resolutions = {m.group(1) for it in links for m in [re.search(r"(\d+/\d{4}/NQ-HĐTP|\d+/VBHN-TANDTC)", it["from"])] if m}
    crimes_of = {}
    for node in g.nodes_of("Law"):
        d = g.node(node)
        key = f"{d.get('bo_luat', 'BLHS')}:{d['entry']}{d.get('suffix', '') or ''}"
        crimes_of[key] = [g.node(c)["description"] for c in g.neighbors(node, "RELATED_CRIME")]
    pool = []
    for q in load_questions(args.questions):
        gold = build_gold(q, items, letters, crimes_of, manual, resolutions)
        if gold["dieu_luat_bat_buoc"]:
            pool.append((q, gold))
    if args.ids_from:
        ids = [r["qa_number"] for r in json.load(open(args.ids_from, encoding="utf-8"))]
        chosen = [p for p in pool if p[1]["qa_number"] in ids]
        missing = set(ids) - {p[1]["qa_number"] for p in chosen}
        if missing:
            raise SystemExit(f"Không còn trong phạm vi: {sorted(missing)}")
    else:
        chosen = [p for p in pool if p[1]["nhom"] == "co_cong_van"]
        chosen += [p for p in pool if p[1]["nhom"] == "khong_cong_van"][: max(0, args.n - len(chosen))]
    order = {id(q): i for i, (q, _) in enumerate(pool)}
    chosen.sort(key=lambda p: order[id(p[0])])
    print(f"Đáp án tự động: {len(pool)} câu có trích BLHS/BLTTHS; chọn {len(chosen)} "
          f"({sum(p[1]['nhom'] == 'co_cong_van' for p in chosen)} có Công văn, "
          f"{sum(bool(p[1]['nghi_quyet_trich']) for p in chosen)} trích Nghị quyết HĐTP có trong graph)")
    return [q for q, _ in chosen], {gold["qa_number"]: gold for _, gold in chosen}


if __name__ == "__main__":
    main()
