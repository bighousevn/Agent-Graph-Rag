#!/usr/bin/env python3
"""Statistics for the ViCSR dataset, restricted to the target chapters.

For every case: split the judgment into sections and label its crimes in
BLHS 2015 numbering (see ``vn_legal_graph.cases.vicsr.label_case``), then
compare with the numeric ground-truth labels, which mix BLHS 1999 and 2015
numbering.

    python scripts/vicsr_stats.py
    python scripts/vicsr_stats.py --data-dir data/raw/dataset_drive --chapters XVI XX

No LLM calls.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.cases.vicsr import (
    build_crime_name_index,
    label_case,
    load_ground_truth,
    load_laws,
    load_records,
)

CHAPTER_ARTICLES = {
    "XVI": range(168, 181),
    "XX": range(247, 260),
}

METHODS = ("ten_toi", "ten_toi_gan_dung", "dieu_luat", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/dataset_drive")
    parser.add_argument("--chapters", nargs="*", default=["XVI", "XX"])
    parser.add_argument("--output", default="data/processed/vicsr_stats.json")
    args = parser.parse_args()

    targets = [a for c in args.chapters for a in CHAPTER_ARTICLES[c.upper()]]
    laws = load_laws(os.path.join(args.data_dir, "law_shorten.txt"))
    gt = load_ground_truth(os.path.join(args.data_dir, "ground_truth.json"))
    details = load_records(os.path.join(args.data_dir, "case_detail.txt"))
    name_index = build_crime_name_index(laws)
    title = {int(l.dieu): l.title for l in laws.values() if l.is_crime}

    section_missing = collections.Counter()
    methods = collections.Counter()
    by_label = collections.Counter()
    by_label_method = collections.defaultdict(collections.Counter)
    by_gt = collections.Counter()
    agree = disagree = 0
    disagreement_pairs = collections.Counter()

    for case_id, detail in details.items():
        result = label_case(detail, laws, name_index)
        for k, v in result["sections"].items():
            if v is None:
                section_missing[k] += 1
        crimes, method = result["toi_danh"], result["cach_gan_nhan"]
        methods[method] += 1
        gt_crimes = [x for x in gt.get(case_id, []) if 108 <= x <= 426]
        for a in crimes:
            by_label[a] += 1
            by_label_method[a][method] += 1
        for a in gt_crimes:
            by_gt[a] += 1
        if crimes:
            if set(crimes) == set(gt_crimes):
                agree += 1
            else:
                disagree += 1
                disagreement_pairs[(tuple(sorted(gt_crimes)), tuple(crimes))] += 1

    n = len(details)
    print(f"Án: {n}")
    print("Thiếu phần:", {k: section_missing[k] for k in ("noi_dung", "nhan_dinh", "quyet_dinh")})
    print("Cách gắn nhãn:", {str(m): methods[m] for m in METHODS})
    print(f"Nhãn mới khớp nhãn số gốc: {agree} | lệch: {disagree}")
    print("Các cặp lệch phổ biến (nhãn gốc -> nhãn mới):")
    for (g, c), k in disagreement_pairs.most_common(10):
        print(f"  {k:5d}  {list(g)} -> {list(c)}")

    print(f"\n{'Điều':>5} {'nhãn mới':>9} {'(tên/gần đúng/điều)':>20} {'nhãn gốc':>9}  Tội danh")
    rows = []
    for a in targets:
        m = by_label_method[a]
        split = f"{m['ten_toi']}/{m['ten_toi_gan_dung']}/{m['dieu_luat']}"
        print(f"{a:>5} {by_label[a]:>9} {split:>20} {by_gt[a]:>9}  {title.get(a, '?')}")
        rows.append(
            {
                "dieu": a,
                "toi_danh": title.get(a),
                "so_an": by_label[a],
                "theo_cach": {str(k): v for k, v in m.items()},
                "so_an_nhan_goc": by_gt[a],
            }
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "so_an": n,
                "thieu_phan": dict(section_missing),
                "cach_gan_nhan": {str(m): methods[m] for m in METHODS},
                "khop_nhan_goc": agree,
                "lech_nhan_goc": disagree,
                "theo_dieu": rows,
                "top_toi": [
                    {"dieu": a, "toi_danh": title.get(a), "so_an": k}
                    for a, k in by_label.most_common(40)
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nĐã ghi {args.output}")


if __name__ == "__main__":
    main()
