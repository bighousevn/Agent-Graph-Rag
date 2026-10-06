#!/usr/bin/env python3
"""Phase 2: build cases_vn.json (corpus) and cases_vn_test.json (test) from
ViCSR. See vn_legal_graph/cases/build_cases.py for the steps.

    python scripts/build_cases.py
    python scripts/build_cases.py --articles 249 251 173 247 --test-per-crime 10 --corpus-max-per-crime 100
    # every crime, ViCSR + anle.toaan.gov.vn, as the paper samples (<=20 corpus/crime);
    # ViCSR cases labelled 410 are dropped (--drop-articles):
    python scripts/build_cases.py --all-crimes --anle data/raw/anle/documents.parquet \
        --corpus-max-per-crime 20 --test-per-crime 5 --test-min-cases 10

No LLM calls.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.cases.build_cases import (
    _crime_names,
    dedupe,
    extract_dieu_khoan,
    extract_facts,
    split_by_crime,
)
from vn_legal_graph.cases.vicsr import (
    build_crime_name_index,
    is_appellate,
    judgment_year,
    label_case,
    load_ground_truth,
    load_laws,
    load_records,
)

# Words that should no longer follow "tội" in a masked fact text if a
# target crime name slipped through (e.g. BLHS 1999 compound names).
LEAK_AUDIT_RE = re.compile(r"\btội (?:tàng trữ|mua bán|trộm cắp|trồng cây|vận chuyển)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw/dataset_drive")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--articles", nargs="*", type=int, default=[249, 251, 173, 247])
    parser.add_argument("--test-per-crime", type=int, default=10)
    parser.add_argument("--corpus-max-per-crime", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-crimes", action="store_true", help="Every BLHS crime, not only --articles.")
    parser.add_argument("--anle", default="", help="anle.toaan.gov.vn parquet (vn_legal_graph/cases/anle.py) to add.")
    parser.add_argument("--law-json", default="data/processed/law_to_crime_vn.json")
    parser.add_argument("--test-min-cases", type=int, default=0, help="Crimes with fewer cases get no test cases.")
    parser.add_argument("--drop-articles", nargs="*", type=int, default=[410],
                        help="ViCSR labels known to be wrong: 410 is Điều 244 matched to a truncated title (HANDOFF 8.1).")
    args = parser.parse_args()

    scope = set(args.articles)
    laws = load_laws(os.path.join(args.data_dir, "law_shorten.txt"))
    gt = load_ground_truth(os.path.join(args.data_dir, "ground_truth.json"))
    details = load_records(os.path.join(args.data_dir, "case_detail.txt"))
    summaries = load_records(os.path.join(args.data_dir, "case_sumary.txt"))
    name_index = build_crime_name_index(laws)
    crime_names = _crime_names(laws)
    title = {int(l.dieu): l.title for l in laws.values() if l.is_crime}

    kept, dropped = dedupe(details)
    print(f"Án: {len(details)} | loại trùng: {len(dropped)} | còn: {len(kept)}")

    drop_reason = collections.Counter()
    records = {}
    for case_id, detail in kept.items():
        if is_appellate(detail):
            drop_reason["án phúc thẩm"] += 1
            continue
        result = label_case(detail, laws, name_index)
        crimes = result["toi_danh"]
        if not crimes:
            drop_reason["không gắn được nhãn"] += 1
            continue
        if set(crimes) & set(args.drop_articles):
            drop_reason["nhãn không tin được"] += 1
            continue
        if not args.all_crimes and not set(crimes) <= scope:
            drop_reason["có tội ngoài phạm vi" if set(crimes) & scope else "ngoài phạm vi"] += 1
            continue
        facts = extract_facts(result["sections"], summaries.get(case_id), crime_names)
        if facts is None:
            drop_reason["diễn biến quá ngắn"] += 1
            continue
        dien_bien, source = facts
        records[str(case_id)] = {
            "id": f"vicsr-{case_id}",
            "nguon": f"ViCSR#{case_id}",
            "nam": judgment_year(detail),
            "bi_cao": "ẩn danh",
            "dien_bien": dien_bien,
            "dien_bien_nguon": source,
            "toi_danh": [title[a] for a in crimes],
            "dieu": crimes,
            "dieu_khoan": extract_dieu_khoan(result["sections"]["quyet_dinh"] or "", crimes),
            "hinh_phat": None,
            "cach_gan_nhan": result["cach_gan_nhan"],
            "nhan_goc": [x for x in gt.get(case_id, []) if 108 <= x <= 426],
        }

    print("Loại vì:", dict(drop_reason))
    if args.anle:
        import pyarrow.parquet as pq

        from vn_legal_graph.cases.anle import records_from_documents

        with open(args.law_json, encoding="utf-8") as f:
            law_json = json.load(f)
        anle = records_from_documents(pq.read_table(args.anle).to_pylist(), law_json)
        print(f"anle.toaan.gov.vn: {len(anle)} án hình sự có nhãn và diễn biến")
        records.update(anle)
    eligible = collections.Counter(a for r in records.values() for a in r["dieu"])
    shown = sorted(eligible, key=lambda a: -eligible[a]) if args.all_crimes else sorted(scope)
    print(f"Án đủ điều kiện theo tội ({len(eligible)} tội):", {a: eligible[a] for a in shown})

    corpus_ids, test_ids = split_by_crime(
        {cid: [str(a) for a in r["dieu"]] for cid, r in records.items()},
        args.test_per_crime,
        args.corpus_max_per_crime,
        args.seed,
        args.test_min_cases,
    )
    assert not set(corpus_ids) & set(test_ids)

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = {"corpus": (corpus_ids, "cases_vn.json"), "test": (test_ids, "cases_vn_test.json")}
    for role, (ids, filename) in outputs.items():
        rows = [dict(records[i], vai_tro=role) for i in ids]
        path = os.path.join(args.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        per_crime = collections.Counter(str(a) for r in rows for a in r["dieu"])
        words = sorted(len(r["dien_bien"].split()) for r in rows)
        leaks = sum(1 for r in rows if LEAK_AUDIT_RE.search(r["dien_bien"]))
        digits = sum(1 for r in rows if re.search(r"điều \d", r["dien_bien"]))
        print(
            f"\n{role}: {len(rows)} án -> {path}\n"
            f"  theo tội ({len(per_crime)} tội): {dict(per_crime.most_common())}\n"
            f"  theo nguồn: {dict(collections.Counter(r['id'].split('-')[0] for r in rows))}\n"
            f"  độ dài diễn biến (từ): min {words[0]}, trung vị {words[len(words) // 2]}, max {words[-1]}\n"
            f"  nguồn diễn biến: {dict(collections.Counter(r['dien_bien_nguon'] for r in rows))}\n"
            f"  kiểm tra rò rỉ: còn 'tội <tên>' ở {leaks} án, còn 'điều <số>' ở {digits} án"
        )


if __name__ == "__main__":
    main()
