#!/usr/bin/env python3
"""Make a clean question file for scripts/answer_batch.py: keep id + question
(+ title), mask phone numbers, drop empty and repeated questions. No LLM.

    python scripts/clean_questions.py --input "data/raw/fb/fb_raw_2026-10-25 (1)_hinh_su.xlsx" \
        --id-col post_id --text-col text --out data/raw/fb/cau_hoi.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.qa.batch import clean_questions, read_table, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--text-col", default="cau_hoi")
    parser.add_argument("--title-col", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, report = clean_questions(read_table(args.input), args.id_col, args.text_col, args.title_col or None)
    write_csv(args.out, rows)
    print(f"{report['vao']} dòng vào -> {report['ra']} câu ({args.out}); bỏ: trống {report['trong']}, "
          f"trùng mã {report['trung_id']}, trùng nội dung {report['trung_noi_dung']}; che {report['che_sdt']} số điện thoại")


if __name__ == "__main__":
    main()
