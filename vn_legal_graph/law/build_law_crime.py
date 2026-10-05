"""Assemble the final Law + Crime layer files from a fully-annotated
article list (parsed structure + judge_dep + related_laws already
attached by parse_blhs.py / judge_dep.py / link_guidance.py).

Outputs two files, mirroring the original repo's data shapes so later
graph-construction code can stay close to `feature_graph.py`:

- `law_to_crime_vn.json`: one entry per crime Điều, in the same
  {"id", "items": [{"text", "crime", "judge_dep", "related_laws"}]}
  shape as the original `law_to_crime.json` (here `items` always has
  exactly one element, since in Vietnamese law one Điều == one Tội,
  unlike the Chinese corpus where one Điều text block could bundle
  several named crimes).
- `crimes_by_part_vn.json`: crime names grouped by Chương (chapter),
  equivalent to the original `crimes_by_part.json`.

Usage:
    python -m vn_legal_graph.law.build_law_crime \
        --input data/processed/criminal_law_vn_linked.json \
        --law-to-crime-output data/processed/law_to_crime_vn.json \
        --crimes-by-part-output data/processed/crimes_by_part_vn.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List


def render_article_full_text(article: Dict[str, Any]) -> str:
    lines = [f"Điều {article['id']}{article.get('suffix', '')}. {article['title']}"]
    if article.get("preamble"):
        lines.append(article["preamble"])
    for k in article["khoan"]:
        lines.append(f"{k['so']}. {k['text']}")
        for d in k["diem"]:
            lines.append(f"{d['ky_hieu']}) {d['text']}")
    return "\n".join(lines)


def build_law_to_crime(
    articles: List[Dict[str, Any]], include_all: bool = False, bo_luat: str = "BLHS"
) -> List[Dict[str, Any]]:
    """One entry per crime Điều. With ``include_all`` every Điều is kept
    (general part such as Điều 51/52 included) and non-crime articles get
    ``crime: []``; the original keeps crime articles only, but legal Q&A
    answers cite the general part often (Điều 51/52 in 62 of 231 questions)."""
    result = []
    for a in articles:
        is_crime = bo_luat == "BLHS" and a.get("title", "").strip().startswith("Tội")
        if not is_crime and not include_all:
            continue
        result.append(
            {
                "id": a["id"],
                "suffix": a.get("suffix", ""),
                "title": a.get("title", ""),
                "bo_luat": bo_luat,
                "chuong": a.get("chuong", ""),
                "phan": a.get("phan", ""),
                "items": [
                    {
                        "text": render_article_full_text(a),
                        "crime": [a["title"]] if is_crime else [],
                        "judge_dep": a.get("judge_dep", []),
                        "related_laws": a.get("related_laws", []),
                    }
                ],
            }
        )
    return result


def build_crimes_by_part(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_chapter: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for a in articles:
        if not a.get("title", "").strip().startswith("Tội"):
            continue
        chuong = a["chuong"]
        if chuong not in by_chapter:
            by_chapter[chuong] = {
                "chuong": chuong,
                "chuong_title": a["chuong_title"],
                "crimes": [],
            }
            order.append(chuong)
        by_chapter[chuong]["crimes"].append(a["title"])
    return [by_chapter[c] for c in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--law-to-crime-output", required=True)
    parser.add_argument("--crimes-by-part-output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)

    law_to_crime = build_law_to_crime(articles)
    crimes_by_part = build_crimes_by_part(articles)

    with open(args.law_to_crime_output, "w", encoding="utf-8") as f:
        json.dump(law_to_crime, f, ensure_ascii=False, indent=2)
    with open(args.crimes_by_part_output, "w", encoding="utf-8") as f:
        json.dump(crimes_by_part, f, ensure_ascii=False, indent=2)

    total_crimes = sum(len(c["crimes"]) for c in crimes_by_part)
    print(f"{len(law_to_crime)} Law/Crime entries across {len(crimes_by_part)} chương "
          f"({total_crimes} tội danh total).")
    print(f"Written {args.law_to_crime_output} and {args.crimes_by_part_output}")


if __name__ == "__main__":
    main()
