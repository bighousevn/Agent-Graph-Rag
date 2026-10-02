"""Generate `judge_dep` (yếu tố cấu thành / constituent-element questions)
for each crime Điều using an LLM.

Equivalent of the original repo's `scripts/prepare_law_judge_dep.py`
(JUDGE_PROMPT), translated to Vietnamese and adapted to operate directly
on the Dieu objects produced by parse_blhs.py instead of a pre-scraped
law_judge_dep.json.

Usage:
    python -m vn_legal_graph.law.judge_dep \
        --input data/processed/criminal_law_vn.json \
        --output data/processed/criminal_law_vn_judge_dep.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from typing import Any, Dict, List

from ..llm import LLMClient
from ..prompts.vi import JUDGE_DEP_PROMPT


def parse_question_list(raw: str) -> List[str]:
    """Extract a Python list[str] from an LLM response, tolerating extra
    surrounding text (matches the defensive parsing style used by
    scripts/prepare_law_judge_dep.py in the original repo)."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    payload = raw[start : end + 1]
    try:
        parsed = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    result = [str(x).strip() for x in parsed if str(x).strip()]
    return result


JUDGE_DEP_MAX_TOKENS = 4096


def generate_judge_dep_for_article(client: LLMClient, dieu_text: str) -> List[str]:
    prompt = JUDGE_DEP_PROMPT.format(dieu_text=dieu_text)
    # Drug articles yield 30-40+ questions; 1024 tokens cut them off
    # (Điều 249, 250, 252 on the first real run).
    response = client.generate(prompt, max_tokens=JUDGE_DEP_MAX_TOKENS)
    questions = parse_question_list(response)
    if not questions:
        print(f"Warning: could not parse judge_dep response: {response[:200]!r}")
    return questions


def annotate_articles(
    articles: List[Dict[str, Any]], client: LLMClient
) -> List[Dict[str, Any]]:
    from .parse_blhs import Dieu, Khoan, Diem  # local import to avoid cycles

    for article in articles:
        if not article.get("title", "").strip().startswith("Tội"):
            article["judge_dep"] = []
            continue
        # Rebuild a Dieu just to reuse full_text(); cheap and avoids
        # duplicating the rendering logic here.
        khoan_objs = [
            Khoan(
                so=k["so"],
                text=k["text"],
                diem=[Diem(**d) for d in k["diem"]],
            )
            for k in article["khoan"]
        ]
        dieu_obj = Dieu(
            id=article["id"],
            suffix=article["suffix"],
            title=article["title"],
            phan=article["phan"],
            phan_title=article["phan_title"],
            chuong=article["chuong"],
            chuong_title=article["chuong_title"],
            muc=article.get("muc"),
            muc_title=article.get("muc_title"),
            khoan=khoan_objs,
            preamble=article.get("preamble", ""),
        )
        article["judge_dep"] = generate_judge_dep_for_article(
            client, dieu_obj.full_text()
        )
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="criminal_law_vn.json from parse_blhs.py")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dotenv-path", default=".env", help="Path to .env with LLM_* settings."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; just report how many crime articles would be processed.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)

    crime_articles = [a for a in articles if a.get("title", "").startswith("Tội")]
    print(f"{len(crime_articles)} / {len(articles)} Điều are crime articles.")

    if args.dry_run:
        return

    from ..config import AppConfig

    config = AppConfig.from_env_file(args.dotenv_path)
    client = LLMClient(config)
    articles = annotate_articles(articles, client)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
