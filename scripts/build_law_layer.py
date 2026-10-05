#!/usr/bin/env python3
"""End-to-end CLI for Phase 1: build the Law + Crime graph layers from the
BLHS .docx source.

Pipeline: parse_blhs -> judge_dep (LLM) -> link_guidance -> build_law_crime

Examples:
    # Structural parse only, no LLM calls, just to sanity-check chapter
    # selection and article counts before spending any API budget:
    python scripts/build_law_layer.py --chapters XVI XX --dry-run

    # Full run for two chapters, with LLM judge_dep generation:
    python scripts/build_law_layer.py --chapters XVI XX

    # Full BLHS, no chapter filter:
    python scripts/build_law_layer.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.config import AppConfig
from vn_legal_graph.law.parse_blhs import parse_docx, articles_to_dicts
from vn_legal_graph.law.judge_dep import annotate_articles
from vn_legal_graph.law.link_guidance import load_guidance_links, link_articles
from vn_legal_graph.law.build_law_crime import build_law_to_crime, build_crimes_by_part
from vn_legal_graph.llm import LLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docx",
        default="data/raw/law/11_VBHN-VPQH_650257.docx",
        help="Path to the BLHS .docx source.",
    )
    parser.add_argument(
        "--chapters",
        nargs="*",
        default=None,
        help="Optional Roman-numeral chapter filter, e.g. XVI XX. Default: all chapters.",
    )
    parser.add_argument(
        "--guidance-links",
        default="data/raw/guidance/guidance_links.json",
        help="Optional curated judicial-guidance link file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write intermediate and final JSON files.",
    )
    parser.add_argument(
        "--dotenv-path", default=".env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse structure and report counts only; skip LLM judge_dep calls.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel LLM calls for judge_dep.")
    parser.add_argument(
        "--bo-luat", default="BLHS",
        help="Code of the .docx (BLHS, BLTTHS, ...). Anything but BLHS is written to "
        "law_<code>_vn.json with every article, no crimes and no judge_dep; add it to "
        "the graph with build_graph.py --extra-laws.",
    )
    parser.add_argument(
        "--all-articles",
        action="store_true",
        help="Keep every Điều as a Law entry (general part too), not only crime articles.",
    )
    parser.add_argument(
        "--skip-judge-dep",
        action="store_true",
        help="Build law_to_crime_vn.json without LLM calls (judge_dep left empty). "
        "Does not read .env. Re-run without this flag to fill judge_dep.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1/4] Parsing {args.docx} ...")
    articles = parse_docx(args.docx)
    if args.chapters:
        wanted = {c.upper() for c in args.chapters}
        articles = [a for a in articles if a.chuong in wanted]
    article_dicts = articles_to_dicts(articles)
    n_crime = sum(1 for a in article_dicts if a["title"].strip().startswith("Tội"))
    print(f"    {len(article_dicts)} Điều parsed, {n_crime} are crime ('Tội ...') articles.")

    parsed_path = os.path.join(
        args.output_dir,
        "criminal_law_vn.json" if args.bo_luat == "BLHS" else f"criminal_law_{args.bo_luat.lower()}_vn.json",
    )
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(article_dicts, f, ensure_ascii=False, indent=2)
    print(f"    Written {parsed_path}")

    if args.dry_run:
        print("Dry run requested: stopping before LLM calls (judge_dep).")
        return

    if args.bo_luat != "BLHS":
        for a in article_dicts:
            a["judge_dep"] = []
        article_dicts = link_articles(article_dicts, load_guidance_links(args.guidance_links), bo_luat=args.bo_luat)
        out = os.path.join(args.output_dir, f"law_{args.bo_luat.lower()}_vn.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(build_law_to_crime(article_dicts, include_all=True, bo_luat=args.bo_luat), f, ensure_ascii=False, indent=2)
        print(f"{args.bo_luat}: {len(article_dicts)} Điều -> {out} (không có tội danh, không judge_dep)")
        return

    if args.skip_judge_dep:
        print("[2/4] Skipping judge_dep (--skip-judge-dep): left empty.")
        for a in article_dicts:
            a["judge_dep"] = []
    else:
        print("[2/4] Generating judge_dep (constituent-element questions) via LLM ...")
        config = AppConfig.from_env_file(args.dotenv_path)
        client = LLMClient(config)
        article_dicts = annotate_articles(article_dicts, client, workers=args.workers)
    with_dep = os.path.join(args.output_dir, "criminal_law_vn_judge_dep.json")
    with open(with_dep, "w", encoding="utf-8") as f:
        json.dump(article_dicts, f, ensure_ascii=False, indent=2)
    print(f"    Written {with_dep}")

    print("[3/4] Linking cross-references and judicial guidance ...")
    guidance_links = load_guidance_links(args.guidance_links)
    article_dicts = link_articles(article_dicts, guidance_links)
    linked_path = os.path.join(args.output_dir, "criminal_law_vn_linked.json")
    with open(linked_path, "w", encoding="utf-8") as f:
        json.dump(article_dicts, f, ensure_ascii=False, indent=2)
    print(f"    Written {linked_path}")

    print("[4/4] Building law_to_crime_vn.json and crimes_by_part_vn.json ...")
    law_to_crime = build_law_to_crime(article_dicts, include_all=args.all_articles)
    crimes_by_part = build_crimes_by_part(article_dicts)
    law_to_crime_path = os.path.join(args.output_dir, "law_to_crime_vn.json")
    crimes_by_part_path = os.path.join(args.output_dir, "crimes_by_part_vn.json")
    with open(law_to_crime_path, "w", encoding="utf-8") as f:
        json.dump(law_to_crime, f, ensure_ascii=False, indent=2)
    with open(crimes_by_part_path, "w", encoding="utf-8") as f:
        json.dump(crimes_by_part, f, ensure_ascii=False, indent=2)
    print(f"    Written {law_to_crime_path} and {crimes_by_part_path}")

    print("\nDone. Law + Crime layers ready for graph construction (Phase 3).")


if __name__ == "__main__":
    main()
