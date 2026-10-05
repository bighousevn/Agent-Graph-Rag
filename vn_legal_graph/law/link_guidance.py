"""Attach `related_laws` to each Điều: cross-references to other Điều
found inside its own text, plus (once curated) links to judicial guidance
documents (Nghị quyết HĐTP, Nghị định, Án lệ, Thông tư liên tịch).

Equivalent of `build_related_laws` in the original repo's
scripts/prepare_law_judge_dep.py, simplified because Vietnamese law
already uses Arabic numerals for "Điều X" (no Chinese-numeral conversion
needed).

Two independent sources of `related_laws`, both optional and additive:

1. **In-text cross-references** — always available: regex-extracted
   "Điều X" mentions inside an Điều's own full text (e.g. Điều 2 khoản 2
   references Điều 76). Needs no external file.

2. **Curated guidance links** — `data/raw/guidance/guidance_links.json`,
   a list of {"explain": <đoạn văn bản hướng dẫn>, "from": <tên văn bản>,
   "laws": [<article ids>]} records, in the same shape as the original
   repo's judicial_explanations.json. This file does not exist until the
   guidance texts in `guidance_manifest.json` have been fetched and their
   relevant passages transcribed/verified; until then this step is a
   no-op and only cross-references are attached.

Usage:
    python -m vn_legal_graph.law.link_guidance \
        --input data/processed/criminal_law_vn.json \
        --guidance-links data/raw/guidance/guidance_links.json \
        --output data/processed/criminal_law_vn_linked.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List

DIEU_REF_RE = re.compile(r"Điều\s+(\d+)([a-zđ]?)\b", re.IGNORECASE)


def build_article_text_index(articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map entry_label (e.g. "168") -> article dict, for quick lookup when
    resolving cross-references."""
    index = {}
    for a in articles:
        entry_label = f"{a['id']}{a.get('suffix', '')}"
        index[entry_label] = a
    return index


def render_article_full_text(article: Dict[str, Any]) -> str:
    lines = [f"Điều {article['id']}{article.get('suffix', '')}. {article['title']}"]
    if article.get("preamble"):
        lines.append(article["preamble"])
    for k in article["khoan"]:
        lines.append(f"{k['so']}. {k['text']}")
        for d in k["diem"]:
            lines.append(f"{d['ky_hieu']}) {d['text']}")
    return "\n".join(lines)


def extract_cross_references(
    article: Dict[str, Any], article_index: Dict[str, Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Find "Điều X" mentions inside an article's own text that point to a
    *different* article, and resolve them to that article's title + text
    (when the referenced article is itself in our parsed corpus)."""
    text = render_article_full_text(article)
    self_label = f"{article['id']}{article.get('suffix', '')}"
    seen = set()
    related: List[Dict[str, str]] = []

    for match in DIEU_REF_RE.finditer(text):
        ref_label = f"{match.group(1)}{match.group(2) or ''}"
        if ref_label == self_label or ref_label in seen:
            continue
        seen.add(ref_label)
        ref_article = article_index.get(ref_label)
        if ref_article is None:
            continue  # referenced article outside this corpus slice (e.g. outside selected chapters)
        related.append(
            {
                "loai": "dan_chieu_dieu_luat",
                "id": f"Điều {ref_label}",
                "text": render_article_full_text(ref_article),
            }
        )
    return related


def load_guidance_links(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _refers_to(ref: Any, entry_int: int, entry_label: str, bo_luat: str) -> bool:
    """A guidance ref is an int or "249" (BLHS), or "CODE:label" such as
    "BLTTHS:155" (BLTTHS Điều 155 is not BLHS Điều 155)."""
    if isinstance(ref, int):
        return bo_luat == "BLHS" and ref == entry_int
    ref = str(ref)
    if ":" in ref:
        code, label = ref.split(":", 1)
        return code.upper() == bo_luat and label.lower() == entry_label.lower()
    return bo_luat == "BLHS" and ref.lower() == entry_label.lower()


def attach_guidance_links(
    article: Dict[str, Any], guidance_links: List[Dict[str, Any]], bo_luat: str = "BLHS"
) -> List[Dict[str, str]]:
    entry_label = f"{article['id']}{article.get('suffix', '')}"
    entry_int = article["id"]
    related = []
    for item in guidance_links:
        laws = item.get("laws", [])
        if not any(_refers_to(r, entry_int, entry_label, bo_luat) for r in laws):
            continue
        related.append(
            {
                "loai": "van_ban_huong_dan",
                "id": str(item.get("from", "")),
                "text": str(item.get("explain", "")),
            }
        )
    return related


def link_articles(
    articles: List[Dict[str, Any]], guidance_links: List[Dict[str, Any]], bo_luat: str = "BLHS"
) -> List[Dict[str, Any]]:
    article_index = build_article_text_index(articles)
    for article in articles:
        cross_refs = extract_cross_references(article, article_index)
        guidance = attach_guidance_links(article, guidance_links, bo_luat)
        article["related_laws"] = cross_refs + guidance
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--guidance-links",
        default="data/raw/guidance/guidance_links.json",
        help="Optional curated guidance-link file; skipped if missing.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        articles = json.load(f)

    guidance_links = load_guidance_links(args.guidance_links)
    if not guidance_links:
        print(
            f"No guidance links found at {args.guidance_links} "
            "(expected until guidance texts are curated) — attaching only "
            "in-text cross-references."
        )
    else:
        print(f"Loaded {len(guidance_links)} guidance-link records.")

    articles = link_articles(articles, guidance_links)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    n_with_related = sum(1 for a in articles if a["related_laws"])
    print(f"{n_with_related} / {len(articles)} Điều now have related_laws.")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
