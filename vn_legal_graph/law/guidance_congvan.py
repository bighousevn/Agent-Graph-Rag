"""Turn TANDTC official letters (Công văn giải đáp vướng mắc, e.g.
163/TANDTC-PC, 250/TANDTC-PC, 01/TANDTC-PC) into guidance_links records,
the judicial-explanation layer of the original method (related_laws on
Law nodes).

A letter is split into numbered items (question + answer). Items are kept
only from criminal parts ("HÌNH SỰ" in the part heading; a letter without
part headings counts as criminal). Each item is linked to the articles it
cites ("Điều 134 của Bộ luật Hình sự", "khoản 3 Điều 155 Bộ luật Tố tụng
hình sự") and to BLHS crimes it names exactly ("Tội lừa đảo chiếm đoạt tài
sản"). Links are "BLHS:134" / "BLTTHS:155".

    python -m vn_legal_graph.law.guidance_congvan --dir data/raw/guidance \
        --law-json data/processed/law_to_crime_vn.json \
        --output data/raw/guidance/guidance_links.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, Iterable, List, Optional

from .parse_blhs import load_paragraphs_from_docx

PART_RE = re.compile(r"^([IVX]+)\.\s+(.+)$")
ITEM_RE = re.compile(r"^(\d+)\.\s+\S")
SO_HIEU_RE = re.compile(r"Số:\s*([\w/.\-]+)")
DATE_RE = re.compile(r"ngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})")

# "Điều 134", "Điều 256a" followed (within a short window) by the code name.
ARTICLE_REF_RE = re.compile(
    r"Điều\s+(\d+[a-zđ]?)\b(?:[^.;\n]{0,25}?)\s*(?:của\s+)?"
    r"(Bộ luật Tố tụng hình sự|Bộ luật Hình sự|BLTTHS|BLHS)",
    re.IGNORECASE,
)
CODE_OF = {"bộ luật tố tụng hình sự": "BLTTHS", "bltths": "BLTTHS", "bộ luật hình sự": "BLHS", "blhs": "BLHS"}


def letter_metadata(paragraphs: List[str], table_cells: List[str]) -> Dict[str, str]:
    text = "\n".join(table_cells + paragraphs[:5])
    so = SO_HIEU_RE.search(text)
    date = DATE_RE.search(text)
    return {
        "so_hieu": so.group(1) if so else "",
        "ngay": f"{int(date.group(1)):02d}/{int(date.group(2)):02d}/{date.group(3)}" if date else "",
    }


def split_items(paragraphs: Iterable[str]) -> List[Dict]:
    """Numbered items of the criminal parts. A new item starts at "N. "
    only when N is the next number of the current part; quoted law text
    inside an answer ("1. ...", "2. ...") stays in its item."""
    items: List[Dict] = []
    part, part_title, criminal, expected = "", "", True, 1
    current: Optional[Dict] = None
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        m = PART_RE.match(p)
        if m:
            part, part_title = m.group(1), m.group(2)
            criminal = "HÌNH SỰ" in part_title.upper()
            current = None
            # Some letters number items across parts (250), others restart (163).
            continue
        m = ITEM_RE.match(p)
        if m and int(m.group(1)) in (expected, 1) and (current is None or int(m.group(1)) == expected):
            expected = int(m.group(1)) + 1
            current = {"phan": part, "phan_title": part_title, "muc": m.group(1), "criminal": criminal, "text": p}
            items.append(current)
            continue
        if current is not None:
            current["text"] += "\n" + p
    return [i for i in items if i["criminal"]]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def law_refs(text: str, crime_titles: Dict[str, int]) -> List[str]:
    refs: List[str] = []
    for m in ARTICLE_REF_RE.finditer(text):
        ref = f"{CODE_OF[m.group(2).lower()]}:{m.group(1).lower()}"
        if ref not in refs:
            refs.append(ref)
    norm = _norm(text)
    for title, article in crime_titles.items():
        if title in norm:
            ref = f"BLHS:{article}"
            if ref not in refs:
                refs.append(ref)
    return refs


def crime_title_index(law_json: str) -> Dict[str, int]:
    """Normalised BLHS crime titles ("tội lừa đảo chiếm đoạt tài sản") ->
    article. Titles of 4 words or fewer are skipped: "Tội giết người" would
    match "tội giết người hoặc ..." inside unrelated items."""
    with open(law_json, encoding="utf-8") as f:
        laws = json.load(f)
    out = {}
    for law in laws:
        for crime in law["items"][0].get("crime", []):
            t = _norm(crime)
            if len(t.split()) > 4:
                out[t] = law["id"]
    return out


def links_from_letter(path: str, crime_titles: Dict[str, int]) -> List[Dict]:
    import docx  # metadata sits in the header table

    from .parse_blhs import _docx_source

    document = docx.Document(_docx_source(path))
    cells = [c.text.strip() for t in document.tables for r in t.rows for c in r.cells if c.text.strip()]
    paragraphs = [p for p in load_paragraphs_from_docx(path) if p]
    meta = letter_metadata(paragraphs, cells)
    links = []
    for item in split_items(paragraphs):
        refs = law_refs(item["text"], crime_titles)
        if not refs:
            continue
        where = f"phần {item['phan']} mục {item['muc']}" if item["phan"] else f"mục {item['muc']}"
        links.append({
            "explain": item["text"],
            "from": f"Công văn {meta['so_hieu']} ngày {meta['ngay']} của Tòa án nhân dân tối cao, {where}",
            "laws": refs,
        })
    return links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/raw/guidance")
    parser.add_argument("--law-json", default="data/processed/law_to_crime_vn.json")
    parser.add_argument("--output", default="data/raw/guidance/guidance_links.json")
    args = parser.parse_args()

    titles = crime_title_index(args.law_json)
    links: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(args.dir, "*TANDTC-PC*.docx"))):
        got = links_from_letter(path, titles)
        print(f"{os.path.basename(path)}: {len(got)} mục hình sự có gắn điều luật")
        links += got
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)
    print(f"Tổng {len(links)} mục -> {args.output}")


if __name__ == "__main__":
    main()
