"""Extract application guidance from TTLT 17/2007 (drug crimes, BLHS 1999)
into guidance_links records for the BLHS 2015 drug articles.

The user chose (2026-10-04) to use TTLT 17/2007 as REFERENCE ONLY: it
guides the expired BLHS 1999 and no in-force guidance exists for BLHS
2015 on what "tàng trữ" and "mua bán" mean. Every record says so in its
"from" field, so the LLM and a reader see the status.

BLHS 1999 Điều 194 (tàng trữ, vận chuyển, mua bán, chiếm đoạt) was split
by BLHS 2015 into Điều 249-252, hence the mapping below. Passages are
copied verbatim from the .docx by their section numbers; TTLT 08/2015
repealed point đ of section II.3.7, which is never selected.

    python -m vn_legal_graph.law.guidance_ttlt \
        --docx data/raw/guidance/17_2007_TTLT-BCA-VKSNDTC-TANDTC-BTP_m_61683.docx \
        --output data/raw/guidance/guidance_links.json
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List

from .parse_blhs import load_paragraphs_from_docx

SOURCE = (
    "TTLT 17/2007/TTLT-BCA-VKSNDTC-TANDTC-BTP, phần II mục {section} "
    "(hướng dẫn BLHS 1999 Điều 194, nay tương ứng Điều 249-252 BLHS 2015; "
    "văn bản đã hết hiệu lực cùng BLHS 1999, chỉ dùng tham khảo)"
)

# section -> BLHS 2015 articles it explains
SECTIONS: Dict[str, List[int]] = {
    "3.1": [249],        # tàng trữ: "... mà không nhằm mục đích mua bán ..."
    "3.2": [250, 251],   # vận chuyển; 2nd paragraph: giữ hộ biết mục đích mua bán -> mua bán
    "3.3": [251, 249],   # mua bán: the 7 acts, all "nhằm bán"; also tells 249 cases apart
    "3.4": [252],        # chiếm đoạt
}
# Single points of section 3.7, kept apart from the repealed point đ.
POINTS_37: Dict[str, List[int]] = {
    "c": [249, 251],     # buying on behalf of a user, for use
    "d": [249, 250],     # driving a buyer and his drugs
}

SECTION_HEAD_RE = re.compile(r"^(\d+\.\d+)\.\s")
POINT_RE = re.compile(r"^([a-zđ])\)\s")


def _part_ii_section_3(paragraphs: List[str]) -> List[str]:
    """Paragraphs of part II, section 3 (Điều 194), heading excluded."""
    start = next(
        i for i, p in enumerate(paragraphs) if p.startswith("3. Tội tàng trữ, vận chuyển, mua bán trái phép")
    )
    end = next(i for i in range(start + 1, len(paragraphs)) if re.match(r"^4\.\s", paragraphs[i]))
    return paragraphs[start + 1 : end]


def extract_section(paragraphs: List[str], number: str) -> str:
    """Section "3.x" of part II.3 with its following paragraphs, verbatim."""
    block = _part_ii_section_3(paragraphs)
    out: List[str] = []
    for p in block:
        m = SECTION_HEAD_RE.match(p)
        if m:
            if out:
                break
            if m.group(1) == number:
                out.append(p)
        elif out:
            out.append(p)
    if not out:
        raise ValueError(f"Section {number} not found")
    return "\n".join(out)


def extract_point_37(paragraphs: List[str], letter: str) -> str:
    block = extract_section(paragraphs, "3.7").split("\n")
    head = block[0]
    for p in block[1:]:
        m = POINT_RE.match(p)
        if m and m.group(1) == letter:
            return f"{head}\n{p}"
    raise ValueError(f"Point 3.7.{letter} not found")


def build_links(paragraphs: List[str]) -> List[Dict]:
    links = [
        {"explain": extract_section(paragraphs, s), "from": SOURCE.format(section=s), "laws": laws}
        for s, laws in SECTIONS.items()
    ]
    links += [
        {"explain": extract_point_37(paragraphs, letter), "from": SOURCE.format(section=f"3.7 điểm {letter}"), "laws": laws}
        for letter, laws in POINTS_37.items()
    ]
    return links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", required=True)
    parser.add_argument("--output", default="data/raw/guidance/guidance_links.json")
    args = parser.parse_args()
    links = build_links([p for p in load_paragraphs_from_docx(args.docx) if p])
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False, indent=2)
    for link in links:
        print(f"{link['from'][:60]}... -> {link['laws']} ({len(link['explain'])} ký tự)")
    print(f"Đã ghi {args.output}")


if __name__ == "__main__":
    main()
