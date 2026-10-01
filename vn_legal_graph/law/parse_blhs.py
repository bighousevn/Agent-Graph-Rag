"""Parse a Bộ luật Hình sự (Vietnamese Penal Code) .docx file into a
structured JSON tree: Phần -> Chương -> Mục (optional) -> Điều -> Khoản -> Điểm.

This is the Vietnamese-law equivalent of how the original LegalGraphRAG
repo represents ``criminal_law_processed.json`` / ``law_to_crime.json``,
except we parse directly from an official .docx source instead of a
pre-scraped corpus, and we keep the full Phần/Chương/Mục/Khoản/Điểm
hierarchy instead of flattening straight to "entry" numbers.

Usage:
    python -m vn_legal_graph.law.parse_blhs \
        --docx data/raw/law/100_2015_QH13_296661.docx \
        --output data/processed/criminal_law_vn.json

Known data quirk handled here: some legacy .docx exports of Vietnamese
legal text use the Latin letter "Ð"/"ð" (U+00D0 / U+00F0, Eth) instead of
the correct Vietnamese "Đ"/"đ" (U+0110 / U+0111) in words like "Điều" and
"Đ)". We normalize this before parsing, otherwise articles/points using
the wrong glyph (e.g. "Ðiều 317") silently fail to match and go missing.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import docx  # python-docx
except ImportError:  # pragma: no cover - exercised only when dependency missing
    docx = None


# --------------------------------------------------------------------------
# Text normalization
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Fix common Unicode look-alike issues found in scraped/exported
    Vietnamese legal .docx files.

    - U+00D0 (Ð, Latin Capital Letter Eth) -> U+0110 (Đ)
    - U+00F0 (ð, Latin Small Letter Eth)   -> U+0111 (đ)
    - U+01A3 / U+01A2 (ƣ / Ƣ, Latin Letter Oi) -> U+01B0 / U+01AF (ư / Ư),
      left by legacy-font converters ("đƣợc"); ~24k times in ViCSR.
    - Non-breaking spaces / stray whitespace -> normal spaces, trimmed.
    """
    text = text.replace("Ð", "Đ").replace("ð", "đ")
    text = text.replace("ƣ", "ư").replace("Ƣ", "Ư")
    text = text.replace("\xa0", " ")
    return text.strip()


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Diem:
    """Điểm (point) — the finest-grained sub-unit, e.g. "a) ...", "b) ..."."""
    ky_hieu: str  # a, b, c, ..., đ, e, g, h, ...
    text: str


@dataclass
class Khoan:
    """Khoản (clause) — numbered paragraph within an Điều, e.g. "1. ...""."""
    so: int
    text: str
    diem: List[Diem] = field(default_factory=list)


@dataclass
class Dieu:
    """Điều (article) — the unit that becomes one Law node in the graph."""
    id: int
    suffix: str  # "" normally; "a"/"b"... for inserted articles like "217a"
    title: str
    phan: str
    phan_title: str
    chuong: str
    chuong_title: str
    muc: Optional[str] = None
    muc_title: Optional[str] = None
    khoan: List[Khoan] = field(default_factory=list)
    # Text that appears directly under the Điều heading before any
    # numbered Khoản (some short articles have no Khoản at all).
    preamble: str = ""

    @property
    def entry_label(self) -> str:
        return f"{self.id}{self.suffix}"

    @property
    def is_crime_article(self) -> bool:
        """Heuristic used later to decide which Điều become Crime+Law
        nodes: articles inside Phần thứ hai ("CÁC TỘI PHẠM") whose title
        starts with "Tội "."""
        return self.title.strip().startswith("Tội")

    def full_text(self) -> str:
        """Reconstruct a human-readable rendering of the article, close to
        the original docx layout (used for embeddings / LLM prompts)."""
        lines = [f"Điều {self.entry_label}. {self.title}"]
        if self.preamble:
            lines.append(self.preamble)
        for k in self.khoan:
            lines.append(f"{k.so}. {k.text}")
            for d in k.diem:
                lines.append(f"{d.ky_hieu}) {d.text}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Regex patterns (applied to already-normalized text)
# --------------------------------------------------------------------------

PHAN_RE = re.compile(r"^Phần\s+thứ\s+\S+$", re.IGNORECASE)
CHUONG_RE = re.compile(r"^Chương\s+[IVXLCDM]+$")
MUC_RE = re.compile(r"^Mục\s+(\d+)\.\s*(.+)$")
DIEU_RE = re.compile(r"^Điều\s+(\d+)([a-zđ]?)\.\s*(.+)$", re.IGNORECASE)
KHOAN_RE = re.compile(r"^(\d+)\.\s*(.*)$")
DIEM_RE = re.compile(r"^([a-zđ])\)\s*(.*)$", re.IGNORECASE)


def _extract_chuong_numeral(line: str) -> str:
    return line.split()[-1]


# --------------------------------------------------------------------------
# Core parser
# --------------------------------------------------------------------------

def parse_paragraphs(paragraphs: List[str]) -> List[Dieu]:
    """Parse a flat list of paragraph strings (already normalized) into a
    list of Dieu objects, in document order."""

    articles: List[Dieu] = []

    cur_phan = ""
    cur_phan_title = ""
    cur_chuong = ""
    cur_chuong_title = ""
    cur_muc: Optional[str] = None
    cur_muc_title: Optional[str] = None

    cur_article: Optional[Dieu] = None
    cur_khoan: Optional[Khoan] = None

    # State for "the next non-empty paragraph is a title line" cases.
    expect_title_for: Optional[str] = None  # "phan" | "chuong"

    def flush_article():
        nonlocal cur_article
        if cur_article is not None:
            articles.append(cur_article)
        cur_article = None

    for raw_line in paragraphs:
        line = raw_line.strip()
        if not line:
            continue

        if expect_title_for == "phan":
            cur_phan_title = line
            cur_chuong = ""
            cur_chuong_title = ""
            cur_muc = None
            cur_muc_title = None
            expect_title_for = None
            continue
        if expect_title_for == "chuong":
            cur_chuong_title = line
            cur_muc = None
            cur_muc_title = None
            expect_title_for = None
            continue

        if PHAN_RE.match(line):
            flush_article()
            cur_article = None
            cur_khoan = None
            cur_phan = line
            expect_title_for = "phan"
            continue

        if CHUONG_RE.match(line):
            flush_article()
            cur_khoan = None
            cur_chuong = _extract_chuong_numeral(line)
            expect_title_for = "chuong"
            continue

        m = MUC_RE.match(line)
        if m:
            flush_article()
            cur_khoan = None
            cur_muc = m.group(1)
            cur_muc_title = m.group(2).strip()
            continue

        m = DIEU_RE.match(line)
        if m:
            flush_article()
            cur_khoan = None
            cur_article = Dieu(
                id=int(m.group(1)),
                suffix=m.group(2) or "",
                title=m.group(3).strip(),
                phan=cur_phan,
                phan_title=cur_phan_title,
                chuong=cur_chuong,
                chuong_title=cur_chuong_title,
                muc=cur_muc,
                muc_title=cur_muc_title,
            )
            continue

        if cur_article is None:
            # Preamble text before Chương I (Quốc hiệu, "Căn cứ Hiến pháp...")
            # — not part of any article, safely ignored.
            continue

        m = KHOAN_RE.match(line)
        if m:
            cur_khoan = Khoan(so=int(m.group(1)), text=m.group(2).strip())
            cur_article.khoan.append(cur_khoan)
            continue

        m = DIEM_RE.match(line)
        if m:
            diem = Diem(ky_hieu=m.group(1), text=m.group(2).strip())
            if cur_khoan is not None:
                cur_khoan.diem.append(diem)
            else:
                # Điểm appearing directly under an Điều with no Khoản
                # number (rare, but handled defensively): fold into the
                # preamble so no text is silently dropped.
                cur_article.preamble = (
                    cur_article.preamble + f"\n{diem.ky_hieu}) {diem.text}"
                ).strip()
            continue

        # Plain continuation line: append to whatever is the most specific
        # open unit (điểm > khoản > preamble), so multi-paragraph clauses
        # don't lose text.
        if cur_khoan is not None and cur_khoan.diem:
            cur_khoan.diem[-1].text = (cur_khoan.diem[-1].text + " " + line).strip()
        elif cur_khoan is not None:
            cur_khoan.text = (cur_khoan.text + " " + line).strip()
        else:
            cur_article.preamble = (cur_article.preamble + " " + line).strip()

    flush_article()
    return articles


def load_paragraphs_from_docx(path: str) -> List[str]:
    if docx is None:
        raise ImportError(
            "python-docx is required to parse .docx files. Install it with "
            "`pip install python-docx`."
        )
    document = docx.Document(path)
    return [normalize_text(p.text) for p in document.paragraphs]


def parse_docx(path: str) -> List[Dieu]:
    paragraphs = load_paragraphs_from_docx(path)
    return parse_paragraphs(paragraphs)


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def articles_to_dicts(articles: List[Dieu]) -> List[dict]:
    return [asdict(a) for a in articles]


def save_json(articles: List[Dieu], output_path: str) -> None:
    data = articles_to_dicts(articles)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", required=True, help="Path to the BLHS .docx file.")
    parser.add_argument(
        "--output",
        default="data/processed/criminal_law_vn.json",
        help="Where to write the parsed article tree as JSON.",
    )
    parser.add_argument(
        "--chapters",
        nargs="*",
        default=None,
        help="Optional list of Roman-numeral chapter filters, e.g. XVI XX.",
    )
    args = parser.parse_args()

    articles = parse_docx(args.docx)
    if args.chapters:
        wanted = {c.upper() for c in args.chapters}
        articles = [a for a in articles if a.chuong in wanted]

    save_json(articles, args.output)

    n_crime = sum(1 for a in articles if a.is_crime_article)
    print(f"Parsed {len(articles)} Điều ({n_crime} are 'Tội ...' crime articles).")
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
