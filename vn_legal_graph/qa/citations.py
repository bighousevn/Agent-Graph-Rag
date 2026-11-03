"""Check the khoản/điểm the answer cites against the article text it was given.

QA_ANSWER_PROMPT asks for each citation's "trich_dan", a verbatim piece of
the cited khoản/điểm. Finding that piece in the article text tells whether
the quote is real, and the khoản/điểm markers before it ("1.", "a)") tell
where it actually sits, so a citation like "khoản 6 điểm a Điều 134"
(khoản 6 has no points) shows up as a mismatch. Nothing is changed in the
answer; the result is reported next to it. No LLM, no graph change: the
markers are read from the stored article text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

from .batch import ALIASES

KHOAN_RE = re.compile(r"^\s*(\d+)\.\s", re.M)
DIEM_RE = re.compile(r"^\s*([a-zđ])\)\s", re.M)
MIN_QUOTE_CHARS = 12


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).lower()


def _fragments(quote: str) -> List[str]:
    """Pieces of the quote between ellipses, longest first, quote marks and
    spaces evened out; pieces shorter than MIN_QUOTE_CHARS are dropped."""
    pieces = re.split(r"\.\.\.|…", _norm(quote))
    out = [" ".join(p.split()).strip(" \"'“”‘’,.;:") for p in pieces]
    return sorted((p for p in out if len(p) >= MIN_QUOTE_CHARS), key=len, reverse=True)


def quote_places(article_text: str, quote: str) -> List[Tuple[str, str]]:
    """Every (khoản, điểm) where the quote occurs in the article text ("" when
    it is before any khoản, or in a khoản without a point); [] when it is not
    in the text. The longest piece of the quote that is found decides."""
    text = _norm(article_text)
    # one space for any run of spaces/newlines, keeping a map back to text
    flat_chars, index = [], []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            prev_space = True
            flat_chars.append(" ")
        else:
            prev_space = False
            flat_chars.append(ch)
        index.append(i)
    flat = "".join(flat_chars)
    for piece in _fragments(quote):
        places, pos = [], flat.find(piece)
        while pos != -1:
            at = index[pos]
            khoan, khoan_at = "", -1
            for m in KHOAN_RE.finditer(text, 0, at + 1):
                khoan, khoan_at = m.group(1), m.start()
            diem = ""
            for m in DIEM_RE.finditer(text, max(khoan_at, 0), at + 1):
                diem = m.group(1)
            if (khoan, diem) not in places:
                places.append((khoan, diem))
            pos = flat.find(piece, pos + 1)
        if places:
            return places
    return []


def locate_quote(article_text: str, quote: str) -> Optional[Tuple[str, str]]:
    """First place of the quote (see quote_places), or None."""
    places = quote_places(article_text, quote)
    return places[0] if places else None


def _code(luat: str) -> str:
    key = " ".join(str(luat or "").split()).lower()
    return ALIASES.get(key, str(luat or "").strip())


def check_citations(citations: Sequence, laws: Dict[str, str]) -> List[Dict]:
    """laws: label ("BLHS Điều 134") -> article text, for the articles the
    answer prompt showed. One row per citation:
      cite          the citation as text
      trong_tai_lieu  the cited article was among those shown
      trich_dan_co_that  the quote was found in that article (None: no quote)
      vi_tri        "khoản K điểm d" where the quote sits, or None
      khop_vi_tri   that place is the cited khoản (and điểm, if one is cited)
    """
    by_label = {_norm(k): v for k, v in laws.items()}
    rows = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        khoan = str(c.get("khoan") or "").strip()
        diem = _norm(str(c.get("diem") or "").strip())
        label = f"{_code(c.get('luat'))} Điều {str(c.get('dieu') or '').strip()}"
        cite = label + (f" khoản {khoan}" if khoan else "") + (f" điểm {diem}" if diem else "")
        text = by_label.get(_norm(label))
        quote = str(c.get("trich_dan") or "").strip()
        row = {"cite": cite, "trong_tai_lieu": text is not None, "trich_dan_co_that": None,
               "vi_tri": None, "khop_vi_tri": None}
        if text is not None and quote:
            places = quote_places(text, quote)
            row["trich_dan_co_that"] = bool(places)
            if places:
                fits = [(k, d) for k, d in places if (not khoan or khoan == k) and (not diem or diem == d)]
                k, d = (fits or places)[0]  # a repeated phrase counts if any occurrence fits
                row["vi_tri"] = " ".join(x for x in (f"khoản {k}" if k else "", f"điểm {d}" if d else "") if x)
                row["khop_vi_tri"] = bool(fits)
        rows.append(row)
    return rows


ARTICLE_RE = re.compile(r"\bĐiều\s+(\d+[a-z]?)\b")


def mentioned_not_cited(answer: Dict) -> List[str]:
    """Articles named in the answer text ("Điều 174") that are missing from
    its "dieu_luat" list. On the FB run, 21 of the 30 answers with an empty
    list named an article in the text: the prompt said to list only the
    articles "used", and a "not a crime" answer listed none."""
    cited = {str(c.get("dieu") or "").strip().lower() for c in answer.get("dieu_luat", []) if isinstance(c, dict)}
    out: List[str] = []
    for m in ARTICLE_RE.finditer(unicodedata.normalize("NFC", str(answer.get("cau_tra_loi") or ""))):
        n = m.group(1).lower()
        if n not in cited and f"Điều {n}" not in out:
            out.append(f"Điều {n}")
    return out
