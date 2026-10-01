"""Loader for the ViCSR dataset (Vietnamese Case-to-Statute Retrieval,
SIGIR 2026), as distributed in the Google Drive "Dataset" folder:

    law_shorten.txt    1122 điều: số hiệu + tên điều
    law_detail.txt     1122 điều: toàn văn
    case_sumary.txt    10001 án: đoạn tóm tắt (phần lớn là "nội dung vụ án")
    case_detail.txt    10001 án: toàn văn bản án
    ground_truth.json  {case_id: [law_id, ...]}

Text files hold records joined as ``<id> [---] <text> [-/-] <id> [---] ...``.
Text is lowercased; ``case_detail`` has its punctuation stripped.

Law ids 1-426 are BLHS điều 1-426 (id == số điều). The rest are Bộ luật Tố
tụng hình sự and Luật Thi hành án hình sự.

Two data issues this module handles:

1. The Latin Eth glyph "ð" appears in place of "đ" (same issue as the BLHS
   .docx), so every text goes through ``normalize_text``.
2. ``ground_truth.json`` mixes BLHS 1999 and BLHS 2015 article numbers.
   E.g. ~755 cases labelled "194" are drug cases citing BLHS 1999 Điều 194,
   which BLHS 2015 split into Điều 249-252, while 2015's Điều 194 is an
   unrelated crime. So the crime label we trust is the crime *name* in the
   verdict's "phạm tội ..." sentence, matched to BLHS 2015 crime titles —
   see ``extract_convicted_crimes``. The numeric label is kept for
   comparison.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..law.parse_blhs import normalize_text

RECORD_SEP = "[-/-]"
FIELD_SEP = "[---]"

BO_LUAT_PREFIXES = [
    ("bltths", "bộ luật tố tụng hình sự"),
    ("lthahs", "luật thi hành án hình sự"),
    ("blhs", "bộ luật hình sự"),
]


@dataclass
class LawEntry:
    id: int
    bo_luat: str  # "blhs" | "bltths" | "lthahs" | "khac"
    dieu: str  # article number within its own code, e.g. "249"
    title: str  # e.g. "tội tàng trữ trái phép chất ma túy"

    @property
    def is_crime(self) -> bool:
        return self.bo_luat == "blhs" and self.title.startswith("tội ")


# --------------------------------------------------------------------------
# Raw file parsing
# --------------------------------------------------------------------------

def parse_records(text: str) -> Dict[int, str]:
    records: Dict[int, str] = {}
    for chunk in text.split(RECORD_SEP):
        chunk = chunk.strip()
        if not chunk:
            continue
        raw_id, sep, body = chunk.partition(FIELD_SEP)
        if not sep:
            raise ValueError(f"Record without {FIELD_SEP!r}: {chunk[:80]!r}")
        records[int(raw_id.strip())] = normalize_text(body)
    return records


def load_records(path: str) -> Dict[int, str]:
    with open(path, encoding="utf-8") as f:
        return parse_records(f.read())


LAW_HEAD_RE = re.compile(r"^điều (\d+[a-z]?) ")


def parse_law_entry(law_id: int, text: str) -> LawEntry:
    m = LAW_HEAD_RE.match(text)
    if not m:
        return LawEntry(id=law_id, bo_luat="khac", dieu="", title=text)
    rest = text[m.end():]
    for code, prefix in BO_LUAT_PREFIXES:
        # "luật thi hành án" is also a suffix of "bộ luật thi hành án", so
        # allow an optional leading "bộ ".
        for p in (prefix, "bộ " + prefix):
            if rest.startswith(p + " "):
                return LawEntry(
                    id=law_id, bo_luat=code, dieu=m.group(1), title=rest[len(p) + 1:].strip()
                )
    return LawEntry(id=law_id, bo_luat="khac", dieu=m.group(1), title=rest)


def load_laws(law_shorten_path: str) -> Dict[int, LawEntry]:
    return {i: parse_law_entry(i, t) for i, t in load_records(law_shorten_path).items()}


def load_ground_truth(path: str) -> Dict[int, List[int]]:
    with open(path, encoding="utf-8") as f:
        return {int(k): [int(x) for x in v] for k, v in json.load(f).items()}


# --------------------------------------------------------------------------
# Judgment structure
# --------------------------------------------------------------------------

NOI_DUNG_MARKER = "nội dung vụ án"
NHAN_DINH_MARKERS = ("nhận định của hội đồng xét xử", "nhận định của tòa án")
# The verdict opens with "vì các lẽ trên, quyết định:" in ~92% of ViCSR
# judgments. The bare word "quyết định" is unreliable: it also appears in
# "quyết định đưa vụ án ra xét xử" and "khi quyết định hình phạt" inside the
# nhận định section.
VERDICT_OPENING_RE = re.compile(r"vì (?:các |những )?lẽ (?:nêu )?trên|từ những nhận định trên")
# Used when the opening phrase is missing or damaged by legacy-font garbling
# ("v c c lï trªn quyõt þnh"): the conviction sentence itself is often
# still readable.
QUYET_DINH_FALLBACK_MARKERS = (
    "tuyên bố bị cáo",
    "tuyên bố các bị cáo",
    "quyết định tuyên bố",
    "quyết định căn cứ",
)


def split_sections(detail: str) -> Dict[str, Optional[str]]:
    """Split a full judgment into noi_dung / nhan_dinh / quyet_dinh.
    A section is None when its marker is not found."""
    nd = detail.find(NOI_DUNG_MARKER)
    search_from = max(nd, 0)

    nh_positions = [p for p in (detail.find(m, search_from) for m in NHAN_DINH_MARKERS) if p != -1]
    nh = min(nh_positions) if nh_positions else -1

    qd_from = nh if nh != -1 else search_from
    openings = [m.start() for m in VERDICT_OPENING_RE.finditer(detail, qd_from)]
    qd = openings[-1] if openings else -1
    if qd == -1:
        positions = [detail.rfind(m, qd_from) for m in QUYET_DINH_FALLBACK_MARKERS]
        positions = [p for p in positions if p != -1]
        qd = min(positions) if positions else -1

    def cut(start: int, *ends: int) -> Optional[str]:
        if start == -1:
            return None
        valid_ends = [e for e in ends if e > start]
        end = min(valid_ends) if valid_ends else len(detail)
        return detail[start:end].strip()

    return {
        "noi_dung": cut(nd, nh, qd),
        "nhan_dinh": cut(nh, qd),
        "quyet_dinh": cut(qd),
    }


# --------------------------------------------------------------------------
# Crime labels from the verdict text
# --------------------------------------------------------------------------

# Old-style tone placement ("tuý", "hoà") -> new style ("túy", "hòa"), only
# when the vowel pair ends the syllable and is not "qu" + vowel ("quý").
_TONE_SHIFT = {
    "uý": "úy", "uỳ": "ùy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "oá": "óa", "oà": "òa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oé": "óe", "oè": "òe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
}
_TONE_SHIFT_RE = re.compile(r"(?<!q)(" + "|".join(_TONE_SHIFT) + r")(?!\w)")


def _norm_name(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = _TONE_SHIFT_RE.sub(lambda m: _TONE_SHIFT[m.group(1)], s)
    return re.sub(r"\s+", " ", s).strip()


# Short names courts use instead of the full statutory title.
CRIME_NAME_ALIASES = {
    "trồng cây thuốc phiện": 247,
    "trồng cây cần sa": 247,
    "trồng cây côca": 247,
}


def build_crime_name_index(laws: Dict[int, LawEntry]) -> Dict[str, int]:
    """Map normalized crime name (without the leading "tội ") -> BLHS 2015
    article number, plus CRIME_NAME_ALIASES."""
    index = {}
    for law in laws.values():
        if law.is_crime:
            index[_norm_name(law.title[len("tội "):])] = int(law.dieu)
    for alias, article in CRIME_NAME_ALIASES.items():
        index.setdefault(_norm_name(alias), article)
    return index


# The conviction sentence: "tuyên bố (các) bị cáo <tên> (đã) phạm (các) tội <tội>".
SENTENCE_RE = re.compile(r"tuyên bố (?:các )?bị cáo .{0,200}?(?:đã )?phạm (?:các )?tội ")
CO_DEFENDANT_RE = re.compile(r"bị cáo .{0,120}?(?:đã )?phạm (?:các )?tội ")
FALLBACK_RE = re.compile(r"phạm (?:các )?tội ")
FUZZY_MIN_LEN = 15
FUZZY_THRESHOLD = 0.9


def _match_name(tail: str, names: List[str]) -> Optional[tuple]:
    """Return (name, fuzzy) for the longest crime name at the start of
    ``tail``. Exact prefix match first; otherwise a close match on names of
    at least FUZZY_MIN_LEN chars, to absorb OCR damage like "tàng trư"."""
    exact = next((n for n in names if tail.startswith(n)), None)
    if exact is not None:
        return exact, False
    best, best_ratio = None, 0.0
    for n in names:
        if len(n) < FUZZY_MIN_LEN:
            continue
        ratio = difflib.SequenceMatcher(None, tail[: len(n)], n).ratio()
        if ratio > best_ratio:
            best, best_ratio = n, ratio
    if best is not None and best_ratio >= FUZZY_THRESHOLD:
        return best, True
    return None


def extract_convicted_crimes(
    quyet_dinh: str, name_index: Dict[str, int], with_fuzzy_flag: bool = False
):
    """Return BLHS 2015 article numbers of the crimes the verdict convicts
    for, matched by name with the longest name winning (so "tàng trữ vận
    chuyển mua bán hoặc chiếm đoạt tiền chất ..." is not read as "tàng trữ
    trái phép chất ma túy").

    Only the conviction sentence ("tuyên bố bị cáo ... phạm tội ...") is
    read; a bare "phạm tội" is used only when no such sentence exists. This
    keeps out crimes mentioned for prior convictions or sentence merging.

    With ``with_fuzzy_flag`` returns (articles, used_fuzzy_match)."""
    text = _norm_name(quyet_dinh)
    names = sorted(name_index, key=len, reverse=True)
    sentences = list(SENTENCE_RE.finditer(text))
    if sentences:
        # One sentence can convict several defendants: "tuyên bố bị cáo A
        # phạm tội X bị cáo B phạm tội Y". Read every "bị cáo ... phạm tội"
        # from the first conviction sentence on ("về tội" for prior
        # convictions does not match).
        starts = [m.end() for m in sentences]
        starts += [m.end() for m in CO_DEFENDANT_RE.finditer(text, sentences[0].end())]
        starts = sorted(set(starts))
    else:
        starts = [m.end() for m in FALLBACK_RE.finditer(text)]

    found: List[int] = []
    used_fuzzy = False
    for start in starts:
        tail = text[start:]
        # Several crimes can be listed: "phạm tội A và tội B".
        while tail:
            tail = tail.removeprefix("tội ")
            hit = _match_name(tail, names)
            if hit is None:
                break
            name, fuzzy = hit
            used_fuzzy = used_fuzzy or fuzzy
            article = name_index[name]
            if article not in found:
                found.append(article)
            tail = tail[len(name):].lstrip()
            if tail.startswith("và tội "):
                tail = tail[len("và "):]
            else:
                break
    return (found, used_fuzzy) if with_fuzzy_flag else found


CITED_ARTICLE_RE = re.compile(r"điều (\d+)")


def extract_cited_crimes(
    quyet_dinh: str, judgment: str, laws: Dict[int, LawEntry]
) -> List[int]:
    """Fallback for verdicts with no "phạm tội <name>" sentence (e.g. "căn
    cứ điểm c khoản 1 điều 249 ... xử phạt bị cáo").

    Takes the article numbers cited in the verdict and keeps a crime article
    only if its BLHS 2015 name (or an alias) also appears somewhere in the
    judgment. That cross-check rejects BLHS 1999 numbers: a drug case citing
    1999's "điều 194" does not mention 2015's Điều 194 crime name."""
    names_by_article: Dict[int, List[str]] = {}
    for law in laws.values():
        if law.is_crime:
            names_by_article.setdefault(int(law.dieu), []).append(
                _norm_name(law.title[len("tội "):])
            )
    for alias, article in CRIME_NAME_ALIASES.items():
        names_by_article.setdefault(article, []).append(_norm_name(alias))

    full = _norm_name(judgment)
    found: List[int] = []
    for m in CITED_ARTICLE_RE.finditer(_norm_name(quyet_dinh)):
        article = int(m.group(1))
        if article in found or article not in names_by_article:
            continue
        if any(name in full for name in names_by_article[article]):
            found.append(article)
    return found


def label_case(detail: str, laws: Dict[int, LawEntry], name_index: Dict[str, int]) -> Dict:
    """Crime labels for one judgment, with how they were obtained:
    "ten_toi" (conviction sentence), "ten_toi_gan_dung" (fuzzy), "dieu_luat"
    (cited article confirmed by name), or None."""
    sections = split_sections(detail)
    qd = sections["quyet_dinh"] or ""
    crimes, fuzzy = extract_convicted_crimes(qd, name_index, with_fuzzy_flag=True)
    if crimes:
        method = "ten_toi_gan_dung" if fuzzy else "ten_toi"
    else:
        crimes = extract_cited_crimes(qd, detail, laws) if qd else []
        method = "dieu_luat" if crimes else None
    return {"sections": sections, "toi_danh": crimes, "cach_gan_nhan": method}


def judgment_year(detail: str) -> Optional[int]:
    head = detail[:400]
    m = re.search(r"bản án số \S+ (20\d\d)", head) or re.search(r"ngày \d+ \d+ (20\d\d)", head)
    return int(m.group(1)) if m else None
