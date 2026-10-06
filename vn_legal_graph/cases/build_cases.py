"""Build the Phase 2 case files from ViCSR:

    data/processed/cases_vn.json       (vai_tro = "corpus", used to build the graph)
    data/processed/cases_vn_test.json  (vai_tro = "test", held out for Recall@k)

Steps:
1. Label every judgment in BLHS 2015 numbering (``vicsr.label_case``).
2. Drop exact duplicate judgments (ViCSR has 135 duplicate groups).
3. Keep judgments whose crimes all fall inside the chosen scope.
4. Take the facts (``dien_bien``) from the "nội dung vụ án" section, cut
   before the indictment / prosecutor part, and mask what still leaks the
   answer ("tội <tên tội>" -> "tội ××", "điều N" -> "điều ××"), as CAIL
   does. Behaviour descriptions ("có hành vi tàng trữ trái phép chất ma
   túy") are facts and stay.
5. Sample per crime and split corpus/test with no judgment in both.

Not done here, and why:
- Splitting a judgment per defendant needs an LLM (CASE_SEG_PROMPT); one
  judgment stays one record with all its crimes.
- Names: ViCSR is already published with given names cut to an initial
  ("phùng thanh h"), as courts do. No extra anonymisation yet.
- Penalty extraction (``hinh_phat``) is left as None for now.
"""
from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .vicsr import CRIME_NAME_ALIASES, LawEntry, _match_name, _norm_name

MIN_FACT_WORDS = 40

# Where the facts end and the indictment / prosecutor part begins.
PROSECUTION_RES = (
    re.compile(r"(?:tại |theo )?(?:bản )?cáo trạng số"),
    # "viện kiểm sát ... truy tố các bị cáo ... về tội ××", without "cáo trạng số"
    re.compile(r"(?:viện kiểm sát|vksnd)(?: \S+){0,10}? truy tố (?:\S+ ){0,12}?về (?:các )?tội"),
    re.compile(r"(?:tại phần tranh luận )?(?:đại diện viện kiểm sát|kiểm sát viên)"),
    re.compile(r"đề nghị hội đồng xét xử"),
)

ARTICLE_RE = re.compile(r"\bđiều \d+\w*")

# Leftover words when the cut lands just after "tại bản" / "theo" (also the
# typo "bán" for "bản").
TRAILING_CONNECTORS = {"tại", "theo", "bản", "bán", "và", "ngày"}

# Characters produced when TCVN3/ABC legacy-font text is decoded as Unicode
# ("héi ång xđt xö", "ngµy"). Measured on ViCSR: ~500 judgments exceed 0.1%.
LEGACY_FONT_CHARS = set("µþª¹®¬¸×ÐÞ¶¼½¾§¦¥¤£¢¡¿º©«»")
LEGACY_FONT_MAX_RATIO = 0.001


def legacy_font_ratio(text: str) -> float:
    return sum(ch in LEGACY_FONT_CHARS for ch in text) / max(1, len(text))


def dedupe(details: Dict[int, str]) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Keep the lowest id of each group of identical judgments.
    Returns (kept, dropped_id -> kept_id)."""
    first_by_hash: Dict[str, int] = {}
    kept: Dict[int, str] = {}
    dropped: Dict[int, int] = {}
    for case_id in sorted(details):
        text = details[case_id]
        key = hashlib.md5(re.sub(r"\s+", " ", text).encode("utf-8")).hexdigest()
        if key in first_by_hash:
            dropped[case_id] = first_by_hash[key]
        else:
            first_by_hash[key] = case_id
            kept[case_id] = text
    return kept, dropped


def cut_before_prosecution(text: str, min_words: int = MIN_FACT_WORDS) -> str:
    """Cut the fact text where the indictment / prosecutor part starts. A
    marker in the first ``min_words`` words is ignored (e.g. the opening
    "bị viện kiểm sát ... truy tố về hành vi phạm tội như sau")."""
    words_before = lambda pos: len(text[:pos].split())
    # Earliest marker over all patterns, not the first pattern that matches.
    starts = [
        m.start()
        for pattern in PROSECUTION_RES
        for m in pattern.finditer(text)
        if words_before(m.start()) >= min_words
    ]
    cut = text[: min(starts)] if starts else text
    words = cut.split()
    while words and words[-1] in TRAILING_CONNECTORS:
        words.pop()
    return " ".join(words)


def _crime_names(laws: Dict[int, LawEntry]) -> List[str]:
    names = {_norm_name(l.title[len("tội "):]) for l in laws.values() if l.is_crime}
    names |= {_norm_name(a) for a in CRIME_NAME_ALIASES}
    return sorted(names, key=len, reverse=True)


def mask_leaks(text: str, crime_names: Sequence[str]) -> str:
    """Replace "tội <crime name>" with "tội ××" and "điều N" with "điều ××".
    ``text`` must already be in ``_norm_name`` form."""
    pattern = re.compile(r"\btội (?:" + "|".join(re.escape(n) for n in crime_names) + r")\b")
    text = pattern.sub("tội ××", text)
    # OCR-damaged names ("tội ta ng trư trái phép chất ma túy") escape the
    # exact pattern; mask those through the fuzzy matcher used for labels.
    out, pos = [], 0
    for m in re.finditer(r"\btội (?!××)", text):
        if m.start() < pos:
            continue
        hit = _match_name(text[m.end():], list(crime_names))
        if hit is not None and hit[1]:
            end = m.end() + len(hit[0])
            out.append(text[pos : m.end()] + "××")
            pos = end
    out.append(text[pos:])
    text = "".join(out)
    return ARTICLE_RE.sub("điều ××", text)


def extract_facts(
    sections: Dict[str, Optional[str]], summary: Optional[str], crime_names: Sequence[str]
) -> Optional[Tuple[str, str]]:
    """Return (dien_bien, source) or None if no usable facts remain.
    Prefers the judgment's own "nội dung vụ án" section; falls back to the
    ViCSR summary."""
    for source, raw in (("noi_dung", sections.get("noi_dung")), ("tom_tat", summary)):
        if not raw or legacy_font_ratio(raw) > LEGACY_FONT_MAX_RATIO:
            continue
        facts = cut_before_prosecution(_norm_name(raw))
        if len(facts.split()) >= MIN_FACT_WORDS:
            return mask_leaks(facts, crime_names), source
    return None


def extract_dieu_khoan(quyet_dinh: str, articles: Iterable[int]) -> List[str]:
    """Cited "điểm x khoản y điều N" for the given articles, as "N.y.x" /
    "N.y" / "N". Only the first citation per article is kept."""
    text = _norm_name(quyet_dinh)
    out: List[str] = []
    for article in articles:
        m = re.search(rf"(?:điểm (\w) )?(?:khoản (\d+) )?điều {article}\b", text)
        if not m:
            continue
        label = str(article)
        if m.group(2):
            label += f".{m.group(2)}"
            if m.group(1):
                label += f".{m.group(1)}"
        out.append(label)
    return out


def split_by_crime(
    case_crimes: Dict,
    test_per_crime: int,
    corpus_max_per_crime: int,
    seed: int = 42,
    test_min_cases: int = 0,
) -> Tuple[List, List]:
    """Per-crime sampling, rarest crime first, so a multi-crime case counts
    toward its rarest crime. Each crime gets up to ``test_per_crime`` test
    cases and up to ``corpus_max_per_crime`` corpus cases. A case is used
    at most once. A crime with fewer than ``test_min_cases`` cases gets no
    test cases (they all go to the corpus: a rare crime is better in the
    graph than as two test cases). Returns (corpus_ids, test_ids)."""
    rng = random.Random(seed)
    by_crime: Dict[int, List[int]] = defaultdict(list)
    for case_id, crimes in sorted(case_crimes.items()):
        for c in crimes:
            by_crime[c].append(case_id)

    used: set = set()
    corpus: List[int] = []
    test: List[int] = []
    for crime in sorted(by_crime, key=lambda c: (len(by_crime[c]), c)):
        pool = [c for c in by_crime[crime] if c not in used]
        rng.shuffle(pool)
        n_test = test_per_crime if len(by_crime[crime]) >= test_min_cases else 0
        t = pool[:n_test]
        k = pool[n_test : n_test + corpus_max_per_crime]
        test += t
        corpus += k
        used.update(t)
        used.update(k)
    return sorted(corpus), sorted(test)
