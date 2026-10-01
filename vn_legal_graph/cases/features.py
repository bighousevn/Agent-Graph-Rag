"""Phase 3, step 1: extract the 4 feature groups of each case with an LLM.

Port of the original repo's feature step:
- corpus cases: ``scripts/prepare_case_features.py`` (prompt includes the
  charge, keywords centred on it);
- query cases: ``core/preprocess/get_features.py`` (facts only).

``concat_feature_description`` turns the features into the text that is
embedded for a Case node AND for a query, so both sides of the similarity
search use one format. (The original uses English labels for nodes and
Chinese labels for queries; here it is one function.)

Callers that touch a real LLM are run by the user, not by Claude: the API
key stays in the user's .env.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from ..prompts.vi import (
    GET_CASE_FEATURES_INPUT,
    GET_CASE_FEATURES_INPUT_WITH_CRIME,
    GET_CASE_FEATURES_PROMPT,
)

FEATURE_KEYS = ("defendant_info", "criminal_acts", "victim_property_details", "intent_remorse")

FEATURE_LABELS = {
    "defendant_info": "Nhân thân bị cáo",
    "criminal_acts": "Hành vi phạm tội",
    "victim_property_details": "Đối tượng/tài sản",
    "intent_remorse": "Lỗi và thái độ",
}

# Input facts are cut to this many characters. Median ViCSR fact text is
# ~3,300 chars, so most cases are sent whole; the cap bounds cost on the
# few very long ones (max ~25,000 chars).
DEFAULT_MAX_CHARS = 6000


def build_prompt(fact: str, crime_names: Optional[Sequence[str]] = None, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    fact = fact[:max_chars]
    if crime_names:
        tail = GET_CASE_FEATURES_INPUT_WITH_CRIME.format(crime="; ".join(crime_names), fact=fact)
    else:
        tail = GET_CASE_FEATURES_INPUT.format(fact=fact)
    return GET_CASE_FEATURES_PROMPT + "\n" + tail


def parse_features(raw: str) -> Optional[Dict[str, List[str]]]:
    """Parse the LLM's JSON answer. Returns None when no valid JSON object
    is found, so callers can count failures instead of silently storing
    empty features."""
    first, last = raw.find("{"), raw.rfind("}")
    if first == -1 or last < first:
        return None
    try:
        data = json.loads(raw[first : last + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for key in FEATURE_KEYS:
        value = data.get(key, [])
        out[key] = [str(x).strip() for x in value if str(x).strip()] if isinstance(value, list) else []
    return out


def concat_feature_description(features: Dict[str, List[str]]) -> str:
    """Text embedded for a Case node and for a query. Empty groups are left
    out, as in the original."""
    parts = [
        f"{FEATURE_LABELS[k]}: {', '.join(features[k])}."
        for k in FEATURE_KEYS
        if features.get(k)
    ]
    return " ".join(parts)


def annotate_cases(
    cases: Iterable[Dict],
    generate: Callable[[str], str],
    use_crime_hint: bool,
    max_chars: int = DEFAULT_MAX_CHARS,
    on_progress: Optional[Callable[[int], None]] = None,
) -> List[Dict]:
    """Add ``dac_trung`` (features), ``mo_ta_dac_trung`` (embedded text)
    and ``dac_trung_loi`` (True when the answer could not be parsed).

    The crime hint is only ever given for corpus cases; a test case gets
    facts only, whatever ``use_crime_hint`` says."""
    out = []
    for i, case in enumerate(cases):
        hint = case.get("toi_danh") if use_crime_hint and case.get("vai_tro") == "corpus" else None
        features = parse_features(generate(build_prompt(case["dien_bien"], hint, max_chars)))
        row = dict(case)
        row["dac_trung"] = features or {k: [] for k in FEATURE_KEYS}
        row["dac_trung_loi"] = features is None
        row["mo_ta_dac_trung"] = concat_feature_description(row["dac_trung"])
        out.append(row)
        if on_progress:
            on_progress(i + 1)
    return out


# ---- audit ---------------------------------------------------------------
# Checks for the hallucinations seen in the trial runs: features the fact
# text does not support. They are heuristics (keyword presence), meant to
# flag cases for a human look, not to prove an error.

_REMORSE_RE = re.compile(r"thành khẩn|khai nhận|khai báo|ăn năn|thừa nhận|tự thú|đầu thú|khắc phục|bồi thường")
_VALUE_IN_TEXT_RE = re.compile(r"trị giá|giá trị|định giá|\d+ ?(?:000|triệu)")
_VALUE_IN_FEATURE_RE = re.compile(r"trị giá|triệu đồng|nghìn đồng")
_GENERIC_VALUE_RE = re.compile(r"giá trị (?:lớn|nhỏ|rất lớn|đặc biệt lớn)")
_PERSONAL_RE = re.compile(r"sinh năm|\bvợ\b|\bchồng\b|con của|em gái|em trai|anh trai|chị gái|\bbố\b|\bmẹ\b")


def audit_features(row: Dict) -> List[str]:
    """Flags for one case: which feature groups look unsupported."""
    f, text = row["dac_trung"], row["dien_bien"]
    flags = []
    if any(_REMORSE_RE.search(x) for x in f["intent_remorse"]) and not _REMORSE_RE.search(text):
        flags.append("thai_do_khong_co_trong_van_ban")
    if any(_GENERIC_VALUE_RE.search(x) for x in f["victim_property_details"]):
        flags.append("gia_tri_chung_chung")
    if any(_VALUE_IN_FEATURE_RE.search(x) for x in f["victim_property_details"]) and not _VALUE_IN_TEXT_RE.search(text):
        flags.append("gia_tri_khong_co_trong_van_ban")
    if any(_PERSONAL_RE.search(x) for x in f["defendant_info"]):
        flags.append("chi_tiet_ca_nhan")
    return flags
