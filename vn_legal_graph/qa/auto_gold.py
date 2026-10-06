"""Reference labels taken automatically from the lawyer's answer, so the Q&A
evaluation can run on 20-50 questions without labelling each by hand
(data/qa/pilot_gold.json stays the hand-checked set for the first 5).

- Citations: the BLHS/BLTTHS articles (with khoản/điểm when given) that the
  lawyer cites in their own sentences. Lines of quoted law ("Điều 155. ...",
  "1. ...", "a) ...") are skipped: they cross-reference other articles
  ("khoản 1 các điều 134, 135 ...") that are not part of the answer.
  A bare "khoản 2 Điều 321" counts only when the same article is cited with
  its code elsewhere in the answer; "Điều 1 Luật sửa đổi ..." never counts.
- Group: a question is "co_cong_van" when the answer cites a TANDTC letter
  that is in the graph, or the question repeats an item of one (word
  3-gram overlap >= 0.5; real matches are >= 0.77, the rest <= 0.35 on
  data/raw/questions.xlsx).

Crimes cannot be labelled reliably this way (an answer cites crime
articles to rule them out, too), so auto gold has no required crimes;
the crimes of the cited articles are only "allowed" (precision).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence

from .scoring import merge_citations

CODE_RE = r",?\s*(?:của\s+)?(Bộ luật Tố tụng hình sự|Bộ luật Hình sự|BLTTHS|BLHS)"
CODES = {"bộ luật tố tụng hình sự": "BLTTHS", "bltths": "BLTTHS", "bộ luật hình sự": "BLHS", "blhs": "BLHS"}
ART = r"[Đđ]iều\s+(\d+[a-zđ]?)"
DETAIL = r"(?:điểm\s+([a-zđ])(?:\s*,\s*[a-zđ])*\s+)?(?:khoản\s+(\d+)\s+)?"
DETAIL_NC = r"(?:điểm\s+[a-zđ](?:\s*,\s*[a-zđ])*\s+)?(?:khoản\s+\d+\s+)?"
# "Điều 142 và Điều 145 của Bộ luật Hình sự", "khoản 2 Điều 321 BLHS"
CODED_RE = re.compile(
    DETAIL + ART + r"((?:\s*(?:,|và|hoặc)\s*" + DETAIL_NC + r"(?:[Đđ]iều\s+)?\d+[a-zđ]?)*)" + CODE_RE,
    re.IGNORECASE,
)
BARE_RE = re.compile(DETAIL + ART + r"(?!\s*(?:của\s+)?(?:Luật|Nghị định|Nghị quyết|Thông tư|Bộ luật|BLHS|BLTTHS)\b)", re.IGNORECASE)
LIST_ITEM_RE = re.compile(DETAIL + r"(?:[Đđ]iều\s+)?(\d+[a-zđ]?)", re.IGNORECASE)
QUOTED_LINE_RE = re.compile(r"^\s*(?:[Đđ]iều\s+\d+[a-zđ]?\.|\d+\.|[a-zđ]\))", re.IGNORECASE)
LETTER_RE = re.compile(r"(\d+)/(?:\d{4}/)?TANDTC")
RESOLUTION_RE = re.compile(r"(\d+/\d{4}/NQ-HĐTP|\d+/VBHN-TANDTC)")
# Resolutions folded into a consolidated text that is in the graph.
RESOLUTION_ALIASES = {"02/2018/NQ-HĐTP": "02/VBHN-TANDTC", "01/2022/NQ-HĐTP": "02/VBHN-TANDTC"}


def cited_resolutions(answer: str, in_graph: Iterable[str]) -> List[str]:
    """Nghị quyết HĐTP the lawyer cites that the graph has (directly or
    through a consolidated text)."""
    have = set(in_graph)
    out = []
    for so in RESOLUTION_RE.findall(answer or ""):
        so = RESOLUTION_ALIASES.get(so, so)
        if so in have and so not in out:
            out.append(so)
    return out


def own_lines(answer: str) -> List[str]:
    return [l for l in (answer or "").splitlines() if l.strip() and not QUOTED_LINE_RE.match(l)]


def _cit(code: str, dieu: str, khoan: Optional[str], diem: Optional[str]) -> Dict[str, str]:
    return {"luat": code, "dieu": dieu.lower(), "khoan": khoan or "", "diem": (diem or "").lower()}


def extract_citations(answer: str) -> List[Dict[str, str]]:
    lines = own_lines(answer)
    cits: List[Dict[str, str]] = []
    for line in lines:
        for m in CODED_RE.finditer(line):
            code = CODES[m.group(5).lower()]
            cits.append(_cit(code, m.group(3), m.group(2), m.group(1)))
            for item in LIST_ITEM_RE.finditer(m.group(4) or ""):
                cits.append(_cit(code, item.group(3), item.group(2), item.group(1)))
    code_of = {}
    for c in cits:
        code_of.setdefault(c["dieu"], c["luat"])
    for line in lines:
        for m in BARE_RE.finditer(line):
            dieu = m.group(3).lower()
            if dieu in code_of:
                cits.append(_cit(code_of[dieu], dieu, m.group(2), m.group(1)))
    return merge_citations(cits)


def _grams(text: str, n: int = 3) -> set:
    w = re.findall(r"\w+", (text or "").lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def guidance_match(question: str, guidance_items: Sequence[Dict], threshold: float = 0.5) -> Optional[Dict]:
    """The guidance item the question repeats, with its overlap, if any."""
    q = _grams(question)
    if not q:
        return None
    best = max(((len(q & _grams(it["explain"])) / len(q), it) for it in guidance_items), key=lambda x: x[0], default=None)
    if best and best[0] >= threshold:
        return {"from": best[1]["from"], "overlap": round(best[0], 2)}
    return None


def build_gold(
    q: Dict,
    guidance_items: Sequence[Dict],
    letters: Iterable[str],
    crimes_of: Dict[str, List[str]],
    manual: Optional[Dict[str, str]] = None,
    resolutions: Iterable[str] = (),
) -> Dict:
    """crimes_of: "BLHS:173" -> crime names of that article (from the graph).
    manual: qa_number -> guidance item, for questions that reword an item
    (data/qa/cong_van_tay.json)."""
    cits = extract_citations(q.get("answer") or "")
    articles = merge_citations({"luat": c["luat"], "dieu": c["dieu"]} for c in cits)
    cited_letters = sorted(set(LETTER_RE.findall(q.get("answer") or "")) & set(letters))
    match = guidance_match(q.get("question") or "", guidance_items)
    if not match and manual and str(q["qa_number"]) in manual:
        match = {"from": manual[str(q["qa_number"])], "overlap": None}
    allowed_crimes: List[str] = []
    for c in articles:
        for name in crimes_of.get(f"{c['luat']}:{c['dieu']}", []):
            if name not in allowed_crimes:
                allowed_crimes.append(name)
    return {
        "qa_number": str(q["qa_number"]),
        "tu_dong": True,
        "dieu_luat_bat_buoc": articles,
        "dieu_luat_tuy_chon": [],
        "trich_dan_luat_su": cits,
        "toi_danh": [],
        "toi_danh_tuy_chon": allowed_crimes,
        "nhom": "co_cong_van" if cited_letters or match else "khong_cong_van",
        "cong_van_trich": cited_letters,
        "cong_van_trung": match,
        "nghi_quyet_trich": cited_resolutions(q.get("answer") or "", resolutions),
    }


def khoan_precision(prediction: Dict, gold: Dict) -> Optional[float]:
    """Share of predicted citations with a khoản whose (code, article,
    khoản[, điểm]) the lawyer also cites. None if the lawyer cites no khoản
    of those articles or the system gives none."""
    lawyer = gold.get("trich_dan_luat_su", [])
    with_k = {(c["luat"], c["dieu"]) for c in lawyer if c["khoan"]}
    from .scoring import norm_citation

    pred = [norm_citation(c) for c in prediction.get("dieu_luat", [])]
    pred = [p for p in pred if p["khoan"] and (p["luat"], p["dieu"]) in with_k]
    if not pred:
        return None
    hits = sum(
        any(
            (c["luat"], c["dieu"], c["khoan"]) == (p["luat"], p["dieu"], p["khoan"]) and (not c["diem"] or not p["diem"] or c["diem"] == p["diem"])
            for c in lawyer
        )
        for p in pred
    )
    return hits / len(pred)
