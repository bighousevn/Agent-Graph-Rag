"""Legal Q&A on top of the HierarGraph: questions, citations and scoring.

The pilot uses data/raw/questions.xlsx (luatvietnam.vn Q&A: title, question,
lawyer's answer) and hand-checked reference labels in
data/qa/pilot_gold.json. Scoring follows the user's criteria: right
articles (and khoản/điểm), right crimes/laws, right conclusion, short.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Sequence

LAW_ALIASES = {
    "blhs": "BLHS",
    "bộ luật hình sự": "BLHS",
    "bltths": "BLTTHS",
    "bộ luật tố tụng hình sự": "BLTTHS",
}


def load_questions(xlsx_path: str) -> List[Dict]:
    import openpyxl

    rows = list(openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True).worksheets[0].iter_rows(values_only=True))
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[1:] if any(r)]


def question_text(q: Dict) -> str:
    """What the system sees: title + question. Never the answer."""
    return f"{q['title']}\n{q['question']}".strip()


def norm_law(name: str) -> str:
    key = (name or "").strip().lower()
    return LAW_ALIASES.get(key, name.strip().upper() if name else "")


def norm_citation(c: Dict) -> Dict[str, str]:
    return {
        "luat": norm_law(c.get("luat", "")),
        "dieu": str(c.get("dieu", "")).strip().lower(),
        "khoan": str(c.get("khoan", "") or "").strip(),
        "diem": str(c.get("diem", "") or "").strip().lower(),
    }


def article_key(c: Dict) -> tuple:
    c = norm_citation(c)
    return (c["luat"], c["dieu"])


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"^tội\s+", "", s.strip())
    return re.sub(r"[^\w\s]", " ", re.sub(r"\s+", " ", s)).split()


def same_crime(a: str, b: str) -> bool:
    """Crime names match if one's words start the other's (the LLM may
    shorten "Tội cố ý gây thương tích hoặc gây tổn hại ..." )."""
    wa, wb = _norm_name(a), _norm_name(b)
    n = min(len(wa), len(wb))
    return n >= 2 and wa[:n] == wb[:n]


def score(prediction: Dict, gold: Dict) -> Dict:
    """Automatic part of the scoring (the conclusion is graded separately).

    - articles: recall over the required articles, precision over required
      + optional ones (an optional article cited is not an error).
    - khoan_diem: share of required citations whose khoản/điểm, when gold
      gives them, are matched exactly.
    - crimes: recall/precision of crime names (optional crimes allowed).
    """
    pred = [norm_citation(c) for c in prediction.get("dieu_luat", [])]
    pred_keys = {(c["luat"], c["dieu"]) for c in pred}
    req = [norm_citation(c) for c in gold["dieu_luat_bat_buoc"]]
    opt = [norm_citation(c) for c in gold.get("dieu_luat_tuy_chon", [])]
    req_keys = {(c["luat"], c["dieu"]) for c in req}
    allowed = req_keys | {(c["luat"], c["dieu"]) for c in opt}

    article_recall = len(req_keys & pred_keys) / len(req_keys) if req_keys else 1.0
    article_precision = len(pred_keys & allowed) / len(pred_keys) if pred_keys else 0.0

    detail_hits, detail_total = 0, 0
    for g in req:
        if not g["khoan"]:
            continue
        detail_total += 1
        if any(
            (p["luat"], p["dieu"]) == (g["luat"], g["dieu"])
            and p["khoan"] == g["khoan"]
            and (not g["diem"] or p["diem"] == g["diem"])
            for p in pred
        ):
            detail_hits += 1

    gold_crimes = gold.get("toi_danh", [])
    ok_crimes = gold_crimes + gold.get("toi_danh_tuy_chon", [])
    pred_crimes = prediction.get("toi_danh", [])
    crime_recall = (
        sum(any(same_crime(g, p) for p in pred_crimes) for g in gold_crimes) / len(gold_crimes)
        if gold_crimes else (1.0 if not pred_crimes else 0.0)
    )
    crime_precision = (
        sum(any(same_crime(p, g) for g in ok_crimes) for p in pred_crimes) / len(pred_crimes)
        if pred_crimes else (1.0 if not gold_crimes else 0.0)
    )
    return {
        "dieu_recall": article_recall,
        "dieu_precision": article_precision,
        "khoan_diem": detail_hits / detail_total if detail_total else None,
        "toi_danh_recall": crime_recall,
        "toi_danh_precision": crime_precision,
        "so_tu": len(prediction.get("cau_tra_loi", "").split()),
    }


GRADES = ("dung", "mot_phan", "sai")


def parse_grade(text: str) -> Optional[Dict]:
    import json

    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last < first:
        return None
    try:
        data = json.loads(text[first : last + 1])
    except json.JSONDecodeError:
        return None
    if data.get("diem") not in GRADES:
        return None
    return {"diem": data["diem"], "ly_do": str(data.get("ly_do", ""))}


def references_text(answer: str) -> str:
    """The lawyer's answer without the luatvietnam boilerplate tail."""
    cut = re.split(r"\nXem thêm:|\nTrên đây là nội dung tư vấn", answer)[0]
    return cut.strip()


def merge_citations(cits: Sequence[Dict]) -> List[Dict]:
    seen, out = set(), []
    for c in cits:
        n = norm_citation(c)
        key = tuple(n.values())
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out
