"""The original judgment pipeline (core/utils/util.py::analyze_case) on a
Vietnamese case: per defendant, Researcher -> Auditor -> Adjudicator.

    0. defendants: LIST_DEFENDANTS_PROMPT (added: CAIL ships the names),
       then CASE_SEG_PROMPT per defendant (segment_case_text_withname)
    1. Researcher: features (query prompt, no crime hint) -> embed ->
       two-stage case retrieval with LLM rerank (top_retrieve +
       direct_retrieve + rerank, <=3 cases) -> laws of those cases, plus
       retrieve_law (LLM names crimes -> Crime -> Law)
    2. Auditor: judge_law on every candidate; rejected ones are dropped
       (hard filter, as the original); filter_facts keeps the retrieved
       cases that share an accepted article
    3. Adjudicator: JUDGE_CRIME_ALL_PROMPT over the accepted articles ->
       crimes, articles, penalty. As in the original, the retrieved cases
       are passed along but not shown to the LLM (judge_crime_all ignores
       its retrieved_facts argument).

Candidates are BLHS crime articles (Law nodes with Crime edges), the only
ones the original judges.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional

from ..cases.features import build_prompt, concat_feature_description, parse_features, sanitize_features
from ..prompts.vi import (
    CASE_SEG_PROMPT,
    JUDGE_CRIME_ALL_INPUT_TEMPLATE,
    JUDGE_CRIME_ALL_PROMPT,
    LIST_DEFENDANTS_PROMPT,
)
from ..qa.pipeline import DEFAULTS as QA_DEFAULTS
from ..qa.pipeline import judge_candidate, llm_crime_laws, parse_crime_list, reranked_cases

Generate = Callable[..., str]
Embed = Callable[[str], object]

DEFAULTS = {
    **QA_DEFAULTS,
    "fact_chars": 4000,  # the original cuts at 1024 Chinese characters
    "max_defendants": 4,
    "judge_mode": "gop",
    # "loc": the original hard filter (rejected articles are dropped).
    # "goi-y": every candidate goes to the Adjudicator with the Auditor's
    # verdict as a hint (the trial run: judge_law rejected the right article
    # in 4/22 cases, then the Adjudicator never saw it).
    "auditor": "loc",
}

PENALTY_KEYS = ("tu_hinh", "tu_co_thoi_han_thang", "chung_than")


def list_defendants(generate: Generate, fact: str, limit: int) -> List[str]:
    names = parse_crime_list(generate(LIST_DEFENDANTS_PROMPT.format(fact=fact), max_tokens=128))
    out: List[str] = []
    for n in names:
        n = n.strip().lower()
        if n and n not in out:
            out.append(n)
    return out[:limit]


def segment(generate: Generate, fact: str, name: str) -> str:
    text = generate(CASE_SEG_PROMPT.format(fact=fact, name=name), max_tokens=1024).strip()
    return text or fact


def is_crime_law(g, law_node: str) -> bool:
    d = g.node(law_node)
    return d.get("bo_luat", "BLHS") == "BLHS" and bool(g.neighbors(law_node, "RELATED_CRIME"))


def format_laws(g, law_nodes: List[str], verdicts: Optional[Dict[str, bool]] = None) -> str:
    """Original format_law: article, the crimes it defines, its text; with
    verdicts, the Auditor's result as a hint."""
    parts = []
    for n in law_nodes:
        d = g.node(n)
        crimes = [g.node(c)["description"] for c in g.neighbors(n, "RELATED_CRIME")]
        hint = ""
        if verdicts is not None:
            hint = f" (kiểm tra yếu tố cấu thành, chỉ tham khảo: {'thỏa mãn' if verdicts.get(n) else 'không thỏa mãn'})"
        parts.append(f"Điều {d['entry']}{d.get('suffix', '') or ''} Bộ luật Hình sự{hint}, tội danh: {', '.join(crimes)}. "
                     f"Nội dung: {d['description']}\n---")
    return "\n".join(parts)


def parse_adjudication(text: str) -> Optional[Dict]:
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last < first:
        return None
    try:
        data = json.loads(text[first : last + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    toi = data.get("toi_danh") or []
    dieu = data.get("dieu_luat") or []
    data["toi_danh"] = [toi] if isinstance(toi, str) else [str(x) for x in toi]
    data["dieu_luat"] = [dieu] if isinstance(dieu, str) else [str(x) for x in dieu]
    data["hinh_phat"] = data.get("hinh_phat") if isinstance(data.get("hinh_phat"), dict) else {}
    return data


ARTICLE_NUM_RE = re.compile(r"(?:Điều|điều)\s*(\d+[a-zđ]?)")


def article_numbers(texts: List[str]) -> List[str]:
    """["Điều 173 khoản 1", "điều 51"] -> ["173", "51"]; bare numbers too."""
    out: List[str] = []
    for t in texts:
        found = ARTICLE_NUM_RE.findall(t) or re.findall(r"^\s*(\d+[a-zđ]?)\b", t)
        for a in found:
            if a not in out:
                out.append(a)
    return out


def filter_facts(g, accepted: List[str], cases: List[str]) -> List[str]:
    """Original filter_facts: retrieved cases citing an accepted article."""
    accepted = set(accepted)
    return [c for c in cases if accepted & set(g.neighbors(c, "RELATES_TO_LAW"))]


def analyze_defendant(g, name: str, description: str, generate: Generate, embed: Embed, cfg: Dict) -> Dict:
    features = parse_features(generate(build_prompt(description)))
    features = sanitize_features(features) if features else None
    query_text = concat_feature_description(features) if features else description
    query_vec = embed(query_text)

    # Researcher
    rerank = reranked_cases(g, query_vec, query_text, generate, cfg)
    case_laws: List[str] = []
    for case in rerank["an"]:
        for law in g.neighbors(case, "RELATES_TO_LAW"):
            if law not in case_laws:
                case_laws.append(law)
    augment, crime_hits = llm_crime_laws(g, description[: cfg["fact_chars"]], generate, embed)
    candidates = [n for n in dict.fromkeys(case_laws + augment) if is_crime_law(g, n)]

    # Auditor
    case_desc = f"Bị cáo: {name}. Diễn biến: {description}"
    judgments = {n: judge_candidate(g, n, case_desc, generate, cfg["judge_mode"]) for n in candidates}
    accepted = [n for n in candidates if judgments[n]["ap_dung"]]
    used_cases = filter_facts(g, accepted, rerank["an"])

    # Adjudicator
    if cfg["auditor"] == "goi-y":
        ordered = accepted + [n for n in candidates if n not in accepted]
        laws_text = format_laws(g, ordered, {n: judgments[n]["ap_dung"] for n in candidates})
    else:
        laws_text = format_laws(g, accepted)
    raw = generate(JUDGE_CRIME_ALL_PROMPT + JUDGE_CRIME_ALL_INPUT_TEMPLATE.format(law=laws_text, case=case_desc), max_tokens=1024)
    verdict = parse_adjudication(raw)
    label = lambda n: f"{g.node(n)['entry']}{g.node(n).get('suffix', '') or ''}"
    return {
        "bi_cao": name,
        "dien_bien_rieng": description,
        "dac_trung": query_text,
        "researcher": {"cum": rerank["cum"], "an": rerank["an"], "dieu_tu_an": [label(n) for n in case_laws],
                       "llm_toi_danh": crime_hits, "dieu_tu_toi_danh": [label(n) for n in augment],
                       "ung_vien": [label(n) for n in candidates]},
        "auditor": {"judge": {label(n): v for n, v in judgments.items()}, "chap_nhan": [label(n) for n in accepted],
                    "an_giu_lai": used_cases},
        "adjudicator": verdict or {"toi_danh": [], "dieu_luat": [], "hinh_phat": {}},
        "adjudicator_loi": verdict is None,
    }


def analyze_case(g, fact: str, generate: Generate, embed: Embed, cfg: Optional[Dict] = None) -> Dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    fact = fact[: cfg["fact_chars"]]
    names = list_defendants(generate, fact, cfg["max_defendants"])
    if names:
        parts = [(n, segment(generate, fact, n)) for n in names]
    else:
        parts = [("bị cáo", fact)]
    per = [analyze_defendant(g, n, d, generate, embed, cfg) for n, d in parts]
    return {"bi_cao": names, "theo_bi_cao": per, **union_prediction(per)}


def union_prediction(per_defendant: List[Dict]) -> Dict:
    """Judgment-level prediction (labels are per judgment): union of the
    defendants' crimes and articles, the longest prison term."""
    crimes: List[str] = []
    articles: List[str] = []
    months = 0
    for p in per_defendant:
        v = p["adjudicator"]
        for c in v.get("toi_danh", []):
            if c not in crimes:
                crimes.append(c)
        for a in article_numbers(v.get("dieu_luat", [])):
            if a not in articles:
                articles.append(a)
        try:
            months = max(months, int(v.get("hinh_phat", {}).get("tu_co_thoi_han_thang") or 0))
        except (TypeError, ValueError):
            pass
    return {"du_doan_toi_danh": crimes, "du_doan_dieu": articles, "du_doan_tu_thang": months}


def adjudicate_without_graph(generate: Generate, fact: str) -> Dict:
    """Baseline: the Adjudicator alone, no retrieval and no candidate
    articles (what the LLM knows by itself)."""
    raw = generate(JUDGE_CRIME_ALL_PROMPT + JUDGE_CRIME_ALL_INPUT_TEMPLATE.format(law="(không có)", case=fact), max_tokens=1024)
    v = parse_adjudication(raw) or {"toi_danh": [], "dieu_luat": [], "hinh_phat": {}}
    return {"adjudicator": v, **union_prediction([{"adjudicator": v}])}


# ---- Scoring, as evaluation/evaluate_results.py: exact-match accuracy and
# micro-F1 for charges and articles, per judgment ---------------------------

def score_case(pred: Dict, gold_articles: List[str], gold_crimes: List[str], crime_articles: set) -> Dict:
    from ..qa.scoring import same_crime

    pa = [a for a in pred.get("du_doan_dieu", []) if a in crime_articles]  # general-part articles do not count
    ga = [str(a) for a in gold_articles]
    pc = pred.get("du_doan_toi_danh", [])
    crime_tp = sum(any(same_crime(p, g) for g in gold_crimes) for p in pc)
    crime_hit_gold = sum(any(same_crime(g, p) for p in pc) for g in gold_crimes)
    return {
        "dieu_tp": len(set(pa) & set(ga)), "dieu_fp": len(set(pa) - set(ga)), "dieu_fn": len(set(ga) - set(pa)),
        "dieu_dung_het": set(pa) == set(ga),
        "toi_tp": crime_tp, "toi_fp": len(pc) - crime_tp, "toi_fn": len(gold_crimes) - crime_hit_gold,
        "toi_dung_het": crime_tp == len(pc) and crime_hit_gold == len(gold_crimes) and bool(pc),
    }


def summarize(scores: List[Dict]) -> Dict:
    def f1(prefix):
        tp = sum(s[f"{prefix}_tp"] for s in scores)
        fp = sum(s[f"{prefix}_fp"] for s in scores)
        fn = sum(s[f"{prefix}_fn"] for s in scores)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    n = len(scores) or 1
    return {
        "so_an": len(scores),
        "toi_danh_accuracy": sum(s["toi_dung_het"] for s in scores) / n,
        "toi_danh_micro_f1": f1("toi"),
        "dieu_accuracy": sum(s["dieu_dung_het"] for s in scores) / n,
        "dieu_micro_f1": f1("dieu"),
    }
