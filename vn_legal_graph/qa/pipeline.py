"""Answer one legal question with the HierarGraph, following the original
pipeline (core/utils/util.py::analyze_case) as far as Q&A allows.

    1. features of the question (query prompt, no crime hint) -> embed
    2. candidate articles from four routes:
       a. direct: nearest Case nodes -> RELATES_TO_LAW          (original)
       b. cluster: nearest Clusters -> their Cases -> laws     (original; clusters by cosine, no LLM rerank)
       c. augment: LLM names <=3 crimes -> nearest Crime -> Law (original retrieve_law)
       d. law text: question vs Law node text                  (added: few cases exist for most
                                                                 crimes in the questions, and the
                                                                 general part has no Crime nodes)
    3. judge every candidate: judge_law with judge_dep when the article has
       it, else one applicability call (original JUDGE_LAW_PROMPT1)
    4. answer from the accepted articles (replaces judge_crime_all, which
       predicts charge and sentence, with a short answer plus the crimes and
       articles as structured fields)

Not done: splitting by defendant (a Q&A situation has no defendant list).
"""
from __future__ import annotations

import ast
import json
from typing import Callable, Dict, List, Optional

from ..cases.features import build_prompt, concat_feature_description, parse_features, sanitize_features
from ..graph.graph_db import HierarGraph
from ..judge.judge_law import judge_law, parse_bool, render_related
from ..prompts.vi import JUDGE_LAW_SIMPLE_PROMPT, QA_ANSWER_PROMPT, RETRIEVE_LAW_PROMPT
from ..retrieval.search import cluster_cases, direct_cases

Generate = Callable[..., str]
Embed = Callable[[str], object]

DEFAULTS = {
    "top_k_cases": 5,
    "n_clusters": 2,
    "top_k_laws_text": 5,
    "max_candidates": 8,
    "judge_mode": "gop",
    "law_text_chars": 4000,
    "guidance_chars": 4000,
    # True = the original: answer only from articles judge_law accepted.
    # False (default for Q&A) = every candidate, with the judge verdict shown
    # as a hint; run 1 lost the right article for hypothetical questions.
    "loc_theo_judge": False,
}


def law_label(g: HierarGraph, law_node: str) -> str:
    d = g.node(law_node)
    code = d.get("bo_luat", "BLHS")
    return f"{code} Điều {d['entry']}{d.get('suffix', '') or ''}"


def parse_crime_list(text: str) -> List[str]:
    first, last = text.find("["), text.rfind("]")
    if first == -1 or last < first:
        return []
    try:
        data = ast.literal_eval(text[first : last + 1])
    except (SyntaxError, ValueError):
        return []
    return [str(x).strip() for x in data if str(x).strip()] if isinstance(data, list) else []


def laws_of_cases(g: HierarGraph, ranked_cases) -> List[str]:
    out: List[str] = []
    for case, _ in ranked_cases:
        for law in g.neighbors(case, "RELATES_TO_LAW"):
            if law not in out:
                out.append(law)
    return out


def retrieve_candidates(
    g: HierarGraph, question: str, query_vec, generate: Generate, embed: Embed, cfg: Dict
) -> Dict:
    routes: Dict[str, List[str]] = {}
    routes["truc_tiep"] = laws_of_cases(g, direct_cases(g, query_vec, cfg["top_k_cases"]))
    routes["qua_cum"] = laws_of_cases(g, cluster_cases(g, query_vec, cfg["n_clusters"], cfg["top_k_cases"]))

    crimes = parse_crime_list(generate(RETRIEVE_LAW_PROMPT.format(fact=question), max_tokens=256))
    augment: List[str] = []
    crime_hits = []
    for name in crimes[:3]:
        hit = g.search(embed(name), "Crime", top_k=1)
        if not hit:
            continue
        crime_node, sim = hit[0]
        crime_hits.append({"ten_llm": name, "crime_node": g.node(crime_node)["description"], "cosine": round(sim, 3)})
        for law in g.predecessors(crime_node, "RELATED_CRIME"):
            if law not in augment:
                augment.append(law)
    routes["llm_doan_toi"] = augment
    routes["van_ban_luat"] = [n for n, _ in g.search(embed(question), "Law", top_k=cfg["top_k_laws_text"])]

    order = ["llm_doan_toi", "van_ban_luat", "truc_tiep", "qua_cum"]
    candidates: List[str] = []
    for r in order:
        for law in routes[r]:
            if law not in candidates:
                candidates.append(law)
    return {
        "tuyen": routes,
        "llm_toi_danh": crime_hits,
        "ung_vien": candidates[: cfg["max_candidates"]],
    }


def judge_candidate(g: HierarGraph, law_node: str, question: str, generate: Generate, mode: str) -> Dict:
    d = g.node(law_node)
    law = {
        "entry": d["entry"],
        "description": d["description"],
        "judge_dep": d.get("judge_dep") or [],
        "related_laws": d.get("related_laws") or [],
    }
    if law["judge_dep"]:
        out = judge_law(generate, question, law, mode)
        return {"cach": "judge_law", "ap_dung": out["ap_dung"], "so_dung": len(out["dung"]), "so_sai": len(out["sai"])}
    answer = generate(JUDGE_LAW_SIMPLE_PROMPT.format(law=d["description"], case=question), max_tokens=16)
    return {"cach": "don_gian", "ap_dung": parse_bool(answer) is True}


def render_laws_for_answer(
    g: HierarGraph, law_nodes: List[str], cfg: Dict, judgments: Optional[Dict[str, Dict]] = None
) -> str:
    parts = []
    for n in law_nodes:
        d = g.node(n)
        text = d["description"][: cfg["law_text_chars"]]
        block = f"[{law_label(g, n)}]"
        if judgments and n in judgments:
            verdict = "thỏa mãn" if judgments[n]["ap_dung"] else "không thỏa mãn"
            block += f" (kiểm tra yếu tố cấu thành, chỉ tham khảo: {verdict})"
        block += f"\n{text}"
        guidance = render_related([r for r in d.get("related_laws") or [] if isinstance(r, dict) and r.get("loai") == "van_ban_huong_dan"])
        if guidance:
            block += f"\nHướng dẫn áp dụng: {guidance[: cfg['guidance_chars']]}"
        parts.append(block)
    return "\n\n".join(parts)


def parse_answer(text: str) -> Optional[Dict]:
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last < first:
        return None
    try:
        data = json.loads(text[first : last + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "cau_tra_loi" not in data:
        return None
    data.setdefault("toi_danh", [])
    data.setdefault("dieu_luat", [])
    return data


def answer_question(
    g: HierarGraph, question: str, generate: Generate, embed: Embed, cfg: Optional[Dict] = None
) -> Dict:
    cfg = {**DEFAULTS, **(cfg or {})}
    features = parse_features(generate(build_prompt(question)))
    features = sanitize_features(features) if features else None
    description = concat_feature_description(features) if features else ""
    query_vec = embed(description or question)

    retrieval = retrieve_candidates(g, question, query_vec, generate, embed, cfg)
    judgments = {n: judge_candidate(g, n, question, generate, cfg["judge_mode"]) for n in retrieval["ung_vien"]}
    accepted = [n for n in retrieval["ung_vien"] if judgments[n]["ap_dung"]]
    fallback = not accepted
    if cfg["loc_theo_judge"]:
        used = accepted or retrieval["ung_vien"][:3]
    else:
        used = accepted + [n for n in retrieval["ung_vien"] if n not in accepted]

    raw = generate(
        QA_ANSWER_PROMPT.format(question=question, laws=render_laws_for_answer(g, used, cfg, judgments)),
        max_tokens=1024,
    )
    answer = parse_answer(raw)
    return {
        "dac_trung": description,
        "truy_xuat": {**retrieval, "ung_vien": [law_label(g, n) for n in retrieval["ung_vien"]],
                      "tuyen": {k: [law_label(g, n) for n in v] for k, v in retrieval["tuyen"].items()}},
        "judge": {law_label(g, n): v for n, v in judgments.items()},
        "dieu_dung_de_tra_loi": [law_label(g, n) for n in used],
        "khong_dieu_nao_duoc_chap_nhan": fallback,
        "tra_loi": answer or {"cau_tra_loi": "", "toi_danh": [], "dieu_luat": []},
        "tra_loi_loi": answer is None,
        "tra_loi_tho": raw if answer is None else None,
    }
