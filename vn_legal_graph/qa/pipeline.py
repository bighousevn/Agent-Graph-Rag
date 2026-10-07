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
       e. guidance: question vs guidance units (Công văn, Nghị quyết HĐTP items in
          related_laws) -> the articles they are attached to  (added: "Như thế nào là
                                                                 lập công chuộc tội?" is answered by a
                                                                 Nghị quyết item, not by Điều 51's text)
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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..cases.features import build_prompt, concat_feature_description, parse_features, sanitize_features
from ..graph.graph_db import HierarGraph
from ..judge.judge_law import judge_law, parse_bool
from ..prompts.vi import (
    JUDGE_LAW_SIMPLE_PROMPT,
    QA_ANSWER_PROMPT,
    RERANK_CASES_PROMPT,
    RERANK_CLUSTERS_PROMPT,
    RETRIEVE_LAW_PROMPT,
)
from ..retrieval.search import cluster_cases, direct_cases

Generate = Callable[..., str]
Embed = Callable[[str], object]

DEFAULTS = {
    "top_k_cases": 5,
    "n_clusters": 2,
    "top_k_laws_text": 5,
    "top_k_guidance": 3,
    "max_candidates": 8,
    "judge_mode": "gop",
    "law_text_chars": 4000,
    "guidance_chars": 4000,
    # Original two-stage case retrieval: LLM reranks the top-5 clusters (keep
    # n_clusters), then the merged direct + cluster cases (keep 3).
    "rerank": False,
    "rerank_clusters_from": 5,
    "rerank_cases_keep": 3,
    # True = the original: answer only from articles judge_law accepted.
    # False (default for Q&A) = every candidate, with the judge verdict shown
    # as a hint; run 1 lost the right article for hypothetical questions.
    "loc_theo_judge": False,
}


# Display names for codes that are not obvious to the LLM.
LAW_NAMES = {"XLVPHC": "Luật Xử lý vi phạm hành chính", "ND282": "Nghị định 282/2025/NĐ-CP"}


def law_label(g: HierarGraph, law_node: str) -> str:
    d = g.node(law_node)
    code = d.get("bo_luat", "BLHS")
    return f"{LAW_NAMES.get(code, code)} Điều {d['entry']}{d.get('suffix', '') or ''}"


def guidance_items(g: HierarGraph, law_node: str) -> List[Dict]:
    return [r for r in g.node(law_node).get("related_laws") or [] if isinstance(r, dict) and r.get("loai") == "van_ban_huong_dan"]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


def guidance_index(g: HierarGraph, embed: Embed) -> List[Tuple[Dict, List[str], np.ndarray]]:
    """Every guidance unit once, with the Law nodes it is attached to.
    Kept on the graph object; embeddings come from the embedder's cache."""
    cached = getattr(g, "_guidance_index", None)
    if cached is not None:
        return cached
    by_key: Dict[Tuple[str, str], Tuple[Dict, List[str]]] = {}
    for law in g.nodes_of("Law"):
        for item in guidance_items(g, law):
            key = (item.get("id", ""), item.get("text", ""))
            by_key.setdefault(key, (item, []))[1].append(law)
    index = [(item, laws, _unit(embed(item.get("text", "")))) for item, laws in by_key.values()]
    g._guidance_index = index
    return index


def guidance_laws(g: HierarGraph, query_vec, embed: Embed, top_k: int) -> List[str]:
    index = guidance_index(g, embed)
    if not index:
        return []
    q = _unit(query_vec)
    ranked = sorted(index, key=lambda x: -float(x[2] @ q))[:top_k]
    out: List[str] = []
    for _, laws, _ in ranked:
        for law in laws:
            if law not in out:
                out.append(law)
    return out


def pick_guidance(items: List[Dict], query_vec, embed: Embed, budget: int) -> str:
    """The guidance units closest to the question that fit in budget
    chars, in that order. Điều 51 alone has 40 units (Nghị quyết
    04/2025); cutting the concatenation lost the one that answers."""
    if query_vec is None or embed is None:
        ranked = items
    else:
        q = _unit(query_vec)
        ranked = sorted(items, key=lambda it: -float(_unit(embed(it.get("text", ""))) @ q))
    parts, used = [], 0
    for it in ranked:
        part = f"{it.get('id', '')}: {it.get('text', '')}"
        if used and used + len(part) > budget:
            continue
        parts.append(part[: budget - used])
        used += len(parts[-1]) + 3
        if used >= budget:
            break
    return " | ".join(parts)


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


def parse_rank(text: str) -> List[int]:
    """[3, 1, 2] or "rank: [3,1,2]" -> [3, 1, 2]; [] if unreadable."""
    first, last = text.find("["), text.rfind("]")
    if first == -1 or last < first:
        return []
    out: List[int] = []
    for tok in text[first + 1 : last].replace(" ", "").split(","):
        if tok.isdigit() and int(tok) not in out:
            out.append(int(tok))
    return out


def _apply_rank(items: List, rank: List[int], keep: int) -> List:
    """Items in the LLM's order (1-based numbers), unknown numbers ignored,
    unranked items after; first ``keep``."""
    picked = [items[i - 1] for i in rank if 1 <= i <= len(items)]
    picked += [x for x in items if x not in picked]
    return picked[:keep]


def llm_crime_laws(g: HierarGraph, text: str, generate: Generate, embed: Embed) -> Tuple[List[str], List[Dict]]:
    """The original retrieve_law: the LLM names <=3 crimes, each goes to
    its nearest Crime node, then back to its Law nodes."""
    crimes = parse_crime_list(generate(RETRIEVE_LAW_PROMPT.format(fact=text), max_tokens=256))
    laws: List[str] = []
    hits = []
    for name in crimes[:3]:
        hit = g.search(embed(name), "Crime", top_k=1)
        if not hit:
            continue
        crime_node, sim = hit[0]
        hits.append({"ten_llm": name, "crime_node": g.node(crime_node)["description"], "cosine": round(sim, 3)})
        for law in g.predecessors(crime_node, "RELATED_CRIME"):
            if law not in laws:
                laws.append(law)
    return laws, hits


def reranked_cases(g: HierarGraph, query_vec, query_text: str, generate: Generate, cfg: Dict) -> Dict:
    """The original top_retrieve + direct_retrieve + rerank."""
    clusters = [c for c, _ in g.search(query_vec, "Cluster", top_k=cfg["rerank_clusters_from"])]
    listing = "\n".join(f"{i}. {g.node(c).get('description', '')}" for i, c in enumerate(clusters, 1))
    rank = parse_rank(generate(RERANK_CLUSTERS_PROMPT.format(cluster_summaries=listing, query_text=query_text), max_tokens=64))
    kept_clusters = _apply_rank(clusters, rank, cfg["n_clusters"])
    pool: List[str] = []
    for c in kept_clusters:
        for case, _ in g.search(query_vec, "Case", top_k=cfg["top_k_cases"], among=g.predecessors(c, "BELONGS_TO")):
            if case not in pool:
                pool.append(case)
    for case, _ in direct_cases(g, query_vec, cfg["top_k_cases"]):
        if case not in pool:
            pool.append(case)
    listing = "\n".join(f"code{i}: {g.node(c).get('description', '')[:800]}" for i, c in enumerate(pool, 1))
    rank = parse_rank(generate(RERANK_CASES_PROMPT.format(neighbor_summaries=listing, query_text=query_text), max_tokens=64))
    kept = _apply_rank(pool, rank, cfg["rerank_cases_keep"])
    return {"cum": kept_clusters, "an_ung_vien": pool, "an": kept}


def retrieve_candidates(
    g: HierarGraph, question: str, query_vec, generate: Generate, embed: Embed, cfg: Dict, query_text: str = ""
) -> Dict:
    routes: Dict[str, List[str]] = {}
    rerank = None
    if cfg["rerank"]:
        rerank = reranked_cases(g, query_vec, query_text or question, generate, cfg)
        routes["an_tuong_tu"] = laws_of_cases(g, [(c, 0.0) for c in rerank["an"]])
    else:
        routes["truc_tiep"] = laws_of_cases(g, direct_cases(g, query_vec, cfg["top_k_cases"]))
        routes["qua_cum"] = laws_of_cases(g, cluster_cases(g, query_vec, cfg["n_clusters"], cfg["top_k_cases"]))

    routes["llm_doan_toi"], crime_hits = llm_crime_laws(g, question, generate, embed)
    question_vec = embed(question)
    routes["van_ban_luat"] = [n for n, _ in g.search(question_vec, "Law", top_k=cfg["top_k_laws_text"])]
    routes["huong_dan"] = guidance_laws(g, question_vec, embed, cfg["top_k_guidance"])

    order = ["llm_doan_toi", "huong_dan", "van_ban_luat", "an_tuong_tu", "truc_tiep", "qua_cum"]
    candidates: List[str] = []
    for r in order:
        for law in routes.get(r, []):
            if law not in candidates:
                candidates.append(law)
    return {
        "tuyen": routes,
        "rerank": rerank,
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
    g: HierarGraph, law_nodes: List[str], cfg: Dict, judgments: Optional[Dict[str, Dict]] = None,
    query_vec=None, embed: Optional[Embed] = None,
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
        guidance = pick_guidance(guidance_items(g, n), query_vec, embed, cfg["guidance_chars"])
        if guidance:
            block += f"\nHướng dẫn áp dụng: {guidance}"
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

    retrieval = retrieve_candidates(g, question, query_vec, generate, embed, cfg, description)
    judgments = {n: judge_candidate(g, n, question, generate, cfg["judge_mode"]) for n in retrieval["ung_vien"]}
    accepted = [n for n in retrieval["ung_vien"] if judgments[n]["ap_dung"]]
    fallback = not accepted
    if cfg["loc_theo_judge"]:
        used = accepted or retrieval["ung_vien"][:3]
    else:
        used = accepted + [n for n in retrieval["ung_vien"] if n not in accepted]

    raw = generate(
        QA_ANSWER_PROMPT.format(
            question=question, laws=render_laws_for_answer(g, used, cfg, judgments, embed(question), embed)
        ),
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
