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
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..cases.features import build_prompt, concat_feature_description, parse_features, sanitize_features
from ..graph.graph_db import HierarGraph
from ..judge.judge_law import judge_law, parse_bool
from ..prompts.vi import (
    JUDGE_LAW_SIMPLE_PROMPT,
    LEGAL_ISSUES_PROMPT,
    QA_REWRITE_PROMPT,
    QA_ANSWER_NO_DOCS_PROMPT,
    QA_ANSWER_PROMPT,
    QA_ANSWER_TOO_SHORT_NOTE,
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
    # Route "van_de_phap_ly": the LLM names general legal issues, matched to
    # titles of BLHS general-part (Điều 1-107) and BLTTHS articles; up to
    # general_slots of them are added after the max_candidates others, so
    # they are not crowded out by crime articles.
    "general_slots": 4,
    "laws_per_issue": 2,
    "max_issues": 5,
    # Crime route: each crime the LLM names goes to its nearest Crime nodes
    # (the original takes 1; 2 also catches a near crime when the name is
    # off, e.g. "làm nhục" vs "đe dọa").
    "crime_nodes_per_guess": 2,
    # Q&A uses the law/crime graph only: on Kaggle run 3 the similar-case
    # route found 49 right articles, every one also found by another route
    # (0 found only by it), at 3 extra LLM calls per question. True brings
    # the case routes (and the feature extraction they need) back.
    "dung_an": False,
    # Restate the question in legal language first and search with that
    # (people's words rarely match the statute's).
    "dien_dat_lai": True,
    # Order of the articles in the answer prompt. False: accepted by judge
    # first, then retrieval order. True: accepted first, and within each
    # group the articles more routes found first (run 5: right 6% of the
    # time when one route found it, 35% two, 54% three or more).
    "xep_theo_do_chac": False,
    # An answer shorter than this (words) is asked again once.
    "min_answer_words": 40,
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


def is_general_law(d: Dict) -> bool:
    """BLHS general part (Điều 1-107) or any BLTTHS article."""
    if d.get("bo_luat") == "BLTTHS":
        return True
    entry = str(d.get("entry", ""))
    return d.get("bo_luat", "BLHS") == "BLHS" and entry.isdigit() and int(entry) <= 107


def general_title_index(g: HierarGraph, embed: Embed) -> List[Tuple[str, np.ndarray]]:
    """(Law node, embedding of its title) for the general articles; titles
    are short and say what the article is about ("Án treo"), the full text
    is long. Kept on the graph object."""
    cached = getattr(g, "_general_title_index", None)
    if cached is not None:
        return cached
    index = []
    for n in g.nodes_of("Law"):
        d = g.node(n)
        if is_general_law(d) and d.get("title"):
            index.append((n, _unit(embed(d["title"]))))
    g._general_title_index = index
    return index


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _grams(text: str, n: int = 3) -> set:
    w = _words(text)
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


# Names of the codes and of liability itself are ordinary words in a legal
# issue, not a sign of copying.
LAW_NAME_GRAMS = set().union(*(_grams(t) for t in (
    "bộ luật hình sự", "bộ luật tố tụng hình sự", "trách nhiệm hình sự", "vụ án hình sự", "tố tụng hình sự",
)))


def copied_from_prompt(item: str, template: str, source: str) -> List[str]:
    """3-word phrases of item that occur in the prompt's own text (the
    template, before the question) but not in source (question + rewrite):
    the model repeated the instructions instead of reading the question.
    Qwen3-8B did it in every run (example articles, example issues, then
    the wording of the issue definition)."""
    fixed = template.split("{")[0]
    return sorted((_grams(item) & _grams(fixed)) - _grams(source) - LAW_NAME_GRAMS)


def parse_issues(text: str) -> List[Dict[str, str]]:
    """[{"van_de", "trich"}] from LEGAL_ISSUES_PROMPT; a plain list of
    strings is accepted with an empty "trich"."""
    first, last = text.find("["), text.rfind("]")
    if first == -1 or last < first:
        return []
    try:
        data = json.loads(text[first : last + 1])
    except json.JSONDecodeError:
        return [{"van_de": x, "trich": ""} for x in parse_crime_list(text)]
    out = []
    for x in data if isinstance(data, list) else []:
        if isinstance(x, dict) and str(x.get("van_de", "")).strip():
            out.append({"van_de": str(x["van_de"]).strip(), "trich": str(x.get("trich", "")).strip()})
        elif isinstance(x, str) and x.strip():
            out.append({"van_de": x.strip(), "trich": ""})
    return out


def grounded_issues(issues: List[Dict[str, str]], source: str, template: str, min_overlap: float = 0.7) -> Tuple[List[str], List[Dict]]:
    """Keep an issue only if its quote is (mostly) words of the question and
    its name repeats no phrase of the prompt that the question lacks.
    Returns (kept names, dropped with the reason)."""
    src = set(_words(source))
    kept, dropped = [], []
    for it in issues:
        quote = _words(it["trich"])
        copied = copied_from_prompt(it["van_de"], template, source)
        if not quote or sum(w in src for w in quote) / len(quote) < min_overlap:
            dropped.append({**it, "ly_do": "trích dẫn không có trong câu hỏi"})
        elif copied:
            dropped.append({**it, "ly_do": "chép chữ của prompt: " + ", ".join(copied)})
        else:
            kept.append(it["van_de"])
    return kept, dropped


def rewrite_question(question: str, generate: Generate) -> str:
    """The question in legal language (QA_REWRITE_PROMPT), or "" if empty."""
    text = generate(QA_REWRITE_PROMPT.format(question=question), max_tokens=256).strip()
    return text.strip("`").strip()


def llm_issue_laws(
    g: HierarGraph, question: str, generate: Generate, embed: Embed, per_issue: int, max_issues: int = 5,
    rewrite: str = "", rank_vec=None,
) -> Tuple[List[str], List[Dict]]:
    """The LLM names <=max_issues general legal issues; each goes to the general
    articles whose titles are closest. With rank_vec, the articles found are
    ordered by how close their titles are to it (the restated question), so
    an issue the LLM named out of habit does not take the reserved slots
    first (Kaggle run 4: "độ tuổi chịu trách nhiệm hình sự" in 146/187
    questions pushed out 16 right articles)."""
    prompt = LEGAL_ISSUES_PROMPT.format(question=question, rewrite=rewrite or question)
    issues, dropped = grounded_issues(
        parse_issues(generate(prompt, max_tokens=512)), f"{question} {rewrite}", LEGAL_ISSUES_PROMPT
    )
    index = general_title_index(g, embed)
    laws: List[str] = []
    hits = []
    for issue in issues[:max_issues]:
        if not index or not issue.strip():
            continue
        q = _unit(embed(issue))
        ranked = sorted(index, key=lambda x: -float(x[1] @ q))[:per_issue]
        hits.append({"van_de": issue, "dieu": [law_label(g, n) for n, _ in ranked],
                     "cosine": [round(float(v @ q), 3) for _, v in ranked]})
        for n, _ in ranked:
            if n not in laws:
                laws.append(n)
    if rank_vec is not None and laws:
        title_vec = dict(index)
        r = _unit(rank_vec)
        laws.sort(key=lambda n: -float(title_vec[n] @ r))
    hits += [{"van_de": d["van_de"], "bo": d["ly_do"], "trich": d["trich"]} for d in dropped]
    return laws, hits


def llm_crime_laws(
    g: HierarGraph, text: str, generate: Generate, embed: Embed, nodes_per_crime: int = 1
) -> Tuple[List[str], List[Dict]]:
    """The original retrieve_law: the LLM names <=3 crimes, each goes to
    its nearest Crime node(s) (1 in the original), then back to their Law
    nodes."""
    crimes = parse_crime_list(generate(RETRIEVE_LAW_PROMPT.format(fact=text), max_tokens=256))
    laws: List[str] = []
    hits = []
    for name in crimes[:3]:
        for crime_node, sim in g.search(embed(name), "Crime", top_k=nodes_per_crime):
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
    g: HierarGraph, question: str, query_vec, generate: Generate, embed: Embed, cfg: Dict, query_text: str = "",
    rewrite: str = "",
) -> Dict:
    """rewrite: the question in legal language; the LLM guesses read both,
    the vector searches use the rewrite when there is one."""
    routes: Dict[str, List[str]] = {}
    rerank = None
    if not cfg["dung_an"]:
        pass
    elif cfg["rerank"]:
        rerank = reranked_cases(g, query_vec, query_text or question, generate, cfg)
        routes["an_tuong_tu"] = laws_of_cases(g, [(c, 0.0) for c in rerank["an"]])
    else:
        routes["truc_tiep"] = laws_of_cases(g, direct_cases(g, query_vec, cfg["top_k_cases"]))
        routes["qua_cum"] = laws_of_cases(g, cluster_cases(g, query_vec, cfg["n_clusters"], cfg["top_k_cases"]))

    guess_text = f"{question}\nDiễn đạt pháp lý: {rewrite}" if rewrite else question
    routes["llm_doan_toi"], crime_hits = llm_crime_laws(g, guess_text, generate, embed, cfg["crime_nodes_per_guess"])
    question_vec = embed(rewrite or question)
    routes["van_ban_luat"] = [n for n, _ in g.search(question_vec, "Law", top_k=cfg["top_k_laws_text"])]
    routes["huong_dan"] = guidance_laws(g, question_vec, embed, cfg["top_k_guidance"])
    issue_hits: List[Dict] = []
    if cfg["general_slots"]:
        routes["van_de_phap_ly"], issue_hits = llm_issue_laws(
            g, question, generate, embed, cfg["laws_per_issue"], cfg["max_issues"], rewrite, question_vec
        )

    order = ["llm_doan_toi", "huong_dan", "van_ban_luat", "an_tuong_tu", "truc_tiep", "qua_cum"]
    candidates: List[str] = []
    for r in order:
        for law in routes.get(r, []):
            if law not in candidates:
                candidates.append(law)
    candidates = candidates[: cfg["max_candidates"]]
    extra = [n for n in routes.get("van_de_phap_ly", []) if n not in candidates][: cfg["general_slots"]]
    return {
        "tuyen": routes,
        "rerank": rerank,
        "llm_toi_danh": crime_hits,
        "llm_van_de": issue_hits,
        "ung_vien": candidates + extra,
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
    rewrite = rewrite_question(question, generate) if cfg["dien_dat_lai"] else ""
    description = ""
    if cfg["dung_an"]:  # features only serve the case routes
        features = parse_features(generate(build_prompt(question)))
        features = sanitize_features(features) if features else None
        description = concat_feature_description(features) if features else ""
    query_vec = embed(description or question)

    retrieval = retrieve_candidates(g, question, query_vec, generate, embed, cfg, description, rewrite)
    judgments = {n: judge_candidate(g, n, question, generate, cfg["judge_mode"]) for n in retrieval["ung_vien"]}
    accepted = [n for n in retrieval["ung_vien"] if judgments[n]["ap_dung"]]
    fallback = not accepted
    if cfg["loc_theo_judge"]:
        used = accepted or retrieval["ung_vien"][:3]
    else:
        used = accepted + [n for n in retrieval["ung_vien"] if n not in accepted]
    if cfg["xep_theo_do_chac"]:
        used = order_by_confidence(used, judgments, retrieval["tuyen"])

    prompt = QA_ANSWER_PROMPT.format(
        question=question, laws=render_laws_for_answer(g, used, cfg, judgments, embed(question), embed)
    )
    raw, answer, retried = answer_with_retry(prompt, generate, cfg["min_answer_words"])
    return {
        "dien_dat_lai": rewrite,
        "dac_trung": description,
        "truy_xuat": {**retrieval, "ung_vien": [law_label(g, n) for n in retrieval["ung_vien"]],
                      "tuyen": {k: [law_label(g, n) for n in v] for k, v in retrieval["tuyen"].items()}},
        "judge": {law_label(g, n): v for n, v in judgments.items()},
        "dieu_dung_de_tra_loi": [law_label(g, n) for n in used],
        "khong_dieu_nao_duoc_chap_nhan": fallback,
        "tra_loi": answer or {"cau_tra_loi": "", "toi_danh": [], "dieu_luat": []},
        "tra_loi_loi": answer is None,
        "tra_loi_tho": raw if answer is None else None,
        "hoi_lai_vi_ngan": retried,
    }


def order_by_confidence(laws: List[str], judgments: Dict[str, Dict], routes: Dict[str, List[str]]) -> List[str]:
    """Accepted by judge first, then by how many routes found the article;
    ties keep the given order."""
    n_routes = {n: sum(n in found for found in routes.values()) for n in laws}
    rank = {n: i for i, n in enumerate(laws)}
    return sorted(laws, key=lambda n: (not judgments.get(n, {}).get("ap_dung"), -n_routes[n], rank[n]))


def answer_with_retry(prompt: str, generate: Generate, min_words: int):
    """(raw, parsed answer, retried). A terse answer is asked again once,
    with a note; the longer of the two is kept."""
    raw = generate(prompt, max_tokens=1024)
    answer = parse_answer(raw)
    n = len((answer or {}).get("cau_tra_loi", "").split())
    if not min_words or n >= min_words:
        return raw, answer, False
    raw2 = generate(prompt + QA_ANSWER_TOO_SHORT_NOTE.format(n=n), max_tokens=1024)
    answer2 = parse_answer(raw2)
    if answer2 and len(answer2.get("cau_tra_loi", "").split()) > n:
        return raw2, answer2, True
    return raw, answer, True


def answer_without_graph(question: str, generate: Generate) -> Dict:
    """Baseline: the LLM answers from its own knowledge, same JSON format,
    so the same scoring applies."""
    raw, answer, _ = answer_with_retry(QA_ANSWER_NO_DOCS_PROMPT.format(question=question), generate, DEFAULTS["min_answer_words"])
    return {
        "truy_xuat": {"ung_vien": []},
        "dieu_dung_de_tra_loi": [],
        "tra_loi": answer or {"cau_tra_loi": "", "toi_danh": [], "dieu_luat": []},
        "tra_loi_loi": answer is None,
        "tra_loi_tho": raw if answer is None else None,
    }
