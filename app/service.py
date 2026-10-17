"""Backend of the web UI (app/ui.py): loads the graph and the embedder,
talks to Ollama, and turns answer_question's trace into what the page shows.

No .env is read: the UI has its own LLM settings (UISettings, overridable
with UI_LLM_BASE_URL / UI_LLM_MODEL), so the key in .env is never loaded.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from vn_legal_graph.graph.graph_db import HierarGraph
from vn_legal_graph.qa.pipeline import LAW_NAMES, answer_question, guidance_items, law_label
from vn_legal_graph.qa.scoring import load_questions, norm_law, question_text

GRAPH_PATH = "outputs/hierargraph.pkl"
QUESTIONS_PATH = "data/raw/questions.xlsx"

# The four progress steps of answer_question, in order, as the page names them.
STEPS = [
    ("dien_dat_lai", "Diễn đạt lại câu hỏi"),
    ("tim_dieu", "Tìm điều luật ứng viên"),
    ("tham_dinh", "Thẩm định từng điều"),
    ("tra_loi", "Viết câu trả lời"),
]

ROUTE_NAMES = {
    "llm_doan_toi": "đoán tội danh",
    "huong_dan": "văn bản hướng dẫn",
    "van_ban_luat": "toàn văn điều luật",
    "van_de_phap_ly": "vấn đề pháp lý",
    "truc_tiep": "án tương tự",
    "qua_cum": "cụm án",
    "an_tuong_tu": "án tương tự",
}


@dataclass
class UISettings:
    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "qwen3:8b-q8_0"  # the model of the Kaggle runs, so their cache is reused
    timeout: int = 600
    cache_dir: str = ".cache/llm"

    @classmethod
    def from_env(cls) -> "UISettings":
        return cls(
            base_url=os.getenv("UI_LLM_BASE_URL", cls.base_url),
            model=os.getenv("UI_LLM_MODEL", cls.model),
        )


def make_client(s: UISettings):
    """An LLMClient for Ollama at temperature 0, like the Kaggle runs."""
    from vn_legal_graph.config import AppConfig, LLMConfig
    from vn_legal_graph.llm import LLMClient

    llm = LLMConfig(base_url=s.base_url, api_key="ollama", model=s.model, temperature=0.0,
                    timeout=s.timeout, provider="ollama")
    return LLMClient(AppConfig(llm=llm, cache_dir=s.cache_dir))


def check_ollama(s: UISettings, get: Optional[Callable] = None) -> Tuple[bool, str]:
    """(ready, message): the server answers and has the model."""
    import requests

    get = get or requests.get
    root = re.sub(r"/v1/?$", "", s.base_url.rstrip("/"))
    try:
        tags = get(f"{root}/api/tags", timeout=5).json()
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False, f"Không kết nối được Ollama ở {root}. Hãy chạy `ollama serve`."
    names = {m.get("name") for m in tags.get("models", [])}
    if s.model not in names:
        return False, f"Ollama chưa có model {s.model}. Hãy chạy `ollama pull {s.model}`."
    return True, f"Ollama sẵn sàng · {s.model}"


def load_graph(path: str = GRAPH_PATH) -> HierarGraph:
    return HierarGraph.load(path)


def load_embedder():
    """The embedding function the experiments use (bkai, disk cache)."""
    from vn_legal_graph.config import EmbeddingConfig
    from vn_legal_graph.embedding import CachedEmbedder, embedder_from_config

    cfg = EmbeddingConfig()
    return CachedEmbedder(embedder_from_config(cfg), cfg.model_name).encode_long_text


def sample_questions(path: str = QUESTIONS_PATH, n: int = 12) -> List[Dict]:
    """The first n questions of the test file, with the lawyer's answer kept
    apart (shown only for comparison, never sent to the model)."""
    return [{"title": q["title"], "text": question_text(q), "answer": q.get("answer") or "", "url": q.get("url") or ""}
            for q in load_questions(path)[:n]]


def citation_label(c: Dict) -> str:
    """The graph's label of a cited article, e.g. "BLHS Điều 321"."""
    code = norm_law(c.get("luat", ""))
    return f"{LAW_NAMES.get(code, code)} Điều {str(c.get('dieu', '')).strip()}"


def citation_text(c: Dict) -> str:
    parts = [f"Điều {c.get('dieu', '')}"]
    if c.get("khoan"):
        parts.append(f"khoản {c['khoan']}")
    if c.get("diem"):
        parts.append(f"điểm {c['diem']}")
    return " ".join(parts) + f" {c.get('luat', '')}".rstrip()


def law_cards(g: HierarGraph, trace: Dict) -> List[Dict]:
    """One card per article shown to the model, in the order it saw them."""
    index = {law_label(g, n): n for n in g.nodes_of("Law")}
    cited = {citation_label(c) for c in trace["tra_loi"].get("dieu_luat", []) if isinstance(c, dict)}
    routes = trace["truy_xuat"]["tuyen"]
    cards = []
    for label in trace["dieu_dung_de_tra_loi"]:
        node = index.get(label)
        if node is None:
            continue
        d = g.node(node)
        judge = trace["judge"].get(label, {})
        cards.append({
            "label": label,
            "title": d.get("title") or "",
            "ap_dung": bool(judge.get("ap_dung")),
            "cach_tham_dinh": judge.get("cach", ""),
            "routes": [ROUTE_NAMES.get(r, r) for r, found in routes.items() if label in found],
            "duoc_trich": label in cited,
            "text": d.get("description") or "",
            "yeu_to": list(d.get("judge_dep") or []),
            "huong_dan": [{"id": it.get("id", ""), "text": it.get("text", "")} for it in guidance_items(g, node)],
        })
    return cards


def ask(g: HierarGraph, question: str, generate: Callable, embed: Callable,
        progress: Optional[Callable[[str, Dict], None]] = None) -> Dict:
    """Answer one question with the experiments' Q&A settings and shape the
    result for the page."""
    start = time.time()
    trace = answer_question(g, question, generate, embed, progress=progress)
    answer = trace["tra_loi"]
    return {
        "answer": answer.get("cau_tra_loi", ""),
        "crimes": answer.get("toi_danh", []),
        "citations": [citation_text(c) for c in answer.get("dieu_luat", []) if isinstance(c, dict)],
        "cards": law_cards(g, trace),
        "parse_error": trace["tra_loi_loi"],
        "seconds": time.time() - start,
        "trace": trace,
    }
