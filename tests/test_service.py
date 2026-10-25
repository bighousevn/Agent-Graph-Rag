"""Tests for app/service.py (the web UI's backend) with a fake LLM."""
import json

from app import service
from tests.test_graph import fake_embed
from tests.test_qa import LAWS
from vn_legal_graph.graph.build import build_base_graph


def fake_llm(prompt, max_tokens=None):
    if "diễn đạt lại câu hỏi của người dân" in prompt:
        return "Hành vi đánh bạc trái phép bằng ứng dụng với số tiền 40 triệu đồng."
    if "tối đa ba tội danh" in prompt:
        return '["Tội đánh bạc"]'
    if "KHÔNG gắn riêng với một tội" in prompt:
        return '[{"van_de": "tình tiết giảm nhẹ", "trich": "Đánh bạc qua app"}]'
    if "Các yếu tố cần xét" in prompt:
        return '{"1": true}'
    if "Yếu tố thỏa mãn" in prompt:
        return "true"
    if "Đánh giá tình huống" in prompt:
        return "false"
    if "luật sư tư vấn" in prompt:
        return json.dumps({"cau_tra_loi": " ".join(["Bị xử lý theo khoản 2 Điều 321."] * 10), "toi_danh": ["Tội đánh bạc"],
                           "dieu_luat": [{"luat": "Bộ luật Hình sự", "dieu": "321", "khoan": "2", "diem": "c"}]},
                          ensure_ascii=False)
    raise AssertionError(prompt[:80])


def test_ask_reports_steps_in_order_and_builds_cards():
    g, _ = build_base_graph(LAWS, [], fake_embed)
    steps = []
    out = service.ask(g, "Đánh bạc qua app 40 triệu, ma túy?", fake_llm, fake_embed,
                      progress=lambda step, info: steps.append((step, info)))

    names = [s for s, _ in steps]
    assert names[0] == "dien_dat_lai" and names[1] == "tim_dieu" and names[-1] == "tra_loi"
    judged = [info for s, info in steps if s == "tham_dinh"]
    assert len(judged) == len(steps[1][1]["ung_vien"]) and judged[-1]["xong"] == judged[-1]["tong"]

    assert out["citations"] == ["Điều 321 khoản 2 điểm c Bộ luật Hình sự"]
    first = out["cards"][0]
    assert first["label"] == "BLHS Điều 321" and first["ap_dung"] and first["duoc_trich"]
    assert "đoán tội danh" in first["routes"]
    assert first["yeu_to"] == ["Có đánh bạc trái phép không?"]
    assert not out["parse_error"]


def test_progress_is_optional_and_does_not_change_the_answer():
    g, _ = build_base_graph(LAWS, [], fake_embed)
    with_cb = service.ask(g, "Đánh bạc qua app?", fake_llm, fake_embed, progress=lambda *a: None)
    without = service.ask(g, "Đánh bạc qua app?", fake_llm, fake_embed)
    assert with_cb["trace"]["tra_loi"] == without["trace"]["tra_loi"]
    assert with_cb["trace"]["judge"] == without["trace"]["judge"]


class _Resp:
    def __init__(self, data, status_code=200):
        self.data = data
        self.status_code = status_code

    def json(self):
        return self.data


def test_check_ollama():
    s = service.UISettings(base_url="http://127.0.0.1:11434/v1", model="qwen3:8b-q8_0")
    seen = []

    def has_model(url, timeout, headers):
        seen.append((url, headers))
        return _Resp({"models": [{"name": "qwen3:8b-q8_0"}]})

    assert service.check_ollama(s, has_model)[0] and seen == [("http://127.0.0.1:11434/api/tags", {})]
    ok, msg = service.check_ollama(s, lambda url, timeout, headers: _Resp({"models": []}))
    assert not ok and "ollama pull qwen3:8b-q8_0" in msg

    def down(url, timeout, headers):
        raise ConnectionError

    ok, msg = service.check_ollama(s, down)
    assert not ok and "ollama serve" in msg


def test_check_ollama_on_modal_sends_token_and_explains_cold_start():
    s = service.UISettings(base_url="https://ws--legal-qwen-serve.modal.run", model="qwen3:8b-q8_0",
                           proxy_key="wk-1", proxy_secret="ws-2")
    assert not s.is_local and s.headers == {"Modal-Key": "wk-1", "Modal-Secret": "ws-2"}
    seen = []

    def ok_model(url, timeout, headers):
        seen.append((url, headers))
        return _Resp({"models": [{"name": "qwen3:8b-q8_0"}]})

    ok, msg = service.check_ollama(s, ok_model)
    assert ok and msg.startswith("Modal sẵn sàng") and seen[0] == ("https://ws--legal-qwen-serve.modal.run/api/tags", s.headers)
    assert "MODAL_PROXY_KEY" in service.check_ollama(s, lambda url, timeout, headers: _Resp({}, 401))[1]

    def cold(url, timeout, headers):
        raise TimeoutError

    assert "khởi động" in service.check_ollama(s, cold)[1]
    assert service.make_client(s).llm_config.headers == s.headers


def test_settings_and_client_do_not_read_dotenv(monkeypatch):
    monkeypatch.setenv("UI_LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("LLM_PROVIDER", "openai")  # what a .env could set: ignored by the UI
    s = service.UISettings.from_env()
    assert s.model == "qwen3:8b" and s.base_url.endswith("/v1")
    llm = service.make_client(s).llm_config
    assert (llm.provider, llm.model, llm.temperature, llm.api_key) == ("ollama", "qwen3:8b", 0.0, "ollama")


def test_citation_label_matches_graph_labels():
    assert service.citation_label({"luat": "Bộ luật Tố tụng hình sự", "dieu": "155"}) == "BLTTHS Điều 155"
    assert service.citation_text({"luat": "BLHS", "dieu": "12"}) == "Điều 12 BLHS"
