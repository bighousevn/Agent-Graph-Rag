"""Tests for vn_legal_graph.llm caching, with a fake OpenAI client."""
from types import SimpleNamespace

from vn_legal_graph.config import AppConfig, LLMConfig
from vn_legal_graph.llm import LLMClient


class FakeOpenAI:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text, finish = self.answers.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=text))]
        )


def client_with(tmp_path, answers):
    cfg = AppConfig(llm=LLMConfig(api_key="fake", model="m"), cache_dir=str(tmp_path))
    c = LLMClient(cfg)
    c._client = FakeOpenAI(answers)
    return c


def test_complete_answer_is_cached(tmp_path):
    c = client_with(tmp_path, [("ok", "stop")])
    assert c.generate("p") == "ok"
    assert c.generate("p") == "ok"  # from cache, no second API call
    assert len(c._client.calls) == 1


def test_truncated_answer_is_not_cached(tmp_path):
    c = client_with(tmp_path, [("[cụt", "length"), ('["đủ"]', "stop")])
    assert c.generate("p") == "[cụt"
    assert c.generate("p") == '["đủ"]'
    assert len(c._client.calls) == 2


def test_explicit_max_tokens_is_part_of_cache_key(tmp_path):
    c = client_with(tmp_path, [("a", "stop"), ("b", "stop")])
    assert c.generate("p") == "a"
    assert c.generate("p", max_tokens=4096) == "b"
    assert c._client.calls[1]["max_tokens"] == 4096
    assert c._cache_path("p") != c._cache_path("p", 4096)


def test_deepseek_request_disables_thinking_by_default(tmp_path):
    cfg = AppConfig(
        llm=LLMConfig(api_key="fake", model="deepseek-v4-pro", provider="deepseek"), cache_dir=str(tmp_path)
    )
    c = LLMClient(cfg)
    c._client = FakeOpenAI([("true", "stop")])
    c.generate("p", max_tokens=16)
    call = c._client.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["temperature"] == cfg.llm.temperature
    assert call["max_tokens"] == 16


def test_deepseek_thinking_enabled_drops_temperature_and_raises_limit(tmp_path):
    cfg = AppConfig(
        llm=LLMConfig(api_key="fake", model="deepseek-v4-pro", provider="deepseek", thinking="enabled"),
        cache_dir=str(tmp_path),
    )
    c = LLMClient(cfg)
    c._client = FakeOpenAI([("true", "stop")])
    c.generate("p", max_tokens=16)
    call = c._client.calls[0]
    assert call["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in call
    assert call["max_tokens"] >= 8192


def test_openai_request_has_no_extra_body(tmp_path):
    c = client_with(tmp_path, [("ok", "stop")])
    c.generate("p")
    assert "extra_body" not in c._client.calls[0]


def test_annotate_articles_parallel_keeps_order(tmp_path):
    from vn_legal_graph.law.judge_dep import annotate_articles

    def art(i, title):
        return {"id": i, "suffix": "", "title": title, "phan": "", "phan_title": "", "chuong": "", "chuong_title": "",
                "khoan": [{"so": 1, "text": f"nội dung {i}", "diem": []}], "preamble": ""}

    arts = [art(1, "Nhiệm vụ"), art(2, "Tội A"), art(3, "Tội B"), art(4, "Tội C")]

    class Fake:
        def generate(self, prompt, max_tokens=None):
            n = prompt.split("Điều ")[-1].split(".")[0]
            return f'["Có {n} không?"]'

    out = annotate_articles(arts, Fake(), workers=3)
    assert [a["judge_dep"] for a in out] == [[], ["Có 2 không?"], ["Có 3 không?"], ["Có 4 không?"]]


def test_qwen3_gets_no_think_and_think_block_is_stripped(tmp_path):
    cfg = AppConfig(llm=LLMConfig(api_key="ollama", model="Qwen/Qwen3-8B", provider="openai"), cache_dir=str(tmp_path))  # e.g. vLLM
    c = LLMClient(cfg)
    c._client = FakeOpenAI([("<think>\n\n</think>\n\n{\"a\": 1}", "stop")])
    assert c.generate("p") == '{"a": 1}'
    assert c._client.calls[0]["messages"][0]["content"] == "p\n/no_think"
    assert "extra_body" not in c._client.calls[0]


def test_other_models_prompt_unchanged(tmp_path):
    c = client_with(tmp_path, [("ok", "stop")])
    c.generate("p")
    assert c._client.calls[0]["messages"][0]["content"] == "p"


def test_ollama_uses_native_chat_without_thinking(tmp_path, monkeypatch):
    import requests

    sent = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "<think>\n\n</think>\n\ntrue"}, "done_reason": "stop"}

    monkeypatch.setattr(requests, "post", lambda url, json, timeout: sent.append((url, json)) or Resp())
    cfg = AppConfig(llm=LLMConfig(api_key="ollama", model="qwen3:8b-q8_0", provider="ollama",
                                  base_url="http://localhost:11434/v1"), cache_dir=str(tmp_path))
    c = LLMClient(cfg)
    assert c.generate("p", max_tokens=16) == "true"
    url, body = sent[0]
    assert url == "http://localhost:11434/api/chat" and body["think"] is False and body["stream"] is False
    assert body["messages"][0]["content"] == "p" and body["options"]["num_predict"] == 128
    assert c.generate("p", max_tokens=16) == "true" and len(sent) == 1  # cached
