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
