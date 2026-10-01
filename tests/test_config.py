"""Tests for vn_legal_graph.config. Uses a .env path that does not exist,
so the real .env is never read."""
from vn_legal_graph.config import AppConfig

NO_DOTENV = "/nonexistent/.env"


def test_openai_api_key_is_accepted(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    assert AppConfig.from_env_file(NO_DOTENV).llm.api_key == "sk-test-openai"


def test_llm_api_key_takes_precedence(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test-llm")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    assert AppConfig.from_env_file(NO_DOTENV).llm.api_key == "sk-test-llm"


def test_no_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert AppConfig.from_env_file(NO_DOTENV).llm.api_key == ""
