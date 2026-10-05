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


def _clear(monkeypatch):
    for k in ("LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_THINKING"):
        monkeypatch.delenv(k, raising=False)


def test_deepseek_key_selects_deepseek_over_openai(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    llm = AppConfig.from_env_file(NO_DOTENV).llm
    assert (llm.provider, llm.api_key) == ("deepseek", "sk-deepseek")
    assert llm.base_url == "https://api.deepseek.com"
    assert llm.model == "deepseek-v4-pro"
    assert llm.thinking == "disabled"


def test_deepseek_model_override(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    assert AppConfig.from_env_file(NO_DOTENV).llm.model == "deepseek-flash"


def test_openai_defaults_unchanged(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    llm = AppConfig.from_env_file(NO_DOTENV).llm
    assert (llm.provider, llm.base_url, llm.model) == ("openai", "https://api.openai.com/v1", "gpt-4o-mini")
