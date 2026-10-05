"""Configuration loading for vn_legal_graph.

Reads settings from a .env file (or process environment) so that API keys
never need to be hard-coded or committed. Mirrors the spirit of the
original LegalGraphRAG repo's env.example / configs/main.env, adapted to
Vietnamese-law field names.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: int = 120
    provider: str = "openai"  # "openai" | "deepseek"
    # DeepSeek only: "disabled" (default) or "enabled". Thinking mode ignores
    # temperature, and its effect on max_tokens is not documented.
    thinking: str = "disabled"


# Provider defaults, from https://api-docs.deepseek.com (checked 2026-10-05):
# models "deepseek-flash" (DeepSeek-V4.1-Flash) and "deepseek-v4-pro"
# (DeepSeek-V4-Pro-0813), both thinking by default.
PROVIDER_DEFAULTS = {
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro"},
}


def _pick_provider_and_key() -> tuple:
    """LLM_API_KEY (with LLM_PROVIDER, default openai) wins, then
    DEEPSEEK_API_KEY, then OPENAI_API_KEY."""
    explicit = _env("LLM_API_KEY", "")
    if explicit:
        return (_env("LLM_PROVIDER", "") or "openai"), explicit
    deepseek = _env("DEEPSEEK_API_KEY", "")
    if deepseek:
        return "deepseek", deepseek
    return (_env("LLM_PROVIDER", "") or "openai"), (_env("OPENAI_API_KEY", "") or "")


@dataclass
class EmbeddingConfig:
    backend: str = "vietnamese-bi-encoder"  # or "bge-m3" / "api"
    model_name: str = "bkai-foundation-models/vietnamese-bi-encoder"
    max_seq_length: int = 256
    device: str = "cpu"
    # Used only when backend == "api" (e.g. an Ollama-served embedding model).
    api_url: str = "http://localhost:11434/api/embed"


@dataclass
class PathsConfig:
    raw_law_docx: str = "data/raw/law/11_VBHN-VPQH_650257.docx"
    raw_guidance_dir: str = "data/raw/guidance"
    raw_cases_dir: str = "data/raw/cases"
    processed_dir: str = "data/processed"
    outputs_dir: str = "outputs"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    prompt_language: str = "vi"
    cache_llm_calls: bool = True
    cache_dir: str = ".cache/llm"

    @classmethod
    def from_env_file(cls, dotenv_path: str = ".env") -> "AppConfig":
        if load_dotenv is not None and os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path, override=False)

        provider, api_key = _pick_provider_and_key()
        defaults = PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"])
        llm = LLMConfig(
            base_url=_env("LLM_BASE_URL", "") or defaults["base_url"],
            api_key=api_key,
            model=_env("LLM_MODEL", "") or defaults["model"],
            provider=provider,
            thinking=_env("LLM_THINKING", "") or "disabled",
            temperature=_env_float("LLM_TEMPERATURE", LLMConfig.temperature),
            max_tokens=_env_int("LLM_MAX_TOKENS", LLMConfig.max_tokens),
            timeout=_env_int("LLM_TIMEOUT", LLMConfig.timeout),
        )
        embedding = EmbeddingConfig(
            backend=_env("EMBEDDING_BACKEND", EmbeddingConfig.backend),
            model_name=_env("EMBEDDING_MODEL", EmbeddingConfig.model_name),
            max_seq_length=_env_int(
                "EMBEDDING_MAX_SEQ_LENGTH", EmbeddingConfig.max_seq_length
            ),
            device=_env("EMBEDDING_DEVICE", EmbeddingConfig.device),
            api_url=_env("EMBEDDING_API_URL", EmbeddingConfig.api_url),
        )
        paths = PathsConfig(
            raw_law_docx=_env("RAW_LAW_DOCX", PathsConfig.raw_law_docx),
            raw_guidance_dir=_env(
                "RAW_GUIDANCE_DIR", PathsConfig.raw_guidance_dir
            ),
            raw_cases_dir=_env("RAW_CASES_DIR", PathsConfig.raw_cases_dir),
            processed_dir=_env("PROCESSED_DIR", PathsConfig.processed_dir),
            outputs_dir=_env("OUTPUTS_DIR", PathsConfig.outputs_dir),
        )
        return cls(
            llm=llm,
            embedding=embedding,
            paths=paths,
            prompt_language=_env("PROMPT_LANGUAGE", "vi"),
            cache_llm_calls=_env_bool("CACHE_LLM_CALLS", True),
            cache_dir=_env("CACHE_DIR", ".cache/llm"),
        )
