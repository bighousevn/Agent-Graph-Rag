"""OpenAI-compatible LLM client used for all extraction steps
(judge_dep questions, judicial-guidance linking, later: case feature
extraction and cluster summarization).

Kept provider-agnostic on purpose: point LLM_BASE_URL / LLM_MODEL at
OpenAI, DeepSeek, or any other OpenAI-compatible endpoint via .env.

Every call is cached to disk by a hash of (model, prompt, temperature) so
re-running a script after a crash, or during development, does not
re-spend API budget on prompts already answered.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Optional

from .config import AppConfig, LLMConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.from_env_file()
        self.llm_config: LLMConfig = self.config.llm
        self._client = None  # lazy: avoid requiring `openai` pkg for dry runs

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise LLMError(
                    "The `openai` package is required to call the LLM. "
                    "Install it with `pip install openai`."
                ) from e
            if not self.llm_config.api_key:
                raise LLMError(
                    "No LLM_API_KEY configured. Set it in .env or the "
                    "environment before making LLM calls."
                )
            self._client = OpenAI(
                base_url=self.llm_config.base_url,
                api_key=self.llm_config.api_key,
            )
        return self._client

    def _cache_path(self, prompt: str) -> str:
        key = json.dumps(
            {
                "model": self.llm_config.model,
                "temperature": self.llm_config.temperature,
                "prompt": prompt,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        cache_dir = self.config.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{digest}.json")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str:
        """Send a single-turn prompt to the configured LLM and return the
        text response. Cached on disk when config.cache_llm_calls is True."""
        cache_path = self._cache_path(prompt) if self.config.cache_llm_calls else None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)["response"]

        client = self._get_client()
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.llm_config.temperature,
                    max_tokens=max_tokens or self.llm_config.max_tokens,
                    timeout=self.llm_config.timeout,
                )
                text = (resp.choices[0].message.content or "").strip()
                if cache_path:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump({"prompt": prompt, "response": text}, f, ensure_ascii=False)
                return text
            except Exception as e:  # noqa: BLE001 - broad on purpose, retried
                last_error = e
                if attempt < retries:
                    time.sleep(retry_delay * attempt)
        raise LLMError(f"LLM call failed after {retries} attempts: {last_error}")
