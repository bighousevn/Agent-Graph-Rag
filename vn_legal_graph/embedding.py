"""Embedding backends for the Vietnamese HierarGraph.

Default backend: `bkai-foundation-models/vietnamese-bi-encoder`, a
PhoBERT-base-v2-derived bi-encoder fine-tuned for retrieval (including
legal-domain data). Unlike raw PhoBERT (a masked-LM checkpoint, not meant
to produce sentence embeddings), this model is suited for the cosine-
similarity search the graph relies on (kNN between Case nodes, cluster
lookup, Crime/Law lookup).

Two constraints this module handles for the caller:
  1. PhoBERT-family tokenizers expect *word-segmented* Vietnamese text
     (underscores joining multi-syllable words), so we segment with
     `pyvi` before encoding.
  2. The model's max sequence length is 256 tokens. Long Điều text (many
     Khoản) is embedded khoản-by-khoản and averaged, so no content is
     silently truncated away; short Case/Crime descriptions are
     unaffected.

A `bge-m3` HTTP backend (matching the original LegalGraphRAG repo, e.g.
served via Ollama) is also provided for side-by-side comparison in
Phase 4.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

import numpy as np

from .config import AppConfig, EmbeddingConfig


def segment_vietnamese(text: str) -> str:
    """Word-segment Vietnamese text for PhoBERT-family tokenizers."""
    try:
        from pyvi import ViTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pyvi is required for Vietnamese word segmentation. "
            "Install it with `pip install pyvi`."
        ) from e
    return ViTokenizer.tokenize(text)


def chunk_by_words(text: str, max_words: int) -> List[str]:
    """Simple whitespace-based chunker used as a fallback splitter before
    handing text to a tokenizer with a hard max-length limit."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


class VietnameseBiEncoder:
    """sentence-transformers wrapper around a PhoBERT-family bi-encoder."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required. Install it with "
                    "`pip install sentence-transformers`."
                ) from e
            self._model = SentenceTransformer(
                self.config.model_name, device=self.config.device
            )
            self._model.max_seq_length = self.config.max_seq_length
        return self._model

    def encode_one(self, text: str) -> np.ndarray:
        segmented = segment_vietnamese(text)
        model = self._get_model()
        vec = model.encode(segmented, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def encode_long_text(self, text: str, approx_words_per_chunk: int = 180) -> np.ndarray:
        """Embed text that may exceed the model's 256-token limit by
        chunking on whitespace (a conservative word-count proxy for token
        count in Vietnamese), embedding each chunk, and mean-pooling.
        Used for full Điều text spanning multiple Khoản.
        """
        chunks = chunk_by_words(text, approx_words_per_chunk)
        vectors = [self.encode_one(c) for c in chunks]
        mean_vec = np.mean(np.stack(vectors, axis=0), axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.astype(np.float32)


class BgeM3ApiEncoder:
    """HTTP backend matching the original LegalGraphRAG repo's Ollama-served
    bge-m3 usage, kept for A/B comparison in Phase 4."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

    def encode_one(self, text: str) -> np.ndarray:
        import requests

        resp = requests.post(
            self.config.api_url,
            json={"model": "bge-m3", "input": text},
            timeout=120,
        )
        resp.raise_for_status()
        embeddings = resp.json().get("embeddings")
        if not embeddings or not embeddings[0]:
            raise ValueError(f"Embedding backend returned no vectors for: {text[:50]!r}")
        return np.asarray(embeddings[0], dtype=np.float32)

    def encode_long_text(self, text: str, approx_words_per_chunk: int = 180) -> np.ndarray:
        # bge-m3 supports much longer contexts; no chunking needed.
        return self.encode_one(text)


def embedder_from_config(config: EmbeddingConfig):
    """Build an embedder from an EmbeddingConfig alone, without reading .env."""
    if config.backend in ("vietnamese-bi-encoder", "phobert"):
        return VietnameseBiEncoder(config)
    if config.backend == "bge-m3":
        return BgeM3ApiEncoder(config)
    raise ValueError(f"Unknown embedding backend: {config.backend!r}")


def get_embedder(config: Optional[AppConfig] = None):
    config = config or AppConfig.from_env_file()
    return embedder_from_config(config.embedding)


class CachedEmbedder:
    """Disk cache in front of an embedder, keyed by (model name, text), so
    rebuilding the graph does not re-encode unchanged texts."""

    def __init__(self, inner, model_name: str, cache_dir: str = ".cache/emb"):
        self.inner = inner
        self.dir = os.path.join(cache_dir, re.sub(r"[^\w.-]", "_", model_name))
        os.makedirs(self.dir, exist_ok=True)
        self.hits = self.misses = 0

    def _path(self, kind: str, text: str) -> str:
        digest = hashlib.sha256(f"{kind}\n{text}".encode("utf-8")).hexdigest()
        return os.path.join(self.dir, f"{digest}.npy")

    def _cached(self, kind: str, text: str, compute) -> np.ndarray:
        path = self._path(kind, text)
        if os.path.exists(path):
            self.hits += 1
            return np.load(path)
        self.misses += 1
        vec = np.asarray(compute(text), dtype=np.float32)
        np.save(path, vec)
        return vec

    def encode_one(self, text: str) -> np.ndarray:
        return self._cached("one", text, self.inner.encode_one)

    def encode_long_text(self, text: str) -> np.ndarray:
        return self._cached("long", text, self.inner.encode_long_text)
