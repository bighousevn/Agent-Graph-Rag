"""Tests for vn_legal_graph.embedding.

Only the pure-Python helpers (word segmentation, chunking) are tested
here — they need no network access and no downloaded model. Encoding with
the actual sentence-transformers model is exercised manually / in later
phases once the model is cached locally, not in this fast unit-test
suite.
"""
import pytest

from vn_legal_graph.embedding import chunk_by_words, segment_vietnamese


def test_segment_vietnamese_joins_compound_words():
    segmented = segment_vietnamese("chiếm đoạt tài sản")
    # pyvi joins multi-syllable Vietnamese words with underscores so the
    # PhoBERT tokenizer sees them as single tokens.
    assert "_" in segmented


def test_chunk_by_words_returns_single_chunk_when_short():
    text = "Người nào dùng vũ lực nhằm chiếm đoạt tài sản"
    chunks = chunk_by_words(text, max_words=180)
    assert chunks == [text]


def test_chunk_by_words_splits_long_text():
    text = " ".join(f"tu{i}" for i in range(500))
    chunks = chunk_by_words(text, max_words=180)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "tu0"
    assert chunks[-1].split()[-1] == "tu499"
    # No word should be dropped or duplicated across chunk boundaries.
    rejoined = " ".join(chunks)
    assert rejoined == text


def test_chunk_by_words_exact_multiple_boundary():
    text = " ".join(f"tu{i}" for i in range(360))  # exactly 2 * 180
    chunks = chunk_by_words(text, max_words=180)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 180
    assert len(chunks[1].split()) == 180
