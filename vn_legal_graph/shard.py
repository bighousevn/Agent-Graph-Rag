"""Split a run over several notebooks/GPUs: --shard K/N keeps every N-th
item starting at the K-th (1-based), so shards are disjoint and of even
size whatever the order; --merge puts the result files back together."""
from __future__ import annotations

from typing import List, Sequence, Tuple, TypeVar

T = TypeVar("T")


def parse_shard(text: str) -> Tuple[int, int]:
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard phải có dạng K/N, ví dụ 1/3: {text!r}") from None
    if not 1 <= k <= n:
        raise ValueError(f"--shard {text}: cần 1 <= K <= N")
    return k, n


def take_shard(items: Sequence[T], text: str) -> List[T]:
    k, n = parse_shard(text)
    return list(items[k - 1 :: n])


def shard_suffix(text: str) -> str:
    if not text or text == "1/1":
        return ""
    k, n = parse_shard(text)
    return f"_phan{k}-{n}"
