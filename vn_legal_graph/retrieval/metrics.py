"""Retrieval metrics. A case's gold labels are the BLHS articles it was
convicted under (often 2-3 for multi-crime cases)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


def recall_at_k(gold: Iterable[int], predicted: Sequence[int], k: int) -> float:
    gold = set(gold)
    return len(gold & set(predicted[:k])) / len(gold) if gold else 0.0


def hit_at_k(gold: Iterable[int], predicted: Sequence[int], k: int) -> float:
    return 1.0 if set(gold) & set(predicted[:k]) else 0.0


def evaluate(rows: List[Dict], ks: Sequence[int] = (1, 2, 3)) -> Dict:
    """``rows``: [{"gold": [...], "pred": [...]}]. Returns mean Recall@k and
    Hit@k over cases, plus per-article recall: for each article, the share
    of the cases containing it where it appears in the top k."""
    out: Dict = {"n": len(rows)}
    for k in ks:
        out[f"R@{k}"] = sum(recall_at_k(r["gold"], r["pred"], k) for r in rows) / len(rows)
        out[f"Hit@{k}"] = sum(hit_at_k(r["gold"], r["pred"], k) for r in rows) / len(rows)
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for a in r["gold"]:
            for k in ks:
                per[a][k].append(1.0 if a in r["pred"][:k] else 0.0)
    out["theo_dieu"] = {
        a: {f"R@{k}": sum(v) / len(v) for k, v in by_k.items()} | {"n": len(by_k[ks[0]])}
        for a, by_k in sorted(per.items())
    }
    return out
