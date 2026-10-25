"""The HierarGraph as CSV files a person can open in Excel (read-only copy;
the pipeline still reads outputs/hierargraph.pkl).

    <out>/nodes.csv          every node: id, type, label
    <out>/law.csv, crime.csv, case.csv, cluster.csv
                             one file per node type, one column per attribute
    <out>/edges.csv          source, target, relation, edge attributes
    <out>/guidance.csv       each guidance unit (Công văn / Nghị quyết item) and the article it is attached to

Lists of strings are joined with " | "; other lists and dicts are JSON.
Vectors are left out. UTF-8 with BOM so Excel shows Vietnamese correctly.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from .graph_db import NODE_TYPES, HierarGraph

SKIP = {"node_type", "embedding"}
FIRST = ["id", "entry", "suffix", "bo_luat", "title", "caseId", "description", "summary"]


def cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return " | ".join(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def label(d: Dict) -> str:
    """A short name for the node: a law's title, else the first line of its text."""
    text = d.get("title") or d.get("description") or d.get("summary") or ""
    return str(text).split("\n")[0][:120]


def _write(path: str, header: List[str], rows: List[List[str]]) -> int:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def export_csv(g: HierarGraph, out_dir: str) -> Dict[str, int]:
    """Write the CSV files; returns rows written per file."""
    os.makedirs(out_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    nodes = list(g.graph.nodes(data=True))
    counts["nodes.csv"] = _write(os.path.join(out_dir, "nodes.csv"), ["id", "type", "label"],
                                 [[n, d["node_type"], label(d)] for n, d in nodes])
    for node_type in NODE_TYPES:
        of_type = [(n, d) for n, d in nodes if d["node_type"] == node_type]
        keys = sorted({k for _, d in of_type for k in d} - SKIP)
        cols = [k for k in FIRST if k in keys or k == "id"] + [k for k in keys if k not in FIRST]
        rows = [[n if k == "id" else cell(d.get(k)) for k in cols] for n, d in of_type]
        counts[f"{node_type.lower()}.csv"] = _write(os.path.join(out_dir, f"{node_type.lower()}.csv"), cols, rows)
    edges = list(g.graph.edges(data=True))
    extra = sorted({k for _, _, d in edges for k in d} - {"relation"})
    counts["edges.csv"] = _write(os.path.join(out_dir, "edges.csv"), ["source", "target", "relation"] + extra,
                                 [[s, t, d["relation"]] + [cell(d.get(k)) for k in extra] for s, t, d in edges])
    guidance = [[n, label(d), item.get("id", ""), item.get("text", "")]
                for n, d in nodes if d["node_type"] == "Law"
                for item in (d.get("related_laws") or [])
                if isinstance(item, dict) and item.get("loai") == "van_ban_huong_dan"]
    counts["guidance.csv"] = _write(os.path.join(out_dir, "guidance.csv"),
                                    ["law_id", "law_label", "guidance_id", "guidance_text"], guidance)
    return counts
