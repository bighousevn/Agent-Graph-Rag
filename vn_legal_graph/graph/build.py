"""Build the HierarGraph, following the original repo's
construct_feature_graph (core/graph_construct/feature_graph.py):

1. Law + Crime nodes from law_to_crime_vn.json, edge Law -RELATED_CRIME-> Crime.
2. Case nodes from the corpus features file, edge Case -RELATES_TO_LAW-> Law.
3. kNN top-3 between cases: Case -SIMILAR_TO-> Case (with score).
4. Louvain communities + PageRank + degree on the whole graph.
5. Per community, an LLM summary becomes a Cluster node:
   Case -BELONGS_TO-> Cluster.

Step 5 needs an LLM, which the user runs, so it is split in two:
``cluster_inputs`` produces the text to summarize and ``attach_clusters``
adds the Cluster nodes once summaries exist.

Kept from the original on purpose (for a faithful reproduction):
- Cases whose features have no "hành vi phạm tội" are left out.
- Cluster representatives are ranked by 0.7*PageRank + 0.3*degree. The
  degree term dominates (degrees are integers, PageRank sums to 1).
"""
from __future__ import annotations

import collections
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..prompts.vi import SUMMARIZE_CLUSTER_PROMPT
from .graph_db import HierarGraph

Embed = Callable[[str], np.ndarray]


def law_id(entry: int, suffix: str = "", bo_luat: str = "BLHS") -> str:
    """BLHS keeps the short id ("law:249"); other codes are namespaced
    ("law:bltths:155"), since BLTTHS Điều 155 is not BLHS Điều 155."""
    if bo_luat == "BLHS":
        return f"law:{entry}{suffix}"
    return f"law:{bo_luat.lower()}:{entry}{suffix}"


def crime_id(entry: int, suffix: str = "") -> str:
    return f"crime:{entry}{suffix}"


def case_id(record_id: str) -> str:
    return f"case:{record_id}"


def build_base_graph(
    law_to_crime: List[Dict], cases: List[Dict], embed: Embed
) -> Tuple[HierarGraph, Dict[str, list]]:
    """Steps 1-2. Returns the graph and a report of skipped cases and
    case->law links that had no Law node."""
    g = HierarGraph()
    for law in law_to_crime:
        item = law["items"][0]
        suffix = law.get("suffix", "")
        bo_luat = law.get("bo_luat", "BLHS")
        lid, cid = law_id(law["id"], suffix, bo_luat), crime_id(law["id"], suffix)
        g.add_node(
            lid,
            "Law",
            embedding=embed(item["text"]),
            entry=law["id"],
            description=item["text"],
            crimes=item["crime"],
            judge_dep=item.get("judge_dep", []),
            related_laws=item.get("related_laws", []),
            insights="",
            title=law.get("title", ""),
            bo_luat=bo_luat,
            suffix=suffix,
        )
        if item["crime"]:  # general-part articles (include_all) have no crime
            crime_title = item["crime"][0]
            g.add_node(cid, "Crime", embedding=embed(crime_title), entry=law["id"], description=crime_title)
            g.add_edge(lid, cid, "RELATED_CRIME", match_type="exact")

    report = {"bo_qua_dac_trung_loi": [], "bo_qua_khong_co_hanh_vi": [], "thieu_law": []}
    for case in cases:
        if case.get("vai_tro") != "corpus":
            raise ValueError(
                f"{case['id']} has vai_tro={case.get('vai_tro')!r}: only corpus cases go into the graph"
            )
        if case.get("dac_trung_loi"):
            report["bo_qua_dac_trung_loi"].append(case["id"])
            continue
        if not case["dac_trung"].get("criminal_acts"):
            report["bo_qua_khong_co_hanh_vi"].append(case["id"])
            continue
        nid = case_id(case["id"])
        g.add_node(
            nid,
            "Case",
            embedding=embed(case["mo_ta_dac_trung"]),
            caseId=case["id"],
            description=case["mo_ta_dac_trung"],
            crime=case["toi_danh"],
            law=case["dieu"],
            dieu_khoan=case.get("dieu_khoan", []),
        )
        for article in case["dieu"]:
            lid = law_id(article)
            if lid in g.graph:
                g.add_edge(nid, lid, "RELATES_TO_LAW")
            else:
                report["thieu_law"].append((case["id"], article))
    return g, report


def add_knn_edges(g: HierarGraph, k: int = 3) -> int:
    """Step 3: each Case gets SIMILAR_TO edges to its k most similar cases."""
    ids, sim = g.similarity_matrix("Case")
    if len(ids) < 2:
        return 0
    np.fill_diagonal(sim, -np.inf)
    added = 0
    for i, source in enumerate(ids):
        for j in np.argsort(-sim[i], kind="stable")[: min(k, len(ids) - 1)]:
            g.add_edge(source, ids[j], "SIMILAR_TO", score=float(sim[i, j]))
            added += 1
    return added


def detect_communities(g: HierarGraph, seed: int = 42) -> Dict[int, List[str]]:
    """Step 4. Louvain and PageRank on the undirected graph, degree on the
    directed multigraph (as the original). Stores communityId / pagerank /
    degree on every node; returns community id -> Case node ids."""
    undirected = nx.Graph(g.graph.to_undirected())
    communities = nx.community.louvain_communities(undirected, seed=seed)
    pagerank = nx.pagerank(undirected)
    for comm_id, members in enumerate(sorted(communities, key=lambda c: min(c))):
        for node in members:
            g.graph.nodes[node].update(
                communityId=comm_id, pagerank=pagerank.get(node, 0.0), degree=g.graph.degree(node)
            )
    by_comm: Dict[int, List[str]] = collections.defaultdict(list)
    for node in g.nodes_of("Case"):
        by_comm[g.graph.nodes[node]["communityId"]].append(node)
    return {c: sorted(v) for c, v in sorted(by_comm.items())}


def cluster_inputs(
    g: HierarGraph, communities: Dict[int, List[str]], top_n: int = 10, top_crimes: int = 5
) -> List[Dict]:
    """Step 5a: for each community, the text the LLM summarizes: the main
    crimes (counted through Case -> Law -> Crime) and the descriptions of
    the top_n most central cases."""
    out = []
    for comm_id, members in communities.items():
        def centrality(n: str) -> float:
            d = g.graph.nodes[n]
            return d["pagerank"] * 0.7 + d["degree"] * 0.3

        ranked = sorted(members, key=lambda n: (-centrality(n), n))
        crime_counts: collections.Counter = collections.Counter()
        for n in members:
            for law in g.neighbors(n, "RELATES_TO_LAW"):
                for crime in g.neighbors(law, "RELATED_CRIME"):
                    crime_counts[g.graph.nodes[crime]["description"]] += 1
        top = sorted(crime_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_crimes]
        crime_line = "; ".join(f"{name} ({count} án)" for name, count in top)
        descriptions = "\n".join(g.graph.nodes[n]["description"] for n in ranked[:top_n])
        text = (
            "Thông tin cụm:\n"
            f"- Tội danh chủ yếu: {crime_line}\n\n"
            f"Mô tả các vụ án tiêu biểu:\n{descriptions}"
        )
        out.append(
            {
                "cluster_id": comm_id,
                "case_ids": members,
                "top_crimes": [name for name, _ in top],
                "top_crime_counts": [count for _, count in top],
                "input": text,
            }
        )
    return out


def cluster_prompt(cluster_input: Dict) -> str:
    return SUMMARIZE_CLUSTER_PROMPT + "\n\n" + cluster_input["input"]


def attach_clusters(
    g: HierarGraph, inputs: List[Dict], summaries: Dict[str, Dict], embed: Embed
) -> Dict[str, list]:
    """Step 5b: add Cluster nodes and BELONGS_TO edges. A summary is used
    only if it was made from exactly the same input text, so summaries from
    an older graph build are never attached to a different community."""
    report: Dict[str, list] = {"gan": [], "thieu_tom_tat": [], "tom_tat_cu": []}
    for ci in inputs:
        key = str(ci["cluster_id"])
        entry: Optional[Dict] = summaries.get(key)
        if entry is None:
            report["thieu_tom_tat"].append(key)
            continue
        if entry.get("input") != ci["input"]:
            report["tom_tat_cu"].append(key)
            continue
        nid = f"cluster:{key}"
        g.add_node(
            nid,
            "Cluster",
            embedding=embed(entry["summary"]),
            summary=entry["summary"],
            top_crimes=ci["top_crimes"],
            top_crime_counts=ci["top_crime_counts"],
        )
        for case in ci["case_ids"]:
            g.add_edge(case, nid, "BELONGS_TO")
        report["gan"].append(key)
    return report
