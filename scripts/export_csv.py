#!/usr/bin/env python3
"""Export the HierarGraph to CSV files (nodes, one file per node type,
edges, guidance units) for reading in Excel. No LLM, no .env.

    python scripts/export_csv.py                       # -> outputs/graph_csv/
    python scripts/export_csv.py --out some/dir
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.graph.csv_export import export_csv
from vn_legal_graph.graph.graph_db import HierarGraph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", default="outputs/hierargraph.pkl")
    parser.add_argument("--out", default="outputs/graph_csv")
    args = parser.parse_args()
    counts = export_csv(HierarGraph.load(args.graph), args.out)
    for name, n in counts.items():
        print(f"{os.path.join(args.out, name)}: {n:,} dòng")


if __name__ == "__main__":
    main()
