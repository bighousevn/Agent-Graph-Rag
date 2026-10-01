#!/usr/bin/env python3
"""Phase 3.2: LLM summary for each case community (becomes a Cluster node).

Run by the user (needs LLM_API_KEY in .env). Claude only runs --dry-run,
which never loads .env.

    python scripts/summarize_clusters.py --dry-run
    python scripts/summarize_clusters.py
    python scripts/build_graph.py      # run again to attach the summaries

Reads data/processed/cluster_inputs.json (written by build_graph.py) and
writes data/processed/cluster_summaries.json as
{cluster_id: {"input": ..., "summary": ...}}. The input text is stored so
build_graph.py can refuse a summary made for a different community.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vn_legal_graph.graph.build import cluster_prompt

CHARS_PER_TOKEN = 3.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "cluster_inputs.json"), encoding="utf-8") as f:
        inputs = json.load(f)
    prompts = [(str(ci["cluster_id"]), ci["input"], cluster_prompt(ci)) for ci in inputs]

    if args.dry_run:
        chars = sum(len(p) for _, _, p in prompts)
        print(f"{len(prompts)} cụm, ~{chars / CHARS_PER_TOKEN:,.0f} token đầu vào")
        print("\nPrompt cụm đầu tiên:\n" + prompts[0][2] if prompts else "Không có cụm nào.")
        return

    from vn_legal_graph.config import AppConfig
    from vn_legal_graph.llm import LLMClient

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    out = {}
    for i, (cid, text, prompt) in enumerate(prompts, 1):
        # The prompt asks for one line; keep the first non-empty one.
        lines = [l.strip() for l in client.generate(prompt, max_tokens=128).splitlines() if l.strip()]
        if not lines:
            print(f"[{i}/{len(prompts)}] cụm {cid}: LLM trả về rỗng, bỏ qua")
            continue
        summary = lines[0]
        out[cid] = {"input": text, "summary": summary}
        print(f"[{i}/{len(prompts)}] cụm {cid}: {summary}")

    path = os.path.join(args.data_dir, "cluster_summaries.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {path}. Chạy lại scripts/build_graph.py để gắn node Cluster.")


if __name__ == "__main__":
    main()
