#!/usr/bin/env python3
"""Phase 3, step 1: extract case features with the LLM.

Run by the user (needs LLM_API_KEY in .env). Claude only runs --dry-run,
which never loads .env.

    # Count prompts / estimate tokens, write a sample prompt. No LLM, no .env.
    python scripts/extract_case_features.py --dry-run

    # Try a few cases first and read the output:
    python scripts/extract_case_features.py --per-crime 2

    # Full run (corpus + test):
    python scripts/extract_case_features.py

Answers are cached in .cache/llm, so re-running (e.g. after --limit 5, or
after a crash) only pays for prompts not answered yet.

Corpus cases get the crime as a hint (as the original repo's corpus
script does); test cases never do. --no-crime-hint turns the hint off for
the corpus too (ablation).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import collections

from vn_legal_graph.cases.features import DEFAULT_MAX_CHARS, annotate_cases, audit_features, build_prompt

INPUTS = {
    "corpus": ("cases_vn.json", "cases_vn_features.json"),
    "test": ("cases_vn_test.json", "cases_vn_test_features.json"),
}

# Rough: Vietnamese text runs ~3-4 characters per token on common tokenizers.
CHARS_PER_TOKEN = 3.5


def print_audit(role, rows) -> None:
    flagged = collections.defaultdict(list)
    for r in rows:
        for flag in audit_features(r):
            flagged[flag].append(r["id"])
    print(f"  kiểm tra đặc trưng ({role}, {len(rows)} án):")
    if not flagged:
        print("    không có án nào bị gắn cờ")
    for flag, ids in sorted(flagged.items()):
        print(f"    {flag}: {len(ids)} {ids[:6]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--roles", nargs="*", default=["corpus", "test"], choices=list(INPUTS))
    parser.add_argument("--limit", type=int, default=None, help="Only the first N cases of each file.")
    parser.add_argument(
        "--per-crime", type=int, default=None,
        help="Only the first N cases per crime of each file (a trial that covers every crime).",
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--no-crime-hint", action="store_true")
    parser.add_argument("--dotenv-path", default=".env")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--audit", action="store_true",
        help="Only audit the existing output files for unsupported features. No LLM, no .env.",
    )
    args = parser.parse_args()

    use_hint = not args.no_crime_hint
    if args.audit:
        for role in args.roles:
            path = os.path.join(args.data_dir, INPUTS[role][1])
            with open(path, encoding="utf-8") as f:
                print_audit(role, json.load(f))
        return

    jobs = []
    for role in args.roles:
        src, dst = INPUTS[role]
        with open(os.path.join(args.data_dir, src), encoding="utf-8") as f:
            cases = json.load(f)
        if args.per_crime:
            seen = {}
            picked = []
            for c in cases:
                key = tuple(c["dieu"])
                if seen.get(key, 0) < args.per_crime:
                    picked.append(c)
                    seen[key] = seen.get(key, 0) + 1
            cases = picked
        if args.limit:
            cases = cases[: args.limit]
        jobs.append((role, cases, os.path.join(args.data_dir, dst)))

    if args.dry_run:
        total_chars = 0
        for role, cases, dst in jobs:
            chars = sum(
                len(build_prompt(c["dien_bien"], c["toi_danh"] if use_hint and role == "corpus" else None, args.max_chars))
                for c in cases
            )
            total_chars += chars
            print(f"{role}: {len(cases)} prompt, ~{chars / CHARS_PER_TOKEN:,.0f} token đầu vào -> {dst}")
        print(f"Tổng: ~{total_chars / CHARS_PER_TOKEN:,.0f} token đầu vào (+ ~150 token đầu ra mỗi án)")
        role, cases, _ = jobs[0]
        sample_path = os.path.join(args.data_dir, "sample_feature_prompt.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(build_prompt(cases[0]["dien_bien"], cases[0]["toi_danh"] if use_hint and role == "corpus" else None, args.max_chars))
        print(f"Prompt mẫu: {sample_path}")
        return

    from vn_legal_graph.config import AppConfig
    from vn_legal_graph.llm import LLMClient

    client = LLMClient(AppConfig.from_env_file(args.dotenv_path))
    for role, cases, dst in jobs:
        progress = lambda n, total=len(cases), role=role: print(f"\r{role}: {n}/{total}", end="", flush=True)
        rows = annotate_cases(cases, client.generate, use_hint, args.max_chars, on_progress=progress)
        print()
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        failed = [r["id"] for r in rows if r["dac_trung_loi"]]
        no_acts = [r["id"] for r in rows if not r["dac_trung"]["criminal_acts"]]
        print(f"{role}: {len(rows)} án -> {dst}")
        print(f"  không đọc được JSON: {len(failed)} {failed[:5]}")
        print(f"  không có 'hành vi phạm tội' (bài gốc loại các án này khỏi graph): {len(no_acts)} {no_acts[:5]}")
        print_audit(role, rows)


if __name__ == "__main__":
    main()
