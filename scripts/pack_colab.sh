#!/usr/bin/env bash
# Data the Colab notebook (notebooks/colab_qwen_neo4j.ipynb) needs that is
# not in git: the graph (exported to Neo4j from there), the test cases, the
# questions and the embedding cache. Upload the archive to Google Drive
# under MyDrive/KhoaLuan/. No .env, no keys.
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf outputs/colab_bundle.tar.gz \
  outputs/hierargraph.pkl \
  data/processed/cases_vn_test.json \
  data/raw/questions.xlsx \
  .cache/emb
ls -lh outputs/colab_bundle.tar.gz
