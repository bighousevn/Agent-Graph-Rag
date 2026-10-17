#!/usr/bin/env bash
# Start the web UI (app/ui.py): Ollama with the Kaggle settings if it is not
# running yet, then Streamlit on http://localhost:8501.
set -euo pipefail
cd "$(dirname "$0")"

if ! curl -s http://127.0.0.1:11434 >/dev/null; then
  echo "Khởi động Ollama (ngữ cảnh 24k)…"
  OLLAMA_CONTEXT_LENGTH=24576 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_KEEP_ALIVE=-1 \
    nohup ollama serve > .cache/ollama.log 2>&1 &
  for _ in $(seq 1 30); do curl -s http://127.0.0.1:11434 >/dev/null && break; sleep 1; done
fi

# Headless: no first-run e-mail prompt; the browser is opened here instead.
URL=http://localhost:8501
( sleep 4; command -v open >/dev/null && open "$URL" || { command -v xdg-open >/dev/null && xdg-open "$URL"; } ) >/dev/null 2>&1 &
echo "Giao diện: $URL"
exec .venv/bin/streamlit run app/ui.py --server.headless true --browser.gatherUsageStats false
