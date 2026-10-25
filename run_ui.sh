#!/usr/bin/env bash
# Start the web UI (app/ui.py) on http://localhost:8501.
#   ./run_ui.sh          Qwen on Modal when .modal_proxy.env exists (modal_app.py), else local Ollama
#   ./run_ui.sh --local  always local Ollama (started here with the Kaggle settings if not running)
# Stop with Ctrl+C. A local Ollama started here keeps running: `ollama stop qwen3:8b-q8_0`, `pkill -f "ollama serve"`.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" != "--local" && -f .modal_proxy.env ]]; then
  # shellcheck disable=SC1091
  source .modal_proxy.env
  echo "Qwen: Modal ($UI_LLM_BASE_URL). GPU tự bật khi có câu hỏi, tự tắt sau 5 phút rảnh."
else
  unset UI_LLM_BASE_URL
  if ! curl -s http://127.0.0.1:11434 >/dev/null; then
    echo "Khởi động Ollama trên máy (ngữ cảnh 24k)…"
    OLLAMA_CONTEXT_LENGTH=24576 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_KEEP_ALIVE=-1 \
      nohup ollama serve > .cache/ollama.log 2>&1 &
    for _ in $(seq 1 30); do curl -s http://127.0.0.1:11434 >/dev/null && break; sleep 1; done
  fi
  echo "Qwen: Ollama trên máy (http://127.0.0.1:11434)."
fi

# Headless: no first-run e-mail prompt; the browser is opened here instead.
URL=http://localhost:8501
( sleep 4; command -v open >/dev/null && open "$URL" || { command -v xdg-open >/dev/null && xdg-open "$URL"; } ) >/dev/null 2>&1 &
echo "Giao diện: $URL   (tắt: Ctrl+C)"
exec .venv/bin/streamlit run app/ui.py --server.headless true --browser.gatherUsageStats false
