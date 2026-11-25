"""A Qwen model (Ollama) on a Modal GPU, behind Modal proxy auth, so the UI
and the scripts can use it instead of a local Ollama.

    pip install modal && modal setup            # once: log in to Modal
    modal run modal_app.py::pull                # model -> Modal Volume (no GPU)
    modal deploy modal_app.py                   # prints the URL of `serve`

Then create a proxy auth token (Modal dashboard: Settings -> Proxy Auth
Tokens) and point the code at the URL:

    export LLM_BASE_URL=https://<workspace>--legal-qwen-serve.modal.run   # scripts
    export UI_LLM_BASE_URL=$LLM_BASE_URL                                 # web UI
    export MODAL_PROXY_KEY=wk-...  MODAL_PROXY_SECRET=ws-...

scripts/run_on_modal.py does all of this and answers a question file, then
stops the app. The GPU stops after SCALEDOWN_SECONDS without requests; the
next request waits for a cold start. Every HTTP request is limited to 150 s
by Modal.

Model history: qwen3:8b-q8_0 on an L4 (as the Kaggle runs until fbv4);
2026-11-21 qwen3.6:35b-a3b (MoE, ~3B active, ~24 GB) on an L40S (48 GB),
user choice, to compare with Qwen3-8B and Qwen3.5-9B on the 106 graded
questions.
"""
import subprocess
import time
import urllib.request

import modal

MODEL = "qwen3.6:35b-a3b"
OLLAMA_VERSION = "0.40.16"  # latest release on 2026-11-21 (0.40.5 predates Qwen3.6)
PORT = 11434
GPU = "L40S"  # 48 GB: the 24 GB model plus two 24k contexts
PARALLEL = 2  # requests one GPU answers at once (one per --shard)
SCALEDOWN_SECONDS = 300
MAX_CONTAINERS = 1  # cost cap: one GPU at a time; it serves PARALLEL requests

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VERSION} sh")
    .env({
        "OLLAMA_HOST": f"0.0.0.0:{PORT}",
        "OLLAMA_MODELS": "/models",
        "OLLAMA_CONTEXT_LENGTH": "24576",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_KEEP_ALIVE": "-1",
        "OLLAMA_NUM_PARALLEL": str(PARALLEL),
    })
)
models = modal.Volume.from_name("legal-qwen-models", create_if_missing=True)
app = modal.App("legal-qwen", image=image)


def start_ollama() -> subprocess.Popen:
    proc = subprocess.Popen(["ollama", "serve"])
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}", timeout=2)
            return proc
        except OSError:
            time.sleep(1)
    raise RuntimeError("ollama serve did not start")


@app.function(volumes={"/models": models}, timeout=3600)
def pull():
    """Download the model into the Volume (CPU only; a model already there is kept)."""
    start_ollama()
    subprocess.run(["ollama", "pull", MODEL], check=True)
    models.commit()
    print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)


@app.function(gpu=GPU, volumes={"/models": models}, timeout=24 * 3600,
              scaledown_window=SCALEDOWN_SECONDS, max_containers=MAX_CONTAINERS)
@modal.concurrent(max_inputs=PARALLEL)
@modal.web_server(port=PORT, startup_timeout=600, requires_proxy_auth=True)
def serve():
    start_ollama()
    # Load the model on the GPU before the first request is let in.
    body = f'{{"model": "{MODEL}", "keep_alive": -1}}'.encode()
    urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/generate", data=body, headers={"Content-Type": "application/json"}), timeout=600)
