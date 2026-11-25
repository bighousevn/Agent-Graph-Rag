#!/usr/bin/env python3
"""Answer a question file with the model of modal_app.py on a Modal GPU, in
one go: download the model (if needed), deploy, check the model answers
without thinking, run answer_batch.py in --shards parallel parts against the
Modal URL, merge, and always stop the Modal app at the end (no GPU left
running).

The pipeline (rewrite, retrieval, prompts, citation check) runs on this
machine; only the LLM runs on Modal. Keep the machine awake: on macOS the
script runs `caffeinate` for its own lifetime. A run that stops halfway is
resumed from the LLM cache (.cache/llm) by running it again.

    python scripts/run_on_modal.py --input data/raw/fb/cau_hoi.csv \
        --ids results/qwen36_cham106/ids_106.txt --tag cham106 --out-dir results/qwen36_cham106
    python scripts/run_on_modal.py ... --dry-run     # plan and question count only, no Modal call

Needs .modal_proxy.env (proxy token and URL, see modal_app.py) and `modal`
logged in (`modal setup`).
"""
import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

APP_NAME = "legal-qwen"
# Modal list prices per second (modal.com/pricing, read 2026-11-14), for the cost line only.
GPU_USD_PER_SECOND = {"T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542,
                      "A100-40GB": 0.000583, "A100-80GB": 0.000694, "H100": 0.001097}


def modal_constants(path: str) -> dict:
    """The upper-case string/int constants of modal_app.py (MODEL, GPU, ...), read without importing modal."""
    out = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out[t.id] = node.value.value
    return out


def load_env_file(path: str, base=None) -> dict:
    """KEY=VALUE lines (optional `export`, quotes, # comments) of a shell env file,
    with $VAR / ${VAR} taken from earlier lines or base (as a shell would;
    single quotes keep $ as is). Values are never printed."""
    base = dict(os.environ if base is None else base)
    env = {}

    def expand(m):
        name = m.group(1) or m.group(2)
        return env.get(name, base.get(name, ""))

    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if m:
            raw = m.group(2).strip()
            if len(raw) >= 2 and raw[0] == raw[-1] == "'":
                env[m.group(1)] = raw[1:-1]
            else:
                env[m.group(1)] = re.sub(r"\$\{(\w+)\}|\$(\w+)", expand, raw.strip('"'))
    return env


def model_slug(model: str) -> str:
    return re.sub(r"[^\w.-]", "_", model)


def shard_names(n: int) -> list:
    return [f"{k}/{n}" for k in range(1, n + 1)] if n > 1 else [""]


def batch_command(args, shard: str = "") -> list:
    cmd = [sys.executable, "scripts/answer_batch.py", "--input", args.input, "--id-col", args.id_col,
           "--text-col", args.text_col, "--tag", args.tag]
    if args.ids:
        cmd += ["--ids", args.ids]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if shard:
        cmd += ["--shard", shard]
    return cmd


def modal_cli() -> str:
    local = os.path.join(os.path.dirname(sys.executable), "modal")
    return local if os.path.exists(local) else "modal"


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def check_model(minutes: int = 15) -> None:
    """Wait for the cold start, then ask a true/false question twice: the model must answer at once, without thinking."""
    from vn_legal_graph.config import AppConfig
    from vn_legal_graph.judge.judge_law import parse_bool
    from vn_legal_graph.llm import LLMClient

    import requests

    client = LLMClient(AppConfig.from_env_file(os.path.join(ROOT, ".env")))
    q = "Trả lời đúng một từ true hoặc false: trộm cắp tài sản là tội phạm?"
    start, last = time.time(), None
    while time.time() - start < minutes * 60:
        try:
            client._ollama_chat(q, 16)
            break
        except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema,
                requests.exceptions.InvalidSchema) as e:  # a config error, waiting will not help
            raise SystemExit(f"URL của Modal không hợp lệ ({type(e).__name__}): kiểm tra LLM_BASE_URL trong .modal_proxy.env")
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise SystemExit("Modal từ chối khóa proxy (401/403): kiểm tra MODAL_PROXY_KEY/SECRET")
            last = e
        except Exception as e:  # cold start: container + model load
            last = e
        if last is not None:
            print(f"  đang khởi động GPU ({(time.time() - start) / 60:.0f} phút)...", flush=True)
            time.sleep(20)
    else:
        raise SystemExit(f"Model trên Modal không trả lời sau {minutes} phút: {type(last).__name__}")
    t = time.time()
    raw, truncated = client._ollama_chat(q, 16)
    took = time.time() - t
    print(f"  thử: {raw!r}, {took:.1f}s (khởi động {(t - start) / 60:.1f} phút)", flush=True)
    if truncated or parse_bool(raw) is not True:
        raise SystemExit("Model vẫn suy nghĩ hoặc không trả lời true/false: dừng, không chạy tiếp")
    if took > 15:
        raise SystemExit(f"Một câu true/false mất {took:.0f}s: quá chậm, dừng")


MAX_RESTARTS = 5  # per part: a network drop (machine asleep, Wi-Fi) kills a part; it resumes from the LLM cache


def on_battery() -> bool:
    """macOS on battery: caffeinate cannot stop system sleep then, and a closed lid always sleeps."""
    try:
        return "Battery Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    except OSError:
        return False


def count_done(log_path: str) -> int:
    if not os.path.exists(log_path):
        return 0
    return sum(1 for line in open(log_path, encoding="utf-8") if line.startswith("["))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--ids", default="", help="Only these question ids (comma list or a file), as answer_batch.py.")
    p.add_argument("--id-col", default="id")
    p.add_argument("--text-col", default="cau_hoi")
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shards", type=int, default=0, help="Parallel parts (default: PARALLEL of modal_app.py).")
    p.add_argument("--out-dir", default="", help="Copy the merged .json/.xlsx and logs here.")
    p.add_argument("--env-file", default=".modal_proxy.env")
    p.add_argument("--skip-pull", action="store_true", help="The model is already in the Modal Volume.")
    p.add_argument("--keep-running", action="store_true", help="Do not stop the Modal app at the end.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    os.chdir(ROOT)

    const = modal_constants(os.path.join(ROOT, "modal_app.py"))
    model, gpu = const["MODEL"], const["GPU"]
    shards = shard_names(args.shards or int(const.get("PARALLEL", 1)))
    slug = model_slug(model)
    print(f"Model {model} trên GPU {gpu} (Modal), {len(shards)} phần song song, tag {args.tag}")
    run(batch_command(args) + ["--dry-run"])
    if args.dry_run:
        pull = "" if args.skip_pull else "modal run modal_app.py::pull -> "
        print(f"Kế hoạch: {pull}modal deploy -> thử model -> chạy -> gộp -> modal app stop")
        return

    os.environ.update(load_env_file(os.path.join(ROOT, args.env_file)))
    # Modal cuts every request at 150 s; a longer client timeout only hangs on a dead connection.
    os.environ.update({"LLM_PROVIDER": "ollama", "LLM_MODEL": model, "LLM_TEMPERATURE": "0", "LLM_TIMEOUT": "170",
                       "LLM_API_KEY": os.environ.get("LLM_API_KEY") or "ollama", "PYTHONUNBUFFERED": "1"})
    if not (os.environ.get("LLM_BASE_URL", "").startswith("https://") and os.environ.get("MODAL_PROXY_KEY")):
        raise SystemExit(f"{args.env_file}: LLM_BASE_URL không phải https://... hoặc thiếu MODAL_PROXY_KEY")
    awake = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())]) if shutil.which("caffeinate") else None
    if on_battery():
        print("CẢNH BÁO: máy đang chạy pin. Hãy cắm sạc và mở nắp, nếu không máy vẫn ngủ và mất mạng giữa chừng.", flush=True)

    t0, gpu_from = time.time(), None
    try:
        if not args.skip_pull:
            run([modal_cli(), "run", "modal_app.py::pull"])
        run([modal_cli(), "deploy", "modal_app.py"])
        gpu_from = time.time()
        check_model()

        os.makedirs("outputs", exist_ok=True)
        logs = {s: f"outputs/tra_loi_{args.tag}{'_phan' + s.replace('/', '-') if s else ''}.log" for s in shards}
        def launch(s, mode="w"):
            return subprocess.Popen(batch_command(args, s), cwd=ROOT, stdout=open(logs[s], mode),
                                    stderr=subprocess.STDOUT)

        procs = {s: launch(s) for s in shards}
        restarts = {s: 0 for s in shards}
        while True:
            for s, pr in procs.items():
                if pr.poll() not in (None, 0) and restarts[s] < MAX_RESTARTS:
                    restarts[s] += 1
                    print(f"  phần {s or 'toàn bộ'} lỗi (mã {pr.returncode}, xem {logs[s]}); chạy lại lần "
                          f"{restarts[s]}/{MAX_RESTARTS} sau 30 giây, các câu đã xong lấy từ cache", flush=True)
                    time.sleep(30)
                    procs[s] = launch(s, "a")
            if all(pr.poll() is not None for pr in procs.values()):
                break
            time.sleep(60)
            done = {s or "toàn bộ": count_done(logs[s]) for s in shards}
            print(time.strftime("%H:%M"), "đã xong (câu):", done,
                  f"| {(time.time() - gpu_from) / 60:.0f} phút GPU", flush=True)
        failed = {s: pr.returncode for s, pr in procs.items() if pr.returncode}
        if failed:
            raise SystemExit(f"Phần lỗi {failed}; xem {[logs[s] for s in failed]}. Chạy lại lệnh này để tiếp tục từ cache.")

        merged = f"outputs/answers_{slug}_{args.tag}.json"
        if shards != [""]:
            parts = [f"outputs/answers_{slug}_{args.tag}_phan{s.replace('/', '-')}.json" for s in shards]
            run([sys.executable, "scripts/answer_batch.py", "--merge", *parts, "--merge-out", merged])
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            for f in [merged, os.path.splitext(merged)[0] + ".xlsx", *logs.values()]:
                if os.path.exists(f):
                    shutil.copy(f, args.out_dir)
            print("Kết quả ở", args.out_dir)
    finally:
        if not args.keep_running:
            subprocess.run([modal_cli(), "app", "stop", "--yes", APP_NAME], cwd=ROOT)
            print("Đã tắt Modal app", APP_NAME, flush=True)
        if gpu_from:
            secs = time.time() - gpu_from
            rate = GPU_USD_PER_SECOND.get(gpu)
            cost = f", tối đa ~{secs * rate:.2f} USD" if rate else ""
            print(f"Tổng {(time.time() - t0) / 60:.0f} phút; GPU có thể bật tối đa ~{secs / 60:.0f} phút{cost} "
                  "(GPU chỉ chạy khi có lời gọi tới; chưa tính CPU và 5 phút chờ tắt). Số thật: Modal -> Usage")
        if awake:
            awake.terminate()


if __name__ == "__main__":
    main()
