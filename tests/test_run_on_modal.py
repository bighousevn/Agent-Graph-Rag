"""Tests for scripts/run_on_modal.py (no Modal call)."""
import importlib.util
import subprocess
import sys

from vn_legal_graph.qa.batch import write_csv

spec = importlib.util.spec_from_file_location("run_on_modal", "scripts/run_on_modal.py")
rom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rom)


def test_modal_constants_read_without_modal():
    c = rom.modal_constants("modal_app.py")
    assert c["MODEL"] and c["GPU"] and c["MAX_CONTAINERS"] == 1
    assert c["PARALLEL"] >= 1 and c["GPU"] in rom.GPU_USD_PER_SECOND


def test_load_env_file(tmp_path):
    f = tmp_path / "p.env"
    f.write_text("# note\nexport A=1\nB='x y'\n\nC=\"z\"\nnot a line\n", encoding="utf-8")
    assert rom.load_env_file(str(f), base={}) == {"A": "1", "B": "x y", "C": "z"}


def test_load_env_file_expands_variables_like_a_shell(tmp_path):
    # .modal_proxy.env writes LLM_BASE_URL=$QWEN_MODAL_URL (run 2026-11-21 failed with MissingSchema)
    f = tmp_path / "p.env"
    f.write_text("URL=https://x.modal.run\nexport LLM_BASE_URL=$URL\nexport UI=\"${URL}/v1\"\n"
                 "LIT='$URL'\nHOME_DIR=$H/a\nMISSING=$NOPE\n", encoding="utf-8")
    env = rom.load_env_file(str(f), base={"H": "/home"})
    assert env["LLM_BASE_URL"] == "https://x.modal.run" and env["UI"] == "https://x.modal.run/v1"
    assert env["LIT"] == "$URL" and env["HOME_DIR"] == "/home/a" and env["MISSING"] == ""


def test_restart_limit_and_battery_check():
    assert rom.MAX_RESTARTS >= 1
    assert isinstance(rom.on_battery(), bool)  # False off macOS (no pmset)


def test_shards_and_command():
    assert rom.shard_names(2) == ["1/2", "2/2"] and rom.shard_names(1) == [""]
    assert rom.model_slug("qwen3.6:35b-a3b") == "qwen3.6_35b-a3b"

    class A:
        input, id_col, text_col, tag, ids, limit = "q.csv", "id", "cau_hoi", "t", "ids.txt", 0
    cmd = rom.batch_command(A, "1/2")
    assert cmd[1:3] == ["scripts/answer_batch.py", "--input"] and "--ids" in cmd and cmd[-2:] == ["--shard", "1/2"]
    assert "--limit" not in cmd


def test_dry_run_calls_nothing_on_modal(tmp_path):
    write_csv(str(tmp_path / "q.csv"), [{"id": str(i), "cau_hoi": f"câu hỏi {i}"} for i in range(5)])
    res = subprocess.run([sys.executable, "scripts/run_on_modal.py", "--input", str(tmp_path / "q.csv"),
                          "--ids", "1,3", "--tag", "thu", "--dry-run", "--env-file", "/nonexistent"],
                         check=True, capture_output=True, text=True)
    assert "2 câu" in res.stdout and "Kế hoạch" in res.stdout and "modal app stop" in res.stdout
