import json
from pathlib import Path
import subprocess
import sys


def test_cli_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    req = repo_root / "examples" / "req_graph_editor.txt"
    out = tmp_path / "out"

    cmd = [sys.executable, "-m", "ucgen", "run", "--req", str(req), "--out", str(out), "--max-iters", "2", "--backend", "mock", "--no-interactive"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (out / "final_usecase.puml").exists()
    assert (out / "report.json").exists()

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "iterations" in report
