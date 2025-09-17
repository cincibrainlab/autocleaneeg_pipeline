import os
import sys
import subprocess
from pathlib import Path


def run_cli(*args):
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-m", "autoclean.cli", *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )


def test_unknown_argument_shows_banner():
    result = run_cli("workspace", "cd", "--shell")
    assert result.returncode == 2
    assert "AutocleanEEG Pipeline" in result.stdout
    assert "Unknown argument" in result.stdout
