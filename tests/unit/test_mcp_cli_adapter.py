from __future__ import annotations

import sys
import time

from autoclean_mcp.cli_adapter import get_canonical_cli_command, run_subprocess


def test_get_canonical_cli_command_uses_python_module_entrypoint() -> None:
    command = get_canonical_cli_command(["version"])

    assert command[:3] == [sys.executable, "-m", "autoclean"]
    assert command[-1] == "version"


def test_run_subprocess_normalizes_result() -> None:
    result = run_subprocess([sys.executable, "-c", "print('ok')"])

    assert result.ok is True
    assert result.exit_code == 0
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""
    assert result.duration_ms >= 0


def test_run_subprocess_normalizes_timeout() -> None:
    result = run_subprocess(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        timeout_seconds=0.1,
    )

    assert result.ok is False
    assert result.exit_code == -1
    assert "timed out" in result.stderr.lower()
