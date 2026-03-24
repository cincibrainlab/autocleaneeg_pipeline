"""Subprocess-backed CLI adapter for the AutoClean MCP server."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

from autoclean_mcp.models import CLIExecutionResult, utc_now_iso

MAX_CAPTURE_CHARS = 200_000


def _truncate_output(text: str) -> str:
    """Truncate captured output to a bounded size."""
    if len(text) <= MAX_CAPTURE_CHARS:
        return text
    suffix = "\n... [truncated by autoclean_mcp]\n"
    return text[: MAX_CAPTURE_CHARS - len(suffix)] + suffix


def get_canonical_cli_command(argv: Sequence[str]) -> list[str]:
    """Return the canonical subprocess command for AutoClean CLI execution."""
    return [sys.executable, "-m", "autoclean", *argv]


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> CLIExecutionResult:
    """Run a subprocess and normalize its result."""
    started_at = utc_now_iso()
    started_perf = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        ok = completed.returncode == 0
    except subprocess.TimeoutExpired as exc:
        exit_code = -1
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + (
            f"\nCommand timed out after {timeout_seconds} seconds."
        )
        ok = False

    finished_at = utc_now_iso()
    duration_ms = int((time.perf_counter() - started_perf) * 1000)
    return CLIExecutionResult(
        command=list(command),
        cwd=str(Path(cwd).resolve()) if cwd is not None else str(Path.cwd()),
        exit_code=exit_code,
        stdout=_truncate_output(stdout),
        stderr=_truncate_output(stderr),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        ok=ok,
    )


def execute_cli(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> CLIExecutionResult:
    """Execute AutoClean CLI with the canonical subprocess entrypoint."""
    return run_subprocess(
        get_canonical_cli_command(argv),
        cwd=cwd,
        env=env,
        timeout_seconds=timeout_seconds,
    )
