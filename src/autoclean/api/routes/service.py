"""Service control endpoints.

Manages the ``autocleaneeg-pipeline serve run`` dispatcher subprocess
that continuously scans ingestion folders and processes matched files.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from autoclean.api.models import ServiceActionResponse, ServiceStatusResponse
from autoclean.api.state import api_state

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Module-level process state ───────────────────────────────────────

_process: Optional[subprocess.Popen] = None
_start_time: Optional[float] = None

# ── Log streaming state ──────────────────────────────────────────────

_log_buffer: deque[str] = deque(maxlen=500)
_log_thread: Optional[threading.Thread] = None

# Protects all reads and writes of _process, _start_time, and _log_buffer.
# Request-handler coroutines and the background log-reader thread both
# access this shared state, so a lock is required to prevent races.
_service_lock = threading.Lock()


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _stream_logs() -> None:
    """Background thread that reads process stdout into the log buffer."""
    with _service_lock:
        proc = _process
    if proc is None or proc.stdout is None:
        return
    try:
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            clean = _ANSI_ESCAPE.sub("", line.rstrip("\n"))
            with _service_lock:
                _log_buffer.append(clean)
    except (ValueError, OSError):
        pass


# ── Service start request model ──────────────────────────────────────


class ServiceStartRequest(BaseModel):
    """Optional settings for starting the service."""

    max_cycles: int = Field(default=0, ge=0, description="Max cycles (0 = unlimited)")
    idle_limit: int = Field(default=2, ge=0, description="Idle cycles before exiting")
    sleep_seconds: float = Field(default=1.0, ge=0, description="Sleep between cycles in seconds")
    no_watch: bool = Field(default=False, description="Disable watchfiles usage")
    no_sentinel: bool = Field(default=False, description="Disable sentinel requirement")


def _require_workspace():
    """Raise 500 if workspace is not configured."""
    if not api_state.workspace_dir:
        raise HTTPException(status_code=500, detail="Workspace not configured")


def get_service_status() -> dict:
    """Return service status as a plain dict (used by /api/status too)."""
    global _process, _start_time

    with _service_lock:
        if _process is not None:
            retcode = _process.poll()
            if retcode is not None:
                # Process has exited
                _process = None
                _start_time = None

        running = _process is not None
        return {
            "running": running,
            "pid": _process.pid if running else None,
            "mode": api_state.mode,
            "uptime_seconds": (time.time() - _start_time) if running and _start_time else None,
        }


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/status", response_model=ServiceStatusResponse)
async def status():
    """Get the current service dispatcher status."""
    return get_service_status()


@router.post("/start", response_model=ServiceActionResponse)
async def start_service(settings: ServiceStartRequest | None = None):
    """Start the serve-run dispatcher subprocess."""
    global _process, _start_time, _log_thread

    _require_workspace()

    if settings is None:
        settings = ServiceStartRequest()

    # Check if already running
    info = get_service_status()
    if info["running"]:
        return ServiceActionResponse(
            success=False,
            message=f"Service already running (pid {info['pid']})",
        )

    # Determine which config to use: deployed (deploy/) if it exists,
    # otherwise the operator config at workspace root.
    deploy_config = api_state.workspace_dir / "deploy" / f"serve-{api_state.mode}.yaml"
    use_operator = not deploy_config.exists()

    cmd = [
        sys.executable,
        "-m",
        "autoclean",
        "serve",
        "run",
        "--path",
        str(api_state.workspace_dir),
        "--mode",
        api_state.mode,
    ]
    if use_operator:
        cmd.append("--use-operator")

    # Append optional settings to command
    if settings.max_cycles > 0:
        cmd.extend(["--max-cycles", str(settings.max_cycles)])
    if settings.idle_limit > 0:
        cmd.extend(["--idle-limit", str(settings.idle_limit)])
    if settings.sleep_seconds != 1.0:
        cmd.extend(["--sleep-seconds", str(settings.sleep_seconds)])
    if settings.no_watch:
        cmd.append("--no-watch")
    if settings.no_sentinel:
        cmd.append("--no-sentinel")

    try:
        env = dict(os.environ, NO_COLOR="1", TERM="dumb")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        with _service_lock:
            # Clear previous log buffer and store new process under the lock.
            _log_buffer.clear()
            _process = proc
            _start_time = time.time()

        # Start background log reader thread
        _log_thread = threading.Thread(target=_stream_logs, daemon=True)
        _log_thread.start()

        logger.info("Started serve dispatcher pid=%d cmd=%s", proc.pid, cmd)
        return ServiceActionResponse(
            success=True,
            message=f"Service started (pid {proc.pid})",
        )
    except Exception as exc:
        logger.exception("Failed to start serve dispatcher")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stop", response_model=ServiceActionResponse)
async def stop_service():
    """Stop the serve-run dispatcher subprocess (SIGTERM)."""
    global _process, _start_time

    info = get_service_status()
    if not info["running"]:
        return ServiceActionResponse(
            success=False,
            message="Service is not running",
        )

    with _service_lock:
        proc = _process
        pid = proc.pid if proc else None  # type: ignore[union-attr]

    if proc is None or pid is None:
        return ServiceActionResponse(success=False, message="Service is not running")

    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.warning("Service pid=%d did not stop gracefully; killed", pid)
    except Exception as exc:
        logger.exception("Error stopping service pid=%d", pid)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        with _service_lock:
            _process = None
            _start_time = None

    logger.info("Stopped serve dispatcher pid=%d", pid)
    return ServiceActionResponse(
        success=True,
        message=f"Service stopped (pid {pid})",
    )


@router.get("/logs")
async def get_service_logs(lines: int = Query(default=100, ge=1, le=500)):
    """Get recent service log output."""
    with _service_lock:
        recent = list(_log_buffer)[-lines:]
        total = len(_log_buffer)
    return {"lines": recent, "total": total}
