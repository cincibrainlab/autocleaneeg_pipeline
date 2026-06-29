"""Service control endpoints.

Manages the ``autocleaneeg-pipeline serve run`` dispatcher subprocess
that continuously scans ingestion folders and processes matched files.
"""

from __future__ import annotations

import asyncio
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
from autoclean.utils.ingestion import load_serve_config, parse_serve_config

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
    idle_limit: int = Field(
        default=0,
        ge=0,
        description="Idle cycles before exiting (0 = keep running)",
    )
    sleep_seconds: float = Field(
        default=1.0, ge=0, description="Sleep between cycles in seconds"
    )
    no_watch: bool = Field(default=False, description="Disable watchfiles usage")
    no_sentinel: bool = Field(default=False, description="Disable sentinel requirement")


def _require_workspace():
    """Raise 500 if workspace is not configured."""
    if not api_state.workspace_dir:
        raise HTTPException(status_code=409, detail="Workspace not configured")


def _get_start_blocker() -> str | None:
    """Return a user-facing reason the dispatcher cannot start."""
    _require_workspace()

    workspace_dir = api_state.workspace_dir
    assert workspace_dir is not None

    config_path = api_state.get_config_path(deployed=False)
    deployed_path = api_state.get_config_path(deployed=True)

    try:
        raw = load_serve_config(config_path)
        parse_serve_config(raw, workspace_dir, strict=True)
    except Exception as exc:
        return f"Fix configuration errors before starting the service: {exc}"

    if not deployed_path.exists():
        return "Apply the current configuration before starting the service."

    try:
        deployed_raw = load_serve_config(deployed_path)
        parse_serve_config(deployed_raw, workspace_dir, strict=True)
    except Exception as exc:
        return f"Re-apply the configuration before starting the service: {exc}"

    try:
        if config_path.read_text() != deployed_path.read_text():
            return "Apply the latest configuration changes before starting the service."
    except OSError as exc:
        return f"Unable to verify deployed configuration: {exc}"

    return None


def get_service_status() -> dict:
    """Return service status as a plain dict (used by /api/status too)."""
    global _process, _start_time

    blocker = _get_start_blocker()
    with _service_lock:
        if _process is not None:
            retcode = _process.poll()
            if retcode is not None:
                # Diagnostic: record how the dispatcher exited before clearing state.
                # retcode < 0 means it was killed by signal number -retcode
                # (-15 SIGTERM, -9 SIGKILL, -1 SIGHUP). Captures even uncatchable kills.
                try:
                    if api_state.workspace_dir:
                        _sig = f" (signal {-retcode})" if retcode < 0 else ""
                        with open(
                            api_state.workspace_dir / "dispatcher-exit.log", "a"
                        ) as _fh:
                            _fh.write(
                                f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                                f"pid={_process.pid} exited retcode={retcode}{_sig}\n"
                            )
                except Exception:  # pylint: disable=broad-except
                    pass
                # Process has exited
                _process = None
                _start_time = None

        running = _process is not None
        return {
            "running": running,
            "pid": _process.pid if running else None,
            "mode": api_state.mode,
            "uptime_seconds": (
                (time.time() - _start_time) if running and _start_time else None
            ),
            "can_start": blocker is None,
            "blocked_reason": blocker,
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

    blocker = _get_start_blocker()
    if blocker:
        raise HTTPException(status_code=409, detail=blocker)

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


def _stop_service_blocking() -> tuple[bool, int]:
    """Synchronous helper to stop the service subprocess.

    Returns (was_running, pid). Called via asyncio.to_thread() to avoid
    blocking the event loop during proc.wait().
    """
    global _process, _start_time

    with _service_lock:
        proc = _process
        if proc is None:
            return False, 0
        pid = proc.pid

    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.warning("Service pid=%d did not stop gracefully; killed", pid)
    except Exception:
        proc.kill()
    finally:
        with _service_lock:
            _process = None
            _start_time = None

    return True, pid


@router.post("/stop", response_model=ServiceActionResponse)
async def stop_service():
    """Stop the serve-run dispatcher subprocess (SIGTERM)."""
    info = get_service_status()
    if not info["running"]:
        return ServiceActionResponse(
            success=False,
            message="Service is not running",
        )

    try:
        was_running, pid = await asyncio.to_thread(_stop_service_blocking)
    except Exception as exc:
        logger.exception("Error stopping service")
        raise HTTPException(status_code=500, detail=str(exc))

    if not was_running:
        return ServiceActionResponse(success=False, message="Service is not running")

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
