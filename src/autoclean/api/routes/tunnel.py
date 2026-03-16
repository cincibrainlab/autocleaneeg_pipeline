"""Public tunnel endpoints.

Manages a Cloudflare Tunnel subprocess that exposes the local API server
on a public HTTPS URL.  Two modes are supported:

1. **Quick Tunnel** (default) — ephemeral ``*.trycloudflare.com`` URL,
   no account required, dies when the process stops.
2. **Named Tunnel** — permanent URL via a Cloudflare account token,
   survives restarts, stable hostname.  Token is stored in
   ``tunnel_config.json`` inside the workspace.

Basic Auth credentials are generated per session and enforced by
middleware in ``server.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from autoclean.api.state import api_state

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Module-level tunnel state ─────────────────────────────────────────

_tunnel_process: Optional[subprocess.Popen] = None
_tunnel_url: Optional[str] = None
_tunnel_password: Optional[str] = None
_tunnel_reader: Optional[threading.Thread] = None
_tunnel_start_time: Optional[float] = None
_tunnel_mode: Optional[str] = None  # "quick" or "named"

# Protects all reads and writes of the module-level state variables above.
_tunnel_lock = threading.Lock()

_URL_PATTERN = re.compile(r"(https://[a-z0-9-]+\.trycloudflare\.com)")


# ── Tunnel config persistence ─────────────────────────────────────────


def _config_path() -> Path | None:
    """Return path to tunnel_config.json in the workspace, or None."""
    if not api_state.workspace_dir:
        return None
    return api_state.workspace_dir / "tunnel_config.json"


def _load_config() -> dict[str, Any]:
    """Load tunnel config from workspace. Returns {} if not configured."""
    path = _config_path()
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _save_config(config: dict[str, Any]) -> None:
    """Save tunnel config to workspace."""
    path = _config_path()
    if path is None:
        raise RuntimeError("Workspace not configured")
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


# ── State helpers ─────────────────────────────────────────────────────


def get_tunnel_state() -> dict:
    """Return current tunnel state (used by auth middleware too)."""
    global _tunnel_process

    with _tunnel_lock:
        if _tunnel_process is not None:
            retcode = _tunnel_process.poll()
            if retcode is not None:
                _tunnel_process = None

        active = _tunnel_process is not None
        return {
            "active": active,
            "url": _tunnel_url if active else None,
            "password": _tunnel_password if active else None,
            "mode": _tunnel_mode if active else None,
        }


def _read_tunnel_stderr() -> None:
    """Background thread: read cloudflared stderr and extract the public URL."""
    global _tunnel_url
    with _tunnel_lock:
        proc = _tunnel_process
        mode = _tunnel_mode
    if proc is None or proc.stderr is None:
        return
    try:
        for raw_line in iter(proc.stderr.readline, ""):
            if not raw_line:
                break
            line = raw_line.rstrip("\n")
            logger.debug("cloudflared: %s", line)
            # Quick tunnels: extract URL from stderr
            if mode == "quick":
                match = _URL_PATTERN.search(line)
                if match:
                    url = match.group(1)
                    with _tunnel_lock:
                        _tunnel_url = url
                    logger.info("Tunnel URL detected: %s", url)
            # Named tunnels: look for "Connection registered" to confirm active
            elif mode == "named":
                if "registered" in line.lower() or "connected" in line.lower():
                    logger.info("Named tunnel connection established")
    except (ValueError, OSError):
        pass


# ── Response models ───────────────────────────────────────────────────


class TunnelStartResponse(BaseModel):
    success: bool
    url: Optional[str] = None
    password: Optional[str] = None
    message: str = ""
    mode: Optional[str] = None


class TunnelStatusResponse(BaseModel):
    active: bool
    url: Optional[str] = None
    password: Optional[str] = None
    mode: Optional[str] = None


class TunnelStopResponse(BaseModel):
    success: bool
    message: str = ""


class TunnelConfigResponse(BaseModel):
    configured: bool
    url: str = ""
    has_token: bool = False


class TunnelConfigInput(BaseModel):
    token: str
    url: str


# ── Config endpoints ──────────────────────────────────────────────────


@router.get("/config", response_model=TunnelConfigResponse)
async def get_tunnel_config():
    """Return whether a named tunnel is configured (never exposes the token)."""
    config = _load_config()
    return TunnelConfigResponse(
        configured=bool(config.get("token")),
        url=config.get("url", ""),
        has_token=bool(config.get("token")),
    )


@router.put("/config")
async def set_tunnel_config(body: TunnelConfigInput):
    """Save a named tunnel token and URL.

    The token comes from the Cloudflare Zero Trust dashboard:
    Networks > Tunnels > Create > copy the token.
    The URL is the public hostname configured for the tunnel
    (e.g., ``https://eeg-lab.example.com``).
    """
    token = body.token.strip()
    url = body.url.strip()

    if not token:
        raise HTTPException(status_code=400, detail="Token is required")
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Normalise URL
    if not url.startswith("https://"):
        url = f"https://{url}"

    _save_config({"token": token, "url": url})
    logger.info("Named tunnel config saved (url=%s)", url)

    return {"success": True, "message": "Tunnel configuration saved"}


@router.delete("/config")
async def clear_tunnel_config():
    """Remove named tunnel configuration, reverting to Quick Tunnel mode."""
    path = _config_path()
    if path and path.exists():
        path.unlink()
    logger.info("Named tunnel config cleared — will use Quick Tunnel")
    return {"success": True, "message": "Tunnel configuration cleared"}


# ── Start / Stop endpoints ────────────────────────────────────────────


@router.get("/status", response_model=TunnelStatusResponse)
async def tunnel_status():
    """Get current tunnel status."""
    state = get_tunnel_state()
    return TunnelStatusResponse(**state)


def _start_tunnel_blocking(port: int, password: str) -> TunnelStartResponse:
    """Synchronous helper that spawns cloudflared and waits for readiness.

    Runs in a thread via asyncio.to_thread() so it does not block the event
    loop.  All state mutations are protected by _tunnel_lock.
    """
    global _tunnel_process, _tunnel_url, _tunnel_password, _tunnel_reader
    global _tunnel_start_time, _tunnel_mode

    config = _load_config()
    use_named = bool(config.get("token"))

    with _tunnel_lock:
        # Double-checked: another concurrent call may have started the tunnel
        if _tunnel_process is not None:
            retcode = _tunnel_process.poll()
            if retcode is None:
                return TunnelStartResponse(
                    success=True,
                    url=_tunnel_url,
                    password=_tunnel_password,
                    message="Tunnel already active",
                    mode=_tunnel_mode,
                )
            _tunnel_process = None

        _tunnel_password = password
        _tunnel_url = None

        if use_named:
            # Named tunnel: use token-based run
            _tunnel_mode = "named"
            _tunnel_url = config.get("url", "")
            cmd = [
                "cloudflared", "tunnel", "run",
                "--token", config["token"],
            ]
        else:
            # Quick tunnel: ephemeral URL
            _tunnel_mode = "quick"
            cmd = [
                "cloudflared", "tunnel",
                "--url", f"http://127.0.0.1:{port}",
            ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            _tunnel_process = proc
            _tunnel_start_time = time.time()
        except Exception:
            _tunnel_password = None
            _tunnel_mode = None
            _tunnel_url = None
            raise

    # Start background reader
    reader = threading.Thread(target=_read_tunnel_stderr, daemon=True)
    with _tunnel_lock:
        _tunnel_reader = reader
    reader.start()

    if use_named:
        # For named tunnels, we already know the URL from config.
        # Wait a few seconds for cloudflared to confirm connection.
        deadline = time.time() + 10
        while time.time() < deadline:
            with _tunnel_lock:
                proc = _tunnel_process
            if proc is not None and proc.poll() is not None:
                _clear_tunnel_state()
                raise RuntimeError(
                    "cloudflared exited unexpectedly. "
                    "Check your tunnel token — it may be invalid or the "
                    "tunnel may have been deleted in the Cloudflare dashboard."
                )
            time.sleep(0.5)

        # If process is still running after wait, assume success
        with _tunnel_lock:
            url = _tunnel_url
            pid = _tunnel_process.pid if _tunnel_process else 0

        logger.info("Named tunnel started: url=%s pid=%d", url, pid)
        return TunnelStartResponse(
            success=True,
            url=url,
            password=password,
            message=f"Named tunnel active (pid {pid})",
            mode="named",
        )
    else:
        # Quick tunnel: wait for URL in stderr
        deadline = time.time() + 15
        url: Optional[str] = None
        while time.time() < deadline:
            with _tunnel_lock:
                url = _tunnel_url
            if url is not None:
                break
            with _tunnel_lock:
                proc = _tunnel_process
            if proc is not None and proc.poll() is not None:
                raise RuntimeError("cloudflared exited unexpectedly")
            time.sleep(0.3)

        if url is None:
            with _tunnel_lock:
                if _tunnel_process is not None:
                    _tunnel_process.kill()
                _tunnel_process = None
                _tunnel_password = None
                _tunnel_mode = None
            raise RuntimeError("Timed out waiting for tunnel URL")

        with _tunnel_lock:
            pid = _tunnel_process.pid if _tunnel_process else 0

        logger.info("Quick tunnel started: url=%s pid=%d", url, pid)
        return TunnelStartResponse(
            success=True,
            url=url,
            password=password,
            message=f"Quick tunnel active (pid {pid})",
            mode="quick",
        )


def _clear_tunnel_state() -> None:
    """Reset all tunnel state globals under the lock."""
    global _tunnel_process, _tunnel_url, _tunnel_password, _tunnel_mode
    with _tunnel_lock:
        _tunnel_process = None
        _tunnel_url = None
        _tunnel_password = None
        _tunnel_mode = None


@router.post("/start", response_model=TunnelStartResponse)
async def start_tunnel():
    """Start a Cloudflare Tunnel to expose the local server.

    If a named tunnel token is configured in the workspace, uses that
    for a persistent URL.  Otherwise falls back to a Quick Tunnel with
    an ephemeral ``*.trycloudflare.com`` URL.
    """
    import os as _os

    port: int = int(_os.environ.get("AUTOCLEAN_API_PORT", "8000"))

    # Already running?
    state = get_tunnel_state()
    if state["active"]:
        return TunnelStartResponse(
            success=True,
            url=state["url"],
            password=state["password"],
            message="Tunnel already active",
            mode=state["mode"],
        )

    if not shutil.which("cloudflared"):
        raise HTTPException(
            status_code=400,
            detail=(
                "cloudflared is not installed. "
                "Install it with: brew install cloudflare/cloudflare/cloudflared"
            ),
        )

    password = secrets.token_urlsafe(12)

    try:
        return await asyncio.to_thread(_start_tunnel_blocking, port, password)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to start tunnel")
        _clear_tunnel_state()
        raise HTTPException(status_code=500, detail=str(exc))


def _stop_tunnel_blocking() -> tuple[bool, int]:
    """Synchronous helper to stop the tunnel subprocess."""
    global _tunnel_process, _tunnel_url, _tunnel_password, _tunnel_start_time, _tunnel_mode

    with _tunnel_lock:
        proc = _tunnel_process
        if proc is None:
            return False, 0
        pid = proc.pid

    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        proc.kill()
    finally:
        with _tunnel_lock:
            _tunnel_process = None
            _tunnel_url = None
            _tunnel_password = None
            _tunnel_start_time = None
            _tunnel_mode = None

    return True, pid


@router.post("/stop", response_model=TunnelStopResponse)
async def stop_tunnel():
    """Stop the active tunnel."""
    try:
        was_active, pid = await asyncio.to_thread(_stop_tunnel_blocking)
    except Exception as exc:
        logger.exception("Error stopping tunnel")
        raise HTTPException(status_code=500, detail=str(exc))

    if not was_active:
        return TunnelStopResponse(success=False, message="No tunnel is active")

    logger.info("Tunnel stopped (pid=%d)", pid)
    return TunnelStopResponse(success=True, message=f"Tunnel stopped (pid {pid})")
