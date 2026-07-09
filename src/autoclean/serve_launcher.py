"""One-command launcher for AutoCleanEEG Serve.

Commands:
    autocleaneeg-serve              Start server (foreground)
    autocleaneeg-serve up           Start server (background daemon)
    autocleaneeg-serve down         Stop the running server
    autocleaneeg-serve restart      Stop + start in background
    autocleaneeg-serve status       Show server status
    autocleaneeg-serve share        Manage public tunnel sharing

A PID file prevents duplicate servers. Use --force to bypass.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
from typing import Any

_PID_FILE = None  # Set at runtime for cleanup


# ── Shared HTTP helper ────────────────────────────────────────────────


def _api_request(
    port: int,
    path: str,
    method: str = "GET",
    body: dict | None = None,
    timeout: int = 20,
) -> dict[str, Any]:
    """Make a JSON request to the local API server."""
    import urllib.request

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
        data=data,
    )
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read())


def _wait_for_health(
    port: int, attempts: int = 30, delay: float = 0.5
) -> dict[str, Any] | None:
    """Wait for the local API to become healthy."""
    import time

    for _ in range(attempts):
        time.sleep(delay)
        try:
            return _api_request(port, "/health", timeout=1)
        except Exception:
            continue
    return None


def _ensure_operational_service(port: int) -> tuple[bool, list[str]]:
    """Ensure the dispatcher is running when the workspace is ready.

    Returns:
        (service_running, messages)
    """
    messages: list[str] = []

    try:
        status = _api_request(port, "/api/status", timeout=3)
    except Exception as exc:
        return False, [f"Could not inspect Serve status: {exc}"]

    if not status.get("configured"):
        messages.append(
            "Workspace not configured yet. Open the UI to choose or create a workspace."
        )
        return False, messages

    workspace_dir = status.get("workspace_dir", "unknown workspace")
    routes = status.get("routes", {}) or {}
    config = status.get("config", {}) or {}
    queue = status.get("queue", {}) or {}
    service = status.get("service", {}) or {}

    if service.get("running"):
        messages.append(f"Processing service already running for {workspace_dir}.")
        return True, messages

    if routes.get("total", 0) == 0:
        messages.append(
            "Serve UI started, but no routes exist yet. Add a route before processing can start."
        )
        return False, messages

    if config.get("errors"):
        messages.append(
            "Serve UI started, but processing was not started because configuration is invalid."
        )
        return False, messages

    if config.get("needs_deploy"):
        messages.append(
            "Serve UI started, but processing was not started because unapplied configuration changes exist."
        )
        messages.append(
            "Apply the latest config from Settings or run 'autocleaneeg-pipeline serve deploy --mode <test|live>' before starting processing."
        )
        return False, messages

    try:
        result = _api_request(
            port,
            "/api/service/start",
            method="POST",
            body={
                "max_cycles": 0,
                "idle_limit": 0,
                "sleep_seconds": 1.0,
                "no_watch": False,
                "no_sentinel": False,
            },
            timeout=10,
        )
    except Exception as exc:
        messages.append(
            f"Serve UI started, but processing service failed to start: {exc}"
        )
        return False, messages

    success = bool(result.get("success"))
    if success:
        messages.append(result.get("message", "Processing service started."))
        pending = queue.get("pending", 0)
        processing = queue.get("processing", 0)
        if pending or processing:
            messages.append(
                f"Queue currently has {pending} pending and {processing} active file(s)."
            )
        return True, messages

    messages.append(result.get("message", "Processing service did not start."))
    return False, messages


# ── Entry point ───────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoCleanEEG Serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n"
        "  (none)    Start server in foreground (Ctrl+C to stop)\n"
        "  up        Start server as background daemon\n"
        "  down      Stop the running server\n"
        "  restart   Stop + start as background daemon\n"
        "  status    Show whether server is running\n"
        "  share     Manage public tunnel [start|stop|status|setup|clear]\n",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default=None,
        choices=["up", "down", "restart", "status", "share"],
        help="Server command (omit for foreground start)",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind address (use 127.0.0.1 for localhost only)",
    )
    parser.add_argument("--path", type=str, default=None, help="Workspace path")
    parser.add_argument("--mode", choices=["test", "live"], default="test")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Allow starting a second server instance"
    )

    # Pre-scan for share sub-action before argparse consumes it
    raw_args = sys.argv[1:]
    share_action = "start"
    if len(raw_args) >= 2 and raw_args[0] == "share":
        candidate = raw_args[1]
        if candidate in ("start", "stop", "status", "setup", "clear"):
            share_action = candidate
            # Remove the sub-action so argparse doesn't choke on it
            sys.argv = [sys.argv[0], "share"] + raw_args[2:]

    args = parser.parse_args()

    rc = 0
    if args.command == "down":
        rc = _cmd_down()
    elif args.command == "status":
        rc = _cmd_status(args.port)
    elif args.command == "up":
        rc = _cmd_up(args)
    elif args.command == "restart":
        rc = _cmd_restart(args)
    elif args.command == "share":
        rc = _cmd_share(args.port, action=share_action)
    else:
        _cmd_foreground(args)  # Blocks, never returns normally
    sys.exit(rc)


# ── Commands ──────────────────────────────────────────────────────────


def _cmd_foreground(args) -> None:
    """Start server in foreground (blocking)."""
    import threading
    import webbrowser

    if not args.force:
        existing = _check_existing_server(args.port)
        if existing:
            pid, port, _health = existing
            print(f"AutoCleanEEG Serve is already running (pid {pid}, port {port}).")
            print(f"  Open: http://127.0.0.1:{port}")
            print("  Use --force to start another instance.")
            sys.exit(0)

    port = _resolve_port(args)
    workspace = _resolve_workspace(args.path)
    host = args.host

    if workspace:
        print(f"AutoCleanEEG Serve starting with workspace: {workspace}")
    else:
        print(
            "AutoCleanEEG Serve starting — workspace setup required (will open in browser)"
        )

    listen_display = "0.0.0.0 (LAN accessible)" if host == "0.0.0.0" else host
    print(f"Listening on http://{listen_display}:{port}")

    _write_pid_file(port)

    def post_start_tasks() -> None:
        health = _wait_for_health(port)
        if health is None:
            return

        if not args.no_browser:
            try:
                webbrowser.open(f"http://127.0.0.1:{port}")
            except Exception:
                pass

        service_running, messages = _ensure_operational_service(port)
        for line in messages:
            print(line)
        if service_running:
            print("Serve is operational: UI and processing service are both running.")
        elif workspace:
            print("Serve UI is running, but processing is not fully active yet.")

    threading.Thread(target=post_start_tasks, daemon=True).start()

    from autoclean.api.server import run_server

    run_server(
        workspace_dir=workspace,
        mode=args.mode,
        host=host,
        port=port,
    )


def _cmd_up(args) -> int:
    """Start server as a background daemon. Returns 0 on success, 1 on failure."""
    import shutil

    existing = _check_existing_server(args.port)
    if existing and not args.force:
        pid, port, _health = existing
        print(f"Already running (pid {pid}, port {port}).")
        _print_server_urls(port)
        return 0

    # Build command
    exe = shutil.which("autocleaneeg-serve")
    if exe:
        cmd = [
            exe,
            "--no-browser",
            "--port",
            str(args.port),
            "--host",
            args.host,
            "--mode",
            args.mode,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "autoclean.serve_launcher",
            "--no-browser",
            "--port",
            str(args.port),
            "--host",
            args.host,
            "--mode",
            args.mode,
        ]

    if args.path:
        cmd.extend(["--path", args.path])
    cmd.append("--force")  # Parent already did the guard check

    # Start detached — use Popen with file descriptors (no shell=True)
    log_path = _log_file_path()
    log_f = open(log_path, "w")
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        env=child_env,
    )
    log_f.close()  # Parent doesn't need the FD

    # Wait for health check
    port = args.port
    child_pid = 0
    health = _wait_for_health(port)
    if health is None:
        existing = _check_existing_server(args.port)
        if existing is not None:
            child_pid, port, health = existing
        else:
            _print_startup_failure_diagnostics(
                log_path=log_path,
                cmd=cmd,
                child=child,
                port=args.port,
                host=args.host,
                mode=args.mode,
                workspace_path=args.path,
            )
            return 1
    else:
        child_pid = _read_pid_from_file()

    if health is not None:
        if child_pid == 0:
            child_pid = _read_pid_from_file()
        print(f"AutoCleanEEG Serve started (pid {child_pid}, port {port}).")
        _print_server_urls(port)
        print(f"  Log:     {log_path}")

        service_running, messages = _ensure_operational_service(port)
        for line in messages:
            print(f"  {line}")
        if service_running:
            print("  Operational state: UI + processing service are running.")
        else:
            print(
                "  Operational state: UI is running; processing still needs attention."
            )
        return 0

    _print_startup_failure_diagnostics(
        log_path=log_path,
        cmd=cmd,
        child=child,
        port=args.port,
        host=args.host,
        mode=args.mode,
        workspace_path=args.path,
    )
    return 1


def _read_log_preview(log_path: str, *, max_lines: int = 12) -> list[str]:
    """Return a short log preview for startup diagnostics."""
    import time

    try:
        time.sleep(0.2)
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            return handle.read().splitlines()[:max_lines]
    except OSError:
        return []


def _foreground_diagnostic_command(
    *,
    port: int,
    host: str,
    mode: str,
    workspace_path: str | None,
) -> str:
    """Build a foreground command that exposes startup tracebacks."""
    cmd = [
        "autocleaneeg-pipeline",
        "serve",
        "api",
        "--api-port",
        str(port),
        "--host",
        host,
        "--mode",
        mode,
    ]
    if workspace_path:
        cmd.extend(["--path", workspace_path])
    return " ".join(cmd)


def _print_startup_failure_diagnostics(
    *,
    log_path: str,
    cmd: list[str],
    child,
    port: int,
    host: str,
    mode: str,
    workspace_path: str | None,
) -> None:
    """Print actionable diagnostics when the daemon fails to become healthy."""
    print(f"Server did not respond within 15s. Check {log_path}")
    print(f"  Child command: {' '.join(cmd)}")

    exit_code = child.poll()
    if exit_code is not None:
        print(f"  Child process exited before health check with code {exit_code}.")

    preview = _read_log_preview(log_path)
    if preview:
        print("  Log preview:")
        for line in preview:
            print(f"    {line}")
    else:
        print("  Log is empty.")

    print("  For an immediate traceback, run the server in the foreground:")
    print(
        "    "
        + _foreground_diagnostic_command(
            port=port,
            host=host,
            mode=mode,
            workspace_path=workspace_path,
        )
    )


def _cmd_down(quiet: bool = False) -> int:
    """Stop the running server. Returns 0 on success, 1 on failure."""
    pid_path = _pid_file_path()
    try:
        with open(pid_path) as f:
            data = f.read().strip().split(":")
            pid = int(data[0])
            port = int(data[1]) if len(data) > 1 else 8000
    except (FileNotFoundError, ValueError):
        if not quiet:
            print("No server is running.")
        return 1

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        try:
            os.unlink(pid_path)
        except OSError:
            pass
        if not quiet:
            print("No server is running (stale PID file cleaned up).")
        return 1

    # Verify it's actually an autoclean server before killing (PID reuse safety)
    try:
        _api_request(port, "/health", timeout=2)
    except Exception:
        try:
            os.unlink(pid_path)
        except OSError:
            pass
        if not quiet:
            print(
                f"PID {pid} is alive but not an AutoClean server (stale PID file cleaned up)."
            )
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        if not quiet:
            print(f"Failed to stop server (pid {pid}): {e}")
        return 1

    import time

    for _ in range(20):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            try:
                os.unlink(pid_path)
            except OSError:
                pass
            if not quiet:
                print(f"Server stopped (pid {pid}, was on port {port}).")
            return 0

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        os.unlink(pid_path)
    except OSError:
        pass
    if not quiet:
        print(f"Server killed (pid {pid}).")
    return 0


def _cmd_restart(args) -> int:
    """Stop the running server and start a new one."""
    import time

    _cmd_down(quiet=True)
    for _ in range(10):
        time.sleep(0.5)
        if _port_available(args.host, args.port):
            break
    return _cmd_up(args)


def _cmd_status(default_port: int) -> int:
    """Print server status. Returns 0 if running, 1 if not."""
    existing = _check_existing_server(default_port)
    if not existing:
        print("AutoCleanEEG Serve is not running.")
        return 1

    pid, port, health = existing
    print(f"AutoCleanEEG Serve is running (pid {pid}, port {port}).")
    _print_server_urls(port)

    if health:
        mode = health.get("mode", "?")
        version = health.get("pipeline_version", "?")
        workspace = (
            "configured" if health.get("workspace_configured") else "not configured"
        )
        print(f"  Mode: {mode}  |  Version: {version}  |  Workspace: {workspace}")

    try:
        status = _api_request(port, "/api/status", timeout=3)
    except Exception as exc:
        print(f"  Status detail unavailable: {exc}")
        return 0

    if not status.get("configured"):
        print("  Workspace setup is incomplete.")
        print("  Next: choose or create a workspace in the web UI.")
        return 0

    service = status.get("service", {}) or {}
    routes = status.get("routes", {}) or {}
    queue = status.get("queue", {}) or {}
    config = status.get("config", {}) or {}

    dispatcher = "running" if service.get("running") else "stopped"
    print(f"  Dispatcher: {dispatcher}")
    if service.get("running") and service.get("pid"):
        print(f"  Dispatcher PID: {service['pid']}")

    print(
        "  Routes: "
        f"{routes.get('active', 0)} active"
        + (
            f", {routes.get('archived', 0)} archived"
            if routes.get("archived", 0)
            else ""
        )
    )
    print(
        "  Queue: "
        f"{queue.get('pending', 0)} pending, "
        f"{queue.get('processing', 0)} processing, "
        f"{queue.get('failed', 0)} failed"
    )

    if config.get("errors"):
        print("  Config: invalid")
    elif config.get("needs_deploy"):
        print("  Config: valid but unapplied changes exist")
    else:
        print("  Config: ready")

    if not service.get("running"):
        if routes.get("total", 0) == 0:
            print("  Next: add a route before processing can begin.")
        elif config.get("errors"):
            print("  Next: fix configuration errors, then start processing.")
        else:
            print(
                "  Next: restart with 'autocleaneeg-pipeline serve up' or start processing from the UI."
            )
    return 0


def _cmd_share(default_port: int, action: str = "start") -> int:
    """Manage public tunnel sharing from the CLI. Returns 0 on success, 1 on failure."""
    existing = _check_existing_server(default_port)
    if not existing:
        print("Server is not running. Start it first with: autocleaneeg-serve up")
        return 1

    _pid, port, _health = existing

    def api(method, path, body=None):
        return _api_request(port, path, method, body)

    if action == "stop":
        try:
            result = api("POST", "/api/tunnel/stop")
            print(
                "Tunnel stopped."
                if result.get("success")
                else result.get("message", "No tunnel is active.")
            )
        except Exception as e:
            print(f"Error: {e}")

    elif action == "status":
        try:
            tunnel = api("GET", "/api/tunnel/status")
            config = api("GET", "/api/tunnel/config")
        except Exception as e:
            print(f"Error: {e}")
            return

        print()
        if tunnel.get("active"):
            label = "Permanent" if tunnel.get("mode") == "named" else "Temporary"
            print(f"  Tunnel:   Active ({label})")
            print(f"  URL:      {tunnel.get('url')}")
            print("  Username: autoclean")
            print(f"  Password: {tunnel.get('password')}")
        else:
            print("  Tunnel:   Not active")

        print()
        if config.get("configured"):
            print(f"  Named tunnel configured: {config.get('url')}")
            print("  Token:    ****  (stored in workspace)")
        else:
            print("  No named tunnel configured (using temporary Quick Tunnels)")
            print("  Run 'autocleaneeg-serve share setup' for a permanent URL")
        print()

    elif action == "setup":
        print()
        print("Named Tunnel Setup")
        print("=" * 40)
        print()
        print("1. Go to https://one.dash.cloudflare.com (free account)")
        print("2. Networks > Tunnels > Create a tunnel")
        print(f"3. Set service to http://localhost:{port}")
        print("4. Copy the tunnel token below")
        print()

        token = input("Tunnel token: ").strip()
        if not token:
            print("Cancelled.")
            return 0

        url = input("Public URL (e.g. https://eeg-lab.example.com): ").strip()
        if not url:
            print("Cancelled.")
            return 0

        if not url.startswith("https://"):
            url = f"https://{url}"

        try:
            result = api("PUT", "/api/tunnel/config", {"token": token, "url": url})
            if result.get("success"):
                print(f"\nSaved. Your permanent URL will be: {url}")
                print("Run 'autocleaneeg-serve share' to start the tunnel.")
            else:
                print(f"Failed: {result.get('message', 'unknown error')}")
        except Exception as e:
            print(f"Error saving config: {e}")

    elif action == "clear":
        try:
            api("DELETE", "/api/tunnel/config")
            print("Named tunnel config cleared. Will use temporary Quick Tunnels.")
        except Exception as e:
            print(f"Error: {e}")

    else:
        # Default: start
        try:
            tunnel = api("GET", "/api/tunnel/status")
        except Exception as e:
            print(f"Error: {e}")
            return 1

        if tunnel.get("active"):
            label = "Permanent" if tunnel.get("mode") == "named" else "Temporary"
            print(f"Tunnel already active ({label}):")
            print(f"  URL:      {tunnel.get('url')}")
            print("  Username: autoclean")
            print(f"  Password: {tunnel.get('password')}")
            return 0

        print("Starting tunnel...")
        try:
            data = api("POST", "/api/tunnel/start")
            if data.get("success"):
                label = "Permanent" if data.get("mode") == "named" else "Temporary"
                print(f"Tunnel active ({label}):")
                print(f"  URL:      {data.get('url')}")
                print("  Username: autoclean")
                print(f"  Password: {data.get('password')}")
                if data.get("mode") != "named":
                    print("\n  This URL is temporary and changes on restart.")
                    print("  Run 'autocleaneeg-serve share setup' for a permanent URL.")
            else:
                print(f"Failed: {data.get('message', 'unknown error')}")
        except Exception as e:
            print(f"Failed to start tunnel: {e}")
            return 1

    return 0


# ── URL display helpers ───────────────────────────────────────────────


def _get_lan_addresses() -> list[str]:
    """Return LAN IP addresses (non-loopback IPv4)."""
    import socket

    addrs = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.append(ip)
    except Exception:
        pass
    seen = set()
    return [a for a in addrs if a not in seen and not seen.add(a)]


def _print_server_urls(port: int) -> None:
    """Print all ways to reach the server."""
    print(f"  Local:   http://127.0.0.1:{port}")
    for ip in _get_lan_addresses():
        print(f"  LAN:     http://{ip}:{port}")
    try:
        data = _api_request(port, "/api/tunnel/status", timeout=2)
        if data.get("active") and data.get("url"):
            print(f"  Public:  {data['url']}")
    except Exception:
        pass


# ── Port resolution ──────────────────────────────────────────────────


def _resolve_port(args) -> int:
    """Resolve available port, respecting --force flag."""
    port = args.port
    if not _port_available(args.host, port):
        if args.force:
            while not _port_available(args.host, port):
                port += 1
                if port > args.port + 10:
                    print(f"No available port found near {args.port}")
                    sys.exit(1)
            print(f"Port {args.port} in use — using port {port}")
        else:
            print(f"Port {args.port} is in use by another process.")
            print("  Use --force to start on the next available port.")
            sys.exit(1)
    return port


# ── PID file management ──────────────────────────────────────────────


def _pid_file_path() -> str:
    """Return fixed PID file path (consistent across uv tool envs)."""
    return os.path.join(os.path.expanduser("~"), ".autocleaneeg-serve.pid")


def _log_file_path() -> str:
    """Return fixed log file path."""
    return os.path.join(os.path.expanduser("~"), ".autocleaneeg-serve.log")


def _read_pid_from_file() -> int:
    """Read the PID from the PID file."""
    try:
        with open(_pid_file_path()) as f:
            return int(f.read().strip().split(":")[0])
    except Exception:
        return 0


def _check_existing_server(default_port: int) -> tuple[int, int, dict | None] | None:
    """Check if a server is already running.

    Returns (pid, port, health_data) or None. The health_data is the
    parsed /health response (avoids a redundant second call by callers).
    """
    pid_path = _pid_file_path()
    try:
        with open(pid_path) as f:
            data = f.read().strip().split(":")
            pid = int(data[0])
            port = int(data[1]) if len(data) > 1 else default_port
    except (FileNotFoundError, ValueError):
        return None

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        try:
            os.unlink(pid_path)
        except OSError:
            pass
        return None

    # Verify it's an autoclean server and capture health data
    try:
        health = _api_request(port, "/health", timeout=2)
        return pid, port, health
    except Exception:
        pass

    try:
        os.unlink(pid_path)
    except OSError:
        pass
    return None


def _write_pid_file(port: int) -> None:
    """Write PID file and register cleanup."""
    global _PID_FILE
    _PID_FILE = _pid_file_path()

    with open(_PID_FILE, "w") as f:
        f.write(f"{os.getpid()}:{port}")

    atexit.register(_cleanup_pid_file)

    prev_handler = signal.getsignal(signal.SIGTERM)

    def _sigterm_handler(signum, frame):
        _cleanup_pid_file()
        if callable(prev_handler) and prev_handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            prev_handler(signum, frame)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)


def _cleanup_pid_file() -> None:
    """Remove PID file on exit."""
    if _PID_FILE:
        try:
            os.unlink(_PID_FILE)
        except OSError:
            pass


# ── Utilities ─────────────────────────────────────────────────────────


def _port_available(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _resolve_workspace(path_arg: str | None):
    from pathlib import Path

    if path_arg:
        return Path(path_arg).expanduser().resolve()

    try:
        from autoclean.utils.user_config import UserConfigManager

        ucm = UserConfigManager()
        stored = ucm.get_serve_workspace()
        if stored and Path(stored).exists():
            return Path(stored)
    except Exception:
        pass

    return None


if __name__ == "__main__":
    main()
