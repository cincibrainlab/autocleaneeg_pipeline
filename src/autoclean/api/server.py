"""FastAPI server for AutoClean API."""

from __future__ import annotations

import base64
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import Response

from autoclean.api.state import APIState, api_state


# OpenAPI tag metadata for better documentation
TAGS_METADATA = [
    {
        "name": "Queue",
        "description": """
Manage the file ingestion queue. Files are discovered by automation routes
and added to the active mode queue for processing.

**Workflow:**
1. Files are detected in monitored folders → status: `pending`
2. `serve run` dispatch picks up file → status: `processing`
3. Processing completes → status: `processed` or `failed`

**Common operations:**
- `GET /api/queue/stats` - Dashboard summary
- `GET /api/queue/entries` - List with filtering
- `POST /api/queue/retry` - Requeue failed items
""",
    },
    {
        "name": "Worker",
        "description": """
Monitor and control RQ (Redis Queue) workers that process EEG files.

This is an advanced path. The operator-facing serve workflow in this repo
uses route specs plus `serve run` with mode-specific queue files.

Keep this worker surface separate from the default route-first operator workflow.
""",
    },
    {
        "name": "Config",
        "description": """
View and manage automation configuration (routes, settings).

**Modes:**
- `test` - Draft lane (`serve-test.yaml`)
- `live` - Production lane (`serve-live.yaml`)

**Routes** define which files to process and how:
- Ingestion folders to monitor
- File patterns (globs) to match
- Task/montage configuration
""",
    },
    {
        "name": "Routes",
        "description": """
Manage individual route-spec YAML files.

Each route spec defines an automation route: which folders to watch,
file patterns to match, and which task/montage to apply.

**Workflow:**
1. Create/edit route specs via `POST /api/routes`
2. Promote tested routes to live via `POST /api/routes/{id}/promote`
3. Compile to serve config via `POST /api/routes/sync`

**Discovery endpoints** help populate dropdowns:
- `GET /api/routes/discovery/tasks` - Available task files
- `GET /api/routes/discovery/montages` - Available montages
""",
    },
    {
        "name": "Service",
        "description": """
Control the serve-run dispatcher subprocess.

- `GET /api/service/status` - Check if dispatcher is running
- `POST /api/service/start` - Launch the dispatcher
- `POST /api/service/stop` - Gracefully stop the dispatcher
""",
    },
    {
        "name": "Tunnel",
        "description": """
Expose the local server on a public HTTPS URL via Cloudflare Quick Tunnel.

- `GET /api/tunnel/status` - Check tunnel state
- `POST /api/tunnel/start` - Create a public tunnel (generates auth credentials)
- `POST /api/tunnel/stop` - Tear down the tunnel

Requires `cloudflared` to be installed (`brew install cloudflare/cloudflare/cloudflared`).
""",
    },
    {
        "name": "Filesystem",
        "description": """
Server-side filesystem browser for the web UI folder picker.

- `GET /api/filesystem/browse` - List subdirectories of a path
""",
    },
    {
        "name": "Tasks",
        "description": """
Browse available processing tasks with full configuration and pipeline details.

- `GET /api/tasks` - All discovered tasks with config + pipeline steps
- `GET /api/tasks/{name}` - Single task detail by name
""",
    },
    {
        "name": "Task Manager",
        "description": """
Unified task catalog with install, create, and sync actions.

- `GET /api/task-manager` - Full catalog merging library, builtin, and workspace tasks
- `POST /api/task-manager/install` - Install a library task to workspace
- `POST /api/task-manager/create` - Create a new task from template
- `POST /api/task-manager/refresh-library` - Refresh GitHub registry cache
- `POST /api/task-manager/{name}/update` - Update workspace task to latest version
- `DELETE /api/task-manager/{name}` - Remove task from workspace
""",
    },
    {
        "name": "WebSocket",
        "description": """
Real-time event streaming via WebSocket.

Connect to `/ws/events` for live updates:
- Queue changes (file added, status changed)
- Job events (started, completed, failed)
- Worker status changes

**Event format:**
```json
{
  "type": "job_completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {"job_id": "abc123", "file": "/path/to/file.set"}
}
```
""",
    },
]


# ── Recent workspaces helpers ─────────────────────────────────────────────────

_RECENT_FILE = Path.home() / ".autoclean" / "recent_workspaces.json"
_WORKSPACE_META_FILE = ".serve-workspace.json"


def _load_recent_workspaces() -> list[str]:
    try:
        return json.loads(_RECENT_FILE.read_text())[:10]
    except Exception:
        return []


def _save_recent_workspace(path: str) -> None:
    recent = _load_recent_workspaces()
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    _RECENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RECENT_FILE.write_text(json.dumps(recent[:10]), encoding="utf-8")


def _load_persisted_serve_workspace() -> Optional[Path]:
    """Load the persisted serve workspace from user config, if available."""
    try:
        from autoclean.utils.user_config import UserConfigManager

        workspace = UserConfigManager().get_serve_workspace()
    except Exception:
        return None

    if not workspace:
        return None
    return Path(workspace).expanduser().resolve()


def _workspace_meta_path(workspace_dir: Path) -> Path:
    """Return the workspace metadata file path."""
    return workspace_dir / _WORKSPACE_META_FILE


def _read_workspace_metadata(workspace_dir: Path) -> dict[str, Any]:
    """Load lightweight Serve metadata for the workspace."""
    try:
        raw = json.loads(_workspace_meta_path(workspace_dir).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_workspace_metadata(workspace_dir: Path, metadata: dict[str, Any]) -> None:
    """Persist lightweight Serve metadata for the workspace."""
    path = _workspace_meta_path(workspace_dir)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _serve_workspace_paths(workspace_dir: Path) -> dict[str, Path]:
    """Return the standard Serve workspace paths."""
    return {
        "serve_test": workspace_dir / "serve-test.yaml",
        "serve_live": workspace_dir / "serve-live.yaml",
        "routes": workspace_dir / "routes",
        "automations": workspace_dir / "automations",
        "runtimes_test": workspace_dir / "runtimes" / "test",
        "runtimes_live": workspace_dir / "runtimes" / "live",
        "deploy": workspace_dir / "deploy",
    }


def _workspace_bootstrap_origin(workspace_dir: Path) -> str:
    """Infer how the current Serve workspace was created."""
    metadata = _read_workspace_metadata(workspace_dir)
    origin = metadata.get("origin")
    if origin in {"bootstrapped_autoclean", "new_serve_workspace"}:
        return origin

    looks_like_shared_workspace = _looks_like_workspace_root(workspace_dir)
    if looks_like_shared_workspace:
        return "bootstrapped_autoclean"
    return "unknown"


def _workspace_checklist(workspace_dir: Path) -> list[dict[str, Any]]:
    """Return the workspace status checklist used by doctor/status views."""
    paths = _serve_workspace_paths(workspace_dir)
    checks: list[dict[str, Any]] = []

    required_items = [
        ("serve-test.yaml", paths["serve_test"]),
        ("serve-live.yaml", paths["serve_live"]),
        ("deploy/", paths["deploy"]),
        ("runtimes/test", paths["runtimes_test"]),
        ("runtimes/live", paths["runtimes_live"]),
    ]
    for label, path in required_items:
        checks.append({"label": label, "ok": path.exists(), "detail": str(path)})

    for mode in ("test", "live"):
        runtime_dir = paths[f"runtimes_{mode}"]
        venv_dir = runtime_dir / ".venv"
        python_candidates = [venv_dir / "bin" / "python", venv_dir / "Scripts" / "python.exe"]
        checks.append(
            {
                "label": f"{mode} runtime ready",
                "ok": any(candidate.exists() for candidate in python_candidates),
                "detail": str(venv_dir),
            }
        )

    return checks


def _workspace_doctor(workspace_dir: Path) -> dict[str, Any]:
    """Return action-oriented workspace diagnostics for the UI."""
    checks = _workspace_checklist(workspace_dir)
    blocking_issues: list[dict[str, str]] = []
    guidance: list[str] = []

    for item in checks:
        if item["ok"]:
            continue
        blocking_issues.append({"label": item["label"], "detail": item["detail"]})

    if any("runtime ready" in issue["label"] for issue in blocking_issues):
        guidance.append(
            "Re-run 'autocleaneeg-pipeline serve workspace --mode existing --path <dir>' to rebuild runtimes."
        )
    if any(issue["label"].startswith("serve-") for issue in blocking_issues):
        guidance.append(
            "Use Setup to open a valid Serve workspace or bootstrap an existing AutoClean workspace."
        )
    if any(issue["label"] == "deploy/" for issue in blocking_issues):
        guidance.append(
            "Apply the current configuration in Settings or run 'autocleaneeg-pipeline serve deploy --mode <test|live>' after validation."
        )

    summary = "Workspace looks healthy" if not blocking_issues else f"Found {len(blocking_issues)} blocking issue(s)"
    return {
        "ok": not blocking_issues,
        "summary": summary,
        "blocking_issues": blocking_issues,
        "guidance": guidance,
    }


def _looks_like_workspace_root(workspace_dir: Path) -> bool:
    """Return True when a directory already looks like a normal AutoClean workspace."""
    return (
        (workspace_dir / "tasks").exists()
        or (workspace_dir / "output").exists()
        or (workspace_dir / "tasks" / "builtin").exists()
    )


def _bootstrap_workspace_for_serve(workspace_dir: Path) -> None:
    """Ensure Serve-specific files exist inside the shared workspace root."""
    try:
        from autoclean.utils.user_config import UserConfigManager

        UserConfigManager()._create_workspace_structure(workspace_dir)  # type: ignore[attr-defined]
    except Exception:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "tasks").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "output").mkdir(parents=True, exist_ok=True)

    (workspace_dir / "routes").mkdir(exist_ok=True)
    (workspace_dir / "automations").mkdir(exist_ok=True)
    (workspace_dir / "deploy").mkdir(exist_ok=True)
    (workspace_dir / "runtimes" / "test").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "runtimes" / "live").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "incoming").mkdir(exist_ok=True)

    for mode_name in ("test", "live"):
        config_path = workspace_dir / f"serve-{mode_name}.yaml"
        if config_path.exists():
            continue
        config_content = f"""# Generated by AutoCleanEEG Serve setup wizard.
mode: {mode_name}
automation_mode: true
runtime: runtimes/{mode_name}
runtime_package: autocleaneeg-pipeline
defaults:
  automation_root: automations
  workspace_name: taskfile-montage-version
  file_globs:
    - "*.set"
    - "*.edf"
    - "*.bdf"
  sentinel_ext: .ready
  recursive: true
automations: []
"""
        config_path.write_text(config_content)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    yield
    # Shutdown - cleanup
    if api_state._redis_connection:
        api_state._redis_connection.close()


def create_app(
    workspace_dir: Optional[Path] = None,
    mode: str = "test",
    redis_url: str = "redis://localhost:6379",
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        workspace_dir: Path to serve workspace directory.
        mode: Configuration mode ("test" or "live").
        redis_url: Redis connection URL.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="AutoClean Automation API",
        description="""
## EEG Processing Automation API

REST API for managing automated EEG file processing pipelines.

### Default architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Monitored  │────▶│ queue-*.json │────▶│  serve run   │
│   Folders   │     │ per mode     │     │  dispatch    │
└─────────────┘     └──────────────┘     └──────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
  File Discovery       Mode queue           EEG Processing
```

### Quick Start

1. **Check status:** `GET /health`
2. **View queue:** `GET /api/queue/stats`
3. **List files:** `GET /api/queue/entries`
4. **View config:** `GET /api/config`

### Lanes

- **test** (port 8000): Draft
- **live** (port 8001): Production
""",
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=TAGS_METADATA,
        docs_url="/docs",
        redoc_url="/redoc",
        contact={
            "name": "Cincinnati Brain Lab",
            "url": "https://github.com/cincibrainlab/autoclean_pipeline",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # Configure CORS — restrict to localhost origins only.
    # allow_origin_regex covers any port on loopback so the port can vary
    # (e.g. --api-port 8001) without widening the policy to the public internet.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if workspace_dir:
        api_state.configure(workspace_dir, mode, redis_url)

    # Import routes here to avoid circular imports
    from autoclean.api import events
    from autoclean.api.routes import config, event_analyzer, exclude, filesystem, montage_browser, queue, results, serve_routes, service, task_browser, task_manager, tunnel, tutorial, worker

    # Include routers
    app.include_router(queue.router, prefix="/api/queue", tags=["Queue"])
    app.include_router(worker.router, prefix="/api/worker", tags=["Worker"])
    app.include_router(config.router, prefix="/api/config", tags=["Config"])
    app.include_router(serve_routes.router, prefix="/api/routes", tags=["Routes"])
    app.include_router(service.router, prefix="/api/service", tags=["Service"])
    app.include_router(tunnel.router, prefix="/api/tunnel", tags=["Tunnel"])
    app.include_router(tutorial.router, prefix="/api/tutorial", tags=["Tutorial"])
    app.include_router(filesystem.router, prefix="/api/filesystem", tags=["Filesystem"])
    app.include_router(task_browser.router, prefix="/api/tasks", tags=["Tasks"])
    app.include_router(task_manager.router, prefix="/api/task-manager", tags=["Task Manager"])
    app.include_router(montage_browser.router, prefix="/api/montages", tags=["Montages"])
    app.include_router(results.router, prefix="/api/results", tags=["Results"])
    app.include_router(exclude.router, prefix="/api/exclude", tags=["Exclude"])
    app.include_router(event_analyzer.router, prefix="/api/events", tags=["Events"])
    app.include_router(events.router, prefix="/ws", tags=["WebSocket"])

    # ── Tunnel auth middleware ────────────────────────────────────────
    @app.middleware("http")
    async def tunnel_auth_middleware(request: Request, call_next):
        """Require Basic Auth for non-local requests when a tunnel is active.

        cloudflared proxies all tunneled requests through localhost, so we
        cannot rely on client.host alone to distinguish local from remote
        traffic.  cloudflared always injects CF-Connecting-IP and
        X-Forwarded-For headers; if either is present the request came
        through the tunnel and must be authenticated.
        """
        # Detect whether this request arrived via the Cloudflare tunnel.
        # cloudflared adds CF-Connecting-IP and X-Forwarded-For to every
        # proxied request, even though the TCP connection originates from
        # localhost.
        is_tunneled = bool(
            request.headers.get("cf-connecting-ip")
            or request.headers.get("x-forwarded-for")
        )

        # Genuine localhost requests (not tunneled) are always permitted.
        if not is_tunneled:
            client_host = request.client.host if request.client else "127.0.0.1"
            if client_host in ("127.0.0.1", "::1"):
                return await call_next(request)

        # Check if tunnel is active — if not, no auth needed.
        state = tunnel.get_tunnel_state()
        if not state["active"] or not state["password"]:
            return await call_next(request)

        # Validate Basic Auth
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                username, password = decoded.split(":", 1)
                if username == "autoclean" and secrets.compare_digest(
                    password, state["password"]
                ):
                    return await call_next(request)
            except Exception as _auth_exc:
                import logging as _logging
                _logging.getLogger(__name__).debug(
                    "tunnel_auth_middleware: credential decode error: %s", _auth_exc
                )

        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="AutoClean"'},
            content="Authentication required",
        )

    @app.post("/api/mode/switch")
    async def switch_mode(body: dict[str, str]) -> dict[str, Any]:
        """Switch between test and live mode (Stripe-style toggle).

        Stops the running service if any, switches api_state.mode,
        and returns the new mode.
        """
        from fastapi import HTTPException as _HTTPExc
        from autoclean.api.routes.service import get_service_status, stop_service

        new_mode = body.get("mode", "").lower()
        if new_mode not in ("test", "live"):
            raise _HTTPExc(status_code=400, detail="Mode must be 'test' or 'live'")

        if new_mode == api_state.mode:
            return {"success": True, "mode": new_mode, "message": f"Already in {new_mode} mode"}

        # Stop running service before switching
        try:
            svc = get_service_status()
            if svc.get("running"):
                await stop_service()
        except Exception:
            pass

        old_mode = api_state.mode
        api_state.mode = new_mode

        return {
            "success": True,
            "mode": new_mode,
            "message": f"Switched from {old_mode} to {new_mode}",
        }

    @app.post("/api/setup/workspace")
    async def setup_workspace(body: dict) -> dict[str, Any]:
        """Configure or switch the serve workspace.

        Called by the workspace picker to open an existing workspace or create
        a new one. Stops the running service first if needed, then configures
        api_state in-place — no server restart required.
        """
        from fastapi import HTTPException as _HTTPException

        path = body.get("path", "")
        create_new = body.get("create_new", False)
        if not path:
            raise _HTTPException(status_code=400, detail="Path is required")

        workspace_dir = Path(path).expanduser().resolve()
        bootstrapped_existing_workspace = False

        # Stop running service before switching workspaces
        try:
            from autoclean.api.routes.service import get_service_status, stop_service
            if get_service_status().get("running"):
                await stop_service()
        except Exception:
            pass

        if create_new:
            if workspace_dir.exists() and any(workspace_dir.iterdir()):
                raise _HTTPException(
                    status_code=400,
                    detail="Workspace directory is not empty; use existing mode to link an existing workspace.",
                )
            _bootstrap_workspace_for_serve(workspace_dir)
        else:
            # Opening existing workspace.
            if not workspace_dir.exists():
                raise _HTTPException(
                    status_code=400,
                    detail=f"Workspace directory does not exist: {workspace_dir}",
                )
            serve_markers = (
                (workspace_dir / "serve-test.yaml").exists()
                and (workspace_dir / "serve-live.yaml").exists()
                and (workspace_dir / "automations").exists()
                and (workspace_dir / "runtimes" / "test").exists()
                and (workspace_dir / "runtimes" / "live").exists()
            )
            if not serve_markers:
                if not _looks_like_workspace_root(workspace_dir):
                    raise _HTTPException(
                        status_code=400,
                        detail="Workspace must already be a valid Serve workspace or an AutoClean workspace with tasks/output.",
                    )
                _bootstrap_workspace_for_serve(workspace_dir)
                bootstrapped_existing_workspace = True

        origin: str | None = None
        if bootstrapped_existing_workspace:
            origin = "bootstrapped_autoclean"
        elif create_new:
            origin = "new_serve_workspace"

        if origin is not None:
            try:
                _write_workspace_metadata(
                    workspace_dir,
                    {
                        "origin": origin,
                        "workspace_dir": str(workspace_dir),
                    },
                )
            except Exception:
                pass

        # Configure API state in-place — no restart required
        api_state.configure(workspace_dir, api_state.mode or "test", api_state.redis_url)

        # Persist workspace path for future launches
        try:
            from autoclean.utils.user_config import UserConfigManager

            ucm = UserConfigManager()
            ucm.set_serve_workspace(workspace_dir)
        except Exception:
            pass

        # Save to recent workspaces list
        _save_recent_workspace(str(workspace_dir))

        return {
            "success": True,
            "workspace_dir": str(workspace_dir),
            "message": f"Workspace configured: {workspace_dir}",
        }

    @app.get("/api/workspace/utilities")
    async def get_workspace_utilities() -> dict[str, Any]:
        """Return workspace status and doctor diagnostics for the Utilities page."""
        if not api_state.workspace_dir:
            return {"configured": False}

        workspace_dir = api_state.workspace_dir
        paths = _serve_workspace_paths(workspace_dir)
        metadata = _read_workspace_metadata(workspace_dir)
        status_checks = _workspace_checklist(workspace_dir)
        doctor = _workspace_doctor(workspace_dir)
        checks_by_label = {item["label"]: item for item in status_checks}

        return {
            "configured": True,
            "workspace_dir": str(workspace_dir),
            "selected_workspace_path": str(workspace_dir),
            "bootstrap_origin": _workspace_bootstrap_origin(workspace_dir),
            "bootstrapped_from_autoclean": _workspace_bootstrap_origin(workspace_dir) == "bootstrapped_autoclean",
            "workspace_details": {
                "serve_test_exists": paths["serve_test"].exists(),
                "serve_live_exists": paths["serve_live"].exists(),
                "deploy_exists": paths["deploy"].exists(),
                "runtimes_test_exists": paths["runtimes_test"].exists(),
                "runtimes_live_exists": paths["runtimes_live"].exists(),
                "test_runtime_ready": checks_by_label["test runtime ready"]["ok"],
                "live_runtime_ready": checks_by_label["live runtime ready"]["ok"],
            },
            "status_checks": status_checks,
            "doctor": doctor,
            "metadata": metadata,
        }

    @app.get("/api/workspaces/recent")
    async def get_recent_workspaces() -> dict[str, Any]:
        """Return list of recently used workspaces with health info."""
        recent = _load_recent_workspaces()
        results = []
        for path_str in recent[:10]:
            ws = Path(path_str)
            if not ws.exists():
                continue
            routes_dir = ws / "routes"
            has_routes = routes_dir.exists() and any(routes_dir.iterdir()) if routes_dir.exists() else False
            n_routes = len(list(routes_dir.glob("*.yaml"))) if routes_dir.exists() else 0
            has_runtime_test = (ws / "runtimes" / "test" / ".venv").exists()
            has_runtime_live = (ws / "runtimes" / "live" / ".venv").exists()
            results.append({
                "path": str(ws),
                "name": ws.name,
                "has_routes": has_routes,
                "n_routes": n_routes,
                "has_runtime_test": has_runtime_test,
                "has_runtime_live": has_runtime_live,
                "is_current": str(ws) == str(api_state.workspace_dir) if api_state.workspace_dir else False,
            })
        return {
            "workspaces": results,
            "current": str(api_state.workspace_dir) if api_state.workspace_dir else None,
        }

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        from autoclean import __version__ as pipeline_version

        redis_ok = api_state.check_redis() if api_state.workspace_dir else False
        return {
            "status": "healthy",
            "workspace_configured": api_state.workspace_dir is not None,
            "mode": api_state.mode,
            "redis_connected": redis_ok,
            "pipeline_version": pipeline_version,
        }

    @app.get("/api/status")
    async def get_status() -> dict[str, Any]:
        """Full workspace status snapshot for the dashboard."""
        from autoclean.utils.serve_routes import load_route_specs

        if not api_state.workspace_dir:
            return {"configured": False}

        # Load routes filtered by current mode
        try:
            all_routes = load_route_specs(api_state.workspace_dir)
            routes = [r for r in all_routes if api_state.mode in r.get("modes", [])]
        except Exception:
            routes = []

        # Load queue stats
        queue_path = api_state.get_queue_path()
        queue_stats = {
            "pending": 0,
            "processing": 0,
            "processed": 0,
            "failed": 0,
            "total": 0,
        }
        try:
            from autoclean.utils.ingestion import IngestionQueue

            q = IngestionQueue(queue_path)
            for data in q.entries().values():
                s = data.get("status", "pending")
                if s in queue_stats:
                    queue_stats[s] += 1
                queue_stats["total"] += 1
        except Exception:
            pass

        # Config status
        config_path = api_state.get_config_path(deployed=False)
        deployed_path = api_state.get_config_path(deployed=True)
        config_valid = False
        config_errors: list[str] = []
        needs_deploy = False
        try:
            from autoclean.utils.ingestion import load_serve_config, parse_serve_config

            raw = load_serve_config(config_path)
            parse_serve_config(raw, api_state.workspace_dir, strict=True)
            config_valid = True
        except Exception as e:
            config_errors = [str(e)]

        if config_path.exists() and deployed_path.exists():
            needs_deploy = config_path.read_text() != deployed_path.read_text()
        elif config_path.exists():
            needs_deploy = True

        # Service status
        from autoclean.api.routes.service import get_service_status

        svc = get_service_status()
        operational_state = "ready"
        next_step = None
        processing_state = "idle"

        if len(routes) == 0:
            operational_state = "setup_incomplete"
            next_step = "Create a route before processing can start."
        elif not config_valid:
            operational_state = "blocked"
            next_step = "Fix configuration errors before starting processing."
        elif needs_deploy:
            operational_state = "needs_apply"
            next_step = "Apply the latest configuration changes."
        elif not svc.get("running"):
            operational_state = "ui_only"
            next_step = "Start the processing service."

        if queue_stats["processing"] > 0:
            processing_state = "processing"
        elif queue_stats["pending"] > 0:
            processing_state = "queued"
        elif queue_stats["failed"] > 0:
            processing_state = "attention"

        # Resolve output root
        output_root = str(api_state.workspace_dir / "automations")

        return {
            "configured": True,
            "mode": api_state.mode,
            "workspace_dir": str(api_state.workspace_dir),
            "output_dir": output_root,
            "routes": {
                "total": len(routes),
                "active": len(
                    [
                        r
                        for r in routes
                        if not r.get("archived", False) and r.get("enabled", True)
                    ]
                ),
                "archived": len(
                    [r for r in routes if r.get("archived", False)]
                ),
            },
            "queue": queue_stats,
            "config": {
                "valid": config_valid,
                "errors": config_errors,
                "needs_deploy": needs_deploy,
                "source": (
                    "deployed"
                    if deployed_path.exists()
                    else ("operator" if config_path.exists() else "missing")
                ),
            },
            "service": svc,
            "operational_state": operational_state,
            "processing_state": processing_state,
            "next_step": next_step,
        }

    # Serve built frontend if available
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir() and (static_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        # SPA catch-all: return index.html for client-side routes only
        _API_PREFIXES = ("api/", "ws/", "health", "docs", "redoc", "openapi.json")

        @app.get("/{path:path}")
        async def spa_fallback(path: str) -> Any:
            # Never intercept API or infrastructure paths
            if path.startswith(_API_PREFIXES):
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": f"Not found: /{path}"}, status_code=404)
            # Serve actual static files if they exist
            file_path = (static_dir / path).resolve()
            if not file_path.is_relative_to(static_dir.resolve()):
                # Path traversal attempt — serve index.html (SPA routing)
                return FileResponse(static_dir / "index.html")
            if path and file_path.is_file():
                return FileResponse(file_path)
            # Everything else gets index.html (client-side routing)
            return FileResponse(static_dir / "index.html")

        # Mount assets directory for hashed JS/CSS bundles
        assets_dir = static_dir / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    else:
        @app.get("/")
        async def root() -> dict[str, str]:
            """Root endpoint with API info."""
            return {
                "name": "AutoClean Automation API",
                "version": "1.0.0",
                "docs": "/docs",
                "ui": "Run 'cd web && npm run build' to enable the web UI",
            }

    return app


def _create_bound_app_factory(
    workspace_dir: Optional[Path],
    mode: str,
    redis_url: str,
):
    """Bind explicit startup arguments into a Uvicorn-compatible app factory."""

    def _factory() -> FastAPI:
        return create_app(workspace_dir=workspace_dir, mode=mode, redis_url=redis_url)

    return _factory


def run_server(
    workspace_dir: Optional[Path] = None,
    mode: str = "test",
    host: str = "127.0.0.1",
    port: int = 8000,
    redis_url: str = "redis://localhost:6379",
    reload: bool = False,
) -> None:
    """Run the API server.

    Args:
        workspace_dir: Path to serve workspace directory. May be None for
            first-run — the web UI setup wizard will configure the workspace
            in-place via POST /api/setup/workspace.
        mode: Configuration mode ("test" or "live").
        host: Host to bind to.
        port: Port to listen on.
        redis_url: Redis connection URL.
        reload: Enable auto-reload for development.
    """
    import os
    import uvicorn

    # Publish the API port so the tunnel route can bind cloudflared to the
    # correct loopback address without accepting a caller-supplied port.
    os.environ["AUTOCLEAN_API_PORT"] = str(port)

    resolved_workspace = workspace_dir or _load_persisted_serve_workspace()

    # Configure global state only if a workspace was provided or previously
    # persisted by the CLI/web setup flow. This keeps the app factory itself
    # deterministic for tests while preserving operator-friendly boot behavior.
    if resolved_workspace is not None:
        api_state.configure(resolved_workspace, mode, redis_url)

    uvicorn.run(
        _create_bound_app_factory(resolved_workspace, mode, redis_url),
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
