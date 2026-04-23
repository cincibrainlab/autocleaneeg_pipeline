"""Route spec CRUD endpoints.

Manages individual route-spec YAML files that define automation routes.
These are the operator-editable source-of-truth; ``POST /sync`` compiles
them into the mode-specific ``serve-*.yaml`` config consumed by
``serve run``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from autoclean.api.models import (
    MontageOption,
    RouteActionResponse,
    RouteSpecResponse,
    RouteUpsertRequest,
    SyncResponse,
    TaskOption,
)
from autoclean.api.state import api_state

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_workspace():
    """Raise 500 if workspace is not configured."""
    if not api_state.workspace_dir:
        raise HTTPException(status_code=409, detail="Workspace not configured")


@contextmanager
def _serve_task_discovery_context(workspace_dir: Path):
    """Temporarily point task discovery at the active Serve workspace."""
    from autoclean.utils.user_config import user_config

    original_tasks_dir = user_config.tasks_dir
    original_config_dir = user_config.config_dir
    serve_tasks_dir = workspace_dir / "tasks"
    serve_tasks_dir.mkdir(parents=True, exist_ok=True)

    try:
        user_config.tasks_dir = serve_tasks_dir
        user_config.config_dir = workspace_dir
        yield
    finally:
        user_config.tasks_dir = original_tasks_dir
        user_config.config_dir = original_config_dir


# ── List / Read ──────────────────────────────────────────────────────

@router.get("/discovery/tasks", response_model=list[TaskOption])
async def list_tasks():
    """Return available task files that can be assigned to routes."""
    _require_workspace()
    try:
        from autoclean.utils.task_discovery import safe_discover_tasks

        with _serve_task_discovery_context(api_state.workspace_dir):
            tasks, _invalid, _skipped = safe_discover_tasks()
        return [
            TaskOption(
                name=t.name,
                source=str(t.source),
                description=getattr(t, "description", ""),
            )
            for t in tasks
        ]
    except Exception as exc:
        logger.warning("Task discovery failed: %s", exc)
        return []


@router.get("/discovery/montages", response_model=list[MontageOption])
async def list_montages():
    """Return available montage configurations."""
    _require_workspace()
    try:
        from autoclean.utils.montage import load_valid_montages

        montages = load_valid_montages()
        if isinstance(montages, dict):
            return [
                MontageOption(name=name, description=description or "")
                for name, description in montages.items()
            ]
        return [MontageOption(name=str(m), description="") for m in montages]
    except Exception as exc:
        logger.warning("Montage catalog load failed: %s", exc)
        return []


@router.get("", response_model=list[RouteSpecResponse])
async def list_routes():
    """List route specs filtered by the current mode."""
    _require_workspace()
    try:
        from autoclean.utils.serve_routes import load_route_specs

        specs = load_route_specs(api_state.workspace_dir)
        # Filter to routes that include the current mode
        mode = api_state.mode
        filtered = [s for s in specs if mode in s.get("modes", [])]
        return [_spec_to_response(s) for s in filtered]
    except Exception as exc:
        logger.exception("Failed to load route specs")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{route_id}", response_model=RouteSpecResponse)
async def get_route(route_id: str):
    """Get a single route spec by ID."""
    _require_workspace()
    from autoclean.utils.serve_routes import load_route_specs, normalize_route_id

    route_id = normalize_route_id(route_id)
    specs = load_route_specs(api_state.workspace_dir)
    for s in specs:
        if s.get("id") == route_id:
            return _spec_to_response(s)

    raise HTTPException(status_code=404, detail=f"Route '{route_id}' not found")


# ── Create / Update ─────────────────────────────────────────────────

@router.post("", response_model=RouteActionResponse)
async def upsert_route(body: RouteUpsertRequest):
    """Create or update a route spec."""
    _require_workspace()
    from autoclean.utils.serve_routes import upsert_route_spec

    updates: dict[str, Any] = body.model_dump(exclude={"id"})
    try:
        _path, _spec, action = upsert_route_spec(
            api_state.workspace_dir, body.id, updates
        )
        return RouteActionResponse(
            success=True,
            message=f"Route '{body.id}' {action}",
            route_id=body.id,
        )
    except Exception as exc:
        logger.exception("Upsert failed for route '%s'", body.id)
        raise HTTPException(status_code=400, detail=str(exc))


# ── Delete ───────────────────────────────────────────────────────────

@router.delete("/{route_id}", response_model=RouteActionResponse)
async def delete_route(route_id: str):
    """Delete a route spec (must be archived first)."""
    _require_workspace()
    from autoclean.utils.serve_routes import delete_route_spec, normalize_route_id

    route_id = normalize_route_id(route_id)
    ok, err = delete_route_spec(api_state.workspace_dir, route_id)
    if not ok:
        raise HTTPException(status_code=400, detail=err or "Delete failed")

    return RouteActionResponse(
        success=True, message=f"Route '{route_id}' deleted", route_id=route_id
    )


# ── Lifecycle actions ────────────────────────────────────────────────

@router.post("/{route_id}/promote", response_model=RouteActionResponse)
async def promote_route(route_id: str):
    """Promote a route to include the 'live' mode."""
    _require_workspace()
    from autoclean.utils.serve_routes import normalize_route_id, promote_route_spec

    route_id = normalize_route_id(route_id)
    try:
        promote_route_spec(api_state.workspace_dir, route_id)
        return RouteActionResponse(
            success=True,
            message=f"Route '{route_id}' promoted to live",
            route_id=route_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{route_id}/archive", response_model=RouteActionResponse)
async def archive_route(route_id: str):
    """Set a route's archived flag to True."""
    _require_workspace()
    from autoclean.utils.serve_routes import normalize_route_id, set_route_archived

    route_id = normalize_route_id(route_id)
    try:
        set_route_archived(api_state.workspace_dir, route_id, archived=True)
        return RouteActionResponse(
            success=True,
            message=f"Route '{route_id}' archived",
            route_id=route_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{route_id}/unarchive", response_model=RouteActionResponse)
async def unarchive_route(route_id: str):
    """Set a route's archived flag to False."""
    _require_workspace()
    from autoclean.utils.serve_routes import normalize_route_id, set_route_archived

    route_id = normalize_route_id(route_id)
    try:
        set_route_archived(api_state.workspace_dir, route_id, archived=False)
        return RouteActionResponse(
            success=True,
            message=f"Route '{route_id}' unarchived",
            route_id=route_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{route_id}/enable", response_model=RouteActionResponse)
async def enable_route(route_id: str):
    """Set a route's enabled flag to True."""
    _require_workspace()
    from autoclean.utils.serve_routes import normalize_route_id, upsert_route_spec

    route_id = normalize_route_id(route_id)
    try:
        upsert_route_spec(api_state.workspace_dir, route_id, {"enabled": True})
        return RouteActionResponse(
            success=True,
            message=f"Route '{route_id}' enabled",
            route_id=route_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/{route_id}/disable", response_model=RouteActionResponse)
async def disable_route(route_id: str):
    """Set a route's enabled flag to False."""
    _require_workspace()
    from autoclean.utils.serve_routes import normalize_route_id, upsert_route_spec

    route_id = normalize_route_id(route_id)
    try:
        upsert_route_spec(api_state.workspace_dir, route_id, {"enabled": False})
        return RouteActionResponse(
            success=True,
            message=f"Route '{route_id}' disabled",
            route_id=route_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ── Sync ─────────────────────────────────────────────────────────────

@router.post("/sync", response_model=SyncResponse)
async def sync_routes():
    """Recompile route spec files into serve-*.yaml configs."""
    _require_workspace()
    from autoclean.utils.serve_routes import sync_route_registry

    try:
        result = sync_route_registry(
            api_state.workspace_dir, modes=("test", "live")
        )
        synced_modes = list(result.keys()) if isinstance(result, dict) else []
        test_info = result.get("test", {}) if isinstance(result, dict) else {}
        live_info = result.get("live", {}) if isinstance(result, dict) else {}
        return SyncResponse(
            success=True,
            message=(
                f"Synced route registry for modes: {', '.join(synced_modes)}"
                if synced_modes
                else "Synced route registry"
            ),
            test_path=str(test_info.get("path")) if test_info.get("path") else None,
            live_path=str(live_info.get("path")) if live_info.get("path") else None,
        )
    except Exception as exc:
        logger.exception("Route sync failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Helpers ──────────────────────────────────────────────────────────

def _compute_output_folder(spec: dict) -> str:
    """Compute the output folder path for a route spec."""
    if not api_state.workspace_dir:
        return ""
    try:
        from autoclean.utils.ingestion import build_workspace_name

        taskfile = spec.get("taskfile", "")
        montage = spec.get("montage", "")
        if not taskfile or not montage:
            return ""
        # Strip path to just the task name
        taskfile_label = taskfile.rsplit("/", 1)[-1].replace(".py", "")
        workspace_name = build_workspace_name(
            spec.get("workspace_name", "taskfile-montage-version"),
            taskfile=taskfile_label,
            montage=montage,
            version=spec.get("version"),
        )
        automation_root = spec.get("automation_root", "automations")
        output = api_state.workspace_dir / str(automation_root) / workspace_name
        return str(output)
    except Exception:
        return ""


def _spec_to_response(spec: dict) -> RouteSpecResponse:
    """Convert a raw route-spec dict to a RouteSpecResponse."""
    return RouteSpecResponse(
        id=spec.get("id", ""),
        modes=spec.get("modes", ["test"]),
        enabled=spec.get("enabled", True),
        archived=spec.get("archived", False),
        priority=spec.get("priority", 0),
        taskfile=spec.get("taskfile", ""),
        montage=spec.get("montage", ""),
        ingestion_folders=spec.get("ingestion_folders", []),
        file_globs=spec.get("file_globs", []),
        recursive=spec.get("recursive", True),
        output_folder=_compute_output_folder(spec),
    )
