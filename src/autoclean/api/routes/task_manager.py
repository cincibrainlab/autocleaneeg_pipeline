"""Task Manager endpoints — unified catalog of all EEG processing tasks.

Merges built-in (registry), discovered (workspace/package), and library tasks
into a single view with install/create/sync actions for operator management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from autoclean.api.routes.task_browser import TaskConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Response models ─────────────────────────────────────────────────

SyncStatus = Literal["installed", "modified", "not_installed", "workspace_only"]


class ManagedTask(BaseModel):
    """Full managed task descriptor for the Task Manager UI."""

    name: str
    description: str
    category: str
    source: str
    sync_status: SyncStatus
    workspace_path: Optional[str] = None
    config: Optional[TaskConfig] = None
    pipeline: list[str] = []
    source_code: str = ""


class RegistryInfo(BaseModel):
    commit: Optional[str] = None
    synced_at: Optional[str] = None
    task_count: int = 0


class TaskManagerResponse(BaseModel):
    tasks: list[ManagedTask]
    registry_status: RegistryInfo
    workspace_dir: str


class InstallRequest(BaseModel):
    task_name: str


class CreateRequest(BaseModel):
    class_name: str


class TaskActionResponse(BaseModel):
    success: bool
    message: str
    task_name: str = ""
    path: Optional[str] = None


# ── Helpers ─────────────────────────────────────────────────────────

def _get_workspace_dir() -> Optional[Path]:
    """Return the user workspace tasks directory, or None if unavailable."""
    try:
        from autoclean.utils.user_config import UserConfigManager
        mgr = UserConfigManager()
        return mgr.tasks_dir
    except Exception as exc:
        logger.debug("UserConfigManager unavailable: %s", exc)
        return None


def _get_workspace_str() -> str:
    """Return workspace dir as string for response payload."""
    d = _get_workspace_dir()
    return str(d) if d else ""


def _registry_status() -> RegistryInfo:
    """Return high-level registry sync metadata."""
    try:
        from autoclean.utils.builtins import BuiltinRegistry
        reg = BuiltinRegistry()
        status = reg.registry_status()
        tasks = reg.list_tasks()
        return RegistryInfo(
            commit=status.get("commit"),
            synced_at=status.get("synced_at"),
            task_count=len(tasks),
        )
    except Exception as exc:
        logger.debug("BuiltinRegistry unavailable: %s", exc)
        return RegistryInfo()


def _build_detail_for(discovered_task: Any) -> tuple[Optional[TaskConfig], list[str], str]:
    """Return (config, pipeline, source_code) by delegating to task_browser logic."""
    try:
        from autoclean.api.routes.task_browser import _build_task_detail
        detail = _build_task_detail(discovered_task)
        if detail is None:
            return None, [], ""
        return detail.config, detail.pipeline, detail.source_code
    except Exception as exc:
        logger.debug("Could not enrich task '%s': %s", discovered_task.name, exc)
        return None, [], ""


def _sync_status_for_builtin(name: str, workspace_dir: Optional[Path]) -> tuple[str, Optional[str]]:
    """Return (sync_status, workspace_path) for a registry task.

    Maps BuiltinRegistry status strings onto our frontend vocabulary:
        synced        -> installed
        modified      -> modified
        not_installed -> not_installed
        unknown/missing -> not_installed (safe fallback)
    """
    try:
        from autoclean.utils.builtins import BuiltinRegistry
        reg = BuiltinRegistry()
        info = reg.task_sync_status(name, workspace_dir)
        raw = info.get("status", "not_installed")
        wp = info.get("workspace_path")
        mapping = {
            "synced": "installed",
            "modified": "modified",
            "not_installed": "not_installed",
            "unknown": "not_installed",
            "missing": "not_installed",
        }
        return mapping.get(raw, "not_installed"), wp
    except Exception:
        return "not_installed", None


# ── GET /api/task-manager ────────────────────────────────────────────

@router.get("", response_model=TaskManagerResponse)
async def get_task_manager() -> TaskManagerResponse:
    """Return unified task catalog merging library, builtin, and workspace tasks."""

    workspace_dir = _get_workspace_dir()
    registry_info = _registry_status()

    # -- 1. Collect library task names from BuiltinRegistry
    library_names: set[str] = set()
    try:
        from autoclean.utils.builtins import BuiltinRegistry
        reg = BuiltinRegistry()
        for bt in reg.list_tasks():
            library_names.add(bt.name)
    except Exception as exc:
        logger.debug("Could not list builtin tasks: %s", exc)

    # -- 2. Discover all tasks (workspace + installed package)
    discovered_tasks: list[Any] = []
    try:
        from autoclean.utils.task_discovery import safe_discover_tasks
        valid_tasks, _invalid, _skipped = safe_discover_tasks()
        discovered_tasks = valid_tasks
    except Exception as exc:
        logger.error("Task discovery failed: %s", exc)

    # Build a name -> discovered_task lookup
    discovered_by_name: dict[str, Any] = {t.name: t for t in discovered_tasks}

    # -- 3. Determine workspace task names (files in workspace tasks dir)
    workspace_task_names: set[str] = set()
    if workspace_dir and workspace_dir.exists():
        for py_file in workspace_dir.rglob("*.py"):
            if not py_file.name.startswith("_"):
                workspace_task_names.add(py_file.stem)

    # -- 4. Assemble the full task list

    tasks: list[ManagedTask] = []

    # Pass A: all discovered tasks (covers both builtin package tasks and workspace tasks)
    for dt in discovered_tasks:
        name = dt.name
        source_path = dt.source

        # Is this task in the workspace directory?
        is_workspace = (
            workspace_dir is not None
            and str(source_path).startswith(str(workspace_dir))
        )

        # Determine source label
        if name in library_names:
            source_label = "library"
        elif is_workspace:
            source_label = "workspace"
        else:
            source_label = "builtin"

        # Determine sync status
        if name in library_names:
            sync_status, wp = _sync_status_for_builtin(name, workspace_dir)
        elif is_workspace:
            # Workspace-only task (not in registry)
            sync_status = "workspace_only"
            wp = str(source_path)
        else:
            # Builtin package task, not in registry
            sync_status = "installed"
            wp = str(source_path)

        config, pipeline, source_code = _build_detail_for(dt)

        # Derive category from task_browser helper
        try:
            from autoclean.api.routes.task_browser import _derive_category
            category = _derive_category(source_path)
        except Exception:
            category = "custom"

        tasks.append(ManagedTask(
            name=name,
            description=dt.description,
            category=category,
            source=source_label,
            sync_status=sync_status,
            workspace_path=wp,
            config=config,
            pipeline=pipeline,
            source_code=source_code,
        ))

    # Pass B: library tasks not yet discovered (i.e., not installed anywhere)
    discovered_names = {t.name for t in discovered_tasks}
    for lib_name in sorted(library_names - discovered_names):
        tasks.append(ManagedTask(
            name=lib_name,
            description="",
            category="builtin",
            source="library",
            sync_status="not_installed",
            workspace_path=None,
            config=None,
            pipeline=[],
            source_code="",
        ))

    return TaskManagerResponse(
        tasks=tasks,
        registry_status=registry_info,
        workspace_dir=_get_workspace_str(),
    )


# ── POST /api/task-manager/install ──────────────────────────────────

@router.post("/install", response_model=TaskActionResponse)
async def install_task(body: InstallRequest) -> TaskActionResponse:
    """Install a task from the library/builtin registry into the workspace."""

    task_name = body.task_name
    workspace_dir = _get_workspace_dir()
    if workspace_dir is None:
        raise HTTPException(status_code=400, detail="Workspace not configured")

    try:
        from autoclean.utils.builtins import BuiltinRegistry
        reg = BuiltinRegistry()
        dest_path = reg.materialize_task_to(task_name, workspace_dir)
        return TaskActionResponse(
            success=True,
            message=f"Task '{task_name}' installed to workspace.",
            task_name=task_name,
            path=str(dest_path),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to install task '%s': %s", task_name, exc)
        raise HTTPException(status_code=500, detail=f"Install failed: {exc}")


# ── POST /api/task-manager/create ───────────────────────────────────

@router.post("/create", response_model=TaskActionResponse)
async def create_task(body: CreateRequest) -> TaskActionResponse:
    """Create a new task from the custom task template."""

    class_name = body.class_name
    workspace_dir = _get_workspace_dir()
    if workspace_dir is None:
        raise HTTPException(status_code=400, detail="Workspace not configured")

    # Validate identifier
    try:
        from autoclean.utils.template_renderer import validate_python_identifier
        validate_python_identifier(class_name, label="class_name")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Locate template
    try:
        import autoclean
        package_dir = Path(autoclean.__file__).parent
    except Exception:
        package_dir = Path(__file__).parent.parent.parent

    template_path = package_dir / "templates" / "custom_task_template.jinja"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail=f"Template not found: {template_path}")

    # Render template
    try:
        from autoclean.utils.template_renderer import render_template
        rendered = render_template(template_path, {"class_name": class_name})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Template render failed: {exc}")

    # Write to workspace
    dest_path = workspace_dir / f"{class_name}.py"
    if dest_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Task file '{dest_path.name}' already exists in workspace.",
        )

    try:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(rendered, encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write task file: {exc}")

    return TaskActionResponse(
        success=True,
        message=f"Task '{class_name}' created in workspace.",
        task_name=class_name,
        path=str(dest_path),
    )


# ── POST /api/task-manager/refresh-library ──────────────────────────

@router.post("/refresh-library", response_model=TaskActionResponse)
async def refresh_library() -> TaskActionResponse:
    """Refresh the GitHub registry cache."""

    try:
        from autoclean.utils.builtins import BuiltinRegistry
        reg = BuiltinRegistry()
        message = reg.update_cache(allow_network=True)
        return TaskActionResponse(
            success=True,
            message=message,
        )
    except Exception as exc:
        logger.error("Registry refresh failed: %s", exc)
        return TaskActionResponse(
            success=False,
            message=f"Refresh failed: {exc}",
        )


# ── POST /api/task-manager/{task_name}/update ───────────────────────

@router.post("/{task_name}/update", response_model=TaskActionResponse)
async def update_task(task_name: str) -> TaskActionResponse:
    """Re-materialize a task from the registry to pick up the latest version."""

    workspace_dir = _get_workspace_dir()
    if workspace_dir is None:
        raise HTTPException(status_code=400, detail="Workspace not configured")

    task_file = workspace_dir / f"{task_name}.py"
    if not task_file.resolve().is_relative_to(workspace_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid task name")

    try:
        from autoclean.utils.builtins import BuiltinRegistry
        reg = BuiltinRegistry()
        dest_path = reg.materialize_task_to(task_name, workspace_dir)
        return TaskActionResponse(
            success=True,
            message=f"Task '{task_name}' updated to latest version.",
            task_name=task_name,
            path=str(dest_path),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to update task '%s': %s", task_name, exc)
        raise HTTPException(status_code=500, detail=f"Update failed: {exc}")


# ── DELETE /api/task-manager/{task_name} ────────────────────────────

@router.delete("/{task_name}", response_model=TaskActionResponse)
async def remove_task(task_name: str) -> TaskActionResponse:
    """Remove a task file from the workspace tasks directory."""

    workspace_dir = _get_workspace_dir()
    if workspace_dir is None:
        raise HTTPException(status_code=400, detail="Workspace not configured")

    task_file = workspace_dir / f"{task_name}.py"
    if not task_file.resolve().is_relative_to(workspace_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid task name")

    if not task_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_name}' not found in workspace.",
        )

    try:
        task_file.unlink()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to remove task: {exc}")

    return TaskActionResponse(
        success=True,
        message=f"Task '{task_name}' removed from workspace.",
        task_name=task_name,
        path=None,
    )
