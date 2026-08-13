"""Route spec CRUD endpoints.

Manages individual route-spec YAML files that define automation routes.
These are the operator-editable source-of-truth; ``POST /sync`` compiles
them into the mode-specific ``serve-*.yaml`` config consumed by
``serve run``.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from autoclean.api.models import (
    MontageOption,
    RouteActionResponse,
    RouteMontageCopyEstimateResponse,
    RouteMontageReviewApplyRequest,
    RouteMontageReviewApplyResponse,
    RouteMontageReviewFile,
    RouteMontageReviewGroup,
    RouteMontageReviewScanRequest,
    RouteMontageReviewScanResponse,
    RouteSpecResponse,
    RouteUpsertRequest,
    SyncResponse,
    TaskOption,
)
from autoclean.api.state import api_state
from autoclean.utils.montage_preflight import (
    SUPPORTED_HYDROCEL_MONTAGES,
    MontageBatchPlan,
    MontagePreflightFileResult,
    MontagePreflightGroup,
    build_batch_plan,
    clone_task_for_montage,
    copy_originals_for_plan,
    estimate_copy_originals_for_plan,
    write_apply_summary,
    write_batch_plan_json,
    write_scan_csv,
)

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
        result = sync_route_registry(api_state.workspace_dir, modes=("test", "live"))
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


# ── Montage review ──────────────────────────────────────────────────


@router.post(
    "/{route_id}/montage-review/scan",
    response_model=RouteMontageReviewScanResponse,
)
async def scan_route_montage_review(
    route_id: str, body: RouteMontageReviewScanRequest | None = None
):
    """Scan a route's input files for Serve montage preflight review."""

    _require_workspace()
    try:
        return _scan_route_montage_review(
            route_id, body or RouteMontageReviewScanRequest()
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Route montage review scan failed for '%s'", route_id)
        raise HTTPException(status_code=400, detail=str(exc))


@router.post(
    "/{route_id}/montage-review/apply",
    response_model=RouteMontageReviewApplyResponse,
)
async def apply_route_montage_review(
    route_id: str, body: RouteMontageReviewApplyRequest
):
    """Apply a confirmed Serve montage preflight review in copy mode."""

    _require_workspace()
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Explicit confirmation is required before applying montage review",
        )
    if body.mode != "copy":
        raise HTTPException(
            status_code=400,
            detail="Only copy mode is supported by Serve montage review",
        )

    try:
        plan, route_spec, suggestions = _build_route_montage_plan(route_id, body)
        if not plan.actionable_files:
            raise HTTPException(
                status_code=400,
                detail="No supported detected montage files are available to apply",
            )

        review = _route_review_response_from_plan(plan, route_spec, suggestions)
        split_output_root = Path(review.split_output_root)

        write_scan_csv(plan, split_output_root)
        write_batch_plan_json(plan, split_output_root)

        copy_result = copy_originals_for_plan(
            plan,
            split_output_root=split_output_root,
            overwrite=body.overwrite_existing,
        )
        write_apply_summary(output_dir=split_output_root, copy_result=copy_result)

        cloned_tasks = []
        route_actions = []

        from autoclean.utils.serve_routes import sync_route_registry, upsert_route_spec

        for detected_montage, suggestion in suggestions.items():
            if detected_montage == "unknown" or not suggestion.get("supported"):
                continue
            if suggestion["suggested_route_id"] == route_spec["id"]:
                continue

            try:
                cloned_task = clone_task_for_montage(
                    source_task_path=Path(plan.task_path),
                    task_output_dir=api_state.workspace_dir / "tasks",
                    target_montage=detected_montage,
                    class_name=suggestion["clone_class_name"],
                    file_stem=Path(str(suggestion["suggested_taskfile"])).stem,
                    overwrite=body.overwrite_existing,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task clone validation failed: {exc}",
                ) from exc
            cloned_tasks.append(asdict(cloned_task))

            _path, _spec, action = upsert_route_spec(
                api_state.workspace_dir,
                str(suggestion["suggested_route_id"]),
                {
                    "modes": route_spec.get("modes", [api_state.mode]),
                    "enabled": route_spec.get("enabled", True),
                    "priority": int(route_spec.get("priority", 0)),
                    "taskfile": str(suggestion["suggested_taskfile"]),
                    "montage": detected_montage,
                    "version": route_spec.get("version"),
                    "ingestion_folders": [
                        str(suggestion["suggested_ingestion_folder"])
                    ],
                    "ingestion_excludes": route_spec.get("ingestion_excludes", []),
                    "file_globs": route_spec.get("file_globs", ["*"]),
                    "recursive": route_spec.get("recursive", True),
                    "sentinel_ext": route_spec.get("sentinel_ext"),
                    "automation_root": route_spec.get("automation_root", "automations"),
                    "workspace_name": route_spec.get(
                        "workspace_name", "taskfile-montage-version"
                    ),
                },
            )
            route_actions.append(
                {
                    "route_id": suggestion["suggested_route_id"],
                    "action": action,
                    "detected_montage": detected_montage,
                }
            )

        if route_actions:
            sync_route_registry(api_state.workspace_dir, modes=("test", "live"))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(api_state.get_queue_path())
        entries = queue.entries()
        enqueued = 0
        updated_queue_entries = 0
        files_by_source = {result.path: result for result in plan.files}

        for copied in copy_result.copied_files:
            destination = Path(str(copied["destination"]))
            source = str(copied["source"])
            file_result = files_by_source[source]
            detected_montage = str(file_result.detected_montage)
            suggestion = suggestions[detected_montage]
            destination_key = str(destination)
            existed = destination_key in entries
            queue.enqueue(
                [destination],
                route_id=str(suggestion["suggested_route_id"]),
                ingestion_root=Path(str(suggestion["suggested_ingestion_folder"])),
            )
            if not existed:
                enqueued += 1
            entry = queue.entries()[destination_key]
            entry.update(
                {
                    "expected_montage": plan.expected_montage,
                    "detected_montage": detected_montage,
                    "taskfile": suggestion["suggested_taskfile"],
                    "route_id": suggestion["suggested_route_id"],
                    "route_review_source_path": source,
                    "route_review_original_route_id": route_spec["id"],
                    "workspace_context": {
                        "workspace_dir": str(api_state.workspace_dir),
                        "workspace_name": suggestion["suggested_workspace_name"],
                        "automation_root": route_spec.get(
                            "automation_root", "automations"
                        ),
                    },
                }
            )
            updated_queue_entries += 1
        queue.save()

        return RouteMontageReviewApplyResponse(
            success=True,
            message=(
                f"Applied montage review for route '{route_spec['id']}' in copy mode"
            ),
            review=review,
            copied_files=copy_result.copied_files,
            skipped_files=copy_result.skipped_files,
            enqueued=enqueued,
            updated_queue_entries=updated_queue_entries,
            route_actions=route_actions,
            cloned_tasks=cloned_tasks,
            warnings=review.warnings,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Route montage review apply failed for '%s'", route_id)
        raise HTTPException(status_code=400, detail=str(exc))


# ── Helpers ──────────────────────────────────────────────────────────


def _scan_route_montage_review(
    route_id: str, body: RouteMontageReviewScanRequest
) -> RouteMontageReviewScanResponse:
    plan, route_spec, suggestions = _build_route_montage_plan(route_id, body)
    return _route_review_response_from_plan(plan, route_spec, suggestions)


def _build_route_montage_plan(
    route_id: str, body: RouteMontageReviewScanRequest
) -> tuple[MontageBatchPlan, dict[str, Any], dict[str, dict[str, Any]]]:
    from autoclean.utils.serve_routes import load_route_specs, normalize_route_id

    route_id = normalize_route_id(route_id)
    route_specs = load_route_specs(api_state.workspace_dir)
    route_spec = next(
        (spec for spec in route_specs if spec.get("id") == route_id),
        None,
    )
    if route_spec is None:
        raise HTTPException(status_code=404, detail=f"Route '{route_id}' not found")

    task_path = _resolve_route_task_path(route_spec)
    input_paths = (
        [Path(body.input_path)]
        if body.input_path
        else [Path(path) for path in route_spec.get("ingestion_folders", [])]
    )
    if not input_paths:
        raise ValueError("Route has no ingestion folders to scan")

    split_output_root = (
        Path(body.split_output_root)
        if body.split_output_root
        else api_state.workspace_dir
        / "montage-preflight"
        / api_state.mode
        / route_spec["id"]
    )

    all_files: list[MontagePreflightFileResult] = []
    expected_montage = None
    multiple_inputs = len(input_paths) > 1
    for index, input_path in enumerate(input_paths, start=1):
        plan = build_batch_plan(
            input_path=input_path,
            task_path=task_path,
            output_dir=split_output_root,
        )
        expected_montage = plan.expected_montage
        prefix = _input_copy_prefix(input_path, index) if multiple_inputs else ""
        for result in plan.files:
            if prefix:
                result = replace(
                    result,
                    relative_path=f"{prefix}/{result.relative_path}",
                )
            all_files.append(result)

    groups = _group_route_results(all_files)
    unknown_files = [
        result.path
        for result in all_files
        if result.status in {"unknown", "unsupported"} or not result.is_actionable
    ]
    actionable_files = [result.path for result in all_files if result.is_actionable]
    plan = MontageBatchPlan(
        input_path=", ".join(str(path) for path in input_paths),
        task_path=str(task_path),
        expected_montage=expected_montage,
        output_dir=str(split_output_root),
        groups=groups,
        files=all_files,
        unknown_files=unknown_files,
        actionable_files=actionable_files,
    )
    suggestions = _build_route_review_suggestions(
        route_spec=route_spec,
        plan=plan,
        split_output_root=split_output_root,
    )
    return plan, route_spec, suggestions


def _resolve_route_task_path(route_spec: dict[str, Any]) -> Path:
    from autoclean.utils.ingestion import resolve_taskfile_path
    from autoclean.utils.task_discovery import safe_discover_tasks

    taskfile = str(route_spec.get("taskfile", ""))
    resolved = resolve_taskfile_path(taskfile, api_state.workspace_dir, strict=False)
    if resolved is not None:
        return resolved

    candidate = api_state.workspace_dir / "tasks" / f"{Path(taskfile).stem}.py"
    if candidate.exists():
        return candidate

    with _serve_task_discovery_context(api_state.workspace_dir):
        tasks, _invalid, _skipped = safe_discover_tasks()
    for task in tasks:
        source = Path(str(task.source))
        if task.name == taskfile or source.stem == taskfile or source.name == taskfile:
            return source

    raise ValueError(
        "Route montage review requires a resolvable Python task file; "
        f"could not resolve {taskfile!r}"
    )


def _build_route_review_suggestions(
    *,
    route_spec: dict[str, Any],
    plan: MontageBatchPlan,
    split_output_root: Path,
) -> dict[str, dict[str, Any]]:
    suggestions: dict[str, dict[str, Any]] = {}
    for detected_montage in sorted(
        {
            result.detected_montage or "unknown"
            for result in plan.files
            if result.detected_montage or result.status == "unknown"
        }
    ):
        supported = detected_montage in SUPPORTED_HYDROCEL_MONTAGES
        clone_needed = supported and detected_montage != plan.expected_montage
        suggested_taskfile = str(route_spec.get("taskfile", ""))
        clone_class_name = None
        if clone_needed:
            clone_class_name = _clone_class_name_for_montage(
                Path(plan.task_path), detected_montage
            )
            suggested_taskfile = f"tasks/{clone_class_name}.py"

        suggested_route_id = (
            route_spec["id"]
            if supported and not clone_needed
            else (
                f"{route_spec['id']}-{_safe_route_suffix(detected_montage)}"
                if supported
                else None
            )
        )
        workspace_name = None
        if supported:
            from autoclean.utils.ingestion import build_workspace_name

            task_label = Path(suggested_taskfile).stem
            workspace_name = build_workspace_name(
                str(route_spec.get("workspace_name", "taskfile-montage-version")),
                taskfile=task_label,
                montage=detected_montage,
                version=route_spec.get("version"),
            )

        suggestions[detected_montage] = {
            "supported": supported,
            "clone_needed": clone_needed,
            "clone_class_name": clone_class_name,
            "suggested_route_id": suggested_route_id,
            "suggested_taskfile": suggested_taskfile if supported else None,
            "suggested_workspace_name": workspace_name,
            "suggested_ingestion_folder": (
                str(split_output_root / detected_montage) if supported else None
            ),
        }
    return suggestions


def _route_review_response_from_plan(
    plan: MontageBatchPlan,
    route_spec: dict[str, Any],
    suggestions: dict[str, dict[str, Any]],
) -> RouteMontageReviewScanResponse:
    split_output_root = Path(plan.output_dir)
    copy_estimate = estimate_copy_originals_for_plan(
        plan, split_output_root=split_output_root
    )
    group_models = []
    for group in plan.groups:
        suggestion = suggestions.get(group.detected_montage, {})
        group_models.append(
            RouteMontageReviewGroup(
                detected_montage=group.detected_montage,
                status=group.status,
                file_count=group.file_count,
                total_size_bytes=group.total_size_bytes,
                examples=group.examples,
                supported=bool(suggestion.get("supported", False)),
                suggested_route_id=suggestion.get("suggested_route_id"),
                suggested_taskfile=suggestion.get("suggested_taskfile"),
                suggested_workspace_name=suggestion.get("suggested_workspace_name"),
                suggested_ingestion_folder=suggestion.get("suggested_ingestion_folder"),
            )
        )

    file_models = []
    for result in plan.files:
        suggestion = suggestions.get(result.detected_montage or "unknown", {})
        copy_destination = (
            str(split_output_root / str(result.detected_montage) / result.relative_path)
            if result.is_actionable
            else None
        )
        file_models.append(
            RouteMontageReviewFile(
                path=result.path,
                relative_path=result.relative_path,
                format_id=result.format_id,
                expected_montage=result.expected_montage,
                detected_montage=result.detected_montage,
                status=result.status,
                eeg_channel_count=result.eeg_channel_count,
                e129_present=result.e129_present,
                reason=result.reason,
                size_bytes=result.size_bytes,
                suggested_route_id=suggestion.get("suggested_route_id"),
                copy_destination=copy_destination,
            )
        )

    warnings = []
    if plan.unknown_files:
        warnings.append(
            f"{len(plan.unknown_files)} file(s) are unknown or unsupported and will not be routed."
        )
    if any(
        group.detected_montage != plan.expected_montage and group.supported
        for group in group_models
    ):
        warnings.append(
            "Mismatched supported montage groups require generated task/route context before processing."
        )

    return RouteMontageReviewScanResponse(
        route_id=route_spec["id"],
        mode=api_state.mode,
        workspace_dir=str(api_state.workspace_dir),
        taskfile=str(route_spec.get("taskfile", "")),
        task_path=plan.task_path,
        configured_route_montage=str(route_spec.get("montage", "")),
        expected_task_montage=plan.expected_montage,
        input_paths=[path.strip() for path in plan.input_path.split(",") if path],
        split_output_root=str(split_output_root),
        groups=group_models,
        files=file_models,
        unknown_files=plan.unknown_files,
        copy_estimate=RouteMontageCopyEstimateResponse(**asdict(copy_estimate)),
        can_apply=bool(plan.actionable_files),
        warnings=warnings,
    )


def _group_route_results(
    results: list[MontagePreflightFileResult],
) -> list[MontagePreflightGroup]:
    grouped: dict[tuple[str, str], list[MontagePreflightFileResult]] = {}
    for result in results:
        detected = result.detected_montage or "unknown"
        grouped.setdefault((detected, result.status), []).append(result)

    return [
        MontagePreflightGroup(
            detected_montage=detected,
            status=status,
            file_count=len(items),
            total_size_bytes=sum(item.size_bytes for item in items),
            examples=[item.relative_path for item in items[:5]],
        )
        for (detected, status), items in sorted(grouped.items())
    ]


def _input_copy_prefix(input_path: Path, index: int) -> str:
    stem = input_path.stem if input_path.is_file() else input_path.name
    return _safe_route_suffix(stem or f"input-{index}")


def _safe_route_suffix(value: str) -> str:
    suffix = re.sub(r"[^0-9A-Za-z]+", "-", value).strip("-").lower()
    return suffix or "montage"


def _clone_class_name_for_montage(task_path: Path, montage: str) -> str:
    source = task_path.read_text(encoding="utf-8")
    match = re.search(
        r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        source,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(
            "Task clone validation failed: could not find a Python task class declaration"
        )
    base = match.group(1)
    suffix = re.sub(r"[^0-9A-Za-z]+", "_", montage).strip("_")
    if suffix and suffix[0].isdigit():
        suffix = f"Montage_{suffix}"
    return f"{base}_{suffix or 'Montage'}"


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
