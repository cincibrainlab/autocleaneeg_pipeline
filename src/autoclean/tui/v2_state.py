"""Derived operator snapshot models for the AutoClean Serve v2 TUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

STATUS_ORDER = {"failed": 0, "processing": 1, "pending": 2, "processed": 3}


@dataclass(slots=True)
class RecommendedAction:
    key: str
    title: str
    description: str
    target_tab: str
    button_label: str
    direct_action: Optional[str] = None


@dataclass(slots=True)
class RouteHealth:
    route_id: str
    label: str
    state: str
    summary: str
    taskfile: str
    montage: str
    modes: list[str] = field(default_factory=list)
    enabled: bool = True
    archived: bool = False
    priority: int = 0
    file_globs: list[str] = field(default_factory=list)
    ingestion_folders: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueueItem:
    path: str
    file_name: str
    status: str
    route_id: str
    added_at: str
    updated_at: str
    last_error: str


@dataclass(slots=True)
class QueueHealth:
    pending: int = 0
    processing: int = 0
    processed: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.processing + self.processed + self.failed


@dataclass(slots=True)
class PublishHealth:
    config_valid: bool
    config_errors: list[str]
    config_warnings: list[str]
    operator_config_path: str
    deployed_config_path: str
    config_source: str
    needs_deploy: bool


@dataclass(slots=True)
class ServiceHealth:
    lane: str
    running: bool
    workspace: str
    queue_path: str
    config_source: str
    config_path: str
    log_path: str
    pid: Optional[int]
    uptime: Optional[str]
    command: str
    completed: Optional[str]
    failed: Optional[str]
    failed_error: Optional[str]


@dataclass(slots=True)
class EventSummary:
    timestamp: str
    event_type: str
    message: str


@dataclass(slots=True)
class ServeWorkspaceSnapshot:
    workspace_dir: Optional[str]
    lane: str
    lane_label: str
    last_refresh: datetime
    routes: list[RouteHealth]
    queue_items: list[QueueItem]
    queue: QueueHealth
    publish: PublishHealth
    service: ServiceHealth
    recommended_action: RecommendedAction
    recent_events: list[EventSummary]


def _short_time(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return "-"
    if "T" in raw:
        raw = raw.split("T", 1)[1]
    if "+" in raw:
        raw = raw.split("+", 1)[0]
    return raw[:8] if len(raw) >= 8 else raw


def _recent_timestamp(entry: dict[str, Any]) -> str:
    for key in ("failed_at", "processed_at", "started_at", "updated_at", "added_at"):
        value = str(entry.get(key) or "")
        if value:
            return value
    return ""


def _route_label(route: dict[str, Any]) -> str:
    route_id = str(route.get("id") or "")
    workspace_name = str(route.get("workspace_name") or "")
    return workspace_name or route_id


def _build_route_health(route: dict[str, Any], lane: str) -> RouteHealth:
    route_id = str(route.get("id") or "")
    taskfile = str(route.get("taskfile") or "")
    montage = str(route.get("montage") or "")
    modes = [str(item) for item in route.get("modes", ["test"]) if str(item)]
    enabled = bool(route.get("enabled", True))
    archived = bool(route.get("archived", False))
    priority = int(route.get("priority", 0) or 0)
    file_globs = [str(item) for item in route.get("file_globs", []) if str(item)]
    folders = [str(item) for item in route.get("ingestion_folders", []) if str(item)]
    issues: list[str] = []

    if archived:
        state = "Archived"
        summary = "Route is archived and excluded from compiled lane configs."
    elif not enabled:
        state = "Disabled"
        summary = "Route exists but is disabled."
    else:
        missing_paths = [path for path in folders if not Path(path).exists()]
        if taskfile and not Path(taskfile).exists():
            issues.append("Task file is missing.")
        if missing_paths:
            issues.append(f"{len(missing_paths)} ingestion folder(s) are missing.")
        if not folders:
            issues.append("No ingestion folders configured.")
        if not file_globs:
            issues.append("No file globs configured.")
        if lane == "live" and "live" not in modes:
            state = "Draft only"
            summary = (
                "Route is available in Draft but has not been promoted to Production."
            )
        elif issues:
            state = "Attention"
            summary = issues[0]
        else:
            state = "Ready"
            summary = "Route is available in the active lane and looks operational."

    return RouteHealth(
        route_id=route_id,
        label=_route_label(route),
        state=state,
        summary=summary,
        taskfile=taskfile,
        montage=montage,
        modes=modes,
        enabled=enabled,
        archived=archived,
        priority=priority,
        file_globs=file_globs,
        ingestion_folders=folders,
        issues=issues,
    )


def _build_queue_items(entries: dict[str, Any]) -> tuple[QueueHealth, list[QueueItem]]:
    health = QueueHealth()
    items: list[QueueItem] = []

    for path_str, payload in entries.items():
        status = str(payload.get("status", "pending"))
        if status == "failed":
            health.failed += 1
        elif status == "processing":
            health.processing += 1
        elif status == "processed":
            health.processed += 1
        else:
            health.pending += 1

        items.append(
            QueueItem(
                path=path_str,
                file_name=Path(path_str).name,
                status=status,
                route_id=str(payload.get("route_id") or "-"),
                added_at=str(payload.get("added_at") or ""),
                updated_at=_recent_timestamp(payload),
                last_error=str(payload.get("last_error") or ""),
            )
        )

    items.sort(
        key=lambda item: (
            STATUS_ORDER.get(item.status, 99),
            item.updated_at or item.added_at,
            item.file_name,
        ),
        reverse=False,
    )
    return health, items


def _config_needs_deploy(operator_path: Path, deployed_path: Path) -> bool:
    if not operator_path.exists():
        return False
    if not deployed_path.exists():
        return True
    try:
        return operator_path.read_text(encoding="utf-8") != deployed_path.read_text(
            encoding="utf-8"
        )
    except Exception:
        return True


def _build_recent_events(activity_log: Sequence[Any]) -> list[EventSummary]:
    events: list[EventSummary] = []
    for entry in list(activity_log)[:8]:
        timestamp = getattr(entry, "timestamp", None)
        events.append(
            EventSummary(
                timestamp=_short_time(timestamp.isoformat() if timestamp else ""),
                event_type=str(getattr(entry, "event_type", "info")),
                message=str(getattr(entry, "message", "")),
            )
        )
    return events


def _build_recommended_action(
    *,
    routes: Sequence[RouteHealth],
    queue: QueueHealth,
    publish: PublishHealth,
    service: ServiceHealth,
) -> RecommendedAction:
    active_routes = [route for route in routes if not route.archived]
    if not active_routes:
        return RecommendedAction(
            key="create_route",
            title="Create the first route",
            description="No active routes exist yet. Start by mapping one input folder to one task and montage.",
            target_tab="tab-routes",
            button_label="Open Routes",
        )
    if not publish.config_valid:
        return RecommendedAction(
            key="fix_config",
            title="Fix the current lane configuration",
            description="The active lane config is invalid. Resolve validation errors before publishing or starting the service.",
            target_tab="tab-publish",
            button_label="Open Publish",
        )
    if queue.failed:
        return RecommendedAction(
            key="review_failed",
            title="Review failed jobs",
            description="There are files that need attention. Inspect the failure details before retrying or clearing anything.",
            target_tab="tab-queue",
            button_label="Open Queue",
        )
    if publish.needs_deploy:
        return RecommendedAction(
            key="deploy_lane",
            title="Publish the current lane config",
            description="The operator config differs from the deployed config. Publish the lane before relying on the service state.",
            target_tab="tab-publish",
            button_label="Open Publish",
        )
    if not service.running:
        return RecommendedAction(
            key="start_service",
            title="Start the lane service",
            description="Routes and config look ready, but the background service is not running.",
            target_tab="tab-service",
            button_label="Start Service",
            direct_action="start_service",
        )
    if queue.pending or queue.processing:
        return RecommendedAction(
            key="monitor_queue",
            title="Monitor queue progress",
            description="The lane is active and there is work in flight. Watch the queue for stuck or failing jobs.",
            target_tab="tab-queue",
            button_label="Open Queue",
        )
    return RecommendedAction(
        key="healthy",
        title="System looks healthy",
        description="The active lane is deployed, the service is running, and there are no outstanding failures.",
        target_tab="tab-home",
        button_label="Refresh",
        direct_action="refresh",
    )


def build_workspace_snapshot(
    *,
    workspace_dir: Optional[Path],
    mode: str,
    config_valid: bool,
    config_errors: Iterable[str],
    config_warnings: Iterable[str],
    route_specs: Sequence[dict[str, Any]],
    queue_entries: dict[str, Any],
    service_snapshot: dict[str, Any],
    activity_log: Sequence[Any],
    operator_config_path: Path,
    deployed_config_path: Path,
    config_source: str,
) -> ServeWorkspaceSnapshot:
    routes = [_build_route_health(route, mode) for route in route_specs]
    queue_health, queue_items = _build_queue_items(queue_entries)
    publish = PublishHealth(
        config_valid=config_valid,
        config_errors=[str(item) for item in config_errors],
        config_warnings=[str(item) for item in config_warnings],
        operator_config_path=str(operator_config_path),
        deployed_config_path=str(deployed_config_path),
        config_source=config_source,
        needs_deploy=_config_needs_deploy(operator_config_path, deployed_config_path),
    )
    service = ServiceHealth(
        lane=str(
            service_snapshot.get("lane")
            or ("Draft" if mode == "test" else "Production")
        ),
        running=bool(service_snapshot.get("pid") or service_snapshot.get("uptime")),
        workspace=str(
            service_snapshot.get("workspace")
            or (str(workspace_dir) if workspace_dir else "Not configured")
        ),
        queue_path=str(service_snapshot.get("queue_path") or "Unavailable"),
        config_source=str(service_snapshot.get("config_source") or config_source),
        config_path=str(service_snapshot.get("config_path") or "Unavailable"),
        log_path=str(service_snapshot.get("log_path") or "Unavailable"),
        pid=service_snapshot.get("pid"),
        uptime=service_snapshot.get("uptime"),
        command=str(service_snapshot.get("command") or "Not started yet"),
        completed=service_snapshot.get("completed"),
        failed=service_snapshot.get("failed"),
        failed_error=service_snapshot.get("failed_error"),
    )
    recommendation = _build_recommended_action(
        routes=routes,
        queue=queue_health,
        publish=publish,
        service=service,
    )
    return ServeWorkspaceSnapshot(
        workspace_dir=str(workspace_dir) if workspace_dir else None,
        lane=mode,
        lane_label="Draft" if mode == "test" else "Production",
        last_refresh=datetime.now(),
        routes=routes,
        queue_items=queue_items,
        queue=queue_health,
        publish=publish,
        service=service,
        recommended_action=recommendation,
        recent_events=_build_recent_events(activity_log),
    )
