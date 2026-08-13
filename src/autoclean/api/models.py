"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueueStatus(str, Enum):
    """Queue entry status.

    - pending: File discovered, waiting to be processed
    - processing: Currently being processed by a worker
    - processed: Successfully completed
    - failed: Processing failed (can be retried)
    """

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class QueueStats(BaseModel):
    """Queue statistics response."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "pending": 12,
                "processing": 1,
                "processed": 45,
                "failed": 2,
                "total": 60,
            }
        }
    )

    pending: int = Field(description="Number of pending entries")
    processing: int = Field(default=0, description="Number of currently processing")
    processed: int = Field(description="Number of processed entries")
    failed: int = Field(description="Number of failed entries")
    total: int = Field(description="Total entries in queue")


class QueueEntry(BaseModel):
    """A single queue entry representing an EEG file to be processed."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "path": "/data/input/sub-001/eeg/sub-001_task-rest_eeg.set",
                "status": "pending",
                "route_id": "resting-state",
                "ingestion_root": "/data/input",
                "added_at": "2024-01-15T10:30:00Z",
                "processed_at": None,
                "failed_at": None,
                "last_error": None,
            }
        }
    )

    path: str = Field(description="Full file path")
    status: QueueStatus = Field(description="Entry status")
    route_id: Optional[str] = Field(default=None, description="Assigned route ID")
    ingestion_root: Optional[str] = Field(
        default=None, description="Ingestion root path"
    )
    added_at: Optional[str] = Field(
        default=None, description="ISO timestamp when added"
    )
    processed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when processed"
    )
    failed_at: Optional[str] = Field(
        default=None, description="ISO timestamp when failed"
    )
    last_error: Optional[str] = Field(default=None, description="Last error message")
    expected_montage: Optional[str] = Field(
        default=None, description="Task montage expected during route review"
    )
    detected_montage: Optional[str] = Field(
        default=None, description="File montage detected during route review"
    )
    taskfile: Optional[str] = Field(
        default=None, description="Task file selected during route review"
    )
    route_review_source_path: Optional[str] = Field(
        default=None, description="Original source path copied by route review"
    )
    route_review_original_route_id: Optional[str] = Field(
        default=None, description="Route ID that initiated route review"
    )
    workspace_context: dict[str, Any] = Field(
        default_factory=dict, description="Suggested workspace context"
    )


class QueueEntriesResponse(BaseModel):
    """Response containing queue entries."""

    entries: list[QueueEntry] = Field(description="List of queue entries")
    total: int = Field(description="Total matching entries")
    filters: dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class EnqueueRequest(BaseModel):
    """Request to manually enqueue files for processing."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "paths": [
                    "/data/input/sub-001_task-rest_eeg.set",
                    "/data/input/sub-002_task-rest_eeg.set",
                ],
                "route_id": "resting-state",
            }
        }
    )

    paths: list[str] = Field(description="File paths to enqueue")
    route_id: Optional[str] = Field(default=None, description="Route ID to assign")


class EnqueueResponse(BaseModel):
    """Response from enqueue operation."""

    enqueued: int = Field(description="Number of files enqueued")
    skipped: int = Field(default=0, description="Number of duplicates skipped")


class RetryRequest(BaseModel):
    """Request to retry failed entries."""

    paths: Optional[list[str]] = Field(
        default=None, description="Specific paths to retry (None = all failed)"
    )


class RetryResponse(BaseModel):
    """Response from retry operation."""

    retried: int = Field(description="Number of entries retried")


class ClearResponse(BaseModel):
    """Response from clear operation."""

    cleared: int = Field(description="Number of entries cleared")


class WorkerStatus(str, Enum):
    """Worker status."""

    IDLE = "idle"
    BUSY = "busy"
    STOPPED = "stopped"
    STARTING = "starting"


class WorkerInfo(BaseModel):
    """Information about a single worker."""

    name: str = Field(description="Worker name")
    status: WorkerStatus = Field(description="Current status")
    current_job: Optional[str] = Field(default=None, description="Current job ID")
    queues: list[str] = Field(
        default_factory=list, description="Queues being processed"
    )
    pid: Optional[int] = Field(default=None, description="Process ID")


class WorkerStatusResponse(BaseModel):
    """Response with worker status."""

    workers: list[WorkerInfo] = Field(description="List of workers")
    total_workers: int = Field(description="Total worker count")
    active_jobs: int = Field(description="Currently processing jobs")
    queued_jobs: int = Field(description="Jobs waiting in queue")
    redis_connected: bool = Field(description="Redis connection status")


class WorkerStartRequest(BaseModel):
    """Request to start workers."""

    count: int = Field(default=1, ge=1, le=10, description="Number of workers to start")
    queues: list[str] = Field(
        default_factory=lambda: ["default"], description="Queues to process"
    )


class WorkerStartResponse(BaseModel):
    """Response from worker start."""

    started: int = Field(description="Number of workers started")
    pids: list[int] = Field(description="Process IDs of started workers")


class WorkerStopRequest(BaseModel):
    """Request to stop workers."""

    graceful: bool = Field(default=True, description="Wait for current jobs to finish")


class WorkerStopResponse(BaseModel):
    """Response from worker stop."""

    stopped: int = Field(description="Number of workers stopped")


class RouteInfo(BaseModel):
    """Information about an automation route."""

    id: str = Field(description="Route ID")
    enabled: bool = Field(description="Whether route is enabled")
    priority: int = Field(description="Route priority")
    taskfile: str = Field(description="Task file or name")
    montage: str = Field(description="Montage configuration")
    version: Optional[str] = Field(default=None, description="Version tag")
    ingestion_folders: list[str] = Field(description="Monitored folders")
    file_globs: list[str] = Field(description="File patterns")
    recursive: bool = Field(description="Recursive scanning")
    sentinel_ext: str = Field(description="Sentinel file extension")


class ConfigResponse(BaseModel):
    """Response with configuration."""

    mode: str = Field(description="Current mode (test/live)")
    workspace_dir: str = Field(description="Workspace directory path")
    runtime_path: str = Field(description="Runtime directory path")
    routes: list[RouteInfo] = Field(description="Configured routes")
    valid: bool = Field(description="Whether config is valid")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class ValidateResponse(BaseModel):
    """Response from config validation."""

    valid: bool = Field(description="Whether config is valid")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class DeployResponse(BaseModel):
    """Response from config deploy."""

    success: bool = Field(description="Whether deploy succeeded")
    source: str = Field(description="Source config path")
    target: str = Field(description="Target config path")
    message: str = Field(description="Status message")


class JobInfo(BaseModel):
    """Information about an RQ job."""

    id: str = Field(description="Job ID")
    status: str = Field(description="Job status")
    func_name: str = Field(description="Function name")
    args: list[Any] = Field(default_factory=list, description="Job arguments")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    started_at: Optional[str] = Field(default=None, description="Start timestamp")
    ended_at: Optional[str] = Field(default=None, description="End timestamp")
    result: Optional[Any] = Field(default=None, description="Job result")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class EventType(str, Enum):
    """WebSocket event types."""

    QUEUE_UPDATE = "queue_update"
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    WORKER_STARTED = "worker_started"
    WORKER_STOPPED = "worker_stopped"
    CONFIG_CHANGED = "config_changed"


class Event(BaseModel):
    """WebSocket event."""

    type: EventType = Field(description="Event type")
    timestamp: str = Field(description="ISO timestamp")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data")


class TaskOption(BaseModel):
    """Task option for route discovery UIs."""

    name: str = Field(description="Task name")
    source: str = Field(description="Task source path")
    description: str = Field(default="", description="Short task description")


class MontageOption(BaseModel):
    """Montage option for route discovery UIs."""

    name: str = Field(description="Montage identifier")
    description: str = Field(default="", description="Human-friendly montage label")


class RouteSpecResponse(BaseModel):
    """Operator-editable route spec payload."""

    id: str = Field(description="Route ID")
    modes: list[str] = Field(default_factory=list, description="Target modes")
    enabled: bool = Field(default=True, description="Whether the route is enabled")
    archived: bool = Field(default=False, description="Whether the route is archived")
    priority: int = Field(default=0, description="Route priority")
    taskfile: str = Field(description="Task file or task name")
    montage: str = Field(description="Montage identifier")
    version: Optional[str] = Field(default=None, description="Optional version tag")
    ingestion_folders: list[str] = Field(
        default_factory=list, description="Input roots"
    )
    ingestion_excludes: list[str] = Field(
        default_factory=list, description="Excluded subpaths"
    )
    file_globs: list[str] = Field(default_factory=list, description="File patterns")
    recursive: bool = Field(default=False, description="Whether scanning is recursive")
    sentinel_ext: Optional[str] = Field(default=None, description="Sentinel extension")
    automation_root: Optional[str] = Field(
        default=None, description="Automation output root"
    )
    workspace_name: Optional[str] = Field(
        default=None, description="Workspace naming template"
    )
    output_path: Optional[str] = Field(
        default=None, description="Resolved automation output path"
    )


class RouteUpsertRequest(BaseModel):
    """Create/update payload for one route spec."""

    id: str = Field(description="Route ID")
    modes: list[str] = Field(
        default_factory=lambda: ["test"], description="Target modes"
    )
    enabled: bool = Field(default=True, description="Whether the route is enabled")
    archived: bool = Field(default=False, description="Whether the route is archived")
    priority: int = Field(default=0, description="Route priority")
    taskfile: str = Field(description="Task file or task name")
    montage: str = Field(description="Montage identifier")
    version: Optional[str] = Field(default=None, description="Optional version tag")
    ingestion_folders: list[str] = Field(
        default_factory=list, description="Input roots"
    )
    ingestion_excludes: list[str] = Field(
        default_factory=list, description="Excluded subpaths"
    )
    file_globs: list[str] = Field(default_factory=list, description="File patterns")
    recursive: bool = Field(default=False, description="Whether scanning is recursive")
    sentinel_ext: Optional[str] = Field(default=None, description="Sentinel extension")
    automation_root: Optional[str] = Field(
        default=None, description="Automation output root"
    )
    workspace_name: Optional[str] = Field(
        default=None, description="Workspace naming template"
    )


class RouteActionResponse(BaseModel):
    """Status response for route mutations."""

    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Status message")
    route_id: Optional[str] = Field(default=None, description="Affected route ID")


class SyncResponse(BaseModel):
    """Response from route registry compilation."""

    success: bool = Field(description="Whether sync succeeded")
    message: str = Field(description="Status message")
    test_path: Optional[str] = Field(
        default=None, description="Compiled test config path"
    )
    live_path: Optional[str] = Field(
        default=None, description="Compiled live config path"
    )


class RouteMontageReviewScanRequest(BaseModel):
    """Request to scan one route's inputs for montage preflight review."""

    input_path: Optional[str] = Field(
        default=None,
        description="Optional single input path to scan instead of the route folders",
    )
    split_output_root: Optional[str] = Field(
        default=None,
        description="Optional copy destination root for the review apply step",
    )


class RouteMontageReviewApplyRequest(RouteMontageReviewScanRequest):
    """Request to apply a confirmed montage preflight review."""

    confirm: bool = Field(
        default=False,
        description="Must be true before filesystem or queue changes are made",
    )
    mode: str = Field(
        default="copy",
        description="Apply mode. Only copy is supported by issue #277.",
    )
    overwrite_existing: bool = Field(
        default=True,
        description="Refresh existing copied files in the review split folder",
    )


class RouteMontageReviewGroup(BaseModel):
    """Grouped route montage preflight result."""

    detected_montage: str = Field(description="Detected montage or unknown")
    status: str = Field(description="Grouped status")
    file_count: int = Field(description="Number of files in the group")
    total_size_bytes: int = Field(description="Total source bytes in the group")
    examples: list[str] = Field(default_factory=list, description="Example paths")
    supported: bool = Field(description="Whether the group can be routed")
    suggested_route_id: Optional[str] = Field(
        default=None, description="Route ID suggested for this group"
    )
    suggested_taskfile: Optional[str] = Field(
        default=None, description="Task file or task name suggested for this group"
    )
    suggested_workspace_name: Optional[str] = Field(
        default=None, description="Serve workspace name for this group"
    )
    suggested_ingestion_folder: Optional[str] = Field(
        default=None, description="Copy-mode ingestion folder for this group"
    )


class RouteMontageReviewFile(BaseModel):
    """Per-file route montage preflight result."""

    path: str
    relative_path: str
    format_id: Optional[str] = None
    expected_montage: Optional[str] = None
    detected_montage: Optional[str] = None
    status: str
    eeg_channel_count: Optional[int] = None
    e129_present: bool = False
    reason: str = ""
    size_bytes: int = 0
    suggested_route_id: Optional[str] = None
    copy_destination: Optional[str] = None


class RouteMontageCopyEstimateResponse(BaseModel):
    """Copy-mode estimate for a route montage review."""

    split_output_root: str
    actionable_file_count: int
    skipped_file_count: int
    required_bytes: int
    free_bytes_before: int
    free_bytes_after_estimate: int


class RouteMontageReviewScanResponse(BaseModel):
    """Structured route montage review scan response."""

    route_id: str
    mode: str
    workspace_dir: str
    taskfile: str
    task_path: Optional[str] = None
    configured_route_montage: str
    expected_task_montage: Optional[str] = None
    input_paths: list[str]
    split_output_root: str
    groups: list[RouteMontageReviewGroup]
    files: list[RouteMontageReviewFile]
    unknown_files: list[str]
    copy_estimate: RouteMontageCopyEstimateResponse
    can_apply: bool
    warnings: list[str] = Field(default_factory=list)


class RouteMontageReviewApplyResponse(BaseModel):
    """Result from applying a route montage review."""

    success: bool
    message: str
    review: RouteMontageReviewScanResponse
    copied_files: list[dict[str, Any]] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    enqueued: int = 0
    updated_queue_entries: int = 0
    route_actions: list[dict[str, Any]] = Field(default_factory=list)
    cloned_tasks: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ServiceStatusResponse(BaseModel):
    """Serve dispatcher status."""

    running: bool = Field(description="Whether the dispatcher is running")
    pid: Optional[int] = Field(default=None, description="Dispatcher PID")
    mode: str = Field(description="Current serve mode")
    uptime_seconds: Optional[float] = Field(
        default=None, description="Dispatcher uptime in seconds"
    )
    can_start: bool = Field(
        default=True,
        description="Whether the dispatcher can be started with the current workspace state",
    )
    blocked_reason: Optional[str] = Field(
        default=None,
        description="Reason the dispatcher cannot be started",
    )


class ServiceActionResponse(BaseModel):
    """Status response for serve dispatcher actions."""

    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Status message")
