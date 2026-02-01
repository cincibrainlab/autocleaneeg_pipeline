"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class QueueStatus(str, Enum):
    """Queue entry status."""

    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class QueueStats(BaseModel):
    """Queue statistics response."""

    pending: int = Field(description="Number of pending entries")
    processing: int = Field(default=0, description="Number of currently processing")
    processed: int = Field(description="Number of processed entries")
    failed: int = Field(description="Number of failed entries")
    total: int = Field(description="Total entries in queue")


class QueueEntry(BaseModel):
    """A single queue entry."""

    path: str = Field(description="File path")
    status: QueueStatus = Field(description="Entry status")
    route_id: Optional[str] = Field(default=None, description="Assigned route ID")
    ingestion_root: Optional[str] = Field(default=None, description="Ingestion root path")
    added_at: Optional[str] = Field(default=None, description="ISO timestamp when added")
    processed_at: Optional[str] = Field(default=None, description="ISO timestamp when processed")
    failed_at: Optional[str] = Field(default=None, description="ISO timestamp when failed")
    last_error: Optional[str] = Field(default=None, description="Last error message")


class QueueEntriesResponse(BaseModel):
    """Response containing queue entries."""

    entries: list[QueueEntry] = Field(description="List of queue entries")
    total: int = Field(description="Total matching entries")
    filters: dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class EnqueueRequest(BaseModel):
    """Request to enqueue files."""

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
    queues: list[str] = Field(default_factory=list, description="Queues being processed")
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
