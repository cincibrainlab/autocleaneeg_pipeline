"""Queue management API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from autoclean.api.models import (
    ClearResponse,
    EnqueueRequest,
    EnqueueResponse,
    QueueEntriesResponse,
    QueueEntry,
    QueueStats,
    QueueStatus,
    RetryRequest,
    RetryResponse,
)
from autoclean.api.state import api_state

router = APIRouter()


def _load_queue():
    """Load the ingestion queue."""
    from autoclean.utils.ingestion import IngestionQueue

    queue_path = api_state.get_queue_path()
    return IngestionQueue(queue_path)


@router.get("/stats", response_model=QueueStats)
async def get_queue_stats() -> QueueStats:
    """Get queue statistics."""
    queue = _load_queue()
    entries = queue.entries()

    stats = {"pending": 0, "processing": 0, "processed": 0, "failed": 0}

    for entry_data in entries.values():
        status = entry_data.get("status", "pending")
        if status in stats:
            stats[status] += 1

    return QueueStats(
        pending=stats["pending"],
        processing=stats["processing"],
        processed=stats["processed"],
        failed=stats["failed"],
        total=len(entries),
    )


@router.get("/entries", response_model=QueueEntriesResponse)
async def get_queue_entries(
    status: Optional[QueueStatus] = Query(default=None, description="Filter by status"),
    route_id: Optional[str] = Query(default=None, description="Filter by route ID"),
    limit: int = Query(default=100, ge=1, le=1000, description="Max entries to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> QueueEntriesResponse:
    """Get queue entries with optional filtering."""
    queue = _load_queue()
    entries = queue.entries()

    result = []
    for path_str, data in entries.items():
        entry_status = data.get("status", "pending")
        entry_route = data.get("route_id")

        # Apply filters
        if status and entry_status != status.value:
            continue
        if route_id and entry_route != route_id:
            continue

        result.append(
            QueueEntry(
                path=path_str,
                status=QueueStatus(entry_status),
                route_id=entry_route,
                ingestion_root=data.get("ingestion_root"),
                added_at=data.get("added_at"),
                processed_at=data.get("processed_at"),
                failed_at=data.get("failed_at"),
                last_error=data.get("last_error"),
            )
        )

    # Sort by added_at (newest first)
    result.sort(key=lambda x: x.added_at or "", reverse=True)

    total = len(result)
    result = result[offset : offset + limit]

    filters = {}
    if status:
        filters["status"] = status.value
    if route_id:
        filters["route_id"] = route_id

    return QueueEntriesResponse(entries=result, total=total, filters=filters)


@router.post("/enqueue", response_model=EnqueueResponse)
async def enqueue_files(request: EnqueueRequest) -> EnqueueResponse:
    """Add files to the queue."""
    from pathlib import Path

    queue = _load_queue()
    existing = set(queue.entries().keys())

    enqueued = 0
    skipped = 0

    for path_str in request.paths:
        if path_str in existing:
            skipped += 1
            continue

        queue.enqueue(
            [Path(path_str)],
            route_id=request.route_id,
        )
        enqueued += 1

    return EnqueueResponse(enqueued=enqueued, skipped=skipped)


@router.post("/retry", response_model=RetryResponse)
async def retry_failed(request: RetryRequest) -> RetryResponse:
    """Retry failed queue entries."""
    queue = _load_queue()
    entries = queue.entries()

    retried = 0

    for path_str, data in entries.items():
        if data.get("status") != "failed":
            continue

        # If specific paths provided, check if this one is included
        if request.paths and path_str not in request.paths:
            continue

        data["status"] = "pending"
        data.pop("last_error", None)
        data.pop("failed_at", None)
        retried += 1

    if retried > 0:
        queue.save()

    return RetryResponse(retried=retried)


@router.delete("/entry/{path:path}", response_model=ClearResponse)
async def remove_entry(path: str) -> ClearResponse:
    """Remove a specific entry from the queue."""
    queue = _load_queue()
    entries = queue.entries()

    # URL decode the path
    from urllib.parse import unquote

    decoded_path = unquote(path)

    if decoded_path not in entries:
        raise HTTPException(status_code=404, detail=f"Entry not found: {decoded_path}")

    del entries[decoded_path]
    queue.save()

    return ClearResponse(cleared=1)


@router.delete("/processed", response_model=ClearResponse)
async def clear_processed() -> ClearResponse:
    """Clear all processed entries from the queue."""
    queue = _load_queue()
    entries = queue.entries()

    to_remove = [
        path for path, data in entries.items() if data.get("status") == "processed"
    ]

    for path in to_remove:
        del entries[path]

    if to_remove:
        queue.save()

    return ClearResponse(cleared=len(to_remove))


@router.delete("/failed", response_model=ClearResponse)
async def clear_failed() -> ClearResponse:
    """Clear all failed entries from the queue."""
    queue = _load_queue()
    entries = queue.entries()

    to_remove = [
        path for path, data in entries.items() if data.get("status") == "failed"
    ]

    for path in to_remove:
        del entries[path]

    if to_remove:
        queue.save()

    return ClearResponse(cleared=len(to_remove))


@router.delete("/all", response_model=ClearResponse)
async def clear_all() -> ClearResponse:
    """Clear all entries from the queue."""
    queue = _load_queue()
    entries = queue.entries()
    count = len(entries)

    entries.clear()
    queue.save()

    return ClearResponse(cleared=count)
