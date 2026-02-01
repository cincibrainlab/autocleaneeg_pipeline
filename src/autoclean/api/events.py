"""WebSocket event broadcasting for live updates."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from autoclean.api.models import Event, EventType

router = APIRouter()


class EventBroadcaster:
    """Manages WebSocket connections and broadcasts events."""

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, event: Event) -> None:
        """Broadcast an event to all connected clients."""
        if not self._connections:
            return

        message = event.model_dump_json()

        async with self._lock:
            dead_connections = set()

            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead_connections.add(ws)

            # Clean up dead connections
            self._connections -= dead_connections

    async def broadcast_dict(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Broadcast an event from type and data dict."""
        event = Event(
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        await self.broadcast(event)

    @property
    def connection_count(self) -> int:
        """Return number of active connections."""
        return len(self._connections)


# Global broadcaster instance
broadcaster = EventBroadcaster()


@router.websocket("/events")
async def websocket_events(websocket: WebSocket) -> None:
    """WebSocket endpoint for event streaming.

    Clients connect here to receive live updates about:
    - Queue changes (files added, processed, failed)
    - Job status (started, completed, failed)
    - Worker status (started, stopped)
    - Config changes (deployed)
    """
    await broadcaster.connect(websocket)

    try:
        # Send initial connection event
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"message": "Connected to event stream"},
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for ping/pong or client messages
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )

                # Handle ping
                if data == "ping":
                    await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(websocket)


# Convenience functions for broadcasting events

async def emit_queue_update(
    action: str,
    path: str,
    status: Optional[str] = None,
    route_id: Optional[str] = None,
) -> None:
    """Emit a queue update event."""
    await broadcaster.broadcast_dict(
        EventType.QUEUE_UPDATE,
        {
            "action": action,
            "path": path,
            "status": status,
            "route_id": route_id,
        },
    )


async def emit_job_started(job_id: str, task_name: str, args: dict[str, Any]) -> None:
    """Emit a job started event."""
    await broadcaster.broadcast_dict(
        EventType.JOB_STARTED,
        {
            "job_id": job_id,
            "task": task_name,
            "args": args,
        },
    )


async def emit_job_completed(job_id: str, result: Any) -> None:
    """Emit a job completed event."""
    await broadcaster.broadcast_dict(
        EventType.JOB_COMPLETED,
        {
            "job_id": job_id,
            "result": result,
        },
    )


async def emit_job_failed(job_id: str, error: str) -> None:
    """Emit a job failed event."""
    await broadcaster.broadcast_dict(
        EventType.JOB_FAILED,
        {
            "job_id": job_id,
            "error": error,
        },
    )


async def emit_worker_started(worker_name: str, pid: int, queues: list[str]) -> None:
    """Emit a worker started event."""
    await broadcaster.broadcast_dict(
        EventType.WORKER_STARTED,
        {
            "worker": worker_name,
            "pid": pid,
            "queues": queues,
        },
    )


async def emit_worker_stopped(worker_name: str, pid: int) -> None:
    """Emit a worker stopped event."""
    await broadcaster.broadcast_dict(
        EventType.WORKER_STOPPED,
        {
            "worker": worker_name,
            "pid": pid,
        },
    )


async def emit_config_changed(mode: str, action: str) -> None:
    """Emit a config changed event."""
    await broadcaster.broadcast_dict(
        EventType.CONFIG_CHANGED,
        {
            "mode": mode,
            "action": action,
        },
    )
