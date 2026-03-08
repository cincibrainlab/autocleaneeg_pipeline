"""FastAPI server for AutoClean API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

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

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure state if workspace provided
    if workspace_dir:
        api_state.configure(workspace_dir, mode, redis_url)

    # Import routes here to avoid circular imports
    from autoclean.api import events
    from autoclean.api.routes import config, queue, worker

    # Include routers
    app.include_router(queue.router, prefix="/api/queue", tags=["Queue"])
    app.include_router(worker.router, prefix="/api/worker", tags=["Worker"])
    app.include_router(config.router, prefix="/api/config", tags=["Config"])
    app.include_router(events.router, prefix="/ws", tags=["WebSocket"])

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint."""
        redis_ok = api_state.check_redis() if api_state.workspace_dir else False
        return {
            "status": "healthy",
            "workspace_configured": api_state.workspace_dir is not None,
            "mode": api_state.mode,
            "redis_connected": redis_ok,
        }

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint with API info."""
        return {
            "name": "AutoClean Automation API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


def run_server(
    workspace_dir: Path,
    mode: str = "test",
    host: str = "127.0.0.1",
    port: int = 8000,
    redis_url: str = "redis://localhost:6379",
    reload: bool = False,
) -> None:
    """Run the API server.

    Args:
        workspace_dir: Path to serve workspace directory.
        mode: Configuration mode ("test" or "live").
        host: Host to bind to.
        port: Port to listen on.
        redis_url: Redis connection URL.
        reload: Enable auto-reload for development.
    """
    import uvicorn

    # Configure global state before starting
    api_state.configure(workspace_dir, mode, redis_url)

    uvicorn.run(
        "autoclean.api.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )
