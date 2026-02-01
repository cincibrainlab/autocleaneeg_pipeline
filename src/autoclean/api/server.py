"""FastAPI server for AutoClean API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from autoclean.api.state import APIState, api_state


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
        description="REST API for managing EEG processing automation",
        version="1.0.0",
        lifespan=lifespan,
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
