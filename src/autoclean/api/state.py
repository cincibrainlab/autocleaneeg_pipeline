"""API state management - separate module to avoid circular imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException


class APIState:
    """Global API state container."""

    def __init__(self) -> None:
        self.workspace_dir: Optional[Path] = None
        self.mode: str = "test"
        self.redis_url: str = "redis://localhost:6379"
        self._redis_connection: Optional[Any] = None
        self._rq_queue: Optional[Any] = None

    def configure(
        self,
        workspace_dir: Path,
        mode: str = "test",
        redis_url: str = "redis://localhost:6379",
    ) -> None:
        """Configure the API state."""
        self.workspace_dir = workspace_dir
        self.mode = mode
        self.redis_url = redis_url

    @property
    def redis(self) -> Any:
        """Get Redis connection (lazy initialization)."""
        if self._redis_connection is None:
            try:
                from redis import Redis

                self._redis_connection = Redis.from_url(self.redis_url)
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"Redis connection failed: {exc}"
                )
        return self._redis_connection

    @property
    def rq_queue(self) -> Any:
        """Get RQ queue (lazy initialization)."""
        if self._rq_queue is None:
            try:
                from rq import Queue

                self._rq_queue = Queue(connection=self.redis)
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail=f"RQ queue initialization failed: {exc}"
                )
        return self._rq_queue

    def get_queue_path(self) -> Path:
        """Get the queue.json path for current mode."""
        if not self.workspace_dir:
            raise HTTPException(status_code=500, detail="Workspace not configured")
        return self.workspace_dir / f"queue-{self.mode}.json"

    def get_config_path(self, deployed: bool = False) -> Path:
        """Get the config path for current mode."""
        if not self.workspace_dir:
            raise HTTPException(status_code=500, detail="Workspace not configured")
        if deployed:
            return self.workspace_dir / "deploy" / f"serve-{self.mode}.yaml"
        return self.workspace_dir / f"serve-{self.mode}.yaml"

    def check_redis(self) -> bool:
        """Check if Redis is connected."""
        try:
            self.redis.ping()
            return True
        except Exception:
            return False


# Global state instance
api_state = APIState()
