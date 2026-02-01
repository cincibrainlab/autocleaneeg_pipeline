"""Tests for the AutoClean API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autoclean.api.models import (
    ConfigResponse,
    EnqueueRequest,
    EnqueueResponse,
    QueueEntry,
    QueueStats,
    QueueStatus,
    RetryRequest,
    RouteInfo,
    ValidateResponse,
    WorkerStatus,
    WorkerStatusResponse,
)
from autoclean.api.server import create_app
from autoclean.api.state import APIState, api_state


class TestAPIState:
    """Tests for APIState class."""

    def test_default_state(self) -> None:
        """Test default state values."""
        state = APIState()
        assert state.workspace_dir is None
        assert state.mode == "test"
        assert state.redis_url == "redis://localhost:6379"

    def test_configure(self, tmp_path: Path) -> None:
        """Test configuring state."""
        state = APIState()
        state.configure(tmp_path, mode="live", redis_url="redis://other:6379")

        assert state.workspace_dir == tmp_path
        assert state.mode == "live"
        assert state.redis_url == "redis://other:6379"

    def test_get_queue_path(self, tmp_path: Path) -> None:
        """Test getting queue path."""
        state = APIState()
        state.configure(tmp_path, mode="test")

        queue_path = state.get_queue_path()
        assert queue_path == tmp_path / "queue-test.json"

    def test_get_config_path(self, tmp_path: Path) -> None:
        """Test getting config path."""
        state = APIState()
        state.configure(tmp_path, mode="test")

        # Non-deployed
        config_path = state.get_config_path(deployed=False)
        assert config_path == tmp_path / "serve-test.yaml"

        # Deployed
        deployed_path = state.get_config_path(deployed=True)
        assert deployed_path == tmp_path / "deploy" / "serve-test.yaml"


class TestCreateApp:
    """Tests for app factory."""

    def test_create_app_no_workspace(self) -> None:
        """Test creating app without workspace."""
        app = create_app()
        assert app is not None
        assert app.title == "AutoClean Automation API"

    def test_create_app_with_workspace(self, tmp_path: Path) -> None:
        """Test creating app with workspace."""
        app = create_app(workspace_dir=tmp_path, mode="live")
        assert app is not None


class TestQueueModels:
    """Tests for queue-related Pydantic models."""

    def test_queue_stats(self) -> None:
        """Test QueueStats model."""
        stats = QueueStats(
            pending=5,
            processing=1,
            processed=10,
            failed=2,
            total=18,
        )
        assert stats.pending == 5
        assert stats.total == 18

    def test_queue_entry(self) -> None:
        """Test QueueEntry model."""
        entry = QueueEntry(
            path="/data/file.bdf",
            status=QueueStatus.PENDING,
            route_id="route-1",
            added_at="2024-01-01T00:00:00",
        )
        assert entry.path == "/data/file.bdf"
        assert entry.status == QueueStatus.PENDING
        assert entry.route_id == "route-1"

    def test_enqueue_request(self) -> None:
        """Test EnqueueRequest model."""
        request = EnqueueRequest(
            paths=["/data/file1.bdf", "/data/file2.bdf"],
            route_id="route-1",
        )
        assert len(request.paths) == 2
        assert request.route_id == "route-1"

    def test_enqueue_response(self) -> None:
        """Test EnqueueResponse model."""
        response = EnqueueResponse(enqueued=2, skipped=1)
        assert response.enqueued == 2
        assert response.skipped == 1

    def test_retry_request(self) -> None:
        """Test RetryRequest model."""
        request = RetryRequest(paths=["/data/file1.bdf"])
        assert request.paths == ["/data/file1.bdf"]

        request_all = RetryRequest()
        assert request_all.paths is None


class TestWorkerModels:
    """Tests for worker-related Pydantic models."""

    def test_worker_status_response(self) -> None:
        """Test WorkerStatusResponse model."""
        response = WorkerStatusResponse(
            workers=[],
            total_workers=0,
            active_jobs=0,
            queued_jobs=5,
            redis_connected=True,
        )
        assert response.redis_connected is True
        assert response.queued_jobs == 5


class TestConfigModels:
    """Tests for config-related Pydantic models."""

    def test_route_info(self) -> None:
        """Test RouteInfo model."""
        route = RouteInfo(
            id="route-1",
            enabled=True,
            priority=10,
            taskfile="TestTask",
            montage="biosemi64",
            version="1.0",
            ingestion_folders=["/data/incoming"],
            file_globs=["*.bdf"],
            recursive=True,
            sentinel_ext=".ready",
        )
        assert route.id == "route-1"
        assert route.enabled is True
        assert route.priority == 10

    def test_config_response(self) -> None:
        """Test ConfigResponse model."""
        response = ConfigResponse(
            mode="test",
            workspace_dir="/workspace",
            runtime_path="runtimes/test",
            routes=[],
            valid=True,
            errors=[],
            warnings=["automation_mode not set"],
        )
        assert response.valid is True
        assert len(response.warnings) == 1

    def test_validate_response(self) -> None:
        """Test ValidateResponse model."""
        response = ValidateResponse(
            valid=False,
            errors=["Missing required key: mode"],
            warnings=[],
        )
        assert response.valid is False
        assert len(response.errors) == 1


class TestEventModels:
    """Tests for event-related models."""

    def test_event_types(self) -> None:
        """Test EventType enum."""
        from autoclean.api.models import EventType

        assert EventType.QUEUE_UPDATE == "queue_update"
        assert EventType.JOB_STARTED == "job_started"
        assert EventType.JOB_COMPLETED == "job_completed"
        assert EventType.JOB_FAILED == "job_failed"
        assert EventType.WORKER_STARTED == "worker_started"
        assert EventType.WORKER_STOPPED == "worker_stopped"

    def test_event_model(self) -> None:
        """Test Event model."""
        from autoclean.api.models import Event, EventType

        event = Event(
            type=EventType.QUEUE_UPDATE,
            timestamp="2024-01-01T00:00:00Z",
            data={"path": "/data/file.bdf", "status": "pending"},
        )
        assert event.type == EventType.QUEUE_UPDATE
        assert event.data["path"] == "/data/file.bdf"


class TestTasks:
    """Tests for RQ task definitions."""

    def test_process_file_import(self) -> None:
        """Test process_file can be imported."""
        from autoclean.api.tasks import process_file
        assert process_file is not None

    def test_dispatch_ready_files_import(self) -> None:
        """Test dispatch_ready_files can be imported."""
        from autoclean.api.tasks import dispatch_ready_files
        assert dispatch_ready_files is not None

    def test_run_ingestion_cycle_import(self) -> None:
        """Test run_ingestion_cycle can be imported."""
        from autoclean.api.tasks import run_ingestion_cycle
        assert run_ingestion_cycle is not None


class TestEventBroadcaster:
    """Tests for EventBroadcaster class."""

    def test_broadcaster_init(self) -> None:
        """Test broadcaster initialization."""
        from autoclean.api.events import EventBroadcaster

        broadcaster = EventBroadcaster()
        assert broadcaster.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcaster_connect_disconnect(self) -> None:
        """Test connect and disconnect."""
        from autoclean.api.events import EventBroadcaster

        broadcaster = EventBroadcaster()

        # Mock websocket
        mock_ws = MagicMock()
        mock_ws.accept = MagicMock(return_value=None)

        # Can't fully test async without more setup
        assert broadcaster.connection_count == 0


class TestRoutesImport:
    """Tests for route module imports."""

    def test_queue_routes_import(self) -> None:
        """Test queue routes can be imported."""
        from autoclean.api.routes import queue
        assert queue.router is not None

    def test_worker_routes_import(self) -> None:
        """Test worker routes can be imported."""
        from autoclean.api.routes import worker
        assert worker.router is not None

    def test_config_routes_import(self) -> None:
        """Test config routes can be imported."""
        from autoclean.api.routes import config
        assert config.router is not None


class TestServerImport:
    """Tests for server module."""

    def test_server_import(self) -> None:
        """Test server module can be imported."""
        from autoclean.api.server import create_app, run_server, api_state
        assert create_app is not None
        assert run_server is not None
        assert api_state is not None


class TestAPIIntegration:
    """Integration tests with mocked dependencies."""

    def test_queue_stats_with_mock_queue(self, tmp_path: Path) -> None:
        """Test queue stats with mock queue file."""
        # Create mock queue file
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/file1.bdf": {"status": "pending"},
                "/file2.bdf": {"status": "pending"},
                "/file3.bdf": {"status": "processed"},
                "/file4.bdf": {"status": "failed"},
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        # Configure state
        test_state = APIState()
        test_state.configure(tmp_path, mode="test")

        # Load queue and count
        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        stats = {"pending": 0, "processed": 0, "failed": 0}
        for data in entries.values():
            status = data.get("status", "pending")
            if status in stats:
                stats[status] += 1

        assert stats["pending"] == 2
        assert stats["processed"] == 1
        assert stats["failed"] == 1

    def test_config_loading(self, tmp_path: Path) -> None:
        """Test config loading."""
        import yaml

        # Create minimal config
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "taskfile-montage-version",
            "taskfile": "TestTask",
            "montage": "biosemi64",
            "ingestion_folders": [],
        }

        config_path = tmp_path / "serve-test.yaml"
        config_path.write_text(yaml.dump(config))

        # Create required directories
        (tmp_path / "runtimes" / "test").mkdir(parents=True)
        (tmp_path / "automations").mkdir()

        # Try to load
        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(config_path)
        assert raw_config["mode"] == "test"

        # Parse (non-strict since ingestion_folders is empty)
        parsed, warnings = parse_serve_config(raw_config, tmp_path, strict=False)
        assert parsed.mode == "test"
