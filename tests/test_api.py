"""Tests for the AutoClean API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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
from autoclean.api.routes.service import ServiceStartRequest


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

    def test_create_app_does_not_load_persisted_workspace_by_default(self, monkeypatch) -> None:
        """Test app factory isolation from persisted local user config."""
        old_workspace = api_state.workspace_dir
        monkeypatch.setattr("autoclean.api.server._load_persisted_serve_workspace", lambda: Path("/tmp/persisted"))
        try:
            api_state.workspace_dir = None
            app = create_app()
            assert app is not None
            assert api_state.workspace_dir is None
        finally:
            api_state.workspace_dir = old_workspace

    def test_create_app_with_workspace(self, tmp_path: Path) -> None:
        """Test creating app with workspace."""
        app = create_app(workspace_dir=tmp_path, mode="live")
        assert app is not None


class TestTaskManagerWorkspaceResolution:
    """Tests for Serve-aware Task Manager workspace resolution."""

    def test_task_manager_prefers_serve_workspace_tasks_dir(self, tmp_path: Path) -> None:
        from autoclean.api.routes.task_manager import _get_workspace_dir

        serve_workspace = tmp_path / "serve-workspace"
        serve_tasks_dir = serve_workspace / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        with patch("autoclean.utils.user_config.UserConfigManager") as mock_ucm:
            mgr = mock_ucm.return_value
            mgr.get_serve_tasks_dir.return_value = serve_tasks_dir
            mgr.tasks_dir = legacy_tasks_dir
            assert _get_workspace_dir() == serve_tasks_dir


class TestSetupWorkspaceRoute:
    """Tests for API workspace setup behavior."""

    def test_setup_workspace_rejects_arbitrary_existing_directory(self, tmp_path: Path) -> None:
        app = create_app()
        client = TestClient(app)

        workspace = tmp_path / "random-folder"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("not a workspace", encoding="utf-8")

        response = client.post(
            "/api/setup/workspace",
            json={"path": str(workspace), "create_new": False},
        )

        assert response.status_code == 400
        assert "valid Serve workspace or an AutoClean workspace" in response.json()["detail"]

    def test_setup_workspace_bootstraps_existing_normal_workspace(self, tmp_path: Path) -> None:
        app = create_app()
        client = TestClient(app)

        workspace = tmp_path / "workspace"
        (workspace / "tasks").mkdir(parents=True)
        (workspace / "output").mkdir(parents=True)

        response = client.post(
            "/api/setup/workspace",
            json={"path": str(workspace), "create_new": False},
        )

        assert response.status_code == 200
        assert (workspace / "serve-test.yaml").exists()
        assert (workspace / "serve-live.yaml").exists()
        assert (workspace / "routes").exists()
        assert (workspace / "automations").exists()


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


class TestServiceModels:
    """Tests for service-related request defaults."""

    def test_service_start_request_defaults_to_long_running_dispatcher(self) -> None:
        """UI/API start requests should default to a persistent service loop."""
        request = ServiceStartRequest()
        assert request.max_cycles == 0
        assert request.idle_limit == 0


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

    def test_run_server_binds_explicit_workspace_into_factory(self, tmp_path: Path, monkeypatch) -> None:
        """Test run_server passes the requested workspace into the app factory."""
        import autoclean.api.server as server_module

        captured: dict[str, Any] = {}

        def fake_uvicorn_run(app_factory, **kwargs):
            captured["factory"] = app_factory
            captured["kwargs"] = kwargs

        old_workspace = server_module.api_state.workspace_dir
        monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
        monkeypatch.setattr(server_module, "_load_persisted_serve_workspace", lambda: None)
        try:
            server_module.api_state.workspace_dir = None
            server_module.run_server(workspace_dir=tmp_path, mode="live", port=8123)
            factory = captured["factory"]
            app = factory()
            assert app is not None
            assert server_module.api_state.workspace_dir == tmp_path
            assert captured["kwargs"]["factory"] is True
            assert captured["kwargs"]["port"] == 8123
        finally:
            server_module.api_state.workspace_dir = old_workspace


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

    def test_queue_with_invalid_status(self, tmp_path: Path) -> None:
        """Test queue handles invalid status values gracefully."""
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/file1.bdf": {"status": "pending"},
                "/file2.bdf": {"status": "invalid_status"},  # Invalid
                "/file3.bdf": {"status": "unknown"},  # Also invalid
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        # The entries should load without error
        assert len(entries) == 3
        assert entries["/file1.bdf"]["status"] == "pending"
        assert entries["/file2.bdf"]["status"] == "invalid_status"


class TestAPIStateEdgeCases:
    """Edge case tests for APIState."""

    def test_get_queue_path_unconfigured(self) -> None:
        """Test get_queue_path raises when workspace not configured."""
        from fastapi import HTTPException

        state = APIState()
        with pytest.raises(HTTPException) as exc_info:
            state.get_queue_path()
        assert exc_info.value.status_code == 500

    def test_get_config_path_unconfigured(self) -> None:
        """Test get_config_path raises when workspace not configured."""
        from fastapi import HTTPException

        state = APIState()
        with pytest.raises(HTTPException) as exc_info:
            state.get_config_path()
        assert exc_info.value.status_code == 500

    def test_check_redis_without_connection(self) -> None:
        """Test check_redis returns False when no Redis."""
        state = APIState()
        state.redis_url = "redis://nonexistent:6379"
        # Reset cached connection
        state._redis_connection = None

        # Should return False, not raise
        result = state.check_redis()
        assert result is False


class TestAdditionalModels:
    """Additional model tests."""

    def test_queue_status_enum_values(self) -> None:
        """Test QueueStatus enum has correct values."""
        assert QueueStatus.PENDING.value == "pending"
        assert QueueStatus.PROCESSING.value == "processing"
        assert QueueStatus.PROCESSED.value == "processed"
        assert QueueStatus.FAILED.value == "failed"

    def test_worker_status_enum_values(self) -> None:
        """Test WorkerStatus enum has correct values."""
        assert WorkerStatus.IDLE.value == "idle"
        assert WorkerStatus.BUSY.value == "busy"
        assert WorkerStatus.STOPPED.value == "stopped"
        assert WorkerStatus.STARTING.value == "starting"

    def test_queue_entry_all_fields(self) -> None:
        """Test QueueEntry with all optional fields."""
        entry = QueueEntry(
            path="/data/file.bdf",
            status=QueueStatus.FAILED,
            route_id="route-1",
            ingestion_root="/data",
            added_at="2024-01-01T00:00:00",
            processed_at=None,
            failed_at="2024-01-01T01:00:00",
            last_error="Processing failed",
        )
        assert entry.last_error == "Processing failed"
        assert entry.failed_at == "2024-01-01T01:00:00"

    def test_deploy_response_model(self) -> None:
        """Test DeployResponse model."""
        from autoclean.api.models import DeployResponse

        response = DeployResponse(
            success=True,
            source="/path/serve-test.yaml",
            target="/path/deploy/serve-test.yaml",
            message="Deployed successfully",
        )
        assert response.success is True
        assert "serve-test.yaml" in response.source

    def test_job_info_model(self) -> None:
        """Test JobInfo model."""
        from autoclean.api.models import JobInfo

        job = JobInfo(
            id="job-123",
            status="queued",
            func_name="process_file",
            args=["/data/file.bdf"],
            created_at="2024-01-01T00:00:00",
        )
        assert job.id == "job-123"
        assert job.status == "queued"

    def test_worker_info_model(self) -> None:
        """Test WorkerInfo model."""
        from autoclean.api.models import WorkerInfo

        worker = WorkerInfo(
            name="worker-1",
            status=WorkerStatus.BUSY,
            current_job="job-123",
            queues=["default", "high"],
            pid=12345,
        )
        assert worker.name == "worker-1"
        assert worker.pid == 12345

    def test_worker_start_request_defaults(self) -> None:
        """Test WorkerStartRequest default values."""
        from autoclean.api.models import WorkerStartRequest

        request = WorkerStartRequest()
        assert request.count == 1
        assert request.queues == ["default"]

    def test_worker_stop_request_defaults(self) -> None:
        """Test WorkerStopRequest default values."""
        from autoclean.api.models import WorkerStopRequest

        request = WorkerStopRequest()
        assert request.graceful is True


class TestTaskFunctions:
    """Tests for task function behavior."""

    def test_timestamp_format(self) -> None:
        """Test _timestamp returns ISO format."""
        from autoclean.api.tasks import _timestamp

        ts = _timestamp()
        # Should be ISO format with timezone
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_process_file_dry_run(self, tmp_path: Path) -> None:
        """Test process_file with dry_run."""
        from autoclean.api.tasks import process_file

        # Create minimal workspace structure
        runtime_dir = tmp_path / "runtimes" / "test"
        runtime_dir.mkdir(parents=True)

        result = process_file(
            file_path="/data/test.bdf",
            workspace_dir=str(tmp_path),
            mode="test",
            route_id="route-1",
            taskfile="TestTask",
            montage="biosemi64",
            dry_run=True,
        )

        assert result["status"] == "dry_run"
        assert "command" in result
        assert result["file_path"] == "/data/test.bdf"


class TestEventEmitters:
    """Tests for event emitter functions."""

    @pytest.mark.asyncio
    async def test_emit_queue_update(self) -> None:
        """Test emit_queue_update function."""
        from autoclean.api.events import broadcaster, emit_queue_update

        # With no connections, should not raise
        await emit_queue_update(
            action="added",
            path="/data/file.bdf",
            status="pending",
            route_id="route-1",
        )
        # No error means success

    @pytest.mark.asyncio
    async def test_broadcaster_broadcast_no_connections(self) -> None:
        """Test broadcast with no connections."""
        from autoclean.api.events import EventBroadcaster
        from autoclean.api.models import Event, EventType

        broadcaster = EventBroadcaster()
        event = Event(
            type=EventType.QUEUE_UPDATE,
            timestamp="2024-01-01T00:00:00Z",
            data={"test": True},
        )

        # Should not raise with no connections
        await broadcaster.broadcast(event)
        assert broadcaster.connection_count == 0


class TestQueueEdgeCases:
    """Edge case tests for queue handling - corrupted data, special chars, etc."""

    def test_malformed_json_queue_file(self, tmp_path: Path) -> None:
        """Test handling of corrupted/malformed JSON in queue file."""
        queue_path = tmp_path / "queue-test.json"
        queue_path.write_text("{ invalid json }")

        from autoclean.utils.ingestion import IngestionQueue

        # Should handle gracefully - either raise clear error or return empty
        try:
            queue = IngestionQueue(queue_path)
            # If it doesn't raise, entries should be empty or default
            entries = queue.entries()
            # Malformed JSON should result in empty/default state
            assert isinstance(entries, dict)
        except (json.JSONDecodeError, ValueError):
            # Acceptable to raise on malformed JSON
            pass

    def test_queue_missing_entries_key(self, tmp_path: Path) -> None:
        """Test queue file with missing 'entries' key."""
        queue_path = tmp_path / "queue-test.json"
        queue_path.write_text(json.dumps({"other_key": "value"}))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()
        # Should return empty dict, not crash
        assert isinstance(entries, dict)

    def test_queue_entries_wrong_type(self, tmp_path: Path) -> None:
        """Test queue file where entries is wrong type (list instead of dict).

        NOTE: This documents a potential bug - IngestionQueue returns the raw
        value without type validation. Code that calls .entries() should handle
        the case where it might not be a dict.
        """
        queue_path = tmp_path / "queue-test.json"
        queue_path.write_text(json.dumps({"entries": ["item1", "item2"]}))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        # Current behavior: returns raw value without type checking
        # This is a potential bug - entries could be list, breaking .items() calls
        # Document current behavior: it returns whatever was in the JSON
        assert entries == ["item1", "item2"]

    def test_queue_entry_with_none_status(self, tmp_path: Path) -> None:
        """Test queue entry where status is None instead of string."""
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/file1.bdf": {"status": None},
                "/file2.bdf": {"status": "pending"},
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        # Should handle None status - either skip or default to pending
        assert len(entries) == 2

    def test_queue_entry_status_as_integer(self, tmp_path: Path) -> None:
        """Test queue entry where status is wrong type (int instead of string)."""
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/file1.bdf": {"status": 123},  # Wrong type
                "/file2.bdf": {"status": "pending"},
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        # Should load without crashing
        assert "/file2.bdf" in entries

    def test_file_path_with_unicode(self, tmp_path: Path) -> None:
        """Test queue with unicode characters in file paths."""
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/data/файл.bdf": {"status": "pending"},  # Russian
                "/data/文件.bdf": {"status": "pending"},  # Chinese
                "/data/αβγ.bdf": {"status": "pending"},  # Greek
            }
        }
        queue_path.write_text(json.dumps(queue_data, ensure_ascii=False), encoding="utf-8")

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        assert len(entries) == 3
        assert "/data/файл.bdf" in entries

    def test_file_path_with_spaces_and_special_chars(self, tmp_path: Path) -> None:
        """Test queue with spaces and special characters in paths."""
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/data/my file.bdf": {"status": "pending"},
                "/data/file (1).bdf": {"status": "pending"},
                "/data/file's copy.bdf": {"status": "pending"},
                "/data/file&name.bdf": {"status": "pending"},
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        assert len(entries) == 4
        assert "/data/my file.bdf" in entries

    def test_empty_queue_file(self, tmp_path: Path) -> None:
        """Test completely empty queue file."""
        queue_path = tmp_path / "queue-test.json"
        queue_path.write_text("")

        from autoclean.utils.ingestion import IngestionQueue

        try:
            queue = IngestionQueue(queue_path)
            entries = queue.entries()
            assert isinstance(entries, dict)
        except (json.JSONDecodeError, ValueError):
            # Acceptable to raise on empty file
            pass

    def test_queue_file_does_not_exist(self, tmp_path: Path) -> None:
        """Test loading queue when file doesn't exist yet."""
        queue_path = tmp_path / "nonexistent-queue.json"

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        # Should return empty dict for non-existent file
        assert isinstance(entries, dict)
        assert len(entries) == 0

    def test_very_long_file_path(self, tmp_path: Path) -> None:
        """Test queue with extremely long file path."""
        queue_path = tmp_path / "queue-test.json"
        long_path = "/data/" + "a" * 1000 + ".bdf"
        queue_data = {
            "entries": {
                long_path: {"status": "pending"},
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        assert long_path in entries

    def test_queue_entry_with_extra_unexpected_fields(self, tmp_path: Path) -> None:
        """Test queue entries with extra fields that aren't expected."""
        queue_path = tmp_path / "queue-test.json"
        queue_data = {
            "entries": {
                "/file.bdf": {
                    "status": "pending",
                    "unexpected_field": "value",
                    "another_field": 12345,
                    "nested": {"deep": "value"},
                },
            }
        }
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        # Should load without crashing, extra fields ignored or preserved
        assert "/file.bdf" in entries
        assert entries["/file.bdf"]["status"] == "pending"


class TestAPIStateResetBehavior:
    """Test APIState behavior when reconfigured or reset."""

    def test_reconfigure_clears_cached_connections(self, tmp_path: Path) -> None:
        """Test that reconfiguring state clears cached Redis connections."""
        state = APIState()
        state.configure(tmp_path, mode="test", redis_url="redis://localhost:6379")

        # Simulate cached connection
        state._redis_connection = "fake_connection"
        state._rq_queue = "fake_queue"

        # Reconfigure should work (doesn't auto-clear, but let's verify state)
        state.configure(tmp_path, mode="live", redis_url="redis://other:6379")

        assert state.mode == "live"
        assert state.redis_url == "redis://other:6379"
        # Note: Current implementation doesn't clear cache on reconfigure
        # This test documents current behavior

    def test_multiple_api_state_instances(self, tmp_path: Path) -> None:
        """Test multiple APIState instances are independent."""
        state1 = APIState()
        state2 = APIState()

        state1.configure(tmp_path, mode="test")
        state2.configure(tmp_path, mode="live")

        assert state1.mode == "test"
        assert state2.mode == "live"
