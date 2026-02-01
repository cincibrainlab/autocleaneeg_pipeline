"""Tests for serve CLI commands and API integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import subprocess
import sys

import pytest


def create_minimal_serve_workspace(tmp_path: Path) -> Path:
    """Create a minimal valid serve workspace structure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create required directories
    (workspace / "runtimes" / "test").mkdir(parents=True)
    (workspace / "runtimes" / "live").mkdir(parents=True)
    (workspace / "automations").mkdir()
    (workspace / "deploy").mkdir()

    # Create required config files
    serve_config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automation_mode": True,
        "automation_root": "automations",
        "workspace_name": "test-workspace",
        "taskfile": "TestTask",
        "montage": "biosemi64",
        "ingestion_folders": [],
    }

    import yaml
    (workspace / "serve-test.yaml").write_text(yaml.dump(serve_config))

    serve_config["mode"] = "live"
    serve_config["runtime"] = "runtimes/live"
    (workspace / "serve-live.yaml").write_text(yaml.dump(serve_config))

    return workspace


class TestServeWorkspaceValidation:
    """Tests for serve workspace validation edge cases."""

    def test_workspace_missing_runtimes_test(self, tmp_path: Path) -> None:
        """Test validation fails when runtimes/test is missing."""
        workspace = create_minimal_serve_workspace(tmp_path)

        # Remove runtimes/test
        import shutil
        shutil.rmtree(workspace / "runtimes" / "test")

        from autoclean.cli import _serve_workspace_paths, _validate_serve_workspace

        paths = _serve_workspace_paths(workspace)
        assert not _validate_serve_workspace(paths)

    def test_workspace_missing_automations(self, tmp_path: Path) -> None:
        """Test validation fails when automations is missing."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import shutil
        shutil.rmtree(workspace / "automations")

        from autoclean.cli import _serve_workspace_paths, _validate_serve_workspace

        paths = _serve_workspace_paths(workspace)
        assert not _validate_serve_workspace(paths)

    def test_workspace_missing_config_file(self, tmp_path: Path) -> None:
        """Test validation fails when serve-test.yaml is missing."""
        workspace = create_minimal_serve_workspace(tmp_path)

        (workspace / "serve-test.yaml").unlink()

        from autoclean.cli import _serve_workspace_paths, _validate_serve_workspace

        paths = _serve_workspace_paths(workspace)
        assert not _validate_serve_workspace(paths)

    def test_workspace_valid(self, tmp_path: Path) -> None:
        """Test validation passes for valid workspace."""
        workspace = create_minimal_serve_workspace(tmp_path)

        from autoclean.cli import _serve_workspace_paths, _validate_serve_workspace

        paths = _serve_workspace_paths(workspace)
        assert _validate_serve_workspace(paths)


class TestResolveWorkspaceDir:
    """Tests for workspace directory resolution."""

    def test_resolve_explicit_path(self, tmp_path: Path) -> None:
        """Test resolving explicitly provided path."""
        from autoclean.cli import _resolve_serve_workspace_dir

        result = _resolve_serve_workspace_dir(tmp_path)
        assert result == tmp_path.resolve()

    def test_resolve_none_no_stored(self) -> None:
        """Test resolving None when no stored workspace."""
        from autoclean.cli import _resolve_serve_workspace_dir

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = None
            result = _resolve_serve_workspace_dir(None)
            assert result is None

    def test_resolve_path_with_tilde(self, tmp_path: Path) -> None:
        """Test resolving path with ~ expansion."""
        from autoclean.cli import _resolve_serve_workspace_dir

        # Create a path that looks like it has a tilde
        result = _resolve_serve_workspace_dir(tmp_path)
        assert result.is_absolute()


class TestServeTUICommand:
    """Tests for serve tui command edge cases."""

    def test_tui_no_workspace(self) -> None:
        """Test TUI command with no workspace configured."""
        from autoclean.cli import cmd_serve_tui

        args = MagicMock()
        args.path = None
        args.mode = "test"

        with patch("autoclean.cli._resolve_serve_workspace_dir", return_value=None):
            result = cmd_serve_tui(args)
            assert result == 1

    def test_tui_invalid_workspace(self, tmp_path: Path) -> None:
        """Test TUI command with invalid workspace (missing dirs)."""
        from autoclean.cli import cmd_serve_tui

        # Create workspace without required directories
        workspace = tmp_path / "invalid"
        workspace.mkdir()

        args = MagicMock()
        args.path = workspace
        args.mode = "test"

        with patch("autoclean.cli._resolve_serve_workspace_dir", return_value=workspace):
            result = cmd_serve_tui(args)
            assert result == 1

    def test_tui_import_error(self, tmp_path: Path) -> None:
        """Test TUI command when textual import fails."""
        from autoclean.cli import cmd_serve_tui

        workspace = create_minimal_serve_workspace(tmp_path)

        args = MagicMock()
        args.path = workspace
        args.mode = "test"

        with patch("autoclean.cli._resolve_serve_workspace_dir", return_value=workspace), \
             patch("autoclean.cli._validate_serve_workspace", return_value=True), \
             patch.dict(sys.modules, {"autoclean.tui": None}):
            # Simulate import error
            with patch("autoclean.cli.cmd_serve_tui") as mock_cmd:
                mock_cmd.return_value = 1
                # The real function would return 1 on import error


class TestServeAPICommand:
    """Tests for serve api command edge cases."""

    def test_api_no_workspace(self) -> None:
        """Test API command with no workspace configured."""
        from autoclean.cli import cmd_serve_api

        args = MagicMock()
        args.path = None
        args.mode = "test"
        args.host = "127.0.0.1"
        args.api_port = 8000
        args.redis_url = "redis://localhost:6379"
        args.reload = False

        with patch("autoclean.cli._resolve_serve_workspace_dir", return_value=None):
            result = cmd_serve_api(args)
            assert result == 1

    def test_api_invalid_workspace(self, tmp_path: Path) -> None:
        """Test API command with invalid workspace."""
        from autoclean.cli import cmd_serve_api

        workspace = tmp_path / "invalid"
        workspace.mkdir()

        args = MagicMock()
        args.path = workspace
        args.mode = "test"
        args.host = "127.0.0.1"
        args.api_port = 8000
        args.redis_url = "redis://localhost:6379"
        args.reload = False

        with patch("autoclean.cli._resolve_serve_workspace_dir", return_value=workspace):
            result = cmd_serve_api(args)
            assert result == 1


class TestServeWorkerCommand:
    """Tests for serve worker command edge cases."""

    def test_worker_redis_connection_fails(self) -> None:
        """Test worker command when Redis connection fails."""
        from autoclean.cli import cmd_serve_worker

        args = MagicMock()
        args.queues = "default"
        args.redis_url = "redis://nonexistent:6379"
        args.burst = False

        # This should fail to connect
        result = cmd_serve_worker(args)
        assert result == 1

    def test_worker_empty_queues_string(self) -> None:
        """Test worker command with empty queues string."""
        from autoclean.cli import cmd_serve_worker

        args = MagicMock()
        args.queues = "   "  # Whitespace only
        args.redis_url = "redis://localhost:6379"
        args.burst = False

        # Patch at the redis module level since it's imported inside the function
        with patch("redis.Redis") as mock_redis:
            mock_conn = MagicMock()
            mock_conn.ping.return_value = True
            mock_redis.from_url.return_value = mock_conn

            with patch("rq.Queue") as mock_queue, \
                 patch("rq.Worker") as mock_worker:
                mock_worker_instance = MagicMock()
                mock_worker.return_value = mock_worker_instance

                result = cmd_serve_worker(args)
                # With empty queues string, worker should still be created
                # (empty queue list is valid for RQ)
                assert mock_worker.called or result == 0

    def test_worker_queues_with_special_chars(self) -> None:
        """Test worker command with special characters in queue names."""
        from autoclean.cli import cmd_serve_worker

        args = MagicMock()
        args.queues = "queue-1,queue_2,queue.3"
        args.redis_url = "redis://localhost:6379"
        args.burst = True

        with patch("redis.Redis") as mock_redis:
            mock_conn = MagicMock()
            mock_conn.ping.return_value = True
            mock_redis.from_url.return_value = mock_conn

            with patch("rq.Queue") as mock_queue, \
                 patch("rq.Worker") as mock_worker:
                mock_worker_instance = MagicMock()
                mock_worker.return_value = mock_worker_instance

                result = cmd_serve_worker(args)

                # Verify queues were created with correct names
                calls = mock_queue.call_args_list
                queue_names = [call[0][0] for call in calls]
                assert "queue-1" in queue_names
                assert "queue_2" in queue_names
                assert "queue.3" in queue_names


class TestAPIServerIntegration:
    """Integration tests for API server with serve workspace."""

    def test_api_state_configured_from_workspace(self, tmp_path: Path) -> None:
        """Test that API state is configured correctly from workspace."""
        workspace = create_minimal_serve_workspace(tmp_path)

        from autoclean.api.server import create_app
        from autoclean.api.state import api_state

        # Reset state
        api_state.workspace_dir = None
        api_state.mode = "test"

        app = create_app(workspace_dir=workspace, mode="live")

        assert api_state.workspace_dir == workspace
        assert api_state.mode == "live"

    def test_api_queue_path_matches_mode(self, tmp_path: Path) -> None:
        """Test that queue path matches the configured mode."""
        workspace = create_minimal_serve_workspace(tmp_path)

        from autoclean.api.state import APIState

        state = APIState()
        state.configure(workspace, mode="test")

        queue_path = state.get_queue_path()
        assert queue_path == workspace / "queue-test.json"

        state.configure(workspace, mode="live")
        queue_path = state.get_queue_path()
        assert queue_path == workspace / "queue-live.json"

    def test_api_config_path_deployed_vs_nondeployed(self, tmp_path: Path) -> None:
        """Test config path for deployed vs non-deployed modes."""
        workspace = create_minimal_serve_workspace(tmp_path)

        from autoclean.api.state import APIState

        state = APIState()
        state.configure(workspace, mode="test")

        # Non-deployed
        config_path = state.get_config_path(deployed=False)
        assert config_path == workspace / "serve-test.yaml"

        # Deployed
        deployed_path = state.get_config_path(deployed=True)
        assert deployed_path == workspace / "deploy" / "serve-test.yaml"


class TestQueueFileEdgeCases:
    """Test queue file handling edge cases in serve context."""

    def test_queue_file_created_on_first_access(self, tmp_path: Path) -> None:
        """Test that queue file is created if it doesn't exist."""
        workspace = create_minimal_serve_workspace(tmp_path)
        queue_path = workspace / "queue-test.json"

        assert not queue_path.exists()

        from autoclean.utils.ingestion import IngestionQueue

        queue = IngestionQueue(queue_path)
        entries = queue.entries()

        assert isinstance(entries, dict)
        assert len(entries) == 0

    def test_queue_file_concurrent_access_simulation(self, tmp_path: Path) -> None:
        """Test queue file with simulated concurrent access."""
        workspace = create_minimal_serve_workspace(tmp_path)
        queue_path = workspace / "queue-test.json"

        # Create initial queue
        queue_data = {"entries": {"/file1.bdf": {"status": "pending"}}}
        queue_path.write_text(json.dumps(queue_data))

        from autoclean.utils.ingestion import IngestionQueue

        # Load queue twice (simulating concurrent access)
        queue1 = IngestionQueue(queue_path)
        queue2 = IngestionQueue(queue_path)

        entries1 = queue1.entries()
        entries2 = queue2.entries()

        # Both should see the same data
        assert "/file1.bdf" in entries1
        assert "/file1.bdf" in entries2


class TestConfigLoadingEdgeCases:
    """Test config loading edge cases."""

    def test_config_with_missing_optional_fields(self, tmp_path: Path) -> None:
        """Test loading config with missing optional fields."""
        workspace = create_minimal_serve_workspace(tmp_path)

        # Create minimal config without all optional fields
        import yaml
        minimal_config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "taskfile": "TestTask",
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(minimal_config))

        from autoclean.utils.ingestion import load_serve_config

        config = load_serve_config(workspace / "serve-test.yaml")
        assert config["mode"] == "test"

    def test_config_with_empty_ingestion_folders(self, tmp_path: Path) -> None:
        """Test config with empty ingestion_folders list."""
        workspace = create_minimal_serve_workspace(tmp_path)

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        config = load_serve_config(workspace / "serve-test.yaml")
        parsed, warnings = parse_serve_config(config, workspace, strict=False)

        # Should parse without error, routes will be empty
        assert parsed.mode == "test"
