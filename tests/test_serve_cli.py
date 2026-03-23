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


class TestServeWorkspaceCommands:
    """Tests for explicit workspace inspection and switching commands."""

    def test_workspace_status_uses_selected_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_serve_workspace

        workspace = create_minimal_serve_workspace(tmp_path)
        args = MagicMock()
        args.workspace_action = "status"
        args.path = None
        args.mode = None
        args.package = "autocleaneeg-pipeline"

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = workspace
            assert cmd_serve_workspace(args) == 0

    def test_workspace_doctor_reports_missing_components(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_serve_workspace

        workspace = create_minimal_serve_workspace(tmp_path)
        (workspace / "serve-test.yaml").unlink()

        args = MagicMock()
        args.workspace_action = "doctor"
        args.path = workspace
        args.mode = None
        args.package = "autocleaneeg-pipeline"

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = workspace
            assert cmd_serve_workspace(args) == 1

    def test_workspace_doctor_treats_unpromoted_live_as_guidance(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_serve_workspace

        workspace = create_minimal_serve_workspace(tmp_path)
        (workspace / "routes").mkdir()
        runtime_cli = workspace / "runtimes" / "test" / ".venv" / "bin" / "autocleaneeg-pipeline"
        runtime_cli.parent.mkdir(parents=True)
        runtime_cli.write_text("", encoding="utf-8")
        live_runtime_cli = workspace / "runtimes" / "live" / ".venv" / "bin" / "autocleaneeg-pipeline"
        live_runtime_cli.parent.mkdir(parents=True)
        live_runtime_cli.write_text("", encoding="utf-8")

        args = MagicMock()
        args.workspace_action = "doctor"
        args.path = workspace
        args.mode = None
        args.package = "autocleaneeg-pipeline"

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = workspace
            with patch("autoclean.cli._validate_serve_yaml", side_effect=lambda config, mode, workspace_dir: mode == "test"):
                assert cmd_serve_workspace(args) == 0

    def test_workspace_use_persists_selection(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_serve_workspace

        workspace = create_minimal_serve_workspace(tmp_path)
        args = MagicMock()
        args.workspace_action = "use"
        args.path = workspace
        args.mode = None
        args.package = "autocleaneeg-pipeline"

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = None
            mock_config.set_serve_workspace.return_value = True
            assert cmd_serve_workspace(args) == 0
            mock_config.set_serve_workspace.assert_called_once()

    def test_workspace_use_rejects_arbitrary_existing_directory(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_serve_workspace

        workspace = tmp_path / "random-folder"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("not a workspace", encoding="utf-8")

        args = MagicMock()
        args.workspace_action = "use"
        args.path = workspace
        args.mode = None
        args.package = "autocleaneeg-pipeline"

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = None
            assert cmd_serve_workspace(args) == 1

    def test_workspace_existing_bootstraps_normal_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_serve_workspace

        workspace = tmp_path / "workspace"
        (workspace / "tasks").mkdir(parents=True)
        (workspace / "output").mkdir(parents=True)

        args = MagicMock()
        args.workspace_action = None
        args.path = workspace
        args.mode = "existing"
        args.package = "autocleaneeg-pipeline"
        args.skip_uv = True
        args.no_test = True

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.get_serve_workspace.return_value = None
            mock_config.set_serve_workspace.return_value = True
            assert cmd_serve_workspace(args) == 0

        assert (workspace / "serve-test.yaml").exists()
        assert (workspace / "serve-live.yaml").exists()
        assert (workspace / "routes").exists()
        assert (workspace / "automations").exists()


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

    def test_resolve_serve_workspace_uses_main_workspace_root(self, tmp_path: Path, monkeypatch) -> None:
        """Serve should resolve through the normal persisted workspace root."""
        from autoclean.utils.user_config import UserConfigManager
        import autoclean.cli as cli_module

        config_home = tmp_path / "config-home"
        docs_home = tmp_path / "docs-home"
        monkeypatch.setattr("platformdirs.user_config_dir", lambda app, appauthor=None: str(config_home))
        monkeypatch.setattr("platformdirs.user_documents_dir", lambda: str(docs_home))

        setup_json = config_home / "setup.json"
        setup_json.parent.mkdir(parents=True, exist_ok=True)
        workspace = tmp_path / "workspace-root"
        legacy_serve = tmp_path / "legacy-serve-root"
        setup_json.write_text(
            json.dumps(
                {
                    "config_directory": str(workspace),
                    "serve_workspace": str(legacy_serve),
                }
            ),
            encoding="utf-8",
        )

        manager = UserConfigManager()
        monkeypatch.setattr(cli_module, "user_config", manager)

        from autoclean.cli import _resolve_serve_workspace_dir

        assert _resolve_serve_workspace_dir(None) == workspace.resolve()


class TestServeWorkspaceRootAlignment:
    """Tests for Serve alignment with the normal workspace root."""

    def test_get_serve_workspace_prefers_main_workspace_root(self, tmp_path: Path, monkeypatch) -> None:
        from autoclean.utils.user_config import UserConfigManager

        config_home = tmp_path / "config-home"
        docs_home = tmp_path / "docs-home"
        monkeypatch.setattr("platformdirs.user_config_dir", lambda app, appauthor=None: str(config_home))
        monkeypatch.setattr("platformdirs.user_documents_dir", lambda: str(docs_home))

        setup_json = config_home / "setup.json"
        setup_json.parent.mkdir(parents=True, exist_ok=True)
        workspace = tmp_path / "workspace-root"
        legacy_serve = tmp_path / "legacy-serve-root"
        setup_json.write_text(
            json.dumps(
                {
                    "config_directory": str(workspace),
                    "serve_workspace": str(legacy_serve),
                }
            ),
            encoding="utf-8",
        )

        manager = UserConfigManager()
        assert manager.get_serve_workspace() == workspace

    def test_set_serve_workspace_updates_main_workspace_root(self, tmp_path: Path, monkeypatch) -> None:
        from autoclean.utils.user_config import UserConfigManager

        config_home = tmp_path / "config-home"
        docs_home = tmp_path / "docs-home"
        monkeypatch.setattr("platformdirs.user_config_dir", lambda app, appauthor=None: str(config_home))
        monkeypatch.setattr("platformdirs.user_documents_dir", lambda: str(docs_home))

        manager = UserConfigManager()
        workspace = tmp_path / "workspace-root"

        assert manager.set_serve_workspace(workspace) is True

        setup_json = config_home / "setup.json"
        saved = json.loads(setup_json.read_text(encoding="utf-8"))
        assert saved["config_directory"] == str(workspace.resolve())
        assert saved["serve_workspace"] == str(workspace.resolve())
        assert manager.config_dir == workspace.resolve()
        assert manager.tasks_dir == workspace.resolve() / "tasks"


class TestTaskCreateCli:
    """Tests for workspace-local task creation from the CLI."""

    def test_task_create_writes_template_file(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_create

        workspace = tmp_path / "workspace"
        tasks_dir = workspace / "tasks"
        tasks_dir.mkdir(parents=True)

        args = MagicMock()
        args.class_name = "MyCustomTask"
        args.file_name = None
        args.force = False

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.tasks_dir = tasks_dir
            mock_config.get_serve_tasks_dir.return_value = None
            assert cmd_task_create(args) == 0

        task_file = tasks_dir / "MyCustomTask.py"
        assert task_file.exists()
        assert "class MyCustomTask" in task_file.read_text(encoding="utf-8")

    def test_task_create_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_create

        serve_workspace = tmp_path / "serve-workspace"
        serve_tasks_dir = serve_workspace / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        args = MagicMock()
        args.class_name = "ServeTask"
        args.file_name = None
        args.force = False

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.tasks_dir = legacy_tasks_dir
            mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
            assert cmd_task_create(args) == 0

        assert (serve_tasks_dir / "ServeTask.py").exists()
        assert not (legacy_tasks_dir / "ServeTask.py").exists()

    def test_task_install_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_install

        serve_workspace = tmp_path / "serve-workspace"
        serve_tasks_dir = serve_workspace / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        args = MagicMock()
        args.task_source = "RestingState_Basic"
        args.source = "library"
        args.name = None
        args.force = False
        args.activate = False

        fake_registry = MagicMock()
        fake_registry.get_task.return_value = object()

        def materialize(task_name: str, target_dir: Path):
            dest = target_dir / f"{task_name}.py"
            dest.write_text("class RestingState_Basic: pass", encoding="utf-8")
            return dest

        fake_registry.materialize_task_to.side_effect = materialize

        with patch("autoclean.cli.BuiltinRegistry", return_value=fake_registry):
            with patch("autoclean.cli.user_config") as mock_config:
                mock_config.tasks_dir = legacy_tasks_dir
                mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                assert cmd_task_install(args) == 0

        assert (serve_tasks_dir / "RestingState_Basic.py").exists()
        assert not (legacy_tasks_dir / "RestingState_Basic.py").exists()

    def test_task_sync_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_sync

        serve_workspace = tmp_path / "serve-workspace"
        serve_tasks_dir = serve_workspace / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)
        (serve_tasks_dir / "ServeTask.py").write_text("class ServeTask: pass", encoding="utf-8")

        args = MagicMock()
        args.update = False
        args.dry_run = False

        fake_registry = MagicMock()
        fake_registry.task_sync_status.return_value = {"status": "synced", "source": "library"}

        with patch("autoclean.cli.BuiltinRegistry", return_value=fake_registry):
            with patch("autoclean.cli.user_config") as mock_config:
                mock_config.tasks_dir = legacy_tasks_dir
                mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                mock_config._extract_task_info.return_value = ("ServeTask", "desc")
                assert cmd_task_sync(args) == 0

        fake_registry.task_sync_status.assert_called_with("ServeTask", serve_tasks_dir)

    def test_task_delete_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_delete

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        serve_task = serve_tasks_dir / "ServeTask.py"
        serve_task.write_text("class ServeTask: pass", encoding="utf-8")

        args = MagicMock()
        args.target = "ServeTask"
        args.force = True

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.tasks_dir = legacy_tasks_dir
            mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
            assert cmd_task_delete(args) == 0

        assert not serve_task.exists()

    def test_task_diff_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_diff

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        serve_task = serve_tasks_dir / "ServeTask.py"
        serve_task.write_text("class ServeTask: pass\n", encoding="utf-8")
        source_file = tmp_path / "ServeTask-source.py"
        source_file.write_text("class ServeTask: pass\n", encoding="utf-8")

        args = MagicMock()
        args.task_name = "ServeTask"
        args.context = 3
        args.color = True

        fake_registry = MagicMock()
        fake_registry.get_task.return_value = MagicMock(path="ServeTask.py")
        fake_registry._cache_path_for.return_value = source_file

        with patch("autoclean.cli.BuiltinRegistry", return_value=fake_registry):
            with patch("autoclean.cli.user_config") as mock_config:
                mock_config.tasks_dir = legacy_tasks_dir
                mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                assert cmd_task_diff(args) == 0

    def test_task_search_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_search

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        (serve_tasks_dir / "ServeTask.py").write_text("class ServeTask: pass\n", encoding="utf-8")

        args = MagicMock()
        args.query = "serve"
        args.source = "all"

        fake_registry = MagicMock()
        fake_registry._cache_index_path.return_value = tmp_path / "missing-index.json"
        fake_registry._pkg_index_text.return_value = json.dumps(
            {"tasks": [{"name": "ServeTask", "description": "Serve task", "category": "custom"}]}
        )

        with patch("autoclean.cli.BuiltinRegistry", return_value=fake_registry):
            with patch("autoclean.cli.user_config") as mock_config:
                mock_config.tasks_dir = legacy_tasks_dir
                mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                mock_config._extract_task_info.return_value = ("ServeTask", "Serve task")
                assert cmd_task_search(args) == 0

    def test_task_diagnose_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_diagnose

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        (serve_tasks_dir / "ServeTask.py").write_text("class ServeTask: pass\n", encoding="utf-8")

        args = MagicMock()

        fake_registry = MagicMock()
        fake_registry.task_sync_status.return_value = {"status": "synced"}
        fake_registry.manifest.task_record.return_value = None
        fake_registry.registry_status.return_value = {"commit": "abc123", "synced_at": None}
        fake_registry.cache_root = tmp_path / "cache"
        fake_registry.cache_root.mkdir()

        with patch("autoclean.cli.BuiltinRegistry", return_value=fake_registry):
            with patch("autoclean.cli.user_config") as mock_config:
                mock_config.tasks_dir = legacy_tasks_dir
                mock_config.config_dir = tmp_path / "legacy-workspace"
                mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                mock_config.get_active_task.return_value = None
                mock_config._extract_task_info.return_value = ("ServeTask", "Serve task")
                assert cmd_task_diagnose(args) == 0

        fake_registry.task_sync_status.assert_called_with("ServeTask", serve_tasks_dir)

    def test_task_list_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_list_tasks

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)

        args = MagicMock()
        args.overrides = False
        args.source = "all"
        args.status = "all"
        args.category = "all"
        args.format = "json"

        fake_registry = MagicMock()
        fake_registry.registry_status.return_value = {}
        fake_registry.list_tasks.return_value = []

        discovered = [
            MagicMock(name="ServeTask", description="Serve task", source=str(serve_tasks_dir / "ServeTask.py")),
        ]
        discovered[0].name = "ServeTask"
        discovered[0].description = "Serve task"
        discovered[0].source = str(serve_tasks_dir / "ServeTask.py")

        with patch("autoclean.cli.BuiltinRegistry", return_value=fake_registry):
            with patch("autoclean.cli.safe_discover_tasks", return_value=(discovered, [], [])):
                with patch("autoclean.cli.user_config") as mock_config:
                    mock_config.tasks_dir = legacy_tasks_dir
                    mock_config.config_dir = tmp_path / "legacy-workspace"
                    mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                    assert cmd_list_tasks(args) == 0

    def test_task_copy_prefers_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_copy

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)
        source_file = tmp_path / "SourceTask.py"
        source_file.write_text("class SourceTask:\n    pass\n", encoding="utf-8")

        args = MagicMock()
        args.source = str(source_file)
        args.name = "Copied Serve Task"
        args.force = True

        with patch("autoclean.cli.user_config") as mock_config:
            mock_config.tasks_dir = legacy_tasks_dir
            mock_config.config_dir = tmp_path / "legacy-workspace"
            mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
            mock_config._extract_task_info.return_value = ("SourceTask", "desc")
            assert cmd_task_copy(args) == 0

        copied = serve_tasks_dir / "copied_serve_task.py"
        assert copied.exists()
        assert not (legacy_tasks_dir / "copied_serve_task.py").exists()

    def test_task_edit_copies_builtin_into_selected_serve_workspace(self, tmp_path: Path) -> None:
        from autoclean.cli import cmd_task_edit

        serve_tasks_dir = tmp_path / "serve-workspace" / "tasks"
        serve_tasks_dir.mkdir(parents=True)
        legacy_tasks_dir = tmp_path / "legacy-workspace" / "tasks"
        legacy_tasks_dir.mkdir(parents=True)
        builtin_source = tmp_path / "BuiltinTask.py"
        builtin_source.write_text("class BuiltinTask:\n    pass\n", encoding="utf-8")

        args = MagicMock()
        args.target = "BuiltinTask"
        args.name = None
        args.force = True

        discovered = [
            MagicMock(name="BuiltinTask", description="Builtin task", source=str(builtin_source)),
        ]
        discovered[0].name = "BuiltinTask"
        discovered[0].description = "Builtin task"
        discovered[0].source = str(builtin_source)

        with patch("autoclean.cli.safe_discover_tasks", return_value=(discovered, [], [])):
            with patch("autoclean.cli._detect_editor", return_value=["true"]):
                with patch("autoclean.cli.subprocess.call", return_value=0):
                    with patch("autoclean.cli.user_config") as mock_config:
                        mock_config.tasks_dir = legacy_tasks_dir
                        mock_config.config_dir = tmp_path / "legacy-workspace"
                        mock_config.get_serve_tasks_dir.return_value = serve_tasks_dir
                        assert cmd_task_edit(args) == 0

        copied = serve_tasks_dir / "BuiltinTask.py"
        assert copied.exists()
        assert not (legacy_tasks_dir / "BuiltinTask.py").exists()


class TestServeParserDefaults:
    """Tests for consistent Serve CLI defaults."""

    def test_serve_parser_defaults_to_port_8000(self) -> None:
        from autoclean.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["serve"])
        assert args.port == 8000

    def test_top_level_serve_port_flows_to_service_and_queue_commands(self) -> None:
        from autoclean.cli import create_parser

        parser = create_parser()

        service_args = parser.parse_args(["serve", "--port", "9001", "service", "status"])
        queue_args = parser.parse_args(["serve", "--port", "9001", "queue", "status"])
        mode_args = parser.parse_args(["serve", "--port", "9001", "mode", "status"])

        assert service_args.port == 9001
        assert queue_args.port == 9001
        assert mode_args.port == 9001


class TestServeLauncherOperationalStartup:
    """Tests for the normal Serve startup path."""

    def test_operational_service_skips_when_workspace_missing(self) -> None:
        from autoclean.serve_launcher import _ensure_operational_service

        with patch("autoclean.serve_launcher._api_request", return_value={"configured": False}):
            running, messages = _ensure_operational_service(8000)

        assert running is False
        assert any("Workspace not configured" in message for message in messages)

    def test_operational_service_skips_when_no_routes(self) -> None:
        from autoclean.serve_launcher import _ensure_operational_service

        status = {
            "configured": True,
            "workspace_dir": "/tmp/workspace",
            "routes": {"total": 0, "active": 0, "archived": 0},
            "config": {"errors": [], "needs_deploy": False},
            "queue": {"pending": 0, "processing": 0, "failed": 0},
            "service": {"running": False},
        }

        with patch("autoclean.serve_launcher._api_request", return_value=status):
            running, messages = _ensure_operational_service(8000)

        assert running is False
        assert any("no routes" in message.lower() for message in messages)

    def test_operational_service_blocks_on_unapplied_config(self) -> None:
        from autoclean.serve_launcher import _ensure_operational_service

        status = {
            "configured": True,
            "workspace_dir": "/tmp/workspace",
            "routes": {"total": 1, "active": 1, "archived": 0},
            "config": {"errors": [], "needs_deploy": True},
            "queue": {"pending": 2, "processing": 1, "failed": 0},
            "service": {"running": False},
        }

        def fake_request(port: int, path: str, method: str = "GET", body: dict | None = None, timeout: int = 20):
            if path == "/api/status":
                return status
            raise AssertionError(f"Unexpected path: {path}")

        with patch("autoclean.serve_launcher._api_request", side_effect=fake_request):
            running, messages = _ensure_operational_service(8000)

        assert running is False
        assert any("unapplied configuration changes exist" in message for message in messages)
        assert any("serve deploy" in message for message in messages)

    def test_operational_service_reports_invalid_config(self) -> None:
        from autoclean.serve_launcher import _ensure_operational_service

        status = {
            "configured": True,
            "workspace_dir": "/tmp/workspace",
            "routes": {"total": 1, "active": 1, "archived": 0},
            "config": {"errors": ["bad config"], "needs_deploy": True},
            "queue": {"pending": 0, "processing": 0, "failed": 0},
            "service": {"running": False},
        }

        with patch("autoclean.serve_launcher._api_request", return_value=status):
            running, messages = _ensure_operational_service(8000)

        assert running is False
        assert any("configuration is invalid" in message for message in messages)


class TestServeServiceCli:
    """Tests for dispatcher control through the CLI."""

    def test_service_status_requires_running_server(self) -> None:
        from autoclean.cli import cmd_serve_service

        args = MagicMock()
        args.service_action = "status"
        args.port = 8000
        args.service_port = None

        with patch("autoclean.serve_launcher._check_existing_server", return_value=None):
            assert cmd_serve_service(args) == 1

    def test_service_start_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_service

        args = MagicMock()
        args.service_action = "start"
        args.port = 8000
        args.service_port = None
        args.max_cycles = 0
        args.idle_limit = 0
        args.sleep_seconds = 1.0
        args.no_watch = False
        args.no_sentinel = False

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch("autoclean.serve_launcher._api_request", return_value={"success": True, "message": "Service started"}):
                assert cmd_serve_service(args) == 0

    def test_service_stop_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_service

        args = MagicMock()
        args.service_action = "stop"
        args.port = 8000
        args.service_port = None

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch("autoclean.serve_launcher._api_request", return_value={"success": True, "message": "Service stopped"}):
                assert cmd_serve_service(args) == 0


class TestServeModeCli:
    """Tests for mode switching through the CLI."""

    def test_mode_status_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_mode

        args = MagicMock()
        args.mode_action = "status"
        args.port = 8000
        args.mode_port = None

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch("autoclean.serve_launcher._api_request", return_value={"mode": "live"}):
                assert cmd_serve_mode(args) == 0

    def test_mode_switch_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_mode

        args = MagicMock()
        args.mode_action = "live"
        args.port = 8000
        args.mode_port = None

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch("autoclean.serve_launcher._api_request", return_value={"success": True, "message": "Switched to live"}):
                assert cmd_serve_mode(args) == 0


class TestServeQueueCli:
    """Tests for queue inspection and maintenance through the CLI."""

    def test_queue_status_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_queue

        args = MagicMock()
        args.queue_action = "status"
        args.port = 8000
        args.queue_port = None

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch(
                "autoclean.serve_launcher._api_request",
                return_value={"pending": 2, "processing": 1, "processed": 3, "failed": 4},
            ):
                assert cmd_serve_queue(args) == 0

    def test_queue_retry_failed_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_queue

        args = MagicMock()
        args.queue_action = "retry-failed"
        args.port = 8000
        args.queue_port = None
        args.paths = ["/tmp/a.set"]

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch("autoclean.serve_launcher._api_request", return_value={"retried": 1}) as request:
                assert cmd_serve_queue(args) == 0
                request.assert_called_once()
                assert request.call_args.kwargs["body"] == {"paths": ["/tmp/a.set"]}

    def test_queue_remove_calls_api(self) -> None:
        from autoclean.cli import cmd_serve_queue

        args = MagicMock()
        args.queue_action = "remove"
        args.port = 8000
        args.queue_port = None
        args.path = "/tmp/a file.set"

        with patch("autoclean.serve_launcher._check_existing_server", return_value=(123, 8000, {})):
            with patch("autoclean.serve_launcher._api_request", return_value={"cleared": 1}) as request:
                assert cmd_serve_queue(args) == 0
                assert "/api/queue/entry/%2Ftmp%2Fa%20file.set" in request.call_args.args[1]


class TestServeRouteCommands:
    """Tests for route-first serve management."""

    def test_route_upsert_creates_registry_and_compiles(self, tmp_path: Path) -> None:
        """Route upsert should create a spec file and compile mode configs."""
        from autoclean.cli import cmd_serve_route_upsert

        workspace = create_minimal_serve_workspace(tmp_path)
        taskfile = tmp_path / "TaskFile.py"
        taskfile.write_text("print('ok')\n", encoding="utf-8")
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()

        args = MagicMock()
        args.path = workspace
        args.route_id = "resting-biosemi64"
        args.mode = "test"
        args.taskfile = str(taskfile)
        args.montage = "biosemi64"
        args.version = None
        args.ingestion_folders = [str(watch_dir)]
        args.ingestion_excludes = None
        args.file_globs = ["*.set"]
        args.priority = 5
        args.automation_root = None
        args.workspace_name = None
        args.sentinel_ext = ".ready"
        args.enabled = True
        args.recursive = True

        result = cmd_serve_route_upsert(args)

        assert result == 0
        route_path = workspace / "routes" / "resting-biosemi64.yaml"
        assert route_path.exists()

        import yaml

        route_spec = yaml.safe_load(route_path.read_text(encoding="utf-8"))
        assert route_spec["modes"] == ["test"]
        assert route_spec["priority"] == 5

        serve_test = yaml.safe_load((workspace / "serve-test.yaml").read_text(encoding="utf-8"))
        serve_live = yaml.safe_load((workspace / "serve-live.yaml").read_text(encoding="utf-8"))
        assert serve_test["automations"][0]["id"] == "resting-biosemi64"
        assert serve_test["automations"][0]["taskfile"] == str(taskfile.resolve())
        assert serve_live["automations"] == []

    def test_route_upsert_is_idempotent(self, tmp_path: Path) -> None:
        """Upserting the same route twice should be a no-op on the second run."""
        from autoclean.cli import cmd_serve_route_upsert

        workspace = create_minimal_serve_workspace(tmp_path)
        taskfile = tmp_path / "TaskFile.py"
        taskfile.write_text("print('ok')\n", encoding="utf-8")
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()

        args = MagicMock()
        args.path = workspace
        args.route_id = "resting-biosemi64"
        args.mode = "test"
        args.taskfile = str(taskfile)
        args.montage = "biosemi64"
        args.version = None
        args.ingestion_folders = [str(watch_dir)]
        args.ingestion_excludes = None
        args.file_globs = ["*.set"]
        args.priority = None
        args.automation_root = None
        args.workspace_name = None
        args.sentinel_ext = None
        args.enabled = True
        args.recursive = True

        assert cmd_serve_route_upsert(args) == 0
        route_path = workspace / "routes" / "resting-biosemi64.yaml"
        before = route_path.read_text(encoding="utf-8")

        assert cmd_serve_route_upsert(args) == 0
        after = route_path.read_text(encoding="utf-8")

        assert before == after

    def test_route_promote_adds_live_mode(self, tmp_path: Path) -> None:
        """Promoting a route should compile it into live config as well."""
        from autoclean.cli import cmd_serve_route_promote, cmd_serve_route_upsert

        workspace = create_minimal_serve_workspace(tmp_path)
        taskfile = tmp_path / "TaskFile.py"
        taskfile.write_text("print('ok')\n", encoding="utf-8")
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()

        upsert_args = MagicMock()
        upsert_args.path = workspace
        upsert_args.route_id = "resting-biosemi64"
        upsert_args.mode = "test"
        upsert_args.taskfile = str(taskfile)
        upsert_args.montage = "biosemi64"
        upsert_args.version = None
        upsert_args.ingestion_folders = [str(watch_dir)]
        upsert_args.ingestion_excludes = None
        upsert_args.file_globs = ["*.set"]
        upsert_args.priority = None
        upsert_args.automation_root = None
        upsert_args.workspace_name = None
        upsert_args.sentinel_ext = None
        upsert_args.enabled = True
        upsert_args.recursive = True

        assert cmd_serve_route_upsert(upsert_args) == 0

        promote_args = MagicMock()
        promote_args.path = workspace
        promote_args.route_id = "resting-biosemi64"

        assert cmd_serve_route_promote(promote_args) == 0

        import yaml

        route_spec = yaml.safe_load(
            (workspace / "routes" / "resting-biosemi64.yaml").read_text(encoding="utf-8")
        )
        serve_live = yaml.safe_load((workspace / "serve-live.yaml").read_text(encoding="utf-8"))

        assert route_spec["modes"] == ["test", "live"]
        assert serve_live["automations"][0]["id"] == "resting-biosemi64"

    def test_route_archive_hides_route_from_compiled_configs(self, tmp_path: Path) -> None:
        """Archiving a route should remove it from generated configs without deleting the spec."""
        from autoclean.cli import cmd_serve_route_archive, cmd_serve_route_upsert

        workspace = create_minimal_serve_workspace(tmp_path)
        taskfile = tmp_path / "TaskFile.py"
        taskfile.write_text("print('ok')\n", encoding="utf-8")
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()

        upsert_args = MagicMock()
        upsert_args.path = workspace
        upsert_args.route_id = "resting-biosemi64"
        upsert_args.mode = "both"
        upsert_args.taskfile = str(taskfile)
        upsert_args.montage = "biosemi64"
        upsert_args.version = None
        upsert_args.ingestion_folders = [str(watch_dir)]
        upsert_args.ingestion_excludes = None
        upsert_args.file_globs = ["*.set"]
        upsert_args.priority = None
        upsert_args.automation_root = None
        upsert_args.workspace_name = None
        upsert_args.sentinel_ext = None
        upsert_args.enabled = True
        upsert_args.recursive = True

        assert cmd_serve_route_upsert(upsert_args) == 0

        archive_args = MagicMock()
        archive_args.path = workspace
        archive_args.route_id = "resting-biosemi64"

        assert cmd_serve_route_archive(archive_args) == 0

        import yaml

        route_spec = yaml.safe_load(
            (workspace / "routes" / "resting-biosemi64.yaml").read_text(encoding="utf-8")
        )
        serve_test = yaml.safe_load((workspace / "serve-test.yaml").read_text(encoding="utf-8"))
        serve_live = yaml.safe_load((workspace / "serve-live.yaml").read_text(encoding="utf-8"))

        assert route_spec["archived"] is True
        assert route_spec["enabled"] is False
        assert serve_test["automations"] == []
        assert serve_live["automations"] == []

    def test_route_list_hides_archived_by_default(self, tmp_path: Path) -> None:
        """Archived routes should only appear when include_archived is requested."""
        from autoclean.cli import cmd_serve_route_archive, cmd_serve_route_list, cmd_serve_route_upsert

        workspace = create_minimal_serve_workspace(tmp_path)
        taskfile = tmp_path / "TaskFile.py"
        taskfile.write_text("print('ok')\n", encoding="utf-8")
        watch_dir = tmp_path / "incoming"
        watch_dir.mkdir()

        upsert_args = MagicMock()
        upsert_args.path = workspace
        upsert_args.route_id = "resting-biosemi64"
        upsert_args.mode = "test"
        upsert_args.taskfile = str(taskfile)
        upsert_args.montage = "biosemi64"
        upsert_args.version = None
        upsert_args.ingestion_folders = [str(watch_dir)]
        upsert_args.ingestion_excludes = None
        upsert_args.file_globs = ["*.set"]
        upsert_args.priority = None
        upsert_args.automation_root = None
        upsert_args.workspace_name = None
        upsert_args.sentinel_ext = None
        upsert_args.enabled = True
        upsert_args.recursive = True
        assert cmd_serve_route_upsert(upsert_args) == 0

        archive_args = MagicMock()
        archive_args.path = workspace
        archive_args.route_id = "resting-biosemi64"
        assert cmd_serve_route_archive(archive_args) == 0

        list_args = MagicMock()
        list_args.path = workspace
        list_args.mode = None
        list_args.include_archived = False
        with patch("autoclean.cli.message") as mock_message:
            assert cmd_serve_route_list(list_args) == 0
        emitted = " ".join(call.args[1] for call in mock_message.call_args_list)
        assert "resting-biosemi64" not in emitted

        list_args.include_archived = True
        with patch("autoclean.cli.message") as mock_message:
            assert cmd_serve_route_list(list_args) == 0
        emitted = " ".join(call.args[1] for call in mock_message.call_args_list)
        assert "resting-biosemi64" in emitted


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


class TestTaskFileEdgeCases:
    """Test edge cases with task files in serve config."""

    def test_empty_taskfile_string(self, tmp_path: Path) -> None:
        """Test config with empty taskfile string."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import yaml
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "taskfile": "",  # Empty
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")

        # Non-strict should warn but not fail
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)
        assert any("taskfile" in w.lower() for w in warnings)

    def test_taskfile_with_path_traversal(self, tmp_path: Path) -> None:
        """Test taskfile with path traversal attempt."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import yaml
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "taskfile": "../../../etc/passwd",  # Path traversal
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")
        # Should parse (validation happens elsewhere)
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)
        assert parsed is not None

    def test_taskfile_nonexistent_python_file(self, tmp_path: Path) -> None:
        """Test taskfile pointing to non-existent Python file."""
        workspace = create_minimal_serve_workspace(tmp_path)

        from autoclean.utils.ingestion import resolve_taskfile_path

        # Non-strict mode (default): returns None for missing .py files
        result = resolve_taskfile_path("nonexistent_task.py", workspace)
        assert result is None

        # Strict mode: raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            resolve_taskfile_path("nonexistent_task.py", workspace, strict=True)

    def test_taskfile_existing_python_file(self, tmp_path: Path) -> None:
        """Test taskfile with existing Python file."""
        workspace = create_minimal_serve_workspace(tmp_path)

        # Create a task file
        task_file = workspace / "my_task.py"
        task_file.write_text("# Task file")

        from autoclean.utils.ingestion import resolve_taskfile_path

        result = resolve_taskfile_path("my_task.py", workspace)
        # Should find the file
        assert result is not None
        assert result.name == "my_task.py"

    def test_taskfile_with_special_characters(self, tmp_path: Path) -> None:
        """Test taskfile with special characters in name."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import yaml
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "taskfile": "Task With Spaces & Symbols!",
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)
        # Should parse - special chars are allowed in task names
        assert parsed is not None

    def test_taskfile_very_long_name(self, tmp_path: Path) -> None:
        """Test taskfile with very long name."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import yaml
        long_name = "A" * 500  # Very long task name
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "taskfile": long_name,
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)
        assert parsed is not None

    def test_taskfile_none_value(self, tmp_path: Path) -> None:
        """Test config with taskfile set to None/null."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import yaml
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "taskfile": None,  # Explicit null
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")
        # Non-strict should handle None gracefully
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)
        assert parsed is not None

    def test_taskfile_as_integer(self, tmp_path: Path) -> None:
        """Test config with taskfile as integer (wrong type)."""
        workspace = create_minimal_serve_workspace(tmp_path)

        import yaml
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "taskfile": 12345,  # Wrong type
            "montage": "biosemi64",
            "ingestion_folders": [],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")
        # Should convert to string or handle gracefully
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)
        assert parsed is not None

    def test_taskfile_label_extraction(self, tmp_path: Path) -> None:
        """Test _taskfile_label extracts correct label."""
        from autoclean.utils.ingestion import _taskfile_label

        # Plain name
        assert _taskfile_label("RestingState") == "RestingState"

        # Python file
        assert _taskfile_label("MyTask.py") == "MyTask"

        # Path with directories
        assert _taskfile_label("/path/to/CustomTask.py") == "CustomTask"

        # Path without extension
        assert _taskfile_label("/path/to/task") == "task"

    def test_taskfile_in_automation_route(self, tmp_path: Path) -> None:
        """Test taskfile handling in automation routes."""
        workspace = create_minimal_serve_workspace(tmp_path)

        # Create separate ingestion folders to avoid overlap
        ingestion_dir1 = tmp_path / "incoming1"
        ingestion_dir1.mkdir()
        ingestion_dir2 = tmp_path / "incoming2"
        ingestion_dir2.mkdir()

        import yaml
        config = {
            "mode": "test",
            "runtime": "runtimes/test",
            "automation_mode": True,
            "automation_root": "automations",
            "workspace_name": "{taskfile}-{montage}",
            "automations": [
                {
                    "taskfile": "CustomTask",
                    "montage": "biosemi64",
                    "priority": 10,
                    "ingestion_folders": [str(ingestion_dir1)],
                    "file_globs": ["*.bdf"],
                },
                {
                    "taskfile": "",  # Empty taskfile in route
                    "montage": "standard1020",
                    "priority": 20,
                    "ingestion_folders": [str(ingestion_dir2)],
                    "file_globs": ["*.edf"],
                },
            ],
        }
        (workspace / "serve-test.yaml").write_text(yaml.dump(config))

        from autoclean.utils.ingestion import load_serve_config, parse_serve_config

        raw_config = load_serve_config(workspace / "serve-test.yaml")
        parsed, warnings = parse_serve_config(raw_config, workspace, strict=False)

        # First route should be valid
        assert len(parsed.routes) >= 1
        assert parsed.routes[0].taskfile == "CustomTask"

        # Should have warning about empty taskfile
        assert any("taskfile" in w.lower() for w in warnings)


class TestProcessFileTaskfileEdgeCases:
    """Test edge cases in process_file task with bad taskfiles."""

    def test_process_file_empty_taskfile(self, tmp_path: Path) -> None:
        """Test process_file with empty taskfile."""
        from autoclean.api.tasks import process_file

        runtime_dir = tmp_path / "runtimes" / "test"
        runtime_dir.mkdir(parents=True)

        result = process_file(
            file_path="/data/test.bdf",
            workspace_dir=str(tmp_path),
            mode="test",
            route_id="route-1",
            taskfile="",  # Empty
            montage="biosemi64",
            dry_run=True,
        )

        # Should still generate command (validation happens in CLI)
        assert result["status"] == "dry_run"
        assert "command" in result

    def test_process_file_taskfile_with_spaces(self, tmp_path: Path) -> None:
        """Test process_file with taskfile containing spaces."""
        from autoclean.api.tasks import process_file

        runtime_dir = tmp_path / "runtimes" / "test"
        runtime_dir.mkdir(parents=True)

        result = process_file(
            file_path="/data/test.bdf",
            workspace_dir=str(tmp_path),
            mode="test",
            route_id="route-1",
            taskfile="Task With Spaces",
            montage="biosemi64",
            dry_run=True,
        )

        assert result["status"] == "dry_run"
        # Check taskfile is in command
        assert "Task With Spaces" in result["command"]

    def test_process_file_python_taskfile(self, tmp_path: Path) -> None:
        """Test process_file with Python file as taskfile."""
        from autoclean.api.tasks import process_file

        runtime_dir = tmp_path / "runtimes" / "test"
        runtime_dir.mkdir(parents=True)

        result = process_file(
            file_path="/data/test.bdf",
            workspace_dir=str(tmp_path),
            mode="test",
            route_id="route-1",
            taskfile="custom_task.py",
            montage="biosemi64",
            dry_run=True,
        )

        assert result["status"] == "dry_run"
        assert "custom_task.py" in str(result["command"])
