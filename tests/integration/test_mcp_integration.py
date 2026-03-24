from __future__ import annotations

import time
from pathlib import Path

import yaml

from autoclean_mcp.cli_adapter import execute_cli, get_canonical_cli_command
from autoclean_mcp.session_manager import SessionManager


def _mcp_test_env(tmp_path: Path) -> dict[str, str]:
    home_dir = tmp_path / "home"
    mne_home = tmp_path / "mne-home"
    home_dir.mkdir(parents=True, exist_ok=True)
    mne_home.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home_dir),
        "MNE_HOME": str(mne_home),
    }


def _create_minimal_serve_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "serve-workspace"
    for subdir in (
        "runtimes/test",
        "runtimes/live",
        "automations",
        "deploy",
        "ingest",
    ):
        (workspace / subdir).mkdir(parents=True, exist_ok=True)

    serve_config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automation_mode": True,
        "automation_root": "automations",
        "defaults": {
            "automation_root": "automations",
            "workspace_name": "test-workspace",
            "file_globs": ["*.set"],
            "sentinel_ext": ".ready",
            "recursive": True,
        },
        "automations": [
            {
                "id": "sample-rest-standard-1020",
                "taskfile": "Resting",
                "montage": "standard_1020",
                "ingestion_folders": [str(workspace / "ingest")],
            }
        ],
    }

    (workspace / "serve-test.yaml").write_text(
        yaml.dump(serve_config), encoding="utf-8"
    )
    (workspace / "deploy" / "serve-test.yaml").write_text(
        yaml.dump(serve_config), encoding="utf-8"
    )

    serve_config["mode"] = "live"
    serve_config["runtime"] = "runtimes/live"
    (workspace / "serve-live.yaml").write_text(
        yaml.dump(serve_config), encoding="utf-8"
    )

    return workspace


def test_execute_cli_covers_representative_cli_families(tmp_path: Path) -> None:
    env = _mcp_test_env(tmp_path)
    workspace = _create_minimal_serve_workspace(tmp_path)

    version = execute_cli(["version"], env=env)
    blocks = execute_cli(["blocks", "list"], env=env)
    tasks = execute_cli(["list-tasks"], env=env)
    serve_workspace = execute_cli(
        ["serve", "workspace", "status", "--path", str(workspace)],
        env=env,
    )

    assert version.ok is True
    assert "Version" in version.stdout

    assert blocks.ok is True
    assert "Processing Blocks" in blocks.stdout

    assert tasks.ok is True
    assert "Tasks" in tasks.stdout

    assert serve_workspace.ok is True
    assert "Workspace used by Serve" in serve_workspace.stdout + serve_workspace.stderr


def test_session_manager_manages_real_serve_run_lifecycle(tmp_path: Path) -> None:
    env = _mcp_test_env(tmp_path)
    workspace = _create_minimal_serve_workspace(tmp_path)
    manager = SessionManager()

    command = get_canonical_cli_command(
        [
            "serve",
            "run",
            "--path",
            str(workspace),
            "--mode",
            "test",
            "--max-cycles",
            "100",
            "--idle-limit",
            "0",
            "--no-watch",
            "--sleep-seconds",
            "0.2",
        ]
    )

    started = manager.start(command, env=env)

    assert started.state == "running"
    assert started.pid is not None

    time.sleep(1.0)
    running = manager.get(started.session_id)

    assert running is not None
    assert running.state == "running"
    assert running.exit_code is None

    stopped = manager.stop(started.session_id)

    assert stopped is not None
    assert stopped.state == "exited"
    assert stopped.exit_code is not None
