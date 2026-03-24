"""Tests for the task library registry helpers."""

from __future__ import annotations

import json
from pathlib import Path

from autoclean.utils.builtins import BuiltinRegistry


def _make_registry(root: Path) -> Path:
    tasks_dir = root / "tasks" / "resting"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / "DemoTask.py"
    task_file.write_text("class DemoTask:\n    pass\n", encoding="utf-8")
    registry = {
        "version": 1,
        "commit": "local-snapshot",
        "tasks": [
            {"name": "DemoTask", "path": "tasks/resting/DemoTask.py"},
        ],
    }
    (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    return task_file


def test_sync_status_roundtrip(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    cache_root = tmp_path / "cache"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    task_file = _make_registry(registry_root)

    registry = BuiltinRegistry(
        raw_base=registry_root.as_uri(),
        cache_root=cache_root,
    )

    def _fake_fetch_bytes(url: str) -> bytes:
        if "registry.json" in url:
            return (registry_root / "registry.json").read_bytes()
        if "tasks/resting/DemoTask.py" in url:
            return task_file.read_bytes()
        raise AssertionError(f"Unexpected registry fetch URL: {url}")

    registry._fetch_bytes = _fake_fetch_bytes  # type: ignore[method-assign]

    # Before update, we should be able to skip network cleanly.
    offline_msg = registry.update_cache(allow_network=False)
    assert "Skipped online check" in offline_msg

    # Perform actual update from local file URI.
    message = registry.update_cache()
    assert "Task Library refreshed" in message
    status = registry.registry_status()
    assert status["commit"] == "local-snapshot"

    # Status before install -> not installed.
    sync_info = registry.task_sync_status("DemoTask", workspace)
    assert sync_info["status"] == "not_installed"

    # Install task -> should be copied and marked synced.
    registry.materialize_task_to("DemoTask", workspace)
    sync_info = registry.task_sync_status("DemoTask", workspace)
    assert sync_info["status"] == "synced"

    # Modify workspace copy -> status flips to customized.
    task_path = workspace / "DemoTask.py"
    task_path.write_text(task_path.read_text(encoding="utf-8") + "# tweak\n", encoding="utf-8")
    sync_info = registry.task_sync_status("DemoTask", workspace)
    assert sync_info["status"] == "modified"
