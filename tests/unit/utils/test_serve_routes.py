"""Tests for route registry helpers."""

from __future__ import annotations

from pathlib import Path

import yaml


def create_route_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    base_config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automation_mode": True,
        "defaults": {
            "automation_root": "automations",
            "workspace_name": "taskfile-montage-version",
            "file_globs": ["*.set"],
            "sentinel_ext": ".ready",
            "recursive": True,
        },
        "automations": [],
    }
    (workspace / "serve-test.yaml").write_text(
        yaml.safe_dump(base_config, sort_keys=False),
        encoding="utf-8",
    )
    base_config["mode"] = "live"
    base_config["runtime"] = "runtimes/live"
    (workspace / "serve-live.yaml").write_text(
        yaml.safe_dump(base_config, sort_keys=False),
        encoding="utf-8",
    )
    return workspace


def test_upsert_route_spec_preserves_existing_values(tmp_path: Path) -> None:
    """Partial upserts should merge with the stored route spec."""
    from autoclean.utils.serve_routes import upsert_route_spec

    workspace = create_route_workspace(tmp_path)
    taskfile = tmp_path / "TaskFile.py"
    taskfile.write_text("print('ok')\n", encoding="utf-8")
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()

    _, first_spec, first_status = upsert_route_spec(
        workspace,
        "route-a",
        {
            "taskfile": str(taskfile.resolve()),
            "montage": "biosemi64",
            "ingestion_folders": [str(watch_dir.resolve())],
            "modes": ["test"],
        },
    )
    _, second_spec, second_status = upsert_route_spec(
        workspace,
        "route-a",
        {
            "priority": 10,
        },
    )

    assert first_status == "created"
    assert second_status == "updated"
    assert first_spec["taskfile"] == second_spec["taskfile"]
    assert second_spec["priority"] == 10
    assert second_spec["modes"] == ["test"]


def test_sync_route_registry_compiles_mode_specific_configs(tmp_path: Path) -> None:
    """Only matching modes should be compiled into each serve config."""
    from autoclean.utils.serve_routes import sync_route_registry, upsert_route_spec

    workspace = create_route_workspace(tmp_path)
    taskfile = tmp_path / "TaskFile.py"
    taskfile.write_text("print('ok')\n", encoding="utf-8")
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()

    upsert_route_spec(
        workspace,
        "draft-only",
        {
            "taskfile": str(taskfile.resolve()),
            "montage": "biosemi64",
            "ingestion_folders": [str(watch_dir.resolve())],
            "modes": ["test"],
            "priority": 1,
        },
    )
    upsert_route_spec(
        workspace,
        "draft-and-live",
        {
            "taskfile": str(taskfile.resolve()),
            "montage": "standard_1020",
            "ingestion_folders": [str(watch_dir.resolve())],
            "modes": ["test", "live"],
            "priority": 5,
        },
    )

    results = sync_route_registry(workspace)
    test_config = yaml.safe_load((workspace / "serve-test.yaml").read_text(encoding="utf-8"))
    live_config = yaml.safe_load((workspace / "serve-live.yaml").read_text(encoding="utf-8"))

    assert results["test"]["route_count"] == 2
    assert results["live"]["route_count"] == 1
    assert [route["id"] for route in test_config["automations"]] == [
        "draft-and-live",
        "draft-only",
    ]
    assert [route["id"] for route in live_config["automations"]] == ["draft-and-live"]
    assert "sentinel_ext" not in test_config["defaults"]
    assert "sentinel_ext" not in live_config["defaults"]


def test_archived_routes_are_excluded_from_compiled_configs(tmp_path: Path) -> None:
    """Archived routes should remain in the registry but stay out of compiled configs."""
    from autoclean.utils.serve_routes import archive_route_spec, sync_route_registry, upsert_route_spec

    workspace = create_route_workspace(tmp_path)
    taskfile = tmp_path / "TaskFile.py"
    taskfile.write_text("print('ok')\n", encoding="utf-8")
    watch_dir = tmp_path / "incoming"
    watch_dir.mkdir()

    upsert_route_spec(
        workspace,
        "retired-route",
        {
            "taskfile": str(taskfile.resolve()),
            "montage": "biosemi64",
            "ingestion_folders": [str(watch_dir.resolve())],
            "modes": ["test", "live"],
            "enabled": True,
        },
    )
    archive_route_spec(workspace, "retired-route")

    results = sync_route_registry(workspace)
    route_spec = yaml.safe_load(
        (workspace / "routes" / "retired-route.yaml").read_text(encoding="utf-8")
    )
    test_config = yaml.safe_load((workspace / "serve-test.yaml").read_text(encoding="utf-8"))
    live_config = yaml.safe_load((workspace / "serve-live.yaml").read_text(encoding="utf-8"))

    assert route_spec["archived"] is True
    assert route_spec["enabled"] is False
    assert results["test"]["route_count"] == 0
    assert results["live"]["route_count"] == 0
    assert test_config["automations"] == []
    assert live_config["automations"] == []
