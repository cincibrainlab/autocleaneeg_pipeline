from __future__ import annotations

from datetime import datetime
from pathlib import Path

from autoclean.tui.v2_app import ActivityEvent
from autoclean.tui.v2_state import build_workspace_snapshot


def test_snapshot_recommends_first_route_when_none(tmp_path: Path) -> None:
    snapshot = build_workspace_snapshot(
        workspace_dir=tmp_path,
        mode="test",
        config_valid=True,
        config_errors=[],
        config_warnings=[],
        route_specs=[],
        queue_entries={},
        service_snapshot={
            "lane": "Draft",
            "workspace": str(tmp_path),
            "queue_path": str(tmp_path / "queue-test.json"),
            "config_source": "operator",
            "config_path": str(tmp_path / "serve-test.yaml"),
            "log_path": str(tmp_path / "serve-test.log"),
            "command": "Not started yet",
            "pid": None,
            "uptime": None,
        },
        activity_log=[],
        operator_config_path=tmp_path / "serve-test.yaml",
        deployed_config_path=tmp_path / "deploy" / "serve-test.yaml",
        config_source="operator",
    )

    assert snapshot.recommended_action.key == "create_route"
    assert snapshot.recommended_action.target_tab == "tab-routes"


def test_snapshot_prioritizes_failed_queue_over_deploy(tmp_path: Path) -> None:
    operator = tmp_path / "serve-test.yaml"
    deployed = tmp_path / "deploy" / "serve-test.yaml"
    operator.write_text("mode: test\n", encoding="utf-8")
    deployed.parent.mkdir(parents=True, exist_ok=True)
    deployed.write_text("mode: test\nautomation_mode: true\n", encoding="utf-8")

    route_task = tmp_path / "Task.py"
    route_task.write_text("print('ok')\n", encoding="utf-8")
    incoming = tmp_path / "incoming"
    incoming.mkdir()

    snapshot = build_workspace_snapshot(
        workspace_dir=tmp_path,
        mode="test",
        config_valid=True,
        config_errors=[],
        config_warnings=[],
        route_specs=[
            {
                "id": "resting-biosemi64",
                "taskfile": str(route_task),
                "montage": "biosemi64",
                "ingestion_folders": [str(incoming)],
                "file_globs": ["*.set"],
                "modes": ["test"],
                "enabled": True,
            }
        ],
        queue_entries={
            str(tmp_path / "bad.set"): {
                "status": "failed",
                "route_id": "resting-biosemi64",
                "added_at": "2026-03-11T10:00:00",
                "failed_at": "2026-03-11T10:10:00",
                "last_error": "bad file",
            }
        },
        service_snapshot={
            "lane": "Draft",
            "workspace": str(tmp_path),
            "queue_path": str(tmp_path / "queue-test.json"),
            "config_source": "operator",
            "config_path": str(operator),
            "log_path": str(tmp_path / "serve-test.log"),
            "command": "Not started yet",
            "pid": None,
            "uptime": None,
        },
        activity_log=[
            ActivityEvent(
                timestamp=datetime(2026, 3, 11, 10, 11, 0),
                event_type="queue_fail",
                message="bad.set failed",
            )
        ],
        operator_config_path=operator,
        deployed_config_path=deployed,
        config_source="operator",
    )

    assert snapshot.queue.failed == 1
    assert snapshot.recommended_action.key == "review_failed"
    assert snapshot.queue_items[0].status == "failed"


def test_snapshot_recommends_start_service_when_ready(tmp_path: Path) -> None:
    operator = tmp_path / "serve-live.yaml"
    operator.write_text("mode: live\n", encoding="utf-8")
    deployed = tmp_path / "deploy" / "serve-live.yaml"
    deployed.parent.mkdir(parents=True, exist_ok=True)
    deployed.write_text(operator.read_text(encoding="utf-8"), encoding="utf-8")
    route_task = tmp_path / "Task.py"
    route_task.write_text("print('ok')\n", encoding="utf-8")
    incoming = tmp_path / "incoming"
    incoming.mkdir()

    snapshot = build_workspace_snapshot(
        workspace_dir=tmp_path,
        mode="live",
        config_valid=True,
        config_errors=[],
        config_warnings=[],
        route_specs=[
            {
                "id": "resting-biosemi64",
                "taskfile": str(route_task),
                "montage": "biosemi64",
                "ingestion_folders": [str(incoming)],
                "file_globs": ["*.set"],
                "modes": ["test", "live"],
                "enabled": True,
            }
        ],
        queue_entries={},
        service_snapshot={
            "lane": "Production",
            "workspace": str(tmp_path),
            "queue_path": str(tmp_path / "queue-live.json"),
            "config_source": "deployed",
            "config_path": str(deployed),
            "log_path": str(tmp_path / "serve-live.log"),
            "command": "autocleaneeg-pipeline serve run --mode live",
            "pid": None,
            "uptime": None,
        },
        activity_log=[],
        operator_config_path=operator,
        deployed_config_path=deployed,
        config_source="deployed",
    )

    assert snapshot.recommended_action.key == "start_service"
    assert snapshot.recommended_action.direct_action == "start_service"
