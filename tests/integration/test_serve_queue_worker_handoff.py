"""Integration tests for serve route -> queue -> worker handoff."""

from __future__ import annotations

import sys
from pathlib import Path

from autoclean.utils.ingestion import IngestionQueue, dispatch_ready_ingestion


def _write_file(path: Path, content: str = "data") -> None:
    path.write_text(content, encoding="utf-8")


def _runtime_cli_path(runtime_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        cli_path = runtime_dir / ".venv" / "Scripts" / "autocleaneeg-pipeline.exe"
    else:
        cli_path = runtime_dir / ".venv" / "bin" / "autocleaneeg-pipeline"
    cli_path.parent.mkdir(parents=True, exist_ok=True)
    _write_file(cli_path, "")
    return cli_path


def _write_single_route_config(workspace: Path, ingestion_root: Path) -> Path:
    config_path = workspace / "serve-test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: test",
                "runtime: runtimes/test",
                "automation_mode: true",
                "defaults:",
                "  automation_root: automations",
                "  workspace_name: taskfile-montage-version",
                "  file_globs: ['*.set']",
                "  sentinel_ext: .ready",
                "  recursive: true",
                "automations:",
                "  - id: sample-rest-standard-1020",
                "    taskfile: Resting",
                "    montage: standard_1020",
                "    ingestion_folders:",
                f"      - {ingestion_root}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_route_queue_worker_handoff_pending_to_processing_to_processed(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)

    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    data_file = ingestion_root / "sample-rest.set"
    _write_file(data_file)
    _write_file(ingestion_root / "sample-rest.set.ready")

    (tmp_path / "automations").mkdir()
    config_path = _write_single_route_config(tmp_path, ingestion_root)

    queue_path = tmp_path / "queue-test.json"
    queue = IngestionQueue(queue_path)
    queue.enqueue([data_file])

    assert queue.entries()[str(data_file)]["status"] == "pending"
    assert queue.entries()[str(data_file)]["route_id"] is None

    statuses_seen_during_runner: list[str] = []

    def runner(cmd: list[str]) -> None:
        _ = cmd
        reloaded = IngestionQueue(queue_path)
        statuses_seen_during_runner.append(reloaded.entries()[str(data_file)]["status"])

    results = dispatch_ready_ingestion(
        config_path=config_path,
        workspace_dir=tmp_path,
        use_watchfiles=False,
        max_events=1,
        runner=runner,
        queue=queue,
    )

    assert len(results) == 1
    assert statuses_seen_during_runner == ["processing"]

    final_entry = queue.entries()[str(data_file)]
    assert final_entry["route_id"] == "sample-rest-standard-1020"
    assert final_entry["status"] == "processed"


def test_route_queue_worker_handoff_failed_has_actionable_error(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)

    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    data_file = ingestion_root / "sample-rest.set"
    _write_file(data_file)
    _write_file(ingestion_root / "sample-rest.set.ready")

    (tmp_path / "automations").mkdir()
    config_path = _write_single_route_config(tmp_path, ingestion_root)

    queue = IngestionQueue(tmp_path / "queue-test.json")
    queue.enqueue([data_file])

    def runner(cmd: list[str]) -> None:
        _ = cmd
        raise RuntimeError("task execution failed: missing task module")

    results = dispatch_ready_ingestion(
        config_path=config_path,
        workspace_dir=tmp_path,
        use_watchfiles=False,
        max_events=1,
        runner=runner,
        queue=queue,
    )

    assert len(results) == 1
    final_entry = queue.entries()[str(data_file)]
    assert final_entry["route_id"] == "sample-rest-standard-1020"
    assert final_entry["status"] == "failed"
    assert "missing task module" in final_entry["last_error"]
