"""Tests for ingestion utilities."""

from __future__ import annotations

import sys
from pathlib import Path

from autoclean.utils.ingestion import (
    DispatchPlan,
    DispatchResult,
    IngestionDispatchResult,
    IngestionLedger,
    append_receipt_revision,
    build_dispatch_plan,
    build_process_command,
    build_receipt,
    build_workspace_name,
    compute_provenance_hash,
    dispatch_ready_ingestion,
    evaluate_readiness,
    execute_dispatch_plan,
    list_ingestion_files,
    load_receipt,
    load_serve_config,
    poll_ready_files,
    receipt_path,
    resolve_provenance_folder,
    run_dispatch_plan,
    run_ingestion_loop,
    scan_ready_files,
    stage_provenance_receipt,
    watch_ready_files,
    write_receipt,
)


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


def test_provenance_hash_deterministic(tmp_path: Path) -> None:
    relative = Path("site/subject/session")
    metadata_a = {"subject_id": "S1", "site_id": "SITE"}
    metadata_b = {"site_id": "SITE", "subject_id": "S1"}
    hash_a = compute_provenance_hash(relative, metadata_a)
    hash_b = compute_provenance_hash(relative, metadata_b)
    assert hash_a == hash_b


def test_resolve_provenance_folder(tmp_path: Path) -> None:
    root = tmp_path / "ingest"
    relative = Path("incoming/sample.set")
    metadata = {"site_id": "SITE", "subject_id": "S1"}
    folder, hash_value = resolve_provenance_folder(root, relative, metadata)
    assert folder == root / hash_value
    assert hash_value == compute_provenance_hash(relative, metadata)


def test_build_workspace_name_template() -> None:
    name = build_workspace_name(
        "taskfile-montage-version", taskfile="Task", montage="Montage"
    )
    assert name == "Task-Montage"
    with_version = build_workspace_name(
        "taskfile-montage-version",
        taskfile="Task",
        montage="Montage",
        version="v1",
    )
    assert with_version == "Task-Montage-v1"


def test_list_ingestion_files_filters_sentinels(tmp_path: Path) -> None:
    data_file = tmp_path / "sample.set"
    sentinel = tmp_path / "sample.set.ready"
    _write_file(data_file)
    _write_file(sentinel)
    files = list_ingestion_files(tmp_path, file_glob="*.set", sentinel_ext=".ready")
    assert data_file in files
    assert sentinel not in files


def test_scan_ready_files_separates_pending(tmp_path: Path) -> None:
    ready_file = tmp_path / "ready.set"
    pending_file = tmp_path / "pending.set"
    _write_file(ready_file)
    _write_file(pending_file)
    _write_file(tmp_path / "ready.set.ready")
    result = scan_ready_files([ready_file, pending_file], sentinel_ext=".ready")
    assert ready_file in result.ready_files
    assert pending_file in result.pending_files
    assert result.missing_sentinels


def test_receipt_roundtrip(tmp_path: Path) -> None:
    folder = tmp_path / "ingest"
    folder.mkdir()
    data_file = folder / "sample.set"
    _write_file(data_file)
    receipt = build_receipt(
        folder=folder,
        relative_path=Path("incoming/sample.set"),
        metadata={"site_id": "SITE", "subject_id": "S1"},
        files=[data_file],
        status="pending",
    )
    path = write_receipt(folder, receipt)
    assert path == receipt_path(folder)
    loaded = load_receipt(folder)
    assert loaded is not None
    assert loaded["status"] == "pending"
    assert len(loaded["files"]) == 1
    updated = append_receipt_revision(folder, status="ready", note="ready for dispatch")
    assert updated["status"] == "ready"
    assert len(updated["revisions"]) == 2


def test_ingestion_ledger(tmp_path: Path) -> None:
    ledger = IngestionLedger(tmp_path / "ledger.json")
    assert ledger.is_duplicate("hash") is False
    ledger.record("hash", {"path": "file.set"})
    assert ledger.is_duplicate("hash") is True


def test_stage_provenance_receipt_records_ledger(tmp_path: Path) -> None:
    root = tmp_path / "root"
    data_file = tmp_path / "sample.set"
    _write_file(data_file)
    ledger = IngestionLedger(tmp_path / "ledger.json")
    result = stage_provenance_receipt(
        root=root,
        relative_path=Path("incoming/sample.set"),
        metadata={"site_id": "SITE"},
        files=[data_file],
        status="pending",
        ledger=ledger,
    )
    assert result["folder"].exists()
    assert receipt_path(result["folder"]).exists()
    assert ledger.is_duplicate(result["hash"]) is True
    repeat = stage_provenance_receipt(
        root=root,
        relative_path=Path("incoming/sample.set"),
        metadata={"site_id": "SITE"},
        files=[data_file],
        status="pending",
        ledger=ledger,
    )
    assert repeat["duplicate"] is True


def test_build_dispatch_plan(tmp_path: Path) -> None:
    config_path = tmp_path / "serve-test.yaml"
    runtime_dir = tmp_path / "runtimes" / "test"
    automation_root = tmp_path / "automations"
    runtime_dir.mkdir(parents=True)
    automation_root.mkdir()
    config_path.write_text(
        "\n".join(
            [
                "mode: test",
                "taskfile: Resting",
                "montage: standard_1020",
                "runtime: runtimes/test",
                "automation_root: automations",
                "workspace_name: taskfile-montage-version",
                "ingestion_folders: []",
            ]
        ),
        encoding="utf-8",
    )
    config = load_serve_config(config_path)
    plan = build_dispatch_plan(
        config=config,
        workspace_dir=tmp_path,
        files=[tmp_path / "file.set"],
        version="v1",
    )
    assert isinstance(plan, DispatchPlan)
    assert plan.runtime_path == runtime_dir
    assert plan.automation_root == automation_root
    assert plan.workspace_name == "Resting-standard_1020-v1"


def test_build_process_command_taskfile(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    cli_path = _runtime_cli_path(runtime_dir)
    taskfile = tmp_path / "Task.py"
    _write_file(taskfile, "class Task: pass")
    plan = DispatchPlan(
        mode="test",
        taskfile=str(taskfile),
        montage="montage",
        runtime_path=runtime_dir,
        automation_root=tmp_path,
        workspace_name="Task-montage",
        workspace_dir=tmp_path / "Task-montage",
        files=[tmp_path / "sample.set"],
    )
    cmd = build_process_command(
        plan=plan,
        file_path=tmp_path / "sample.set",
        runtime_cli=cli_path,
    )
    assert "--task-file" in cmd
    assert str(taskfile) in cmd


def test_build_process_command_task_name(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    cli_path = _runtime_cli_path(runtime_dir)
    plan = DispatchPlan(
        mode="test",
        taskfile="Resting",
        montage="montage",
        runtime_path=runtime_dir,
        automation_root=tmp_path,
        workspace_name="Resting",
        workspace_dir=tmp_path / "Resting",
        files=[tmp_path / "sample.set"],
    )
    cmd = build_process_command(
        plan=plan,
        file_path=tmp_path / "sample.set",
        runtime_cli=cli_path,
    )
    assert "--task" in cmd
    assert "Resting" in cmd


def test_run_dispatch_plan_uses_runner(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    _runtime_cli_path(runtime_dir)
    file_a = tmp_path / "a.set"
    file_b = tmp_path / "b.set"
    _write_file(file_a)
    _write_file(file_b)
    plan = DispatchPlan(
        mode="test",
        taskfile="Resting",
        montage="montage",
        runtime_path=runtime_dir,
        automation_root=tmp_path,
        workspace_name="Resting",
        workspace_dir=tmp_path / "Resting",
        files=[file_a, file_b],
    )
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> None:
        calls.append(cmd)

    result = run_dispatch_plan(plan, runner=runner, max_attempts=1)
    assert not result.failed
    assert len(calls) == 2


def test_dispatch_ready_ingestion(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    data_file = ingestion_root / "sample.set"
    _write_file(data_file)
    _write_file(ingestion_root / "sample.set.ready")
    config_path = tmp_path / "serve-test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: test",
                "taskfile: Resting",
                "montage: standard_1020",
                "runtime: runtimes/test",
                "automation_root: automations",
                "workspace_name: taskfile-montage-version",
                "ingestion_folders:",
                "  - ingest",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "automations").mkdir()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> None:
        calls.append(cmd)

    result = dispatch_ready_ingestion(
        config_path=config_path,
        workspace_dir=tmp_path,
        ingestion_root=ingestion_root,
        file_glob="*.set",
        sentinel_ext=".ready",
        use_watchfiles=False,
        max_events=1,
        runner=runner,
    )
    assert isinstance(result, IngestionDispatchResult)
    assert result.ingestion_root == ingestion_root
    assert result.plan is not None
    assert result.result is not None
    assert len(calls) == 1


def test_run_ingestion_loop(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    data_file = ingestion_root / "sample.set"
    _write_file(data_file)
    _write_file(ingestion_root / "sample.set.ready")
    config_path = tmp_path / "serve-test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: test",
                "taskfile: Resting",
                "montage: standard_1020",
                "runtime: runtimes/test",
                "automation_root: automations",
                "workspace_name: taskfile-montage-version",
                "ingestion_folders:",
                "  - ingest",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "automations").mkdir()
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> None:
        calls.append(cmd)

    loop_result = run_ingestion_loop(
        config_path=config_path,
        workspace_dir=tmp_path,
        max_cycles=1,
        file_glob="*.set",
        sentinel_ext=".ready",
        use_watchfiles=False,
        max_events=1,
        runner=runner,
        sleep_fn=lambda _: None,
    )
    assert loop_result.iterations == 1
    assert loop_result.dispatch_results
    assert len(calls) == 1


def test_execute_dispatch_plan_retries(tmp_path: Path) -> None:
    file_ok = tmp_path / "ok.set"
    file_fail = tmp_path / "fail.set"
    _write_file(file_ok)
    _write_file(file_fail)
    plan = DispatchPlan(
        mode="test",
        taskfile="Task",
        montage="Montage",
        runtime_path=tmp_path,
        automation_root=tmp_path,
        workspace_name="Task-Montage",
        workspace_dir=tmp_path / "Task-Montage",
        files=[file_ok, file_fail],
    )
    attempts = {"fail": 0}

    def processor(path: Path, _: DispatchPlan) -> None:
        if path == file_fail and attempts["fail"] == 0:
            attempts["fail"] += 1
            raise RuntimeError("boom")

    result = execute_dispatch_plan(plan, processor=processor, max_attempts=2)
    assert isinstance(result, DispatchResult)
    assert not result.failed
    assert set(result.processed) == {file_ok, file_fail}
    assert result.attempts == 2


def test_execute_dispatch_plan_failure(tmp_path: Path) -> None:
    file_fail = tmp_path / "fail.set"
    _write_file(file_fail)
    plan = DispatchPlan(
        mode="test",
        taskfile="Task",
        montage="Montage",
        runtime_path=tmp_path,
        automation_root=tmp_path,
        workspace_name="Task-Montage",
        workspace_dir=tmp_path / "Task-Montage",
        files=[file_fail],
    )

    def processor(_: Path, __: DispatchPlan) -> None:
        raise RuntimeError("boom")

    result = execute_dispatch_plan(plan, processor=processor, max_attempts=1)
    assert file_fail in result.failed
    assert result.attempts == 1


def test_poll_ready_files(tmp_path: Path) -> None:
    data_file = tmp_path / "sample.set"
    _write_file(data_file)
    result = poll_ready_files(
        tmp_path,
        file_glob="*.set",
        sentinel_ext=".ready",
        poll_interval_seconds=0,
        max_loops=1,
        sleep_fn=lambda _: None,
    )
    assert result.ready is False
    _write_file(tmp_path / "sample.set.ready")
    ready_result = poll_ready_files(
        tmp_path,
        file_glob="*.set",
        sentinel_ext=".ready",
        poll_interval_seconds=0,
        max_loops=1,
        sleep_fn=lambda _: None,
    )
    assert ready_result.ready is True


def test_watch_ready_files_fallback(tmp_path: Path) -> None:
    data_file = tmp_path / "sample.set"
    _write_file(data_file)
    _write_file(tmp_path / "sample.set.ready")
    result = watch_ready_files(
        tmp_path,
        file_glob="*.set",
        sentinel_ext=".ready",
        poll_interval_seconds=0,
        max_events=1,
        use_watchfiles=False,
    )
    assert result.ready is True


def test_readiness_requires_sentinel(tmp_path: Path) -> None:
    data_file = tmp_path / "sample.set"
    _write_file(data_file)
    sentinel = tmp_path / "sample.set.ready"
    _write_file(sentinel, "")
    result = evaluate_readiness([data_file], sentinel_ext=".ready")
    assert result.ready is True
    result_missing = evaluate_readiness([data_file], sentinel_ext=".done")
    assert result_missing.ready is False
    assert result_missing.missing_sentinels
