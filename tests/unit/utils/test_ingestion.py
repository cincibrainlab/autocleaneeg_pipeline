"""Tests for ingestion utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autoclean.utils.ingestion import (
    DispatchPlan,
    DispatchResult,
    IngestionDispatchResult,
    IngestionLedger,
    IngestionQueue,
    IngestionServiceResult,
    ServeConfigError,
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
    parse_serve_config,
    poll_ready_files,
    receipt_path,
    resolve_provenance_folder,
    run_dispatch_plan,
    run_ingestion_loop,
    run_ingestion_service,
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


def test_list_ingestion_files_non_recursive(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    root_file = tmp_path / "root.set"
    nested_file = nested / "nested.set"
    _write_file(root_file)
    _write_file(nested_file)
    files = list_ingestion_files(
        tmp_path, file_glob="*.set", sentinel_ext=".ready", recursive=False
    )
    assert root_file in files
    assert nested_file not in files


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


def test_ingestion_queue_persistence(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue = IngestionQueue(queue_path)
    file_a = tmp_path / "a.set"
    file_b = tmp_path / "b.set"
    _write_file(file_a)
    _write_file(file_b)
    queue.enqueue([file_a, file_b])
    assert len(queue.pending()) == 2
    reloaded = IngestionQueue(queue_path)
    assert len(reloaded.pending()) == 2


def test_ingestion_queue_route_metadata(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue = IngestionQueue(queue_path)
    file_a = tmp_path / "a.set"
    _write_file(file_a)
    queue.enqueue([file_a], route_id="route-a", ingestion_root=tmp_path)
    entry = queue.entries()[str(file_a)]
    assert entry["route_id"] == "route-a"
    assert entry["ingestion_root"] == str(tmp_path)


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


def test_parse_serve_config_defaults(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    automation_root = tmp_path / "automations"
    automation_root.mkdir()
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config_path = tmp_path / "serve-test.yaml"
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
                "  - taskfile: Resting",
                "    montage: standard_1020",
                "    version: v1",
                "    ingestion_folders:",
                "      - ingest",
            ]
        ),
        encoding="utf-8",
    )
    config = load_serve_config(config_path)
    serve_config, warnings = parse_serve_config(config, tmp_path, strict=False)
    assert not warnings
    assert serve_config.mode == "test"
    route = serve_config.routes[0]
    assert route.file_globs == ["*.set"]
    assert route.recursive is True
    assert route.automation_root == automation_root
    assert route.id == "Resting-standard_1020-v1"


def test_parse_serve_config_strict_requires_ingestion_folders_exist(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    automation_root = tmp_path / "automations"
    automation_root.mkdir()
    config_path = tmp_path / "serve-test.yaml"
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
                "  - taskfile: Resting",
                "    montage: standard_1020",
                "    ingestion_folders:",
                "      - missing-root",
            ]
        ),
        encoding="utf-8",
    )
    config = load_serve_config(config_path)
    with pytest.raises(ServeConfigError, match="ingestion_folders"):
        parse_serve_config(config, tmp_path, strict=True)


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
    queue = IngestionQueue(tmp_path / "queue.json")

    def runner(cmd: list[str]) -> None:
        calls.append(cmd)

    results = dispatch_ready_ingestion(
        config_path=config_path,
        workspace_dir=tmp_path,
        file_glob="*.set",
        sentinel_ext=".ready",
        use_watchfiles=False,
        max_events=1,
        runner=runner,
        queue=queue,
    )
    assert len(results) == 1
    result = results[0]
    assert isinstance(result, IngestionDispatchResult)
    assert result.route_id
    assert ingestion_root in result.ingestion_roots
    assert result.plan is not None
    assert result.result is not None
    assert len(calls) == 1
    assert queue.entries()[str(data_file)]["status"] == "processed"


def test_dispatch_ready_ingestion_marks_unroutable_pending_entries_failed(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    (tmp_path / "automations").mkdir()
    config_path = tmp_path / "serve-test.yaml"
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
                "  - taskfile: Resting",
                "    montage: standard_1020",
                "    ingestion_folders:",
                "      - ingest",
            ]
        ),
        encoding="utf-8",
    )
    queue = IngestionQueue(tmp_path / "queue.json")
    orphan = tmp_path / "orphan.set"
    _write_file(orphan)
    queue.enqueue([orphan])

    dispatch_ready_ingestion(
        config_path=config_path,
        workspace_dir=tmp_path,
        use_watchfiles=False,
        max_events=1,
        queue=queue,
    )

    entry = queue.entries()[str(orphan)]
    assert entry["status"] == "failed"
    assert "Unable to resolve route for queued file" in entry["last_error"]


def test_dispatch_ready_ingestion_route_priority(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    data_file = ingestion_root / "sample.set"
    _write_file(data_file)
    _write_file(ingestion_root / "sample.set.ready")
    (tmp_path / "automations").mkdir()
    config_path = tmp_path / "serve-test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mode: test",
                "runtime: runtimes/test",
                "defaults:",
                "  automation_root: automations",
                "  workspace_name: taskfile-montage-version",
                "  file_globs: ['*.set']",
                "  sentinel_ext: .ready",
                "  recursive: true",
                "automations:",
                "  - id: low",
                "    priority: 1",
                "    taskfile: RestingLow",
                "    montage: standard_1020",
                "    ingestion_folders:",
                "      - ingest",
                "  - id: high",
                "    priority: 10",
                "    taskfile: RestingHigh",
                "    montage: standard_1020",
                "    ingestion_folders:",
                "      - ingest",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def runner(cmd: list[str]) -> None:
        calls.append(cmd)

    results = dispatch_ready_ingestion(
        config_path=config_path,
        workspace_dir=tmp_path,
        use_watchfiles=False,
        max_events=1,
        runner=runner,
    )
    result_map = {result.route_id: result for result in results}
    assert data_file in result_map["high"].ready.ready_files
    assert not result_map["low"].ready.ready_files
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
    assert loop_result.dispatch_results[0].route_id
    assert len(calls) == 1


def test_run_ingestion_service_idle(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtimes" / "test"
    _runtime_cli_path(runtime_dir)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    data_file = ingestion_root / "sample.set"
    _write_file(data_file)
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

    result = run_ingestion_service(
        config_path=config_path,
        workspace_dir=tmp_path,
        max_cycles=2,
        idle_limit=1,
        file_glob="*.set",
        sentinel_ext=".ready",
        use_watchfiles=False,
        max_events=1,
        runner=runner,
        sleep_fn=lambda _: None,
    )
    assert isinstance(result, IngestionServiceResult)
    assert result.cycles == 1
    assert result.idle_cycles == 1
    assert not calls


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


# --- Non-strict validation tests ---


def test_parse_serve_config_non_strict_missing_file_globs(tmp_path: Path) -> None:
    """file_globs defaults to ['*'] with warning in non-strict mode."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    automation_root = tmp_path / "automations"
    automation_root.mkdir()
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automation_mode": True,
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "automation_root": "automations",
                "workspace_name": "taskfile-montage",
                "ingestion_folders": ["ingest"],
                # file_globs intentionally missing
            }
        ],
    }
    serve_config, warnings = parse_serve_config(config, tmp_path, strict=False)
    assert any("file_globs is empty" in w for w in warnings)
    assert serve_config.routes[0].file_globs == ["*"]


def test_parse_serve_config_strict_missing_file_globs(tmp_path: Path) -> None:
    """file_globs is required error in strict mode."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    automation_root = tmp_path / "automations"
    automation_root.mkdir()
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "automation_root": "automations",
                "workspace_name": "taskfile-montage",
                "ingestion_folders": ["ingest"],
            }
        ],
    }
    with pytest.raises(ServeConfigError, match="file_globs is required"):
        parse_serve_config(config, tmp_path, strict=True)


def test_parse_serve_config_non_strict_missing_automation_root(tmp_path: Path) -> None:
    """automation_root generates warning in non-strict mode."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "workspace_name": "taskfile-montage",
                "file_globs": ["*.set"],
                "ingestion_folders": ["ingest"],
                # automation_root intentionally missing
            }
        ],
    }
    serve_config, warnings = parse_serve_config(config, tmp_path, strict=False)
    assert any("automation_root is empty" in w for w in warnings)
    # Falls back to workspace_dir when empty
    assert serve_config.routes[0].automation_root == tmp_path


def test_parse_serve_config_strict_missing_automation_root(tmp_path: Path) -> None:
    """automation_root is required error in strict mode."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "workspace_name": "taskfile-montage",
                "file_globs": ["*.set"],
                "ingestion_folders": ["ingest"],
            }
        ],
    }
    with pytest.raises(ServeConfigError, match="automation_root is required"):
        parse_serve_config(config, tmp_path, strict=True)


def test_parse_serve_config_non_strict_missing_workspace_name(tmp_path: Path) -> None:
    """workspace_name generates warning and uses default in non-strict mode."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    automation_root = tmp_path / "automations"
    automation_root.mkdir()
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "automation_root": "automations",
                "file_globs": ["*.set"],
                "ingestion_folders": ["ingest"],
                # workspace_name intentionally missing
            }
        ],
    }
    serve_config, warnings = parse_serve_config(config, tmp_path, strict=False)
    assert any("workspace_name is empty" in w for w in warnings)
    # Uses default template
    assert serve_config.routes[0].workspace_name == "Resting-standard_1020"


def test_parse_serve_config_strict_missing_workspace_name(tmp_path: Path) -> None:
    """workspace_name is required error in strict mode."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    automation_root = tmp_path / "automations"
    automation_root.mkdir()
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "automation_root": "automations",
                "file_globs": ["*.set"],
                "ingestion_folders": ["ingest"],
            }
        ],
    }
    with pytest.raises(ServeConfigError, match="workspace_name is required"):
        parse_serve_config(config, tmp_path, strict=True)


def test_parse_serve_config_non_strict_missing_path_skips_existence(
    tmp_path: Path,
) -> None:
    """Non-strict mode skips path existence checks for automation_root."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "automation_root": "missing-dir",  # doesn't exist
                "workspace_name": "taskfile-montage",
                "file_globs": ["*.set"],
                "ingestion_folders": ["ingest"],
            }
        ],
    }
    # Should NOT raise even though automation_root doesn't exist
    serve_config, warnings = parse_serve_config(config, tmp_path, strict=False)
    assert serve_config.routes[0].automation_root == tmp_path / "missing-dir"


def test_parse_serve_config_strict_missing_path_raises(tmp_path: Path) -> None:
    """Strict mode raises for missing automation_root path."""
    runtime_dir = tmp_path / "runtimes" / "test"
    runtime_dir.mkdir(parents=True)
    ingestion_root = tmp_path / "ingest"
    ingestion_root.mkdir()
    config = {
        "mode": "test",
        "runtime": "runtimes/test",
        "automations": [
            {
                "taskfile": "Resting",
                "montage": "standard_1020",
                "automation_root": "missing-dir",
                "workspace_name": "taskfile-montage",
                "file_globs": ["*.set"],
                "ingestion_folders": ["ingest"],
            }
        ],
    }
    with pytest.raises(ServeConfigError, match="Automation root not found"):
        parse_serve_config(config, tmp_path, strict=True)
