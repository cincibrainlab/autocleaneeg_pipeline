"""Tests for ingestion utilities."""

from __future__ import annotations

from pathlib import Path

from autoclean.utils.ingestion import (
    IngestionLedger,
    append_receipt_revision,
    build_receipt,
    compute_provenance_hash,
    evaluate_readiness,
    list_ingestion_files,
    load_receipt,
    poll_ready_files,
    receipt_path,
    resolve_provenance_folder,
    scan_ready_files,
    stage_provenance_receipt,
    watch_ready_files,
    write_receipt,
)


def _write_file(path: Path, content: str = "data") -> None:
    path.write_text(content, encoding="utf-8")


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
