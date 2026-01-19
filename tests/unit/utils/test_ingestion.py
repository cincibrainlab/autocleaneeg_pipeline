"""Tests for ingestion utilities."""

from __future__ import annotations

from pathlib import Path

from autoclean.utils.ingestion import (
    IngestionLedger,
    append_receipt_revision,
    build_receipt,
    compute_provenance_hash,
    evaluate_readiness,
    load_receipt,
    receipt_path,
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
