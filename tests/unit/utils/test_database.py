"""Unit tests for database utilities (utils/database.py)."""

import sqlite3
from pathlib import Path

import pytest

try:
    from autoclean.utils.database import (
        DatabaseError,
        RecordNotFoundError,
        _create_isolated_schema,
        _serialize_for_json,
        manage_database,
        merge_reprocess_database,
        set_database_path,
    )

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not DATABASE_AVAILABLE, reason="Database module not available"
)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Each test gets its own empty database directory."""
    set_database_path(tmp_path)
    manage_database("create_collection")
    yield tmp_path
    # Cleanup: reset DB path so other tests aren't affected
    set_database_path(None)


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------


class TestDatabaseSchemaCreation:
    def test_create_collection_creates_pipeline_runs_table(self, isolated_db):
        db_path = isolated_db / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
        )
        result = cursor.fetchone()
        conn.close()
        assert result is not None

    def test_create_collection_creates_update_audit_log_table(self, isolated_db):
        db_path = isolated_db / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='update_audit_log'"
        )
        result = cursor.fetchone()
        conn.close()
        assert result is not None

    def test_create_collection_creates_database_access_log_table(self, isolated_db):
        db_path = isolated_db / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='database_access_log'"
        )
        result = cursor.fetchone()
        conn.close()
        assert result is not None


# ---------------------------------------------------------------------------
# Store and retrieve records
# ---------------------------------------------------------------------------


class TestStoreAndRetrieve:
    def test_store_run_record_and_retrieve_by_run_id(self, isolated_db):
        record = {
            "run_id": "test_run_001",
            "task": "TestTask",
            "unprocessed_file": "/path/to/file.fif",
            "status": "processing",
        }
        manage_database("store", run_record=record)

        retrieved = manage_database(
            "get_record", run_record={"run_id": "test_run_001"}
        )
        assert retrieved is not None
        assert retrieved["run_id"] == "test_run_001"
        assert retrieved["task"] == "TestTask"

    def test_get_record_raises_for_unknown_run_id(self, isolated_db):
        """Querying a non-existent run_id raises DatabaseError."""
        with pytest.raises(DatabaseError):
            manage_database(
                "get_record", run_record={"run_id": "nonexistent_id"}
            )

    def test_store_multiple_runs_all_retrievable(self, isolated_db):
        for i in range(3):
            manage_database(
                "store",
                run_record={
                    "run_id": f"run_{i}",
                    "task": "TestTask",
                    "unprocessed_file": f"/path/to/file_{i}.fif",
                    "status": "processing",
                },
            )

        for i in range(3):
            retrieved = manage_database(
                "get_record", run_record={"run_id": f"run_{i}"}
            )
            assert retrieved is not None
            assert retrieved["run_id"] == f"run_{i}"

    def test_get_collection_returns_all_stored_runs(self, isolated_db):
        for i in range(3):
            manage_database(
                "store",
                run_record={
                    "run_id": f"batch_run_{i}",
                    "task": "TestTask",
                    "unprocessed_file": f"/path/to/file_{i}.fif",
                    "status": "processing",
                },
            )

        collection = manage_database("get_collection")
        assert collection is not None
        assert len(collection) >= 3


# ---------------------------------------------------------------------------
# Update operations
# ---------------------------------------------------------------------------


class TestUpdateOperations:
    def test_update_status_changes_run_status(self, isolated_db):
        manage_database(
            "store",
            run_record={
                "run_id": "update_test_001",
                "task": "TestTask",
                "unprocessed_file": "/path.fif",
                "status": "processing",
            },
        )
        manage_database(
            "update_status",
            update_record={"run_id": "update_test_001", "status": "completed"},
        )

        retrieved = manage_database(
            "get_record", run_record={"run_id": "update_test_001"}
        )
        assert "completed" in retrieved["status"]


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


class TestSerializeForJson:
    def test_path_objects_serialized_to_string(self):
        result = _serialize_for_json(Path("/some/path"))
        assert isinstance(result, str)
        assert result == "/some/path"

    def test_dict_with_path_values_serialized(self):
        obj = {"output": Path("/some/output"), "name": "test"}
        result = _serialize_for_json(obj)
        assert result["output"] == "/some/output"
        assert result["name"] == "test"

    def test_list_with_mixed_types_serialized(self):
        obj = [1, "two", Path("/three"), {"key": Path("/val")}]
        result = _serialize_for_json(obj)
        assert result[2] == "/three"
        assert result[3]["key"] == "/val"

    def test_plain_types_pass_through_unchanged(self):
        assert _serialize_for_json(42) == 42
        assert _serialize_for_json("hello") == "hello"
        assert _serialize_for_json(3.14) == 3.14
        assert _serialize_for_json(None) is None


# ---------------------------------------------------------------------------
# Data integrity and failure handling
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    def test_run_record_can_be_marked_failed(self, isolated_db):
        """update_status to 'failed' is persisted and readable."""
        manage_database(
            "store",
            run_record={
                "run_id": "fail_test_001",
                "task": "TestTask",
                "unprocessed_file": "/path.fif",
                "status": "processing",
            },
        )
        manage_database(
            "update_status",
            update_record={"run_id": "fail_test_001", "status": "failed"},
        )
        retrieved = manage_database("get_record", run_record={"run_id": "fail_test_001"})
        assert "failed" in retrieved["status"]

    def test_sql_trigger_prevents_completed_run_modification(self, isolated_db):
        """Trigger blocks UPDATE on rows with status 'completed' or 'failed'."""
        db_path = isolated_db / "pipeline.db"

        # Insert a row with status='completed' directly via SQL
        # (bypassing manage_database which appends a timestamp to status)
        from datetime import datetime
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, created_at, task, status) VALUES (?, ?, ?, ?)",
            ("trigger_test_001", datetime.now().isoformat(), "TestTask", "completed"),
        )
        conn.commit()

        # Attempt to UPDATE this completed row — trigger should fire
        with pytest.raises((sqlite3.OperationalError, sqlite3.IntegrityError)):
            conn.execute(
                "UPDATE pipeline_runs SET status = 'reprocessing' WHERE run_id = 'trigger_test_001'"
            )
            conn.commit()
        conn.close()

    def test_concurrent_writes_do_not_corrupt(self, isolated_db):
        """Two threads writing different run IDs simultaneously both succeed."""
        import threading

        errors = []

        def write_run(run_id):
            try:
                manage_database(
                    "store",
                    run_record={
                        "run_id": run_id,
                        "task": "TestTask",
                        "unprocessed_file": "/path.fif",
                        "status": "processing",
                    },
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=write_run, args=(f"concurrent_{i}",))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes failed: {errors}"

        # Verify all records are present
        for i in range(4):
            record = manage_database(
                "get_record", run_record={"run_id": f"concurrent_{i}"}
            )
            assert record is not None


def _create_run_database(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        _create_isolated_schema(conn)
    finally:
        conn.close()


def test_merge_reprocess_database_reassigns_update_audit_log_ids(tmp_path: Path):
    original_db_path = tmp_path / "original.db"
    reprocess_db_path = tmp_path / "reprocess.db"
    _create_run_database(original_db_path)
    _create_run_database(reprocess_db_path)

    original_conn = sqlite3.connect(str(original_db_path))
    reprocess_conn = sqlite3.connect(str(reprocess_db_path))

    try:
        original_conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, created_at, task, unprocessed_file, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "original-run-001",
                "2026-04-06T10:00:00",
                "RestingState",
                "/tmp/subject01_comp_epo.set",
                "completed",
            ),
        )
        original_conn.execute(
            """
            INSERT INTO update_audit_log (
                id, run_id, timestamp, old_status, new_status, operation_type, user_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "original-run-001",
                "2026-04-06T10:01:00",
                "processing",
                "completed",
                "status_change",
                "{}",
            ),
        )
        original_conn.execute(
            """
            INSERT INTO database_access_log (
                log_id, timestamp, operation, user_context, details, log_hash, previous_hash, auth0_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "2026-04-06T10:00:00",
                "isolated_database_creation",
                "{}",
                "{}",
                "original-genesis-hash",
                "genesis_hash_empty_log",
                None,
            ),
        )
        original_conn.commit()

        reprocess_conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, created_at, task, unprocessed_file, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "reprocess-run-001",
                "2026-04-06T11:00:00",
                "RestingState",
                "/tmp/reprocess/subject01_comp_epo.set",
                "completed",
            ),
        )
        reprocess_conn.execute(
            """
            INSERT INTO update_audit_log (
                id, run_id, timestamp, old_status, new_status, operation_type, user_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "reprocess-run-001",
                "2026-04-06T11:01:00",
                "processing",
                "completed",
                "status_change",
                "{}",
            ),
        )
        reprocess_conn.execute(
            """
            INSERT INTO database_access_log (
                log_id, timestamp, operation, user_context, details, log_hash, previous_hash, auth0_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "2026-04-06T11:00:00",
                "isolated_database_creation",
                "{}",
                "{}",
                "reprocess-genesis-hash",
                "genesis_hash_empty_log",
                None,
            ),
        )
        reprocess_conn.execute(
            """
            INSERT INTO database_access_log (
                log_id, timestamp, operation, user_context, details, log_hash, previous_hash, auth0_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                2,
                "2026-04-06T11:01:00",
                "merge_candidate",
                "{}",
                '{"run_id":"reprocess-run-001"}',
                "stale-hash",
                "reprocess-genesis-hash",
                None,
            ),
        )
        reprocess_conn.commit()
    finally:
        original_conn.close()
        reprocess_conn.close()

    original_run_id, reprocess_run_id = merge_reprocess_database(
        original_db_path=original_db_path,
        reprocess_db_path=reprocess_db_path,
        stem="subject01_comp_epo",
    )

    assert original_run_id == "original-run-001"
    assert reprocess_run_id == "reprocess-run-001"

    merged_conn = sqlite3.connect(str(original_db_path))
    try:
        audit_rows = merged_conn.execute(
            """
            SELECT id, run_id, old_status, new_status
            FROM update_audit_log
            ORDER BY id
            """
        ).fetchall()
        assert len(audit_rows) == 2
        assert audit_rows[0] == (1, "original-run-001", "processing", "completed")
        assert audit_rows[1] == (2, "reprocess-run-001", "processing", "completed")

        merged_run = merged_conn.execute(
            """
            SELECT run_id, supersedes_run_id
            FROM pipeline_runs
            WHERE run_id = ?
            """,
            ("reprocess-run-001",),
        ).fetchone()
        assert merged_run == ("reprocess-run-001", "original-run-001")
    finally:
        merged_conn.close()
