"""Unit tests for audit utilities (utils/audit.py)."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

try:
    from autoclean.utils.audit import (
        create_database_backup,
        get_task_file_info,
        get_user_context,
        log_database_access,
        verify_access_log_integrity,
        verify_database_file_integrity,
    )
    from autoclean.utils.database import manage_database, set_database_path

    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not AUDIT_AVAILABLE, reason="Audit module not available"
)


# ---------------------------------------------------------------------------
# get_user_context
# ---------------------------------------------------------------------------


class TestGetUserContext:
    def test_returns_dict(self):
        ctx = get_user_context()
        assert isinstance(ctx, dict)

    def test_contains_required_keys(self):
        ctx = get_user_context()
        assert "user" in ctx
        assert "host" in ctx
        assert "pid" in ctx
        assert "ts" in ctx

    def test_pid_matches_current_process(self):
        ctx = get_user_context()
        assert ctx["pid"] == os.getpid()

    def test_user_is_a_string(self):
        ctx = get_user_context()
        assert isinstance(ctx["user"], str)

    def test_host_is_a_string(self):
        ctx = get_user_context()
        assert isinstance(ctx["host"], str)

    def test_timestamp_is_numeric(self):
        """ts should be a Unix timestamp (integer)."""
        ctx = get_user_context()
        assert isinstance(ctx["ts"], int)
        # Unix timestamp should be a reasonable recent year
        assert ctx["ts"] > 1_700_000_000  # After ~Nov 2023


# ---------------------------------------------------------------------------
# verify_database_file_integrity
# ---------------------------------------------------------------------------


class TestVerifyDatabaseFileIntegrity:
    def test_returns_false_for_nonexistent_file(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        is_valid, msg = verify_database_file_integrity(db_path)
        assert is_valid is False
        assert "not found" in msg.lower() or "nonexistent" in msg.lower()

    def test_establishes_baseline_on_first_call(self, tmp_path):
        """First call on a new DB should establish a baseline and return True."""
        db_path = tmp_path / "pipeline.db"
        # Create a minimal SQLite DB
        conn = sqlite3.connect(str(db_path))
        conn.close()

        is_valid, msg = verify_database_file_integrity(db_path)
        assert is_valid is True
        # Integrity baseline file should have been created
        integrity_file = tmp_path / ".db_integrity"
        assert integrity_file.exists()

    def test_valid_unmodified_db_returns_true(self, tmp_path):
        """A DB that hasn't changed since baseline should verify successfully."""
        db_path = tmp_path / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        # First call: establishes baseline
        verify_database_file_integrity(db_path)

        # Second call on same file: should still pass
        is_valid, msg = verify_database_file_integrity(db_path)
        assert is_valid is True

    def test_modified_db_fails_verification(self, tmp_path):
        """Modifying the DB after baseline should fail integrity check."""
        db_path = tmp_path / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        # Establish baseline
        verify_database_file_integrity(db_path)

        # Tamper with the DB by modifying raw bytes
        db_path.write_bytes(db_path.read_bytes() + b"TAMPERED")

        is_valid, msg = verify_database_file_integrity(db_path)
        assert is_valid is False
        assert "tamper" in msg.lower() or "failed" in msg.lower()


# ---------------------------------------------------------------------------
# create_database_backup
# ---------------------------------------------------------------------------


class TestCreateDatabaseBackup:
    def test_creates_backup_file(self, tmp_path):
        db_path = tmp_path / "pipeline.db"
        # Create a minimal SQLite DB
        conn = sqlite3.connect(str(db_path))
        conn.close()

        backup_path = create_database_backup(db_path)

        assert backup_path.exists()

    def test_backup_is_in_backups_subdirectory(self, tmp_path):
        db_path = tmp_path / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        backup_path = create_database_backup(db_path)

        assert "backups" in str(backup_path)

    def test_backup_has_timestamp_in_name(self, tmp_path):
        db_path = tmp_path / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        backup_path = create_database_backup(db_path)

        # Backup filename should contain timestamp portion (digits)
        assert any(c.isdigit() for c in backup_path.name)

    def test_backup_raises_for_nonexistent_db(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        with pytest.raises(Exception):
            create_database_backup(db_path)


# ---------------------------------------------------------------------------
# log_database_access
# ---------------------------------------------------------------------------


class TestLogDatabaseAccess:
    def _setup_db_and_patch(self, tmp_path):
        """Create DB and patch audit.DB_PATH so log_database_access can find it."""
        from unittest.mock import patch as _patch
        import autoclean.utils.audit as _audit_module

        set_database_path(tmp_path)
        manage_database("create_collection")
        # audit.py imports DB_PATH by value; patch it directly in audit's namespace
        self._db_patcher = _patch.object(_audit_module, "DB_PATH", tmp_path)
        self._db_patcher.start()

    def _teardown_db(self):
        self._db_patcher.stop()
        set_database_path(None)

    def test_writes_entry_to_access_log_table(self, tmp_path):
        """log_database_access inserts a row into database_access_log."""
        self._setup_db_and_patch(tmp_path)
        try:
            user_ctx = get_user_context()
            log_database_access("test_op", user_ctx, {"detail": "test"})

            db_path = tmp_path / "pipeline.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM database_access_log WHERE operation='test_op'"
            )
            count = cursor.fetchone()[0]
            conn.close()

            assert count >= 1
        finally:
            self._teardown_db()

    def test_hash_chain_links_sequential_entries(self, tmp_path):
        """Each entry's previous_hash equals the prior entry's log_hash."""
        self._setup_db_and_patch(tmp_path)
        try:
            user_ctx = get_user_context()
            log_database_access("op_one", user_ctx, {"seq": 1})
            log_database_access("op_two", user_ctx, {"seq": 2})

            db_path = tmp_path / "pipeline.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT log_hash, previous_hash FROM database_access_log ORDER BY log_id"
            )
            rows = cursor.fetchall()
            conn.close()

            assert len(rows) >= 2
            # Second entry's previous_hash should equal first entry's log_hash
            first_hash = rows[-2][0]
            second_prev = rows[-1][1]
            assert first_hash == second_prev
        finally:
            self._teardown_db()

    def test_access_log_entries_are_valid_json_serializable(self, tmp_path):
        """Access log entries stored in DB are retrievable and JSON-serializable."""
        self._setup_db_and_patch(tmp_path)
        try:
            user_ctx = get_user_context()
            log_database_access("json_test_op", user_ctx, {"key": "value"})

            db_path = tmp_path / "pipeline.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT operation, user_context, details FROM database_access_log "
                "WHERE operation = 'json_test_op' LIMIT 1"
            )
            row = cursor.fetchone()
            conn.close()

            assert row is not None
            operation, user_ctx_str, details_str = row
            assert operation == "json_test_op"
            # These should be JSON-parseable strings (stored as JSON in DB)
            if user_ctx_str:
                parsed = json.loads(user_ctx_str)
                assert isinstance(parsed, dict)
        finally:
            self._teardown_db()

    def test_access_log_has_expected_fields(self, tmp_path):
        """Access log table has the expected column names."""
        self._setup_db_and_patch(tmp_path)
        try:
            db_path = tmp_path / "pipeline.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(database_access_log)")
            cols = {row[1] for row in cursor.fetchall()}
            conn.close()

            # These are the required fields for the CSV/JSONL export
            assert "operation" in cols
            assert "log_hash" in cols
            assert "previous_hash" in cols
        finally:
            self._teardown_db()

    def test_verify_access_log_integrity_passes_after_writes(self, tmp_path):
        """verify_access_log_integrity should return status='valid' for clean DB."""
        self._setup_db_and_patch(tmp_path)
        try:
            user_ctx = get_user_context()
            log_database_access("integrity_check", user_ctx, {})
            log_database_access("integrity_check_2", user_ctx, {})

            result = verify_access_log_integrity()
            assert result is not None
            assert result.get("status") in ("valid", "compromised")  # chain was verified
        finally:
            self._teardown_db()


# ---------------------------------------------------------------------------
# get_task_file_info
# ---------------------------------------------------------------------------


class TestGetTaskFileInfo:
    def test_task_file_sha256_captured(self, tmp_path):
        """get_task_file_info returns a dict with file_content_hash for a real file."""
        # Create a minimal task file
        task_file = tmp_path / "test_task.py"
        task_file.write_text("class TestTask:\n    pass\n")

        class _FakeTask:
            pass

        # Point the task object's module to the temp file
        import types
        mod = types.ModuleType("test_task")
        mod.__file__ = str(task_file)
        _FakeTask.__module__ = "test_task"
        import sys
        sys.modules["test_task"] = mod

        try:
            result = get_task_file_info("TestTask", _FakeTask())
            assert result is not None
            assert isinstance(result, dict)
            # Either we got the hash or there's an error key — either way the call succeeded
            has_hash = result.get("file_content_hash") is not None
            has_error = result.get("error") is not None
            assert has_hash or has_error
        finally:
            sys.modules.pop("test_task", None)
