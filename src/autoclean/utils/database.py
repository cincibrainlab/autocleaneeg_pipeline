# src/autoclean/utils/database.py
"""Database utilities for the autoclean package using SQLite."""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from autoclean.utils.logging import message

# Global lock for thread safety
_db_lock = threading.Lock()

# Global database path
DB_PATH = None


def set_database_path(path: Path) -> None:
    """Set the global database path.

    Parameters
    ----------
    path : Path
        The path to the autoclean directory.
    """
    global DB_PATH  # pylint: disable=global-statement
    DB_PATH = path


class DatabaseError(Exception):
    """Custom exception for database operations."""

    def __init__(self, error_message: str):
        self.message = error_message
        super().__init__(self.message)


class RecordNotFoundError(Exception):
    """Custom exception for when a database record is not found."""

    def __init__(self, error_message: str):
        self.message = error_message
        super().__init__(self.message)


def _serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable object.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        # Preserve all dictionary keys and values, converting non-serializable values to strings
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def get_run_record(run_id: str) -> dict:
    """Get a run record from the database by run ID.

    Parameters
    ----------
    run_id : str
        The string ID of the run to retrieve.

    Returns
    -------
    run_record : dict
        The run record if found, None if not found
    """
    run_record = manage_database(operation="get_record", run_record={"run_id": run_id})
    return run_record


def manage_database_with_audit_protection(
    operation: str,
    run_record: Optional[Dict[str, Any]] = None,
    update_record: Optional[Dict[str, Any]] = None,
) -> Any:
    """Enhanced database management with audit protection and logging.

    This wrapper adds audit protection features to the standard database operations:
    - Access logging for compliance tracking
    - Database integrity verification
    - Automatic backup creation
    - User context tracking

    Parameters
    ----------
    operation : str
        Database operation type (same as manage_database)
    run_record : dict, optional
        Record data for operations
    update_record : dict, optional
        Update data for operations

    Returns
    -------
    Any
        Operation result from underlying manage_database call
    """
    # Import audit functions (avoid circular imports)
    from autoclean.utils.audit import (
        create_database_backup,
        get_user_context,
        log_database_access,
    )

    user_ctx = get_user_context()

    # Log database access attempt
    access_details = {
        "operation": operation,
        "run_id": (
            run_record.get("run_id")
            if run_record
            else update_record.get("run_id") if update_record else None
        ),
    }
    log_database_access(f"{operation}_attempt", user_ctx, access_details)

    # Database integrity is protected by SQLite triggers, not file-level hashing
    # The triggers prevent unauthorized modifications much more effectively than file hashes

    # Perform the actual database operation
    try:
        result = manage_database(operation, run_record, update_record)

        # Log successful operation
        log_database_access(
            f"{operation}_completed", user_ctx, {"result": str(result)[:200]}
        )

        # Create backup and update integrity baseline after significant operations
        if operation in ["create_collection", "store"] and DB_PATH:
            try:
                create_database_backup(DB_PATH / "pipeline.db")
            except Exception as backup_error:
                message("warning", f"Backup creation failed: {backup_error}")
                log_database_access(
                    "backup_failed", user_ctx, {"error": str(backup_error)}
                )

            # Database integrity is maintained by triggers, not file hashes

        return result

    except Exception as e:
        # Log failed operation
        log_database_access(f"{operation}_failed", user_ctx, {"error": str(e)})
        raise


def _validate_metadata(metadata: dict) -> bool:
    """Validates metadata structure and types.

    Parameters
    ----------
    metadata : dict
        The metadata to validate.

    Returns
    -------
    bool
        True if the metadata is valid, False otherwise.
    """
    if not isinstance(metadata, dict):
        return False
    return all(isinstance(k, str) for k in metadata.keys())


def _get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get a database connection with proper configuration.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.

    Returns
    -------
    sqlite3.Connection
        Configured database connection.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable row factory for dict-like access
    return conn


def manage_database(
    operation: str,
    run_record: Optional[Dict[str, Any]] = None,
    update_record: Optional[Dict[str, Any]] = None,
) -> Any:
    """Manage database operations with thread safety.

    Parameters
    ----------
    operation : str
        Operations can be:

        - **create_collection**: Create a new collection.
        - **store**: Store a new record.
        - **update**: Update an existing record.
        - **update_status**: Update the status of an existing record.
        - **drop_collection**: Drop the collection.
        - **get_collection**: Get the collection.
        - **get_record**: Get a record from the collection.

    run_record : dict
        The record to store.
    update_record : dict
        The record updates.

    Returns
    -------
    Any
        Operation-specific return value.
    """
    db_path = DB_PATH / "pipeline.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _db_lock:  # Ensure only one thread can access the database at a time
        try:
            conn = _get_db_connection(db_path)
            cursor = conn.cursor()

            if operation == "create_collection":
                # Create table only if it doesn't exist
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pipeline_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT UNIQUE NOT NULL,
                        created_at TEXT NOT NULL,
                        task TEXT,
                        unprocessed_file TEXT,
                        status TEXT,
                        success BOOLEAN,
                        json_file TEXT,
                        report_file TEXT,
                        user_context TEXT,
                        metadata TEXT,
                        task_file_info TEXT,
                        error TEXT
                    )
                """
                )

                # Create update audit log table for tracking changes
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS update_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        old_status TEXT,
                        new_status TEXT,
                        operation_type TEXT,
                        user_context TEXT,
                        FOREIGN KEY (run_id) REFERENCES pipeline_runs (run_id)
                    )
                """
                )

                # Create database access log table (tamper-proof)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS database_access_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        operation TEXT NOT NULL,
                        user_context TEXT NOT NULL,
                        details TEXT,
                        log_hash TEXT NOT NULL,
                        previous_hash TEXT
                    )
                """
                )

                # Add user_context column to existing tables if it doesn't exist
                try:
                    cursor.execute(
                        "ALTER TABLE pipeline_runs ADD COLUMN user_context TEXT"
                    )
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

                # Add status-based protection triggers
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS prevent_completed_record_updates
                    BEFORE UPDATE ON pipeline_runs
                    FOR EACH ROW
                    WHEN (
                        OLD.status IN ('completed', 'failed') 
                    )
                    BEGIN
                        SELECT RAISE(ABORT, 
                            'Cannot modify audit record - run already completed'
                        );
                    END
                """
                )

                # Never allow deletions of any records
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS prevent_all_deletions
                    BEFORE DELETE ON pipeline_runs
                    BEGIN
                        SELECT RAISE(ABORT, 
                            'Audit records cannot be deleted'
                        );
                    END
                """
                )

                # Log all updates for audit trail
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS log_record_updates
                    AFTER UPDATE ON pipeline_runs
                    FOR EACH ROW
                    BEGIN
                        INSERT INTO update_audit_log (
                            run_id, 
                            timestamp, 
                            old_status, 
                            new_status,
                            operation_type,
                            user_context
                        ) VALUES (
                            NEW.run_id,
                            datetime('now'),
                            OLD.status,
                            NEW.status,
                            CASE 
                                WHEN OLD.status != NEW.status THEN 'status_change'
                                WHEN OLD.metadata != NEW.metadata THEN 'metadata_update'
                                ELSE 'general_update'
                            END,
                            COALESCE(NEW.user_context, OLD.user_context)
                        );
                    END
                """
                )

                # Create tamper protection triggers for database_access_log table
                # Prevent any UPDATE operations on access log (write-only)
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS prevent_access_log_updates
                    BEFORE UPDATE ON database_access_log
                    BEGIN
                        SELECT RAISE(ABORT, 
                            'Access log records are immutable - no updates allowed'
                        );
                    END
                """
                )

                # Prevent any DELETE operations on access log (write-only)
                cursor.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS prevent_access_log_deletions
                    BEFORE DELETE ON database_access_log
                    BEGIN
                        SELECT RAISE(ABORT, 
                            'Access log records cannot be deleted'
                        );
                    END
                """
                )

                conn.commit()
                
                # Initialize access log with genesis entry if empty
                cursor.execute("SELECT COUNT(*) FROM database_access_log")
                log_count = cursor.fetchone()[0]
                
                if log_count == 0:
                    # Create genesis entry to start the hash chain
                    from autoclean.utils.audit import get_user_context, calculate_access_log_hash
                    
                    genesis_timestamp = datetime.now().isoformat()
                    genesis_user_context = get_user_context()
                    genesis_operation = "database_initialization"
                    genesis_details = {"action": "genesis_entry", "database_created": str(db_path)}
                    genesis_previous_hash = "genesis_hash_empty_log"
                    
                    # Calculate hash for genesis entry
                    genesis_hash = calculate_access_log_hash(
                        genesis_timestamp, genesis_operation, genesis_user_context, 
                        "", genesis_details, genesis_previous_hash
                    )
                    
                    # Insert genesis entry
                    cursor.execute(
                        """
                        INSERT INTO database_access_log (
                            timestamp, operation, user_context, 
                            details, log_hash, previous_hash
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            genesis_timestamp,
                            genesis_operation,
                            json.dumps(genesis_user_context),
                            json.dumps(genesis_details),
                            genesis_hash,
                            genesis_previous_hash
                        )
                    )
                    conn.commit()
                
                message(
                    "info", f"✓ Database and audit protection established in {db_path}"
                )

            elif operation == "store":
                if not run_record:
                    raise ValueError("Missing run_record for store operation")

                # Convert metadata to JSON string, handling Path objects
                metadata_json = json.dumps(
                    _serialize_for_json(run_record.get("metadata", {}))
                )

                # Convert user_context to JSON string if present
                user_context_json = None
                if run_record.get("user_context"):
                    user_context_json = json.dumps(
                        _serialize_for_json(run_record["user_context"])
                    )

                cursor.execute(
                    """
                    INSERT INTO pipeline_runs (
                        run_id, created_at, task, unprocessed_file, status,
                        success, json_file, report_file, user_context, metadata, task_file_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_record["run_id"],
                        run_record.get("timestamp", datetime.now().isoformat()),
                        run_record.get("task"),
                        (
                            str(run_record.get("unprocessed_file"))
                            if run_record.get("unprocessed_file")
                            else None
                        ),
                        run_record.get("status"),
                        run_record.get("success", False),
                        (
                            str(run_record.get("json_file"))
                            if run_record.get("json_file")
                            else None
                        ),
                        (
                            str(run_record.get("report_file"))
                            if run_record.get("report_file")
                            else None
                        ),
                        user_context_json,
                        metadata_json,
                        (
                            json.dumps(_serialize_for_json(run_record["task_file_info"]))
                            if run_record.get("task_file_info")
                            else None
                        ),
                    ),
                )
                conn.commit()
                record_id = cursor.lastrowid
                message("info", f"✓ Stored new record with ID: {record_id}")
                return record_id

            elif operation in ["update", "update_status"]:
                if not update_record or "run_id" not in update_record:
                    raise ValueError("Missing run_id in update_record")

                run_id = update_record["run_id"]

                # Check if record exists
                cursor.execute(
                    "SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,)
                )
                existing_record = cursor.fetchone()

                if not existing_record:
                    raise RecordNotFoundError(f"No record found for run_id: {run_id}")

                if operation == "update_status":
                    cursor.execute(
                        """
                        UPDATE pipeline_runs
                        SET status = ?
                        WHERE run_id = ?
                    """,
                        (
                            f"{update_record['status']} at {datetime.now().isoformat()}",
                            run_id,
                        ),
                    )
                else:
                    update_components = []
                    current_update_values = []  # Using a distinct name for clarity

                    # Handle metadata update if 'metadata' key exists in update_record
                    if "metadata" in update_record:
                        metadata_to_update = update_record["metadata"]
                        if not _validate_metadata(metadata_to_update):
                            raise ValueError("Invalid metadata structure for update")

                        # Fetch existing metadata
                        cursor.execute(
                            "SELECT metadata FROM pipeline_runs WHERE run_id = ?",
                            (run_id,),
                        )
                        row_with_metadata = cursor.fetchone()
                        existing_metadata_str = (
                            row_with_metadata["metadata"] if row_with_metadata else "{}"
                        )
                        current_metadata = json.loads(existing_metadata_str or "{}")

                        # Serialize the new metadata fragment and merge it
                        serialized_new_metadata_fragment = _serialize_for_json(
                            metadata_to_update
                        )
                        current_metadata.update(serialized_new_metadata_fragment)
                        final_metadata_json = json.dumps(current_metadata)

                        update_components.append("metadata = ?")
                        current_update_values.append(final_metadata_json)

                    # Handle task_file_info serialization
                    if "task_file_info" in update_record:
                        task_file_info_json = json.dumps(_serialize_for_json(update_record["task_file_info"]))
                        update_components.append("task_file_info = ?")
                        current_update_values.append(task_file_info_json)

                    # Handle other fields present in update_record
                    for key, value in update_record.items():
                        if key == "run_id" or key == "metadata" or key == "task_file_info":
                            continue

                        update_components.append(f"{key} = ?")
                        if isinstance(value, Path):
                            current_update_values.append(str(value))
                        else:
                            current_update_values.append(value)

                    # Only execute the UPDATE SQL statement if there are actual fields to set
                    if update_components:
                        # Add the run_id for the WHERE clause; it's the last parameter for the query
                        current_update_values.append(run_id)

                        set_clause_sql = ", ".join(update_components)
                        query = f"UPDATE pipeline_runs SET {set_clause_sql} WHERE run_id = ?"

                        cursor.execute(query, tuple(current_update_values))
                    else:
                        message(
                            "debug",
                            f"For 'update' operation on run_id '{run_id}', no non-metadata fields were identified for SET clause. update_record: {update_record}. Metadata might have been updated if processed.",
                        )

                conn.commit()
                message("debug", f"Record {operation} successful for run_id: {run_id}")

            elif operation == "drop_collection":
                cursor.execute("DROP TABLE IF EXISTS pipeline_runs")
                conn.commit()
                message("warning", f"'pipeline_runs' table dropped from {db_path}")

            elif operation == "get_collection":
                cursor.execute("SELECT * FROM pipeline_runs")
                records = [dict(row) for row in cursor.fetchall()]
                return records

            elif operation == "get_record":
                if not run_record or "run_id" not in run_record:
                    raise ValueError("Missing run_id in run_record")

                cursor.execute(
                    "SELECT * FROM pipeline_runs WHERE run_id = ?",
                    (run_record["run_id"],),
                )
                record = cursor.fetchone()

                if not record:
                    raise RecordNotFoundError(
                        f"No record found for run_id: {run_record['run_id']}"
                    )

                # Convert record to dict and parse JSON fields
                record_dict = dict(record)
                if record_dict.get("metadata"):
                    record_dict["metadata"] = json.loads(record_dict["metadata"])
                if record_dict.get("user_context"):
                    record_dict["user_context"] = json.loads(
                        record_dict["user_context"]
                    )
                if record_dict.get("task_file_info"):
                    record_dict["task_file_info"] = json.loads(
                        record_dict["task_file_info"]
                    )
                return record_dict

            elif operation == "add_access_log":
                if not run_record:
                    raise ValueError("Missing log entry data for add_access_log operation")
                
                # Handle both timestamp formats for backward compatibility
                timestamp = run_record.get("timestamp")
                if isinstance(timestamp, int):
                    # Unix timestamp - convert to ISO for storage
                    timestamp = datetime.fromtimestamp(timestamp).isoformat()
                elif not timestamp:
                    # No timestamp provided - generate current one
                    timestamp = datetime.now().isoformat()
                operation_type = run_record.get("operation", "unknown")
                user_context = run_record.get("user_context", {})
                details = run_record.get("details", {})
                
                # Get previous hash for chain integrity (using same connection)
                cursor.execute(
                    "SELECT log_hash FROM database_access_log ORDER BY log_id DESC LIMIT 1"
                )
                result = cursor.fetchone()
                previous_hash = result[0] if result else "genesis_hash_empty_log"
                
                # Calculate hash for this entry (without database_file)
                from autoclean.utils.audit import calculate_access_log_hash
                log_hash = calculate_access_log_hash(
                    timestamp, operation_type, user_context, "", details, previous_hash
                )
                
                # Insert the access log entry
                cursor.execute(
                    """
                    INSERT INTO database_access_log (
                        timestamp, operation, user_context, 
                        details, log_hash, previous_hash
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        operation_type,
                        json.dumps(user_context),
                        json.dumps(details),
                        log_hash,
                        previous_hash
                    )
                )
                
                conn.commit()
                log_id = cursor.lastrowid
                return log_id

            conn.close()

        except Exception as e:
            error_context = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
            message("error", f"Database operation failed: {error_context}")
            raise DatabaseError(f"Operation '{operation}' failed: {e}") from e
