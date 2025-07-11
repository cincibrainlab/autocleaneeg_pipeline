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


def get_database_info(db_path: Optional[Path] = None) -> dict:
    """
    Get information about the database including compliance mode and access requirements.
    
    Parameters
    ----------
    db_path : Optional[Path]
        Path to database file. If None, uses default DB_PATH.
        
    Returns
    -------
    dict
        Database information including compliance mode, access requirements, etc.
    """
    if db_path is None:
        if DB_PATH is None:
            return {"error": "Database path not set"}
        db_path = DB_PATH / "pipeline.db"
    
    if not db_path.exists():
        return {"error": f"Database file not found: {db_path}"}
    
    try:
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Check if database_metadata table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='database_metadata'"
        )
        
        if not cursor.fetchone():
            conn.close()
            return {
                "database_path": str(db_path),
                "compliance_mode": False,
                "access_requirements": "none", 
                "legacy_database": True,
                "created_timestamp": "unknown"
            }
        
        # Get database metadata
        cursor.execute(
            """
            SELECT database_path, compliance_mode, created_timestamp, 
                   created_by_auth0_user_id, access_requirements, schema_version
            FROM database_metadata WHERE id = 1
            """
        )
        
        metadata_row = cursor.fetchone()
        
        # Get additional statistics
        cursor.execute("SELECT COUNT(*) FROM pipeline_runs")
        run_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM database_access_log")
        access_log_count = cursor.fetchone()[0]
        
        conn.close()
        
        if metadata_row:
            return {
                "database_path": metadata_row[0],
                "compliance_mode": bool(metadata_row[1]),
                "created_timestamp": metadata_row[2],
                "created_by_auth0_user_id": metadata_row[3],
                "access_requirements": metadata_row[4],
                "schema_version": metadata_row[5],
                "run_count": run_count,
                "access_log_count": access_log_count,
                "legacy_database": False
            }
        else:
            return {
                "database_path": str(db_path),
                "compliance_mode": False,
                "access_requirements": "none",
                "legacy_database": True,
                "run_count": run_count,
                "access_log_count": access_log_count,
                "created_timestamp": "unknown"
            }
            
    except Exception as e:
        return {"error": f"Failed to read database info: {str(e)}"}


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


def manage_database_conditionally(
    operation: str,
    run_record: Optional[Dict[str, Any]] = None,
    update_record: Optional[Dict[str, Any]] = None,
) -> Any:
    """Use audit protection only when compliance mode is enabled.
    
    This function routes database operations to the appropriate handler based
    on compliance mode settings, ensuring audit logging only occurs when
    FDA 21 CFR Part 11 compliance is required.
    
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
        Operation result from underlying database function
    """
    from autoclean.utils.config import is_compliance_mode_enabled
    
    if is_compliance_mode_enabled():
        return manage_database_with_audit_protection(operation, run_record, update_record)
    else:
        return manage_database(operation, run_record, update_record)


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


def _validate_database_access(db_path: Path, operation: str) -> tuple[bool, str]:
    """
    Validate that the current user has permission to access this database.
    
    Part 11 databases require authenticated users in compliance mode.
    
    Parameters
    ----------
    db_path : Path
        Path to the database file
    operation : str
        The database operation being attempted
        
    Returns
    -------
    tuple[bool, str]
        (is_allowed, error_message) 
    """
    try:
        from autoclean.utils.config import is_compliance_mode_enabled
        
        # Check if database exists
        if not db_path.exists():
            # For create operations, check current mode requirements
            if operation == "create_collection":
                if is_compliance_mode_enabled():
                    from autoclean.utils.auth import get_auth0_manager
                    auth_manager = get_auth0_manager()
                    
                    if not auth_manager.is_configured():
                        return False, "Compliance mode enabled but Auth0 not configured. Run 'autoclean-eeg auth0 set' first."
                    
                    if not auth_manager.is_authenticated():
                        return False, "Part 11 compliance mode requires authentication. Run 'autoclean-eeg login' first."
                
                return True, ""  # New database creation is allowed
            else:
                return False, f"Database does not exist: {db_path}"
        
        # Database exists - check its metadata for access requirements
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Check if database_metadata table exists (older databases might not have it)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='database_metadata'"
        )
        
        if not cursor.fetchone():
            # Legacy database without metadata - allow access but warn
            conn.close()
            return True, ""
        
        # Get database metadata
        cursor.execute(
            "SELECT compliance_mode, access_requirements FROM database_metadata WHERE id = 1"
        )
        
        metadata_row = cursor.fetchone()
        conn.close()
        
        if not metadata_row:
            # No metadata record - allow access but warn
            return True, ""
        
        db_compliance_mode = bool(metadata_row[0])
        access_requirements = metadata_row[1]
        
        # If database was created in Part 11 mode, enforce authentication
        if db_compliance_mode and access_requirements == "authenticated_user":
            current_compliance_mode = is_compliance_mode_enabled()
            
            if not current_compliance_mode:
                return False, (
                    "This database was created in Part 11 compliance mode and requires "
                    "compliance mode to be enabled. Run 'autoclean-eeg setup --compliance-mode' first."
                )
            
            from autoclean.utils.auth import get_auth0_manager
            auth_manager = get_auth0_manager()
            
            if not auth_manager.is_configured():
                return False, (
                    "This Part 11 database requires authentication. "
                    "Run 'autoclean-eeg auth0 set' to configure Auth0 first."
                )
            
            if not auth_manager.is_authenticated():
                return False, (
                    "This Part 11 database requires authentication. "
                    "Run 'autoclean-eeg login' first."
                )
        
        return True, ""
        
    except Exception as e:
        return False, f"Database access validation failed: {str(e)}"


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

    # Validate database access permissions before any operations
    access_allowed, access_error = _validate_database_access(db_path, operation)
    if not access_allowed:
        raise DatabaseError(f"Database access denied: {access_error}")

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

                # Create authenticated users table for compliance mode
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS authenticated_users (
                        auth0_user_id TEXT PRIMARY KEY,
                        email TEXT NOT NULL,
                        name TEXT,
                        first_login TEXT,
                        last_login TEXT,
                        token_expires TEXT,
                        user_metadata TEXT
                    )
                """
                )

                # Create electronic signatures table for compliance mode
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS electronic_signatures (
                        signature_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        auth0_user_id TEXT NOT NULL,
                        signature_data TEXT NOT NULL,
                        signature_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id),
                        FOREIGN KEY (auth0_user_id) REFERENCES authenticated_users(auth0_user_id)
                    )
                """
                )

                # Create database metadata table for tracking compliance requirements
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS database_metadata (
                        id INTEGER PRIMARY KEY,
                        database_path TEXT NOT NULL,
                        compliance_mode BOOLEAN NOT NULL DEFAULT FALSE,
                        created_timestamp TEXT NOT NULL,
                        created_by_auth0_user_id TEXT,
                        created_by_user_context TEXT,
                        access_requirements TEXT NOT NULL DEFAULT 'none',
                        schema_version TEXT NOT NULL DEFAULT '1.1'
                    )
                """
                )

                # Create encrypted outputs table for compliance mode
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS encrypted_outputs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        output_type TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        original_path TEXT,
                        encrypted_data BLOB NOT NULL,
                        encryption_algorithm TEXT NOT NULL DEFAULT 'fernet',
                        encryption_key_id TEXT,
                        file_size INTEGER NOT NULL,
                        original_size INTEGER,
                        content_hash TEXT NOT NULL,
                        compression_used BOOLEAN DEFAULT FALSE,
                        metadata TEXT,
                        created_at TEXT NOT NULL,
                        created_by_auth0_user_id TEXT,
                        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id),
                        FOREIGN KEY (created_by_auth0_user_id) REFERENCES authenticated_users(auth0_user_id)
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

                # Add auth0_user_id column for compliance mode
                try:
                    cursor.execute(
                        "ALTER TABLE pipeline_runs ADD COLUMN auth0_user_id TEXT"
                    )
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

                # Add auth0_user_id column to database_access_log for compliance mode
                try:
                    cursor.execute(
                        "ALTER TABLE database_access_log ADD COLUMN auth0_user_id TEXT"
                    )
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

                # Add compliance mode tracking columns to pipeline_runs
                try:
                    cursor.execute(
                        "ALTER TABLE pipeline_runs ADD COLUMN compliance_mode BOOLEAN DEFAULT FALSE"
                    )
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

                try:
                    cursor.execute(
                        "ALTER TABLE pipeline_runs ADD COLUMN outputs_encrypted BOOLEAN DEFAULT FALSE"
                    )
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

                try:
                    cursor.execute(
                        "ALTER TABLE pipeline_runs ADD COLUMN encryption_key_id TEXT"
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
                    from autoclean.utils.audit import (
                        calculate_access_log_hash,
                        get_user_context,
                    )

                    genesis_timestamp = datetime.now().isoformat()
                    genesis_user_context = get_user_context()
                    genesis_operation = "database_initialization"
                    genesis_details = {
                        "action": "genesis_entry",
                        "database_created": str(db_path),
                    }
                    genesis_previous_hash = "genesis_hash_empty_log"

                    # Calculate hash for genesis entry
                    genesis_hash = calculate_access_log_hash(
                        genesis_timestamp,
                        genesis_operation,
                        genesis_user_context,
                        "",
                        genesis_details,
                        genesis_previous_hash,
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
                            genesis_previous_hash,
                        ),
                    )
                    conn.commit()

                # Record database creation metadata  
                from autoclean.utils.config import is_compliance_mode_enabled
                
                compliance_enabled = is_compliance_mode_enabled()
                current_user_id = None
                current_user_context = None
                
                if compliance_enabled:
                    # Get authenticated user information for Part 11 databases
                    from autoclean.utils.auth import get_current_user_for_audit
                    user_context = get_current_user_for_audit()
                    current_user_id = user_context.get("auth0_user_id")
                    current_user_context = json.dumps(user_context)
                
                # Insert database metadata record
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO database_metadata (
                        id, database_path, compliance_mode, created_timestamp,
                        created_by_auth0_user_id, created_by_user_context, 
                        access_requirements
                    ) VALUES (1, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(db_path),
                        compliance_enabled,
                        datetime.now().isoformat(),
                        current_user_id,
                        current_user_context,
                        "authenticated_user" if compliance_enabled else "none"
                    )
                )
                conn.commit()

                mode_text = "Part 11 compliance" if compliance_enabled else "standard"
                message(
                    "info", f"✓ Database and audit protection established in {mode_text} mode: {db_path}"
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
                            json.dumps(
                                _serialize_for_json(run_record["task_file_info"])
                            )
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
                        task_file_info_json = json.dumps(
                            _serialize_for_json(update_record["task_file_info"])
                        )
                        update_components.append("task_file_info = ?")
                        current_update_values.append(task_file_info_json)

                    # Handle other fields present in update_record
                    for key, value in update_record.items():
                        if (
                            key == "run_id"
                            or key == "metadata"
                            or key == "task_file_info"
                        ):
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
                    raise ValueError(
                        "Missing log entry data for add_access_log operation"
                    )

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
                        previous_hash,
                    ),
                )

                conn.commit()
                log_id = cursor.lastrowid
                return log_id

            elif operation == "store_authenticated_user":
                if not run_record:
                    raise ValueError(
                        "Missing user data for store_authenticated_user operation"
                    )

                # Extract user information
                auth0_user_id = run_record.get("auth0_user_id")
                email = run_record.get("email")
                name = run_record.get("name")
                user_metadata = run_record.get("user_metadata", {})

                if not auth0_user_id or not email:
                    raise ValueError(
                        "auth0_user_id and email are required for authenticated users"
                    )

                current_time = datetime.now().isoformat()

                # Check if user already exists
                cursor.execute(
                    "SELECT auth0_user_id, first_login FROM authenticated_users WHERE auth0_user_id = ?",
                    (auth0_user_id,),
                )
                existing_user = cursor.fetchone()

                if existing_user:
                    # Update existing user
                    cursor.execute(
                        """
                        UPDATE authenticated_users 
                        SET email = ?, name = ?, last_login = ?, user_metadata = ?
                        WHERE auth0_user_id = ?
                        """,
                        (
                            email,
                            name,
                            current_time,
                            json.dumps(user_metadata),
                            auth0_user_id,
                        ),
                    )
                    message("debug", f"Updated authenticated user: {email}")
                else:
                    # Insert new user
                    cursor.execute(
                        """
                        INSERT INTO authenticated_users (
                            auth0_user_id, email, name, first_login, 
                            last_login, user_metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            auth0_user_id,
                            email,
                            name,
                            current_time,
                            current_time,
                            json.dumps(user_metadata),
                        ),
                    )
                    message("debug", f"Stored new authenticated user: {email}")

                conn.commit()
                return auth0_user_id

            elif operation == "get_authenticated_user":
                if not run_record or "auth0_user_id" not in run_record:
                    raise ValueError(
                        "Missing auth0_user_id for get_authenticated_user operation"
                    )

                cursor.execute(
                    "SELECT * FROM authenticated_users WHERE auth0_user_id = ?",
                    (run_record["auth0_user_id"],),
                )
                user_record = cursor.fetchone()

                if not user_record:
                    return None

                # Convert to dict and parse JSON fields
                user_dict = dict(user_record)
                if user_dict.get("user_metadata"):
                    user_dict["user_metadata"] = json.loads(user_dict["user_metadata"])

                return user_dict

            elif operation == "store_electronic_signature":
                if not run_record:
                    raise ValueError(
                        "Missing signature data for store_electronic_signature operation"
                    )

                required_fields = [
                    "signature_id",
                    "run_id",
                    "auth0_user_id",
                    "signature_data",
                    "signature_type",
                ]
                for field in required_fields:
                    if field not in run_record:
                        raise ValueError(
                            f"Missing required field '{field}' for electronic signature"
                        )

                current_time = datetime.now().isoformat()

                cursor.execute(
                    """
                    INSERT INTO electronic_signatures (
                        signature_id, run_id, auth0_user_id, signature_data,
                        signature_type, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_record["signature_id"],
                        run_record["run_id"],
                        run_record["auth0_user_id"],
                        json.dumps(run_record["signature_data"]),
                        run_record["signature_type"],
                        current_time,
                    ),
                )

                conn.commit()
                message(
                    "debug",
                    f"Stored electronic signature for run: {run_record['run_id']}",
                )
                return run_record["signature_id"]

            elif operation == "get_electronic_signatures":
                if not run_record or "run_id" not in run_record:
                    raise ValueError(
                        "Missing run_id for get_electronic_signatures operation"
                    )

                cursor.execute(
                    """
                    SELECT es.*, au.email, au.name 
                    FROM electronic_signatures es
                    LEFT JOIN authenticated_users au ON es.auth0_user_id = au.auth0_user_id
                    WHERE es.run_id = ?
                    ORDER BY es.timestamp
                    """,
                    (run_record["run_id"],),
                )

                signatures = []
                for row in cursor.fetchall():
                    sig_dict = dict(row)
                    if sig_dict.get("signature_data"):
                        sig_dict["signature_data"] = json.loads(
                            sig_dict["signature_data"]
                        )
                    signatures.append(sig_dict)

                return signatures

            elif operation == "store_encrypted_output":
                if not run_record:
                    raise ValueError("Missing output data for store_encrypted_output operation")

                required_fields = [
                    "run_id", "output_type", "file_name", "encrypted_data", 
                    "file_size", "content_hash"
                ]
                for field in required_fields:
                    if field not in run_record:
                        raise ValueError(f"Missing required field '{field}' for encrypted output")

                # Get current user for compliance mode
                current_user_id = None
                from autoclean.utils.config import is_compliance_mode_enabled
                
                if is_compliance_mode_enabled():
                    from autoclean.utils.auth import get_current_user_for_audit
                    user_context = get_current_user_for_audit()
                    current_user_id = user_context.get("auth0_user_id")

                current_time = datetime.now().isoformat()

                cursor.execute(
                    """
                    INSERT INTO encrypted_outputs (
                        run_id, output_type, file_name, original_path, encrypted_data,
                        encryption_algorithm, encryption_key_id, file_size, original_size,
                        content_hash, compression_used, metadata, created_at, created_by_auth0_user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_record["run_id"],
                        run_record["output_type"],
                        run_record["file_name"],
                        run_record.get("original_path"),
                        run_record["encrypted_data"],
                        run_record.get("encryption_algorithm", "fernet"),
                        run_record.get("encryption_key_id"),
                        run_record["file_size"],
                        run_record.get("original_size"),
                        run_record["content_hash"],
                        run_record.get("compression_used", False),
                        json.dumps(run_record.get("metadata", {})),
                        current_time,
                        current_user_id
                    )
                )

                output_id = cursor.lastrowid
                conn.commit()
                
                message("debug", f"Stored encrypted output: {run_record['file_name']} for run {run_record['run_id']}")
                return output_id

            elif operation == "get_encrypted_outputs":
                if not run_record or "run_id" not in run_record:
                    raise ValueError("Missing run_id for get_encrypted_outputs operation")

                cursor.execute(
                    """
                    SELECT id, run_id, output_type, file_name, original_path,
                           encryption_algorithm, encryption_key_id, file_size, 
                           original_size, content_hash, compression_used, metadata,
                           created_at, created_by_auth0_user_id
                    FROM encrypted_outputs 
                    WHERE run_id = ?
                    ORDER BY created_at
                    """,
                    (run_record["run_id"],)
                )

                outputs = []
                for row in cursor.fetchall():
                    output_dict = dict(row)
                    if output_dict.get("metadata"):
                        output_dict["metadata"] = json.loads(output_dict["metadata"])
                    outputs.append(output_dict)

                return outputs

            elif operation == "get_encrypted_output_data":
                if not run_record or "output_id" not in run_record:
                    raise ValueError("Missing output_id for get_encrypted_output_data operation")

                cursor.execute(
                    """
                    SELECT encrypted_data, encryption_algorithm, file_name, content_hash
                    FROM encrypted_outputs 
                    WHERE id = ?
                    """,
                    (run_record["output_id"],)
                )

                result = cursor.fetchone()
                if not result:
                    raise RecordNotFoundError(f"Encrypted output not found: {run_record['output_id']}")

                return {
                    "encrypted_data": result[0],
                    "encryption_algorithm": result[1], 
                    "file_name": result[2],
                    "content_hash": result[3]
                }

            elif operation == "delete_encrypted_output":
                if not run_record or "output_id" not in run_record:
                    raise ValueError("Missing output_id for delete_encrypted_output operation")

                cursor.execute(
                    "DELETE FROM encrypted_outputs WHERE id = ?",
                    (run_record["output_id"],)
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                message("debug", f"Deleted encrypted output: {run_record['output_id']}")
                return deleted_count > 0

            conn.close()

        except Exception as e:
            error_context = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
            message("error", f"Database operation failed: {error_context}")
            raise DatabaseError(f"Operation '{operation}' failed: {e}") from e


# Convenience functions for encrypted outputs management

def store_encrypted_output(
    run_id: str,
    output_type: str, 
    file_name: str,
    encrypted_data: bytes,
    file_size: int,
    content_hash: str,
    original_path: Optional[str] = None,
    original_size: Optional[int] = None,
    encryption_algorithm: str = "fernet",
    encryption_key_id: Optional[str] = None,
    compression_used: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> int:
    """
    Store encrypted output data in the database.
    
    Parameters
    ----------
    run_id : str
        The run ID this output belongs to
    output_type : str
        Type of output (e.g., 'report_pdf', 'metadata_json', etc.)
    file_name : str
        Original filename
    encrypted_data : bytes
        Encrypted file content
    file_size : int
        Size of encrypted data in bytes
    content_hash : str
        Hash of original unencrypted content for integrity verification
    original_path : str, optional
        Original filesystem path before encryption
    original_size : int, optional
        Size of original unencrypted data
    encryption_algorithm : str, default 'fernet'
        Encryption algorithm used
    encryption_key_id : str, optional
        ID of encryption key for rotation support
    compression_used : bool, default False
        Whether compression was applied before encryption
    metadata : dict, optional
        Additional metadata about the output
        
    Returns
    -------
    int
        Database ID of stored encrypted output
    """
    output_record = {
        "run_id": run_id,
        "output_type": output_type,
        "file_name": file_name,
        "encrypted_data": encrypted_data,
        "file_size": file_size,
        "content_hash": content_hash,
        "original_path": original_path,
        "original_size": original_size,
        "encryption_algorithm": encryption_algorithm,
        "encryption_key_id": encryption_key_id,
        "compression_used": compression_used,
        "metadata": metadata or {}
    }
    
    return manage_database_conditionally("store_encrypted_output", output_record)


def get_encrypted_outputs(run_id: str) -> list:
    """
    Get all encrypted outputs for a run.
    
    Parameters
    ----------
    run_id : str
        The run ID to get outputs for
        
    Returns
    -------
    list
        List of encrypted output records (without encrypted_data)
    """
    return manage_database_conditionally("get_encrypted_outputs", {"run_id": run_id})


def get_encrypted_output_data(output_id: int) -> Dict[str, Any]:
    """
    Get encrypted data for a specific output.
    
    Parameters
    ----------
    output_id : int
        Database ID of the encrypted output
        
    Returns
    -------
    dict
        Dictionary containing encrypted_data, encryption_algorithm, file_name, content_hash
    """
    return manage_database_conditionally("get_encrypted_output_data", {"output_id": output_id})


def delete_encrypted_output(output_id: int) -> bool:
    """
    Delete an encrypted output from the database.
    
    Parameters
    ----------
    output_id : int
        Database ID of the encrypted output to delete
        
    Returns
    -------
    bool
        True if output was deleted, False if not found
    """
    return manage_database_conditionally("delete_encrypted_output", {"output_id": output_id})


def has_encrypted_outputs(run_id: str) -> bool:
    """
    Check if a run has any encrypted outputs stored.
    
    Parameters
    ---------- 
    run_id : str
        The run ID to check
        
    Returns
    -------
    bool
        True if run has encrypted outputs, False otherwise
    """
    outputs = get_encrypted_outputs(run_id)
    return len(outputs) > 0


def get_database_schema_version() -> str:
    """
    Get the current database schema version.
    
    Returns
    -------
    str
        Schema version (e.g., '1.1') or '1.0' for legacy databases
    """
    try:
        db_info = get_database_info()
        return db_info.get("schema_version", "1.0")
    except Exception:
        return "1.0"


def database_supports_encryption() -> bool:
    """
    Check if the current database supports encrypted outputs.
    
    Returns
    -------
    bool
        True if database has encrypted_outputs table, False otherwise
    """
    if DB_PATH is None:
        return False
        
    db_path = DB_PATH / "pipeline.db"
    if not db_path.exists():
        return False
        
    try:
        conn = _get_db_connection(db_path)
        cursor = conn.cursor()
        
        # Check if encrypted_outputs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='encrypted_outputs'"
        )
        
        table_exists = cursor.fetchone() is not None
        conn.close()
        
        return table_exists
        
    except Exception:
        return False
