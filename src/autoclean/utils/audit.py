# src/autoclean/utils/audit.py
"""Audit utilities for enhanced tracking and compliance."""

import getpass
import gzip
import hashlib
import inspect
import json
import os
import shutil
import socket
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

from autoclean.utils.logging import message

# Optional dependencies - may not be available in all contexts
try:
    from autoclean.utils.database import DB_PATH, manage_database

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    DB_PATH = None

    def manage_database(*args, **kwargs):
        return None


def get_user_context() -> Dict[str, Any]:
    """Get current user context for audit trail.

    Captures basic system and user information for tracking who
    performed operations without requiring authentication.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - username: Current system username
        - hostname: Machine hostname
        - pid: Process ID of the pipeline
        - session_start: When this context was created

    Examples
    --------
    >>> context = get_user_context()
    >>> print(context['username'])
    'researcher1'
    """
    try:
        username = getpass.getuser()
    except Exception:
        username = "unknown"

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    # Optimize for storage space
    try:
        # Abbreviate hostname (keep first part only, max 12 chars)
        hostname = hostname.split(".")[0][:12]
    except Exception:
        pass

    return {
        "user": username[:20],  # Shorter key, limit length
        "host": hostname,  # Shorter key, abbreviated hostname
        "pid": os.getpid(),
        # Use Unix timestamp instead of ISO string to save ~15 chars
        "ts": int(datetime.now().timestamp()),
    }


def verify_database_file_integrity(
    db_path: Path, expected_operation: str = None
) -> Tuple[bool, str]:
    """Verify database file hasn't been tampered with.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message) indicating if database integrity is verified

    Examples
    --------
    >>> is_valid, msg = verify_database_file_integrity(Path("pipeline.db"))
    >>> print(f"Database valid: {is_valid}")
    Database valid: True
    """
    if not db_path.exists():
        return False, f"Database file not found: {db_path}"

    # Calculate current database file hash
    try:
        with open(db_path, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return False, f"Failed to read database file: {e}"

    # Check against stored integrity baseline
    integrity_file = db_path.parent / ".db_integrity"

    if integrity_file.exists():
        try:
            with open(integrity_file, "r") as f:
                stored_data = json.load(f)
                stored_hash = stored_data["hash"]
                last_verified = stored_data["timestamp"]

                if current_hash != stored_hash:
                    return (
                        False,
                        f"Database integrity check FAILED! Database may have been tampered with since {last_verified}",
                    )
                else:
                    # Update last verified timestamp but keep same hash
                    stored_data["last_verified"] = datetime.now().isoformat()
                    with open(integrity_file, "w") as f_update:
                        json.dump(stored_data, f_update, indent=2)
                    return True, "Database integrity verified"
        except Exception as e:
            return False, f"Failed to verify integrity file: {e}"
    else:
        # First time - establish integrity baseline
        try:
            with open(integrity_file, "w") as f:
                json.dump(
                    {
                        "hash": current_hash,
                        "timestamp": datetime.now().isoformat(),
                        "created_by": get_user_context(),
                        "last_verified": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            return True, "Database integrity baseline established"
        except Exception as e:
            return False, f"Failed to create integrity baseline: {e}"


def update_database_integrity_baseline(db_path: Path) -> bool:
    """Update integrity baseline after legitimate database changes.

    Call this after database schema updates or major legitimate changes.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file

    Returns
    -------
    bool
        True if baseline was updated successfully
    """
    try:
        with open(db_path, "rb") as f:
            new_hash = hashlib.sha256(f.read()).hexdigest()

        integrity_file = db_path.parent / ".db_integrity"
        with open(integrity_file, "w") as f:
            json.dump(
                {
                    "hash": new_hash,
                    "timestamp": datetime.now().isoformat(),
                    "updated_by": get_user_context(),
                    "last_verified": datetime.now().isoformat(),
                    "reason": "Schema update or legitimate database change",
                },
                f,
                indent=2,
            )

        message("info", "🔒 Database integrity baseline updated")
        return True
    except Exception as e:
        message("error", f"Failed to update integrity baseline: {e}")
        return False


def create_database_backup(db_path: Path) -> Path:
    """Create timestamped backup of database file.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file to backup

    Returns
    -------
    Path
        Path to the created backup file

    Examples
    --------
    >>> backup_path = create_database_backup(Path("pipeline.db"))
    >>> print(f"Backup created: {backup_path}")
    Backup created: backups/pipeline_backup_20250618_143022.db
    """
    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    # Create timestamped backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"pipeline_backup_{timestamp}.db"

    # Copy database file
    try:
        shutil.copy2(db_path, backup_file)
        message("info", f"✅ Database backup created: {backup_file}")

        # Clean up old backups and compress old ones
        _manage_backup_retention(backup_dir)

        return backup_file
    except Exception as e:
        message("error", f"Failed to create database backup: {e}")
        raise


def _manage_backup_retention(
    backup_dir: Path, keep_days: int = 30, compress_after_days: int = 7
):
    """Manage backup file retention and compression.

    Parameters
    ----------
    backup_dir : Path
        Directory containing backup files
    keep_days : int
        Number of days to keep backup files
    compress_after_days : int
        Number of days after which to compress backup files
    """
    cutoff_compress = datetime.now() - timedelta(days=compress_after_days)
    cutoff_delete = datetime.now() - timedelta(days=keep_days)

    for backup_file in backup_dir.glob("*.db"):
        file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)

        if file_time < cutoff_delete:
            # Delete very old backups
            try:
                backup_file.unlink()
                message("debug", f"Deleted old backup: {backup_file}")
            except Exception as e:
                message("warning", f"Failed to delete old backup {backup_file}: {e}")
        elif file_time < cutoff_compress:
            # Compress older backups
            compressed_file = backup_file.with_suffix(".db.gz")
            if not compressed_file.exists():
                try:
                    with open(backup_file, "rb") as f_in:
                        with gzip.open(compressed_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    backup_file.unlink()  # Delete original after compression
                    message("debug", f"Compressed backup: {compressed_file}")
                except Exception as e:
                    message("warning", f"Failed to compress backup {backup_file}: {e}")


def log_database_access(
    operation: str, user_context: Dict[str, Any], details: Dict[str, Any] = None
):
    """Log database access to tamper-proof database table.

    Parameters
    ----------
    operation : str
        Type of database operation (store, update, get_record, etc.)
    user_context : Dict[str, Any]
        User context information
    details : Dict[str, Any], optional
        Additional operation details

    Examples
    --------
    >>> log_database_access("store", get_user_context(), {"run_id": "ABC123"})
    """
    if DB_PATH is None:
        return  # Database not initialized yet

    try:
        # Check if database and table exist before trying to log
        db_path = DB_PATH / "pipeline.db"
        if not db_path.exists():
            return  # Database not created yet

        # Check if access log table exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='database_access_log'"
        )
        table_exists = cursor.fetchone() is not None
        conn.close()

        if not table_exists:
            return  # Table not created yet, skip logging

        # Create log entry for database storage (optimized for size)
        log_entry = {
            # Use Unix timestamp to save ~15 characters vs ISO string
            "timestamp": int(datetime.now().timestamp()),
            "operation": operation,
            "user_context": user_context,
            "details": details or {},
        }

        # Store in tamper-proof database table with hash chain
        manage_database(operation="add_access_log", run_record=log_entry)

    except Exception as e:
        # Fallback: log to stderr if database logging fails
        message("warning", f"Failed to log database access to secure table: {e}")
        # Don't create file logs anymore - database is the authoritative source


def get_task_file_info(task_name: str, task_object: Any) -> Dict[str, Any]:
    """Get hash and content of the task file used for compliance tracking.

    Parameters
    ----------
    task_name : str
        Name of the task being executed
    task_object : Any
        The task object instance

    Returns
    -------
    Dict[str, Any]
        Dictionary containing task file hash, content, and metadata
    """
    task_file_info = {
        "task_name": task_name,
        "capture_timestamp": datetime.now().isoformat(),
        "file_path": None,
        "file_content_hash": None,
        "file_content": None,
        "error": None,
    }

    try:
        # Try to get the source file from the task object
        task_file_path = None

        # Method 1: Check if task object has __file__ attribute
        if hasattr(task_object.__class__, "__module__"):
            module = inspect.getmodule(task_object.__class__)
            if module and hasattr(module, "__file__") and module.__file__:
                task_file_path = Path(module.__file__)

        # Method 2: Look in workspace tasks directory
        if not task_file_path or not task_file_path.exists():
            workspace_tasks = Path.home() / ".autoclean" / "tasks"
            if workspace_tasks.exists():
                # Look for Python files containing the task class
                for task_file in workspace_tasks.glob("*.py"):
                    try:
                        content = task_file.read_text(encoding="utf-8")
                        # Check if file contains the task class definition
                        if f"class {task_name}" in content:
                            task_file_path = task_file
                            break
                    except Exception:
                        continue

        # Method 3: Check if it's a built-in task
        if not task_file_path or not task_file_path.exists():
            # Try to find in built-in tasks directory
            try:
                module_path = inspect.getfile(task_object.__class__)
                task_file_path = Path(module_path)
            except (TypeError, OSError):
                pass

        if task_file_path and task_file_path.exists():
            # Read file content
            task_content = task_file_path.read_text(encoding="utf-8")

            # Calculate SHA256 hash
            task_hash = hashlib.sha256(task_content.encode("utf-8")).hexdigest()

            # Store information
            task_file_info.update(
                {
                    "file_path": str(task_file_path),
                    "file_content_hash": task_hash,
                    "file_content": task_content,
                    "file_size_bytes": len(task_content.encode("utf-8")),
                    "line_count": len(task_content.splitlines()),
                }
            )

        else:
            task_file_info["error"] = "Task source file not found or not accessible"

    except Exception as e:
        task_file_info["error"] = f"Failed to capture task file info: {str(e)}"

    return task_file_info


def get_last_access_log_hash() -> str:
    """Get the hash of the most recent access log entry for chain integrity.

    Returns
    -------
    str
        Hash of the last access log entry, or genesis hash if no entries exist
    """
    try:
        if DB_PATH is None:
            return "genesis_hash_no_database"

        # Get the most recent access log entry
        db_path = DB_PATH / "pipeline.db"
        if not db_path.exists():
            return "genesis_hash_no_database"

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT log_hash FROM database_access_log 
            ORDER BY log_id DESC 
            LIMIT 1
            """
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            return result[0]
        else:
            # No entries yet, return genesis hash
            return "genesis_hash_empty_log"

    except Exception as e:
        message("warning", f"Failed to get last access log hash: {e}")
        return "genesis_hash_error"


def calculate_access_log_hash(
    timestamp: str,
    operation: str,
    user_context: Dict[str, Any],
    database_file: str,
    details: Dict[str, Any],
    previous_hash: str,
) -> str:
    """Calculate hash for access log entry to maintain integrity chain.

    Parameters
    ----------
    timestamp : str
        ISO timestamp of the log entry
    operation : str
        Database operation type
    user_context : Dict[str, Any]
        User context information
    database_file : str
        Path to database file
    details : Dict[str, Any]
        Additional operation details
    previous_hash : str
        Hash of the previous log entry

    Returns
    -------
    str
        SHA256 hash of the log entry data
    """
    # Create canonical representation for hashing
    log_data = {
        "timestamp": timestamp,
        "operation": operation,
        "user_context": user_context,
        "database_file": database_file,
        "details": details or {},
        "previous_hash": previous_hash,
    }

    # Convert to canonical JSON (sorted keys)
    canonical_json = json.dumps(log_data, sort_keys=True, separators=(",", ":"))

    # Calculate SHA256 hash
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def verify_access_log_integrity() -> Dict[str, Any]:
    """Verify the integrity of the access log hash chain.

    Returns
    -------
    Dict[str, Any]
        Verification results including status and any issues found
    """
    try:
        if DB_PATH is None:
            return {"status": "error", "message": "Database not initialized"}

        db_path = DB_PATH / "pipeline.db"
        if not db_path.exists():
            return {"status": "error", "message": "Database file not found"}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all access log entries in order
        cursor.execute(
            """
            SELECT log_id, timestamp, operation, user_context, 
                   details, log_hash, previous_hash
            FROM database_access_log 
            ORDER BY log_id ASC
            """
        )
        entries = cursor.fetchall()
        conn.close()

        if not entries:
            return {"status": "valid", "message": "No access log entries to verify"}

        # Verify each entry's hash
        issues = []
        expected_previous_hash = "genesis_hash_empty_log"

        for entry in entries:
            (
                log_id,
                timestamp,
                operation,
                user_context_str,
                details_str,
                stored_hash,
                previous_hash,
            ) = entry

            # Parse JSON fields
            try:
                user_context = json.loads(user_context_str) if user_context_str else {}
                details = json.loads(details_str) if details_str else {}
            except json.JSONDecodeError as e:
                issues.append(f"Entry {log_id}: JSON decode error - {e}")
                continue

            # Verify previous hash matches expected
            if previous_hash != expected_previous_hash:
                issues.append(
                    f"Entry {log_id}: Hash chain broken - expected previous_hash {expected_previous_hash}, got {previous_hash}"
                )

            # Recalculate hash for this entry (without database_file)
            calculated_hash = calculate_access_log_hash(
                timestamp, operation, user_context, "", details, previous_hash
            )

            # Verify stored hash matches calculated hash
            if stored_hash != calculated_hash:
                issues.append(
                    f"Entry {log_id}: Hash mismatch - stored {stored_hash[:16]}..., calculated {calculated_hash[:16]}..."
                )

            # Set up for next iteration
            expected_previous_hash = stored_hash

        if issues:
            return {
                "status": "compromised",
                "message": f"Found {len(issues)} integrity issues",
                "issues": issues,
            }
        else:
            return {
                "status": "valid",
                "message": f"All {len(entries)} access log entries verified successfully",
            }

    except Exception as e:
        return {"status": "error", "message": f"Verification failed: {str(e)}"}
