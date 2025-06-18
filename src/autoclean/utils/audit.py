# src/autoclean/utils/audit.py
"""Audit utilities for enhanced tracking and compliance."""

import getpass
import gzip
import hashlib
import json
import os
import shutil
import socket
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

from autoclean.utils.logging import message


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

    return {
        "username": username,
        "hostname": hostname,
        "pid": os.getpid(),
        "session_start": datetime.now().isoformat(),
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
    """Log database access to separate audit file.

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
    # Import here to avoid circular imports
    from autoclean.utils.database import DB_PATH

    if DB_PATH is None:
        return  # Database not initialized yet

    access_log_dir = DB_PATH.parent / "access_logs"
    access_log_dir.mkdir(exist_ok=True)

    # Daily log file
    log_file = access_log_dir / f"db_access_{datetime.now().strftime('%Y%m%d')}.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
        "user_context": user_context,
        "database_file": str(DB_PATH),
        "details": details or {},
    }

    try:
        # Append to daily log file (JSONL format)
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        message("warning", f"Failed to log database access: {e}")
