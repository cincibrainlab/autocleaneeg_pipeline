# src/autoclean/utils/database.py
"""Database utilities for the autoclean package using UnQLite."""

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from unqlite import UnQLite # pylint: disable=no-name-in-module

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
    global DB_PATH # pylint: disable=global-statement
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

    """

    db_path = DB_PATH / "pipeline.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with _db_lock:  # Ensure only one thread can access the database at a time
        try:
            # Get database connection
            db = UnQLite(str(db_path))
            collection = db.collection("pipeline_runs")

            if operation == "create_collection":
                if not collection.exists():
                    collection.create()
                message("info", f"✓ Created 'pipeline_runs' collection in {db_path}")

            elif operation == "store":
                if not run_record:
                    raise ValueError("Missing run_record for store operation")
                record_id = collection.store(run_record)
                message("info", f"✓ Stored new record with ID: {record_id}")
                return record_id

            elif operation in ["update", "update_status"]:
                if not update_record or "run_id" not in update_record:
                    raise ValueError("Missing run_id in update_record")

                run_id = update_record["run_id"]
                existing_record = collection.filter(lambda x: x["run_id"] == run_id)

                if not existing_record:
                    raise RecordNotFoundError(f"No record found for run_id: {run_id}")

                record = existing_record[0]
                record_id = record.pop("__id")

                if operation == "update_status":
                    record["status"] = (
                        f"{update_record['status']} at {datetime.now().isoformat()}"
                    )
                else:
                    if "metadata" in update_record:
                        if not _validate_metadata(update_record["metadata"]):
                            raise ValueError("Invalid metadata structure")

                        if "metadata" not in record:
                            record["metadata"] = {}
                        elif not isinstance(record["metadata"], dict):
                            record["metadata"] = {}

                        if isinstance(update_record["metadata"], dict):
                            record["metadata"].update(update_record["metadata"])
                        else:
                            raise ValueError("Metadata must be a dictionary")

                    record.update(
                        {k: v for k, v in update_record.items() if k != "metadata"}
                    )

                collection.update(record_id=record_id, record=record)
                message("debug", f"Record {operation} successful for run_id: {run_id}")

            elif operation == "drop_collection":
                if collection.exists():
                    collection.drop()
                message("warning", f"'pipeline_runs' collection dropped from {db_path}")

            elif operation == "get_collection":
                if not collection.exists():
                    raise ValueError("Collection 'pipeline_runs' not found")
                return collection

            elif operation == "get_record":
                if not run_record or "run_id" not in run_record:
                    raise ValueError("Missing run_id in run_record")

                record = collection.filter(
                    lambda x: x["run_id"] == run_record["run_id"]
                )

                if not record:
                    raise RecordNotFoundError(
                        f"No record found for run_id: {run_record['run_id']}"
                    )
                return record[0]

        except Exception as e:
            error_context = {
                "operation": operation,
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
            message("error", f"Database operation failed: {error_context}")
            raise DatabaseError(f"Operation '{operation}' failed: {e}") from e
