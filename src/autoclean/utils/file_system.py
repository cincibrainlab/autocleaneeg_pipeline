# src/autoclean/utils/file_system.py
"""
This module contains functions for setting up and validating directory structures.
"""
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from autoclean import __version__
from autoclean.utils.logging import message

# Cache to ensure we only perform a backup once per directory in a single
# Python process. If the user aborts with Ctrl-C, the process ends and this
# cache is cleared automatically on the next run.
_PREPARED_TASK_ROOTS: set[Path] = set()
_CACHE_LOCK = threading.Lock()

STATUS_DIR_NAME = ".autoclean-status"


def step_prepare_directories(
    task: str, autoclean_dir_str: Path, dataset_name: str = None
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path | None]:
    """Set up and validate BIDS-compliant directory structure for processing pipeline.

    Parameters
    ----------
    task : str
        The name of the processing task.
    autoclean_dir_str : Path
        The path to the autoclean directory.
    dataset_name : str, optional
        Optional dataset name to use instead of task name for directory structure.
        If provided, creates directories using dataset_name + timestamp format.

    Returns
    -------
    Tuple of Path objects for key directories:
    (
        autoclean_dir,
        bids_dir,
        metadata_dir,
        clean_dir,
        stage_dir,
        reports_dir,
        logs_dir,
        ica_dir,
        final_files_dir,
        backup_info,
    )

    """
    # Generate directory name - use dataset_name if provided, otherwise task name
    if dataset_name:
        dir_name = dataset_name
        message(
            "header",
            f"Setting up BIDS-compliant directories for dataset: {dataset_name} (task: {task})",
        )
    else:
        dir_name = task
        message("header", f"Setting up BIDS-compliant directories for task: {task}")
    autoclean_dir = Path(autoclean_dir_str)
    if not autoclean_dir.exists() and not autoclean_dir.parent.exists():
        raise EnvironmentError(
            f"Parent directory for AUTOCLEAN_DIR does not exist: {autoclean_dir.parent}"
        )

    # Task root directory that will contain BIDS data plus task-level artifacts
    task_root = autoclean_dir / dir_name
    bids_root = task_root / "bids"

    # ------------------------------------------------------------------
    # In-process cache: ensure the backup logic runs at most once per
    # directory within the lifetime of this Python process.  This guards
    # against creating a new backup for every file when batch-processing,
    # while remaining safe if the user interrupts the run with Ctrl-C.
    task_root_resolved = task_root.resolve()

    with _CACHE_LOCK:
        first_time = task_root_resolved not in _PREPARED_TASK_ROOTS
        if first_time:
            _PREPARED_TASK_ROOTS.add(task_root_resolved)

    # Perform backup only the first time we encounter this directory in
    # the current process.
    backup_info = None
    if first_time and task_root.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{dir_name}_backup_{timestamp}"
        backup_path = (autoclean_dir / backup_name).resolve()

        message(
            "warning", f"Directory '{dir_name}' exists, backing up to: {backup_name}"
        )
        shutil.move(str(task_root_resolved), str(backup_path))
        message("info", "Backup complete, creating fresh directory")
        backup_info = {
            "moved_from": str(task_root_resolved),
            "moved_to": str(backup_path),
            "effective_at": datetime.now().isoformat(),
            "initiated_by_run_id": None,  # Filled by caller with actual run_id
            "scope": {"task_root": str(task_root)},
            "reason": "existing directory found; moved to backup",
        }

    # Derivatives for this pipeline under BIDS (versionless)
    derivatives_root = bids_root / "derivatives"

    dirs = {
        "bids": bids_root,
        # Metadata directory removed; repurpose to reports root for compatibility
        "metadata": task_root / "reports",
        "clean": derivatives_root,  # Legacy compatibility (BIDS derivatives root)
        "logs": task_root / "logs",
        # Write per-stage outputs directly under BIDS derivatives
        "stage": derivatives_root,
        "reports": task_root / "reports",
        "ica": task_root / "ica",
        "exports": task_root / "exports",
        "qa": task_root / "qa",
    }

    # Create directories with error handling
    message("info", "Creating directories...")
    try:
        for name, dir_path in dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permission for directory: {dir_path}")
    except Exception as e:
        message("error", f"Failed to create/validate directory {dir_path}: {str(e)}")
        raise

    # Log directory structure
    message("info", "Directory Structure:")
    message("info", f"root: {autoclean_dir}")
    for name, path in dirs.items():
        message("info", f"{name}: {path}")

    message("success", "Directories ready")

    return (
        autoclean_dir,
        dirs["bids"],
        dirs["metadata"],
        dirs["clean"],
        dirs["stage"],
        dirs["reports"],
        dirs["logs"],
        dirs["ica"],
        dirs["exports"],
        backup_info,
    )


def update_status_marker(
    task_root: Path,
    status: str,
    *,
    run_id: str | None = None,
    details: dict | None = None,
    create_task_root: bool = True,
) -> None:
    """Write a JSON status marker for a task run.

    Parameters
    ----------
    task_root : Path
        Root directory for the task (contains bids/, logs/, etc.).
    status : str
        Status label, e.g., 'running', 'completed', 'interrupted'.
    run_id : str, optional
        Unique run identifier. If omitted, writes a task-level status marker.
    details : dict, optional
        Additional metadata to include in the status file.
    create_task_root : bool, optional
        Whether to create the task root directory if it does not exist.
    """

    try:
        if not task_root.exists():
            if not create_task_root:
                return
            task_root.mkdir(parents=True, exist_ok=True)

        status_dir = task_root / STATUS_DIR_NAME
        status_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if run_id:
            payload["run_id"] = run_id
        if details:
            payload["details"] = details

        marker_name = f"{run_id or 'task'}.json"
        marker_path = status_dir / marker_name
        with marker_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    except Exception as exc:  # pylint: disable=broad-except
        message("warning", f"Failed to write status marker: {exc}")
