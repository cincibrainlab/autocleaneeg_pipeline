# src/autoclean/utils/file_system.py
"""
This module contains functions for setting up and validating directory structures.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from autoclean import __version__
from autoclean.utils.logging import message


def step_prepare_directories(
    task: str, autoclean_dir_str: Path, dataset_name: str = None
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
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
    (autoclean_dir, bids_dir, metadata_dir, clean_dir, stage_dir, logs_dir, final_files_dir)

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

    # BIDS-compliant directory structure - everything under derivatives
    bids_root = autoclean_dir / dir_name / "bids"

    # Backup existing directory if it exists
    task_root = autoclean_dir / dir_name
    if task_root.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{dir_name}_backup_{timestamp}"
        backup_path = autoclean_dir / backup_name
        
        message("warning", f"Directory '{dir_name}' exists, backing up to: {backup_name}")
        shutil.move(str(task_root), str(backup_path))
        message("info", "Backup complete, creating fresh directory")

    # Use version for derivatives directory naming
    derivatives_root = bids_root / "derivatives" / f"autoclean-v{__version__}"

    dirs = {
        "bids": bids_root,
        "metadata": derivatives_root / "metadata",
        "clean": derivatives_root,  # Legacy compatibility
        "logs": derivatives_root / "logs",
        "stage": derivatives_root / "intermediate",
        "final_files": bids_root / "final_files",  # New dedicated final files directory
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
        dirs["logs"],
        dirs["final_files"],
    )
