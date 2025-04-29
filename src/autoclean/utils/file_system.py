# src/autoclean/utils/file_system.py
"""
This module contains functions for setting up and validating directory structures.
"""
import os
from pathlib import Path

from autoclean.utils.logging import message


def step_prepare_directories(
    task: str, autoclean_dir_str: Path
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    """Set up and validate directory structure for processing pipeline.

    Parameters
    ----------
    task : str
        The name of the processing task.
    autoclean_dir_str : Path
        The path to the autoclean directory.

    Returns
    -------
    Tuple of Path objects for key directories:
    (autoclean_dir, bids_dir, metadata_dir, clean_dir, stage_dir, logs_dir, flagged_dir)

    """
    message("header", f"Setting up directories for task: {task}")

    autoclean_dir = Path(autoclean_dir_str)
    if not autoclean_dir.exists() and not autoclean_dir.parent.exists():
        raise EnvironmentError(
            f"Parent directory for AUTOCLEAN_DIR does not exist: {autoclean_dir.parent}"
        )

    # Define directory structure
    dirs = {
        "bids": autoclean_dir / task / "bids",
        "metadata": autoclean_dir / task / "metadata",
        "clean": autoclean_dir / task / "postcomps",
        "logs": autoclean_dir / task / "logs",
        "stage": autoclean_dir / task / "stage",
        "flagged": autoclean_dir / task / "flagged",
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
        dirs["flagged"],
    )
