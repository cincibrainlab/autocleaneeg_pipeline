# src/autoclean/core/pipeline.py
"""Core pipeline class for EEG processing.

This module provides the main interface for automated EEG data processing.
The Pipeline class handles:

1. Configuration Management:
   - Loading and validating processing settings
   - Managing output directories
   - Task-specific parameter validation

2. Data Processing:
   - Single file processing
   - Batch processing of multiple files
   - Progress tracking and error handling

3. Results Management:
   - Saving processed data
   - Generating reports
   - Database logging

Examples
--------
Basic usage for processing a single file:

>>> from autoclean import Pipeline
>>> pipeline = Pipeline(output_dir="/path/to/output")
>>> pipeline.process_file(
...     file_path="/path/to/data.set",
...     task="rest_eyesopen"
... )

Processing multiple files:

>>> pipeline.process_directory(
...     directory="/path/to/data",
...     task="rest_eyesopen",
...     pattern="*.raw"
... )

Async processing of multiple files:

>>> pipeline.process_directory_async(
...     directory="/path/to/data",
...     task="rest_eyesopen",
...     pattern="*.raw",
...     max_concurrent=5
... )
"""

import asyncio
import importlib.util
import inspect

# Standard library imports
import json
import sys
import threading  # Add threading import
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Type, Union

import matplotlib

# Third-party imports
import mne
from tqdm import tqdm
from ulid import ULID

# IMPORT TASKS HERE
from autoclean.core.task import Task
from autoclean.io.export import copy_final_files, save_epochs_to_set, save_raw_to_set
from autoclean.io.import_ import discover_event_processors, discover_plugins
from autoclean.step_functions.reports import (
    create_json_summary,
    create_run_report,
    generate_bad_channels_tsv,
    update_task_processing_log,
)
from autoclean.tasks import task_registry
from autoclean.utils.audit import get_task_file_info
from autoclean.utils.auth import (
    create_electronic_signature,
    get_current_user_for_audit,
    require_authentication,
)
from autoclean.utils.config import (
    hash_and_encode_yaml,
)
from autoclean.utils.database import (
    get_run_record,
    manage_database_conditionally,
    set_database_path,
)
from autoclean.utils.file_system import step_prepare_directories
from autoclean.utils.logging import configure_logger, message
from autoclean.utils.user_config import user_config

# Try to import optional GUI dependencies
try:
    from autoclean.tools.autoclean_review import run_autoclean_review

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Force matplotlib to use non-interactive backend for async operations
# This prevents GUI thread conflicts during parallel processing
matplotlib.use("Agg")


def _expand_brace_glob(pattern: str) -> list[str]:
    """Expand simple brace globs like '*.{raw,set}' into ['*.raw', '*.set'].

    This supports a single pair of braces with comma-separated options.
    If no braces are present, returns the pattern as a single-item list.
    """
    if "{" not in pattern or "}" not in pattern:
        return [pattern]

    start = pattern.find("{")
    end = pattern.find("}", start + 1)
    if start == -1 or end == -1 or end < start:
        return [pattern]

    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    body = pattern[start + 1 : end]
    options = [opt.strip() for opt in body.split(",") if opt.strip()]
    if not options:
        return [pattern]
    return [f"{prefix}{opt}{suffix}" for opt in options]


class Pipeline:
    """Pipeline class for EEG processing.

    Parameters
    ----------
    output_dir : str or Path
        Root directory where all processing outputs will be saved.
        The pipeline will create subdirectories for each task.
    autoclean_config : str or Path
        Path to the YAML configuration file that defines
        processing parameters for all tasks.
    verbose : bool, str, int, or None, optional
        Controls logging verbosity, by default None.

        * bool: True for INFO, False for WARNING.
        * str: One of 'debug', 'info', 'warning', 'error', or 'critical'.
        * int: Standard Python logging level (10=DEBUG, 20=INFO, etc.).
        * None: Reads MNE_LOGGING_LEVEL environment variable, defaults to INFO.

    Attributes
    ----------
    TASK_REGISTRY : Dict[str, Type[Task]]
        Automatically generated dictionary of all task classes in the `autoclean.tasks` module.

    See Also
    --------
    autoclean.core.task.Task : Base class for all processing tasks.
    autoclean.io.import_ : I/O functions for data loading and saving.

    Examples
    --------
    >>> from autoclean import Pipeline
    >>> pipeline = Pipeline(
    ...     output_dir="results/",
    ...     autoclean_config="configs/default.yaml",
    ...     verbose="debug"  # Enable detailed logging
    ... )
    >>> pipeline.process_file('data/sub-01_task-rest_eeg.raw', 'rest_eyesopen')
    """

    TASK_REGISTRY: Dict[str, Type[Task]] = task_registry

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ):
        """Initialize a new processing pipeline.

        Parameters
        ----------
        output_dir : str or Path, optional
            Root directory where all processing outputs will be saved.
            The pipeline will create subdirectories for each task.
            If None, defaults to the user's workspace output directory.
        verbose : bool, str, int, or None, optional
            Controls logging verbosity, by default None.

            * bool: True for INFO, False for WARNING.
            * str: One of 'debug', 'info', 'warning', 'error', or 'critical'.
            * int: Standard Python logging level (10=DEBUG, 20=INFO, etc.).
            * None: Reads MNE_LOGGING_LEVEL environment variable, defaults to INFO.


        Examples
        --------
        >>> # Simple usage with custom output directory
        >>> pipeline = Pipeline(output_dir="results/", verbose="debug")
        >>> pipeline.process_file("data.raw", task="RestingEyesOpen")

        >>> # Use default workspace output directory
        >>> pipeline = Pipeline()  # Uses ~/Documents/Autoclean-EEG/output
        >>> pipeline.process_file("data.raw", task="MyCustomTask")

        >>> # Add custom task and use it
        >>> pipeline = Pipeline()
        >>> pipeline.add_task("my_custom_task.py")
        >>> pipeline.process_file("data.raw", task="MyCustomTask")
        """
        # Use default output directory if none provided
        if output_dir is None:
            output_dir = user_config.get_default_output_dir()
            message("info", f"Using default output directory: {output_dir}")

        # Convert paths to absolute Path objects
        self.output_dir = Path(output_dir).absolute()

        # Configure logging first with output directory
        self.verbose = verbose
        mne_verbose = configure_logger(verbose, output_dir=self.output_dir)
        mne.set_log_level(mne_verbose)

        # Add a threading lock for the participants.tsv file
        self.participants_tsv_lock = threading.Lock()

        # Create session-specific task registry (copy of built-in + user tasks)
        self.session_task_registry: Dict[str, Type[Task]] = task_registry.copy()

        message("header", "Welcome to AutoClean!")

        # All configuration now comes from task files directly
        # No external YAML configuration needed

        # Set global database path
        set_database_path(self.output_dir)

        # Initialize SQLite collection for run tracking with audit protection
        # This creates tables if they don't exist and establishes security triggers
        manage_database(operation="create_collection")

        # Pre-initialize plugins to avoid race conditions in async processing
        message("debug", "Pre-initializing plugins for thread safety...")
        discover_plugins()
        discover_event_processors()

        message(
            "success",
            f"✓ Pipeline initialized with output directory: {self.output_dir}",
        )

    def _entrypoint(
        self,
        unprocessed_file: Path,
        task: str,
        run_id: Optional[str] = None,
        montage_override: Optional[str] = None,
    ) -> None:
        """Main processing entrypoint that orchestrates the complete pipeline.

        Parameters
        ----------
        unprocessed_file : Path
            Path to the raw EEG data file.
        task : str
            Name of the processing task to run.
        run_id : str, optional
            Optional identifier for the processing run, by default None.
        montage_override : str, optional
            If provided, forces the task to use this montage instead of the
            value defined in the task configuration.

        Returns
        -------
        str
            The run identifier.

        Notes
        -----
        This is an internal method called by process_file and process_directory.
        Users should not call this method directly.
        """
        task = self._validate_task(task)
        # Either create new run record or resume existing one
        if run_id is None:
            # Generate time-ordered unique ID for run tracking
            run_id = str(ULID())
            # Initialize run record with metadata
            run_record = {
                "run_id": run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "task": task,
                "unprocessed_file": str(unprocessed_file),
                "status": "unprocessed",
                "success": False,
                # Define output filenames based on input file
                "json_file": f"{unprocessed_file.stem}_autoclean_metadata.json",
                "report_file": f"{unprocessed_file.stem}_autoclean_report.pdf",
                "user_context": get_current_user_for_audit(),
                "metadata": {},
            }

            # Store initial run record and get database ID with audit protection
            run_record["record_id"] = manage_database(
                operation="store", run_record=run_record
            )

        else:
            # Convert run_id to string for consistency
            run_id = str(run_id)
            # Load existing run record for resumed processing
            run_record = get_run_record(run_id)
            message("info", f"Resuming run {run_id}")
            message("info", f"Run record: {run_record}")

        # Initialize run_dict early for error handling
        run_dict = None

        try:
            # Perform core validation steps
            self._validate_file(unprocessed_file)

            # Extract dataset_name from task configuration if available
            from autoclean.utils.task_discovery import extract_config_from_task

            dataset_name = extract_config_from_task(task, "dataset_name")

            # Prepare directory structure for processing outputs
            (
                autoclean_dir,  # Root output directory
                bids_dir,  # BIDS-compliant data directory
                metadata_dir,  # Processing metadata storage
                clean_dir,  # Cleaned data output (legacy)
                stage_dir,  # Intermediate processing stages
                reports_dir,  # Centralized report artifacts
                logs_dir,  # Debug information and logs
                ica_dir,  # ICA FIF storage directory
                final_files_dir,  # Final processed files directory
                backup_info,  # Optional backup move details
            ) = step_prepare_directories(task, self.output_dir, dataset_name)

            # If an auto-backup occurred, persist minimal info in current run metadata
            if backup_info:
                backup_info["initiated_by_run_id"] = run_id
                try:
                    manage_database(
                        operation="update",
                        update_record={
                            "run_id": run_id,
                            "metadata": {"directory_backup": backup_info},
                        },
                    )
                except Exception as e:  # pylint: disable=broad-except
                    message("warning", f"Failed to write backup info to DB metadata: {e}")
                # Also add an audit/access log entry
                try:
                    manage_database(
                        operation="add_access_log",
                        run_record={
                            "timestamp": datetime.now().isoformat(),
                            "operation": "directory_backup",
                            "user_context": get_current_user_for_audit(),
                            "details": backup_info,
                        },
                    )
                except Exception as e:  # pylint: disable=broad-except
                    message("warning", f"Failed to add audit log for backup: {e}")

            # Update database with directory structure using audit protection
            manage_database(
                operation="update",
                update_record={
                    "run_id": run_id,
                    "metadata": {
                        "step_prepare_directories": {
                            "bids": str(bids_dir),
                            "metadata": str(metadata_dir),
                            "clean": str(clean_dir),
                            "logs": str(logs_dir),
                            "stage": str(stage_dir),
                            "reports": str(reports_dir),
                            "ica_fif": str(ica_dir),
                            "final_files": str(final_files_dir),
                        }
                    },
                },
            )

            # Create minimal config for Python task tracking
            task_config = {"type": "python_task", "class_name": task}
            b64_config, config_hash = hash_and_encode_yaml(
                {"version": "1.0", "type": "python_tasks_only"}, is_file=False
            )
            b64_task, task_hash = hash_and_encode_yaml(task_config, is_file=False)

            # Prepare configuration for task execution
            run_dict = {
                "run_id": run_id,
                "task": task,
                "unprocessed_file": unprocessed_file,
                "output_dir": self.output_dir,
                "bids_dir": bids_dir,
                "metadata_dir": metadata_dir,
                "clean_dir": clean_dir,  # Legacy compatibility
                "logs_dir": logs_dir,
                "stage_dir": stage_dir,
                "reports_dir": reports_dir,
                "ica_dir": ica_dir,
                "final_files_dir": final_files_dir,  # New final files directory
                "config_hash": config_hash,
                "config_b64": b64_config,
                "task_hash": task_hash,
                "task_b64": b64_task,
            }
            if montage_override:
                run_dict["override_montage"] = montage_override
                message("info", f"Applying montage override: {montage_override}")
            run_dict["participants_tsv_lock"] = self.participants_tsv_lock

            # Record full run configuration using audit protection
            manage_database(
                operation="update",
                update_record={"run_id": run_id, "metadata": {"entrypoint": run_dict}},
            )

            # Reconfigure logger with task-specific directory
            mne_verbose = configure_logger(self.verbose, logs_dir=logs_dir)
            mne.set_log_level(mne_verbose)

            message("header", f"Starting processing for task: {task}")
            # Instantiate and run task processor
            try:
                task_object = self.session_task_registry[task.lower()](run_dict)
            except KeyError:
                message(
                    "error",
                    f"Task '{task}' not found in task registry. Class name in task file must match task name exactly.",  # pylint: disable=line-too-long
                )
                raise

            # Capture task file information for compliance tracking
            task_file_info = get_task_file_info(task, task_object)

            # Store task file information in database
            manage_database(
                operation="update",
                update_record={"run_id": run_id, "task_file_info": task_file_info},
            )

            task_object.run()

            try:
                flagged, flagged_reasons = task_object.get_flagged_status()
                comp_data = task_object.get_epochs()
                if comp_data is not None:
                    save_epochs_to_set(
                        epochs=comp_data,
                        autoclean_dict=run_dict,
                        stage="post_comp",
                        flagged=flagged,
                    )
                else:
                    comp_data = task_object.get_raw()
                    save_raw_to_set(
                        raw=comp_data,
                        autoclean_dict=run_dict,
                        stage="post_comp",
                        flagged=flagged,
                    )

                # Copy final files to the dedicated final_files directory
                if not flagged or not run_dict.get("move_flagged_files", True):
                    copy_final_files(run_dict)

            except Exception as e:  # pylint: disable=broad-except
                message("error", f"Failed to save completion data: {str(e)}")

            message("success", f"✓ Task {task} completed successfully")

            # Set success status FIRST so JSON summary can detect success correctly
            manage_database(
                operation="update",
                update_record={
                    "run_id": run_record["run_id"],
                    "success": True,
                },
            )

            message("success", f"✓ Task {task} completed successfully")

            # Create a run summary in JSON format
            json_summary = create_json_summary(run_id, flagged_reasons)

            # Only proceed with processing log update if we have a valid summary
            if json_summary:
                # Update processing log
                update_task_processing_log(json_summary, flagged_reasons)
                try:
                    generate_bad_channels_tsv(json_summary)
                except Exception as tsv_error:  # pylint: disable=broad-except
                    message(
                        "warning",
                        f"Failed to generate bad channels tsv: {str(tsv_error)}",
                    )
            else:
                message(
                    "warning",
                    "Could not create JSON summary, processing log will not be updated",
                )

            # Generate PDF report if processing succeeded
            try:
                create_run_report(run_id, run_dict, json_summary)
            except Exception as report_error:  # pylint: disable=broad-except
                message("error", f"Failed to generate report: {str(report_error)}")
            else:
                # Attempt LLM-backed textual reports using processing log + PDF
                try:
                    task_object.emit_llm_reports()
                except Exception as llm_err:  # pylint: disable=broad-except
                    message("warning", f"LLM reporting skipped: {llm_err}")

            # Create electronic signature for compliance mode
            signature_id = create_electronic_signature(
                run_record["run_id"], "processing_completion"
            )
            if signature_id:
                message("debug", f"Electronic signature created: {signature_id}")

            # Mark run as completed LAST - this locks the record from further modifications
            # Include JSON summary in the completion update to avoid audit record conflicts
            update_record = {
                "run_id": run_record["run_id"],
                "status": "completed",
            }
            if json_summary:
                update_record["metadata"] = {"json_summary": json_summary}

            manage_database(
                operation="update",
                update_record=update_record,
            )

            # Get final run record for JSON export
            run_record = get_run_record(run_id)

            # Export run metadata to JSON file
            json_file = metadata_dir / run_record["json_file"]
            with open(json_file, "w", encoding="utf8") as f:
                json.dump(run_record, f, indent=4)
            message("success", f"✓ Run record exported to {json_file}")

        except Exception as e:
            # Get flagged status before creating summary for error case
            try:
                flagged, error_flagged_reasons = task_object.get_flagged_status()
            except Exception:  # pylint: disable=broad-except
                error_flagged_reasons = []

            json_summary = create_json_summary(run_id, error_flagged_reasons)

            # Update database with failure status using audit protection
            # Include JSON summary in the failure update to avoid audit record conflicts
            update_record = {
                "run_id": run_record["run_id"],
                "status": "failed",
                "error": str(e),
                "success": False,
            }
            if json_summary:
                update_record["metadata"] = {"json_summary": json_summary}

            manage_database(
                operation="update",
                update_record=update_record,
            )

            # Try to update processing log even in error case
            if json_summary:
                try:
                    update_task_processing_log(json_summary, error_flagged_reasons)
                except Exception as log_error:  # pylint: disable=broad-except
                    message(
                        "warning", f"Failed to update processing log: {str(log_error)}"
                    )
                try:
                    generate_bad_channels_tsv(json_summary)
                except Exception as tsv_error:  # pylint: disable=broad-except
                    message(
                        "warning",
                        f"Failed to generate bad channels tsv: {str(tsv_error)}",
                    )
            else:
                message("warning", "Could not create JSON summary for error case")

            # Attempt to generate error report
            try:
                if run_dict is not None:
                    create_run_report(run_id, run_dict)
                else:
                    create_run_report(run_id)
            except Exception as report_error:  # pylint: disable=broad-except
                message(
                    "error", f"Failed to generate error report: {str(report_error)}"
                )

            message("error", f"Run {run_record['run_id']} Pipeline failed: {e}")
            raise

        return run_record["run_id"]

    async def _entrypoint_async(
        self,
        unprocessed_file: Path,
        task: str,
        run_id: Optional[str] = None,
        montage_override: Optional[str] = None,
    ) -> None:
        """Async version of _entrypoint for concurrent processing.

        Parameters
        ----------
        unprocessed_file : Path
            Path to the raw EEG data file.
        task : str
            Name of the processing task to run.
        run_id : str, optional
            Optional identifier for the processing run, by default None.

        Notes
        -----
        Wraps synchronous processing in asyncio thread pool to enable
        non-blocking concurrent execution while maintaining database
        and filesystem operation safety.
        """
        try:
            # Run the processing in a thread to avoid blocking
            await asyncio.to_thread(
                self._entrypoint,
                unprocessed_file,
                task,
                run_id,
                montage_override,
            )
        except Exception as e:
            message("error", f"Failed to process {unprocessed_file}: {str(e)}")
            raise

    @require_authentication
    def process_file(
        self,
        file_path: Optional[str | Path] = None,
        task: str = "",
        run_id: Optional[str] = None,
        montage_override: Optional[str] = None,
    ) -> None:
        """Process a single EEG data file.

        Parameters
        ----------
        file_path : str or Path, optional
            Path to the raw EEG data file. If not provided, will attempt to use
            input_path from the task configuration.
        task : str
            Name of the processing task to run (e.g., 'rest_eyesopen').
        run_id : str, optional
            Optional identifier for the processing run, by default None.
            If not provided, a unique ID will be generated.
        montage_override : str, optional
            Overrides the montage specified by the task configuration.

        See Also
        --------
        process_directory : Process multiple files in a directory.
        process_directory_async : Process files asynchronously.

        Examples
        --------
        >>> pipeline.process_file(
        ...     file_path='data/sub-01_task-rest_eeg.raw',
        ...     task='rest_eyesopen'
        ... )
        >>> # Or use input_path from task config
        >>> pipeline.process_file(task='rest_eyesopen')
        """
        # Use input_path from task config if file_path not provided
        if file_path is None:
            from autoclean.utils.task_discovery import extract_config_from_task

            task_input_path = extract_config_from_task(task, "input_path")
            if task_input_path:
                file_path = Path(task_input_path)
                message("info", f"Using input path from task config: {file_path}")
            else:
                raise ValueError(
                    "file_path must be provided or task must have input_path in config"
                )

        self._entrypoint(Path(file_path), task, run_id, montage_override)

    @require_authentication
    def process_directory(
        self,
        directory: Optional[str | Path] = None,
        task: str = "",
        pattern: str = "*.{raw,set}",
        recursive: bool = False,
        montage_override: Optional[str] = None,
    ) -> None:
        """Processes all files matching a pattern within a directory sequentially.

        Parameters
        ----------
        directory : str or Path, optional
            Path to the directory containing the EEG files. If not provided, will
            attempt to use input_path from the task configuration.
        task : str
            The name of the task to perform (e.g., 'RestingEyesOpen').
        pattern : str, optional
            Glob pattern to match files within the directory, default is `*.{raw,set}`.
        recursive : bool, optional
            If True, searches subdirectories recursively, by default False.
        montage_override : str, optional
            Override montage configuration for all files processed in this batch.

        See Also
        --------
        process_file : Process a single file.
        process_directory_async : Process files asynchronously.

        Notes
        -----
        If processing fails for one file, the pipeline will continue
        with the remaining files and report all errors at the end.

        Examples
        --------
        >>> pipeline.process_directory(
        ...     directory='data/rest_state/',
        ...     task='rest_eyesopen',
        ...     pattern='*.{raw,set}',
        ...     recursive=True
        ... )
        >>> # Or use input_path from task config
        >>> pipeline.process_directory(task='rest_eyesopen', pattern='*.{raw,set}')
        """
        # Use input_path from task config if directory not provided
        if directory is None:
            from autoclean.utils.task_discovery import extract_config_from_task

            task_input_path = extract_config_from_task(task, "input_path")
            if task_input_path:
                directory = Path(task_input_path)
                message("info", f"Using input path from task config: {directory}")
            else:
                raise ValueError(
                    "directory must be provided or task must have input_path in config"
                )

        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        # Auto-fix common pattern mistakes
        if pattern.startswith(".") and not pattern.startswith("*."):
            pattern = f"*{pattern}"

        # Find all matching files
        if recursive:
            search_pattern = f"**/{pattern}"
        else:
            search_pattern = pattern

        # Debug: Show what we're searching for
        message("debug", f"Searching in directory: {directory}")
        message("debug", f"Using search pattern: {search_pattern}")

        # List all files in directory for debugging
        all_files = list(directory.iterdir())
        message(
            "debug",
            f"All files in directory ({len(all_files)}): {[f.name for f in all_files if f.is_file()]}",
        )

        # Support brace expansion patterns like '*.{raw,set}'
        files: list[Path] = []
        seen: set[Path] = set()
        for pat in _expand_brace_glob(search_pattern):
            for f in directory.glob(pat):
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        message("debug", f"Files matching pattern: {[f.name for f in files]}")

        if not files:
            message("warning", f"No files matching '{pattern}' found in {directory}")
            all_file_names = [f.name for f in all_files if f.is_file()]
            message("info", f"Available files: {all_file_names}")

            # No need for manual suggestion since auto-correction happens above

            return

        message("info", f"Found {len(files)} files to process")

        # Process each file
        for file_path in files:
            try:
                self._entrypoint(file_path, task, montage_override=montage_override)
            except Exception as e:  # pylint: disable=broad-except
                message("error", f"Failed to process {file_path}: {str(e)}")
                continue

    @require_authentication
    async def process_directory_async(
        self,
        directory_path: Optional[str | Path] = None,
        task: str = "",
        pattern: str = "*.{raw,set}",
        sub_directories: bool = False,
        max_concurrent: int = 3,
        montage_override: Optional[str] = None,
    ) -> None:
        """Processes all files matching a pattern within a directory asynchronously.

        Parameters
        ----------
        directory_path : str or Path, optional
            Path to the directory containing the EEG files. If not provided, will
            attempt to use input_path from the task configuration.
        task : str
            The name of the task to perform (e.g., 'RestingEyesOpen').
        pattern : str, optional
            Glob pattern to match files within the directory, default is `*.{raw,set}`.
        sub_directories : bool, optional
            If True, searches subdirectories recursively, by default False.
        max_concurrent : int, optional
            Maximum number of files to process concurrently, by default 3.
        montage_override : str, optional
            Override montage configuration for all files processed in this batch.

        See Also
        --------
        process_file : Process a single file.
        process_directory : Process files synchronously.

        Notes
        -----
        Implements concurrent batch processing using asyncio semaphores
        for resource management. Processes files in optimized batches
        while maintaining progress tracking and error isolation.
        """
        # Use input_path from task config if directory_path not provided
        if directory_path is None:
            from autoclean.utils.task_discovery import extract_config_from_task

            task_input_path = extract_config_from_task(task, "input_path")
            if task_input_path:
                directory_path = Path(task_input_path)
                message("info", f"Using input path from task config: {directory_path}")
            else:
                raise ValueError(
                    "directory_path must be provided or task must have input_path in config"
                )

        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        # Auto-fix common pattern mistakes
        if pattern.startswith(".") and not pattern.startswith("*."):
            pattern = f"*{pattern}"

        # Find all matching files using glob pattern
        if sub_directories:
            search_pattern = f"**/{pattern}"  # Search in subdirectories
        else:
            search_pattern = pattern  # Search only in current directory

        # Debug: Show what we're searching for
        message("debug", f"Async searching in directory: {directory_path}")
        message("debug", f"Using search pattern: {search_pattern}")

        # List all files in directory for debugging
        all_files = list(directory_path.iterdir())
        message(
            "debug",
            f"All files in directory ({len(all_files)}): {[f.name for f in all_files if f.is_file()]}",
        )

        # Support brace expansion patterns like '*.{raw,set}'
        files: list[Path] = []
        seen: set[Path] = set()
        for pat in _expand_brace_glob(search_pattern):
            for f in directory_path.glob(pat):
                if f not in seen:
                    seen.add(f)
                    files.append(f)
        message("debug", f"Files matching pattern: {[f.name for f in files]}")

        if not files:
            message(
                "warning", f"No files matching '{pattern}' found in {directory_path}"
            )
            all_file_names = [f.name for f in all_files if f.is_file()]
            message("info", f"Available files: {all_file_names}")

            # No need for manual suggestion since auto-correction happens above

            return

        message(
            "info",
            f"\nStarting processing of {len(files)} files with {max_concurrent} concurrent workers",
        )

        # Create semaphore to prevent resource exhaustion
        sem = asyncio.Semaphore(max_concurrent)

        # Initialize progress tracking
        pbar = tqdm(total=len(files), desc="Processing files", unit="file")

        async def process_with_semaphore(file_path: Path) -> None:
            """Process a single file with semaphore control."""
            async with sem:  # Limit overall concurrent processing
                try:
                    # Pass the acquired lock information (implicitly via self if needed later,
                    # but the lock is mainly for the bids step itself)
                    await self._entrypoint_async(
                        file_path, task, montage_override=montage_override
                    )
                    pbar.write(f"✓ Completed: {file_path.name}")
                except Exception as e:  # pylint: disable=broad-except
                    pbar.write(f"✗ Failed: {file_path.name} - {str(e)}")
                finally:
                    pbar.update(1)  # Update progress regardless of outcome

        try:
            # Process files in batches to optimize memory usage
            # Batch size is double the concurrent limit to ensure worker saturation
            batch_size = max_concurrent * 2
            for i in range(0, len(files), batch_size):
                batch = files[i : i + batch_size]
                # Create task list for current batch
                tasks = [process_with_semaphore(f) for f in batch]
                # Process batch with error handling
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            pbar.close()

        # Print processing summary
        message("info", "\nProcessing Summary:")
        message("info", f"Total files processed: {len(files)}")
        message("info", "Check individual file logs for detailed status")

    def list_tasks(self) -> list[str]:
        """Get a list of available processing tasks.

        Returns
        -------
        list of str
            Names of all available tasks (both built-in and user-registered).

        Notes
        -----
        Exposes all tasks available in this pipeline session, including
        built-in tasks and any user-registered Python task files.

        Examples
        --------
        >>> pipeline.list_tasks()
        ['rest_eyesopen', 'assr_default', 'chirp_default', 'mycustomtask']
        """
        return list(self.session_task_registry.keys())

    def list_stage_files(self) -> list[str]:
        """Get a list of default stage file types.

        Returns
        -------
        list of str
            Names of all default stage file types.

        Provides access to intermediate processing stage definitions.
        Critical for understanding processing flow and debugging pipeline state.

        Examples
        --------
        >>> pipeline.list_stage_files()
        ['post_import', 'post_basic_steps', 'post_clean_raw', 'post_epochs', 'post_comp']
        """
        return [
            "post_import",
            "post_basic_steps",
            "post_clean_raw",
            "post_epochs",
            "post_comp",
        ]

    def start_autoclean_review(self):
        """Launch the AutoClean Review GUI tool.

        Notes
        -----
        This method requires the GUI dependencies to be installed.
        Install them with: pip install autocleaneeg-pipeline[gui]

        Note: The ideal use of the Review tool is as a docker container.
        """
        if not GUI_AVAILABLE:
            message(
                "error",
                "GUI dependencies not installed. To use the review tool, install:",
            )
            message("error", "pip install autocleaneeg-pipeline[gui]")
            raise ImportError("GUI dependencies not available")

        run_autoclean_review(self.output_dir)

    def add_task(self, task_file_path: Union[str, Path]) -> str:
        """Register a Python task file for use in this pipeline session.

        Parameters
        ----------
        task_file_path : str or Path
            Path to the Python file containing the task class definition.

        Returns
        -------
        str
            The name of the registered task class.

        Examples
        --------
        >>> pipeline = Pipeline(output_dir="output/")
        >>> task_name = pipeline.add_task("my_resting_task.py")
        >>> pipeline.process_file("data.set", task=task_name)
        """
        task_file_path = Path(task_file_path)
        if not task_file_path.exists():
            raise FileNotFoundError(f"Task file not found: {task_file_path}")

        message("info", f"Loading Python task file: {task_file_path}")
        task_class = self._load_python_task(task_file_path)

        # Register in session registry (case-insensitive key)
        task_name = task_class.__name__.lower()
        self.session_task_registry[task_name] = task_class

        message(
            "success",
            f"✓ Registered task '{task_class.__name__}' from {task_file_path}",
        )
        return task_class.__name__  # Return original case class name

    def _load_python_task(self, task_file_path: Path) -> Type[Task]:
        """Dynamically load a Task class from a Python file.

        Parameters
        ----------
        task_file_path : Path
            Path to the Python file containing the task class.

        Returns
        -------
        Type[Task]
            The loaded Task class.

        Raises
        ------
        ImportError
            If the file cannot be imported or contains no Task classes.
        """
        # Create module spec from file
        module_name = f"user_task_{task_file_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, task_file_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {task_file_path}")

        # Import the module
        module = importlib.util.module_from_spec(spec)

        # Add to sys.modules to support relative imports
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Clean up sys.modules on failure
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise ImportError(
                f"Failed to execute module {task_file_path}: {str(e)}"
            ) from e

        # Find Task subclasses in the module
        task_classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, Task) and obj != Task and obj.__module__ == module_name
            ):  # Only classes defined in this module
                task_classes.append(obj)

        if not task_classes:
            # Clean up sys.modules
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise ImportError(f"No Task subclasses found in {task_file_path}")

        if len(task_classes) > 1:
            message(
                "warning",
                f"Multiple Task classes found in {task_file_path}, using first: {task_classes[0].__name__}",
            )

        return task_classes[0]

    def _validate_task(self, task: str) -> str:
        """Validate that a task type is supported and properly configured.

        Parameters
        ----------
        task : str
            Name of the task to validate (e.g., 'rest_eyesopen').

        Returns
        -------
        str
            The validated task name.

        Notes
        -----
        Ensures task exists in configuration and has required parameters.
        Acts as a guard clause for task instantiation, preventing invalid
        task configurations from entering the processing pipeline.

        Examples
        --------
        >>> pipeline._validate_task('rest_eyesopen')
        'rest_eyesopen'
        """
        message("debug", "Validating task")

        # First check if task exists in session registry (includes Python tasks)
        if task.lower() not in self.session_task_registry:
            # Check if it's a custom task in user config (auto-discovers new tasks)
            custom_task_path = user_config.get_custom_task_path(task)
            if custom_task_path:
                # Load the custom task into session registry
                loaded_task_name = self.add_task(custom_task_path)
                message("info", f"Loaded custom task '{task}' from user configuration")
                return loaded_task_name
            else:
                available_tasks = list(self.session_task_registry.keys())
                custom_tasks = list(user_config.list_custom_tasks().keys())
                if custom_tasks:
                    available_tasks.extend([f"{t} (custom)" for t in custom_tasks])
                raise ValueError(
                    f"Task '{task}' not found. Available tasks: {available_tasks}"
                )

        message("success", f"✓ Task '{task}' validated")
        return task

    def _validate_file(self, file_path: str | Path) -> Path:
        """Validate that an input file exists and is accessible.

        Parameters
        ----------
        file_path : str or Path
            Path to the EEG data file to validate.

        Returns
        -------
        Path
            The validated file path.

        Notes
        -----
        Performs filesystem-level validation using pathlib, ensuring atomic
        file operations can proceed. Normalizes paths for cross-platform
        compatibility.

        Examples
        --------
        >>> pipeline._validate_file('data/sub-01_task-rest_eeg.raw')
        Path('data/sub-01_task-rest_eeg.raw')
        """
        message("debug", "Validating file")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        message("success", f"✓ File '{file_path}' found")
        return path
# Backward compatibility: expose manage_database for test patches
manage_database = manage_database_conditionally
