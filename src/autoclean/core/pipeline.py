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
>>> pipeline = Pipeline(
...     autoclean_dir="/path/to/output",
...     autoclean_config="config.yaml"
... )
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

# Standard library imports
import json
import threading  # Add threading import
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Type, Union

import matplotlib

# Third-party imports
import mne
import yaml
from tqdm import tqdm
from ulid import ULID

# IMPORT TASKS HERE
from autoclean.core.task import Task
from autoclean.io.export import save_epochs_to_set, save_raw_to_set
from autoclean.step_functions.reports import (
    create_json_summary,
    create_run_report,
    generate_bad_channels_tsv,
    update_task_processing_log,
)
from autoclean.tasks import task_registry
from autoclean.utils.config import (
    hash_and_encode_yaml,
    load_config,
    validate_eeg_system,
)
from autoclean.utils.database import get_run_record, manage_database, set_database_path
from autoclean.utils.file_system import step_prepare_directories
from autoclean.utils.logging import configure_logger, message

# Force matplotlib to use non-interactive backend for async operations
# This prevents GUI thread conflicts during parallel processing
matplotlib.use("Agg")


class Pipeline:
    """Pipeline class for EEG processing.

    Parameters
    ----------
    autoclean_dir : str or Path
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
    ...     autoclean_dir="results/",
    ...     autoclean_config="configs/default.yaml",
    ...     verbose="debug"  # Enable detailed logging
    ... )
    >>> pipeline.process_file('data/sub-01_task-rest_eeg.raw', 'rest_eyesopen')
    """

    TASK_REGISTRY: Dict[str, Type[Task]] = task_registry

    def __init__(
        self,
        autoclean_dir: str | Path,
        autoclean_config: str | Path,
        verbose: Optional[Union[bool, str, int]] = None,
    ):
        """Initialize a new processing pipeline.

        Parameters
        ----------
        autoclean_dir : str or Path
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


        Examples
        --------
        >>> pipeline = Pipeline(
        ...     autoclean_dir="results/",
        ...     autoclean_config="configs/default.yaml",
        ...     verbose="debug"  # Enable detailed logging
        ... )
        """
        # Convert paths to absolute Path objects
        self.autoclean_dir = Path(autoclean_dir).absolute()
        self.autoclean_config = Path(autoclean_config).absolute()
        # Configure logging first with output directory
        self.verbose = verbose
        mne_verbose = configure_logger(verbose, output_dir=self.autoclean_dir)
        mne.set_log_level(mne_verbose)

        # Add a threading lock for the participants.tsv file
        self.participants_tsv_lock = threading.Lock()

        message("header", "Welcome to AutoClean!")

        # Load YAML config into memory for repeated access during processing
        self.autoclean_dict = load_config(self.autoclean_config)

        # Set global database path
        set_database_path(self.autoclean_dir)

        # Initialize SQLite collection for run tracking
        # This creates tables if they don't exist
        manage_database(operation="create_collection")

        message(
            "success",
            f"✓ Pipeline initialized with output directory: {self.autoclean_dir}",
        )

    def _entrypoint(
        self, unprocessed_file: Path, task: str, run_id: Optional[str] = None
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
            If not provided, a unique ID will be generated.

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
                "lossless_config": self.autoclean_dict["tasks"][task][
                    "lossless_config"
                ],
                "status": "unprocessed",
                "success": False,
                # Define output filenames based on input file
                "json_file": f"{unprocessed_file.stem}_autoclean_metadata.json",
                "report_file": f"{unprocessed_file.stem}_autoclean_report.pdf",
                "metadata": {},
            }

            # Store initial run record and get database ID
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

        try:
            # Perform core validation steps
            self._validate_file(unprocessed_file)
            # Validate EEG system configuration for task
            eeg_system = validate_eeg_system(self.autoclean_dict, task)

            # Prepare directory structure for processing outputs
            (
                autoclean_dir,  # Root output directory
                bids_dir,  # BIDS-compliant data directory
                metadata_dir,  # Processing metadata storage
                clean_dir,  # Cleaned data output
                stage_dir,  # Intermediate processing stages
                logs_dir,  # Debug information and logs
                flagged_dir,  # Flagged data output
            ) = step_prepare_directories(task, self.autoclean_dir)

            # Update database with directory structure
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
                            "flagged": str(flagged_dir),
                        }
                    },
                },
            )

            # Secure configuration files
            b64_config, config_hash = hash_and_encode_yaml(
                self.autoclean_config, is_file=True
            )
            b64_task, task_hash = hash_and_encode_yaml(
                self.autoclean_dict["tasks"][task], is_file=False
            )
            if self.autoclean_dict["tasks"][task]["lossless_config"]:
                b64_ll_config, ll_config_hash = hash_and_encode_yaml(
                    self.autoclean_dict["tasks"][task]["lossless_config"], is_file=True
                )
            else:
                b64_ll_config, ll_config_hash = None, None

            # Prepare configuration for task execution
            run_dict = {
                "run_id": run_id,
                "task": task,
                "eeg_system": eeg_system,
                "config_file": self.autoclean_config,
                "stage_files": self.autoclean_dict["stage_files"],
                "unprocessed_file": unprocessed_file,
                "autoclean_dir": autoclean_dir,
                "bids_dir": bids_dir,
                "metadata_dir": metadata_dir,
                "clean_dir": clean_dir,
                "logs_dir": logs_dir,
                "stage_dir": stage_dir,
                "flagged_dir": flagged_dir,
                "config_hash": config_hash,
                "config_b64": b64_config,
                "task_hash": task_hash,
                "task_b64": b64_task,
                "ll_config_hash": ll_config_hash,
                "ll_config_b64": b64_ll_config,
            }

            lossless_config_path = self.autoclean_dict["tasks"][task]["lossless_config"]
            lossless_config = yaml.safe_load(
                open(lossless_config_path, encoding="utf8")
            )

            # Merge task-specific config with base config
            run_dict = {**run_dict, **self.autoclean_dict, **lossless_config}
            run_dict["participants_tsv_lock"] = self.participants_tsv_lock

            # Record full run configuration
            manage_database(
                operation="update",
                update_record={"run_id": run_id, "metadata": {"entrypoint": run_dict}},
            )

            message("header", f"Starting processing for task: {task}")
            # Instantiate and run task processor
            try:
                task_object = self.TASK_REGISTRY[task.lower()](run_dict)
            except KeyError:
                message(
                    "error",
                    f"Task '{task}' not found in task registry. Class name in task file must match task name exactly.",  # pylint: disable=line-too-long
                )
                raise
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
            except Exception as e:  # pylint: disable=broad-except
                message("error", f"Failed to save completion data: {str(e)}")

            # Mark run as successful in database
            manage_database(
                operation="update",
                update_record={
                    "run_id": run_record["run_id"],
                    "status": "completed",
                    "success": True,
                },
            )

            message("success", f"✓ Task {task} completed successfully")

            # Create a run summary in JSON format
            json_summary = create_json_summary(run_id)

            # Get final run record for report generation
            run_record = get_run_record(run_id)

            # Export run metadata to JSON file
            json_file = metadata_dir / run_record["json_file"]
            with open(json_file, "w", encoding="utf8") as f:
                json.dump(run_record, f, indent=4)
            message("success", f"✓ Run record exported to {json_file}")

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
                create_run_report(run_id, run_dict)
            except Exception as report_error:  # pylint: disable=broad-except
                message("error", f"Failed to generate report: {str(report_error)}")

        except Exception as e:
            # Update database with failure status
            manage_database(
                operation="update",
                update_record={
                    "run_id": run_record["run_id"],
                    "status": "failed",
                    "error": str(e),
                    "success": False,
                },
            )

            json_summary = create_json_summary(run_id)

            # Try to update processing log even in error case
            if json_summary:
                try:
                    flagged, flagged_reasons = task_object.get_flagged_status()
                    update_task_processing_log(json_summary, flagged_reasons)
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
                if run_dict:
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
        self, unprocessed_file: Path, task: str, run_id: Optional[str] = None
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
            await asyncio.to_thread(self._entrypoint, unprocessed_file, task, run_id)
        except Exception as e:
            message("error", f"Failed to process {unprocessed_file}: {str(e)}")
            raise

    def process_file(
        self, file_path: str | Path, task: str, run_id: Optional[str] = None
    ) -> None:
        """Process a single EEG data file.

        Parameters
        ----------
        file_path : str or Path
            Path to the raw EEG data file.
        task : str
            Name of the processing task to run (e.g., 'rest_eyesopen').
        run_id : str, optional
            Optional identifier for the processing run, by default None.
            If not provided, a unique ID will be generated.

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
        """
        self._entrypoint(Path(file_path), task, run_id)

    def process_directory(
        self,
        directory: str | Path,
        task: str,
        pattern: str = "*.set",
        recursive: bool = False,
    ) -> None:
        """Processes all files matching a pattern within a directory sequentially.

        Parameters
        ----------
        directory : str or Path
            Path to the directory containing the EEG files.
        task : str
            The name of the task to perform (e.g., 'RestingEyesOpen').
        pattern : str, optional
            Glob pattern to match files within the directory, default is `*.set`.
        recursive : bool, optional
            If True, searches subdirectories recursively, by default False.

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
        ...     pattern='*.raw',
        ...     recursive=True
        ... )
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        # Find all matching files
        if recursive:
            search_pattern = f"**/{pattern}"
        else:
            search_pattern = pattern

        files = list(directory.glob(search_pattern))
        if not files:
            message("warning", f"No files matching '{pattern}' found in {directory}")
            return

        message("info", f"Found {len(files)} files to process")

        # Process each file
        for file_path in files:
            try:
                self._entrypoint(file_path, task)
            except Exception as e:  # pylint: disable=broad-except
                message("error", f"Failed to process {file_path}: {str(e)}")
                continue

    async def process_directory_async(
        self,
        directory: str | Path,
        task: str,
        pattern: str = "*.raw",
        sub_directories: bool = False,
        max_concurrent: int = 3,
    ) -> None:
        """Processes all files matching a pattern within a directory asynchronously.

        Parameters
        ----------
        directory : str or Path
            Path to the directory containing the EEG files.
        task : str
            The name of the task to perform (e.g., 'RestingEyesOpen').
        pattern : str, optional
            Glob pattern to match files within the directory, default is `*.raw`.
        sub_directories : bool, optional
            If True, searches subdirectories recursively, by default False.
        max_concurrent : int, optional
            Maximum number of files to process concurrently, by default 3.

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
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        # Find all matching files using glob pattern
        if sub_directories:
            search_pattern = f"**/{pattern}"  # Search in subdirectories
        else:
            search_pattern = pattern  # Search only in current directory

        files = list(directory.glob(search_pattern))
        if not files:
            message("warning", f"No files matching '{pattern}' found in {directory}")
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
                    await self._entrypoint_async(file_path, task)
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
            Names of all configured tasks.

        Notes
        -----
        Exposes configured tasks from YAML configuration, providing runtime
        introspection of available processing options. Used for validation
        and user interface integration.

        Examples
        --------
        >>> pipeline.list_tasks()
        ['rest_eyesopen', 'assr_default', 'chirp_default']
        """
        return list(self.TASK_REGISTRY.keys())

    def list_stage_files(self) -> list[str]:
        """Get a list of configured stage file types.

        Returns
        -------
        list of str
            Names of all configured stage file types.

        Provides access to intermediate processing stage definitions from
        configuration. Critical for understanding processing flow and
        debugging pipeline state.

        Examples
        --------
        >>> pipeline.list_stage_files()
        ['post_import', 'post_prepipeline', 'post_clean']
        """
        return list(self.autoclean_dict["stage_files"].keys())

    def start_autoclean_review(self):
        """Launch the AutoClean Review GUI tool.

        Notes
        -----
        This method requires the GUI dependencies to be installed.
        Install them with: pip install autocleaneeg[gui]

        Note: The ideal use of the Review tool is as a docker container.
        """
        try:
            from autoclean.tools.autoclean_review import run_autoclean_review  # pylint: disable=import-outside-toplevel

            run_autoclean_review(self.autoclean_dir)
        except ImportError:
            message(
                "error",
                "GUI dependencies not installed. To use the review tool, install:",
            )
            message("error", "pip install autocleaneeg[gui]")
            raise

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

        if task not in self.autoclean_dict["tasks"]:
            raise ValueError(f"Task '{task}' not found in configuration")

        message("success", f"✓ Task '{task}' found in configuration")
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
