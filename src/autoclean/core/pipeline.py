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

Example:
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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Type, Union

import matplotlib

# Third-party imports
import mne
from tqdm import tqdm
from ulid import ULID
import yaml

# IMPORT TASKS HERE
from autoclean.core.task import Task
from autoclean.step_functions.io import save_epochs_to_set, save_raw_to_set
# NOTE: The following imports are using deprecated functions. 
# They should eventually be migrated to use the ReportingMixin instead.
from autoclean.step_functions.reports import (
    create_json_summary,
    create_run_report,
    update_task_processing_log,
    generate_bad_channels_tsv,
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
    """Main pipeline class for EEG processing.

    This class serves as the primary interface for the autoclean package.
    It manages the complete processing workflow including:

    - Configuration loading and validation
    - Directory structure setup
    - Task instantiation and execution
    - Progress tracking and error handling
    - Results saving and report generation

    The pipeline supports multiple EEG processing paradigms through its task registry,
    allowing researchers to process different types of EEG recordings with appropriate
    analysis pipelines.

    Core orchestrator that maintains state through a SQLite-based run tracking system,
    manages file system operations via pathlib, and coordinates task execution while
    ensuring atomic operations and proper error handling.

    Attributes:
        TASK_REGISTRY (Dict[str, Type[Task]]): Available processing task types.
            Current tasks include:
            - 'rest_eyesopen': Resting state with eyes open
            - 'assr_default': Auditory steady-state response
            - 'chirp_default': Auditory chirp paradigm
    """

    TASK_REGISTRY: Dict[str, Type[Task]] = task_registry

    def __init__(
        self,
        autoclean_dir: str | Path,
        autoclean_config: str | Path,
        verbose: Optional[Union[bool, str, int]] = None,
    ):
        """Initialize a new processing pipeline.

        Establishes core pipeline state by initializing SQLite database connection,
        loading YAML configuration into memory, and setting up filesystem paths.
        Handles path normalization and validation of core dependencies.

        Args:
            autoclean_dir: Root directory where all processing outputs will be saved.
                          The pipeline will create subdirectories for each task.
            autoclean_config: Path to the YAML configuration file that defines
                            processing parameters for all tasks.
            use_async: Whether to use asynchronous processing. Currently not
                      implemented, defaults to False.
            verbose: Controls logging verbosity. Can be:
                    - bool: True for INFO, False for WARNING
                    - str: One of 'debug', 'info', 'warning', 'error', or 'critical'
                    - int: Standard Python logging level (10=DEBUG, 20=INFO, etc.)
                    - None: Reads MNE_LOGGING_LEVEL environment variable, defaults to INFO

        Raises:
            FileNotFoundError: If config_file doesn't exist
            ValueError: If configuration is invalid
            PermissionError: If output directory is not writable

        Example:
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
        configure_logger(verbose, output_dir=self.autoclean_dir)
        mne.set_log_level(verbose)

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

        Implements core processing logic with ACID-compliant database operations,
        filesystem management, and error handling. Uses ULID for time-ordered
        run tracking and maintains atomic operation guarantees.

        Args:
            unprocessed_file: Path to the raw EEG data file
            task: Name of the processing task to run
            run_id: Optional identifier for the processing run. If not provided,
                   a unique ID will be generated.

        Raises:
            ValueError: If task is not registered or configuration is invalid
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If processing fails

        Note:
            This is an internal method called by process_file and process_directory.
            Users should not call this method directly.
        """
        task = self._validate_task(task)
        configure_logger(self.verbose, output_dir=self.autoclean_dir, task=task)
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
            lossless_config = yaml.safe_load(open(lossless_config_path))

            # Merge task-specific config with base config
            run_dict = {**run_dict, **self.autoclean_dict, **lossless_config}

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
                message("error", f"Task '{task}' not found in task registry. Class name in task file must match task name exactly.")
                raise
            task_object.run()

            try:
                flagged, flagged_reasons = task_object.get_flagged_status()
                comp_data = task_object.get_epochs()
                if comp_data is not None:
                    save_epochs_to_set(epochs = comp_data, autoclean_dict = run_dict, stage = "post_comp", flagged = flagged)
                else:
                    comp_data = task_object.get_raw()
                    save_raw_to_set(raw = comp_data, autoclean_dict = run_dict, stage = "post_comp", flagged = flagged)
            except Exception as e:
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

            #Create a run summary in JSON format
            json_summary = create_json_summary(run_id)

            # Get final run record for report generation
            run_record = get_run_record(run_id)

            # Export run metadata to JSON file
            json_file = metadata_dir / run_record["json_file"]
            with open(json_file, "w") as f:
                json.dump(run_record, f, indent=4)
            message("success", f"✓ Run record exported to {json_file}")
            
            # Only proceed with processing log update if we have a valid summary
            if json_summary:
                # Update processing log
                update_task_processing_log(json_summary, flagged_reasons)
                try:
                    generate_bad_channels_tsv(json_summary)
                except Exception as tsv_error:
                    message("warning", f"Failed to generate bad channels tsv: {str(tsv_error)}")
            else:
                message("warning", "Could not create JSON summary, processing log will not be updated")

            # Generate PDF report if processing succeeded
            try:
                create_run_report(run_id, run_dict)
            except Exception as report_error:
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
                except Exception as log_error:
                    message("warning", f"Failed to update processing log: {str(log_error)}")
                try:
                    generate_bad_channels_tsv(json_summary)
                except Exception as tsv_error:
                    message("warning", f"Failed to generate bad channels tsv: {str(tsv_error)}")
            else:
                message("warning", "Could not create JSON summary for error case")

            # Attempt to generate error report
            try:
                if run_dict:
                    create_run_report(run_id, run_dict)
                else:
                    create_run_report(run_id)
            except Exception as report_error:
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

        Public interface for single-file processing that ensures proper path
        handling and maintains processing state through database-backed run
        tracking. Delegates core processing to _entrypoint.

        Args:
            file_path: Path to the raw EEG data file
            task: Name of the processing task to run (e.g., 'rest_eyesopen')
            run_id: Optional identifier for the processing run. If not provided,
                   a unique ID will be generated.

        Raises:
            ValueError: If task is not registered or configuration is invalid
            FileNotFoundError: If input file doesn't exist
            RuntimeError: If processing fails

        Example:
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
        """Process all matching EEG files in a directory.

        Implements fault-tolerant batch processing using pathlib for filesystem
        operations. Maintains independent error handling per file to prevent
        cascade failures in batch operations.

        Args:
            directory: Path to the directory containing EEG files
            task: Name of the processing task to run (e.g., 'RestingEyesOpen')
            pattern: Glob pattern to match files (default: "*.raw")
            recursive: Whether to search in subdirectories (default: False)

        Raises:
            NotADirectoryError: If directory doesn't exist
            ValueError: If task is not registered
            RuntimeError: If processing fails for any file

        Example:
            >>> pipeline.process_directory(
            ...     directory='data/rest_state/',
            ...     task='rest_eyesopen',
            ...     pattern='*.raw',
            ...     recursive=True
            ... )

        Note:
            If processing fails for one file, the pipeline will continue
            with the remaining files and report all errors at the end.
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
            except Exception as e:
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
        """Process all matching EEG files in a directory asynchronously.

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
            async with sem:  # Limit concurrent processing
                try:
                    await self._entrypoint_async(file_path, task)
                    pbar.write(f"✓ Completed: {file_path.name}")
                except Exception as e:
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

        Exposes configured tasks from YAML configuration, providing runtime
        introspection of available processing options. Used for validation
        and user interface integration.

        Returns:
            list[str]: Names of all configured tasks

        Example:
            >>> pipeline.list_tasks()
            ['rest_eyesopen', 'assr_default', 'chirp_default']
        """
        return list(self.task_registry.keys())


    def list_stage_files(self) -> list[str]:
        """Get a list of configured stage file types.

        Provides access to intermediate processing stage definitions from
        configuration. Critical for understanding processing flow and
        debugging pipeline state.

        Returns:
            list[str]: Names of all configured stage file types

        Example:
            >>> pipeline.list_stage_files()
            ['post_import', 'post_prepipeline', 'post_clean']
        """
        return list(self.autoclean_dict["stage_files"].keys())


    def start_autoclean_review(self):
        """Launch the AutoClean Review GUI tool.
        
        This method requires the GUI dependencies to be installed.
        Install them with: pip install autocleaneeg[gui]
        """
        try:
            from autoclean.tools.autoclean_review import run_autoclean_review
            run_autoclean_review(self.autoclean_dir)
        except ImportError:
            message("error", "GUI dependencies not installed. To use the review tool, install:")
            message("error", "pip install autocleaneeg[gui]")
            raise

    def _validate_task(self, task: str) -> None:
        """Validate that a task type is supported and properly configured.

        Ensures task exists in configuration and has required parameters.
        Acts as a guard clause for task instantiation, preventing invalid
        task configurations from entering the processing pipeline.

        Args:
            task: Name of the task to validate (e.g., 'rest_eyesopen')

        Returns:
            str: The validated task name

        Raises:
            ValueError: If the task is not found in configuration

        Example:
            >>> pipeline.validate_task('rest_eyesopen')
            'rest_eyesopen'
        """
        message("debug", "Validating task")

        if task not in self.autoclean_dict["tasks"]:
            raise ValueError(f"Task '{task}' not found in configuration")

        message("success", f"✓ Task '{task}' found in configuration")
        return task

    def _validate_file(self, file_path: str | Path) -> None:
        """Validate that an input file exists and is accessible.

        Performs filesystem-level validation using pathlib, ensuring atomic
        file operations can proceed. Normalizes paths for cross-platform
        compatibility.

        Args:
            file_path: Path to the EEG data file to validate

        Returns:
            Path: The validated file path

        Raises:
            FileNotFoundError: If the file doesn't exist

        Example:
            >>> pipeline.validate_file('data/sub-01_task-rest_eeg.raw')
            Path('data/sub-01_task-rest_eeg.raw')
        """
        message("debug", "Validating file")

        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        message("success", f"✓ File '{file_path}' found")
        return file_path
