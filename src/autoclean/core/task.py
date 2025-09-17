"""Base class for all EEG processing tasks."""

# Standard library imports
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import mne  # Core EEG processing library for data containers and processing

from autoclean.io.export import save_epochs_to_set, save_raw_to_set
from autoclean.io.import_ import import_eeg

# Local imports
try:
    from autoclean.mixins import DISCOVERED_MIXINS

    if not DISCOVERED_MIXINS:
        print("🚨 CRITICAL ERROR: DISCOVERED_MIXINS is empty!")
        print("Task class will be missing all mixin functionality!")
        print("Check autoclean.mixins package for import errors.")

        # Create a minimal fallback
        class _EmptyMixinFallback:
            def __getattr__(self, name):
                raise AttributeError(
                    f"Method '{name}' not available - mixin discovery failed. "
                    f"Check autoclean.mixins package for import errors."
                )

        DISCOVERED_MIXINS = (_EmptyMixinFallback,)
except ImportError as e:
    print("🚨 CRITICAL ERROR: Could not import DISCOVERED_MIXINS!")
    print(f"Import error: {e}")
    print("Task class will be missing all mixin functionality!")

    # Create a minimal fallback
    class _ImportErrorMixinFallback:
        def __getattr__(self, name):
            raise AttributeError(f"Method '{name}' not available - mixin import failed")

    DISCOVERED_MIXINS = (_ImportErrorMixinFallback,)

from autoclean.utils.auth import require_authentication


class Task(ABC, *DISCOVERED_MIXINS):
    """Base class for all EEG processing tasks.

    This class defines the interface that all specific EEG tasks must implement.
    It provides the basic structure for:
    1. Loading and validating configuration
    2. Importing raw EEG data
    3. Running preprocessing steps
    4. Applying task-specific processing
    5. Saving results

    It should be inherited from to create new tasks in the autoclean.tasks module.

    Notes
    -----
    Abstract base class that enforces a consistent interface across all EEG processing
    tasks through abstract methods and strict type checking. Manages state through
    MNE objects (Raw and Epochs) while maintaining processing history in a dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize a new task instance.

        Parameters
        ----------
        config : Dict[str, Any]
            A dictionary containing all configuration settings for the task.
            Must include:

            - run_id (str): Unique identifier for this processing run
            - unprocessed_file (Path): Path to the raw EEG data file
            - task (str): Name of the task (e.g., "rest_eyesopen")

            The base class automatically detects a module-level 'config' variable
            and uses it for self.settings in Python-based tasks.

        Examples
        --------
        >>> # Python task file approach - no __init__ needed!
        >>> config = {'resample': {'enabled': True, 'value': 250}}
        >>> class MyTask(Task):
        ...     def run(self):
        ...         self.import_raw()
        ...         # Processing steps here
        """
        # Auto-detect module-level config for Python tasks
        if not hasattr(self, "settings"):
            # First try to restore from database (for resume operations)
            if "run_id" in config:
                try:
                    from autoclean.utils.database import get_run_record
                    run_record = get_run_record(config["run_id"])
                    if run_record and "task_config" in run_record.get("metadata", {}):
                        task_config = run_record["metadata"]["task_config"]
                        self.settings = task_config.get("settings")
                        message("info", f"Restored task configuration from database for run {config['run_id']}")
                    else:
                        self.settings = None
                except Exception as e:
                    message("warning", f"Could not restore task config from database: {e}")
                    self.settings = None
            else:
                self.settings = None

            # If still no settings, try module-level config
            if self.settings is None:
                module = inspect.getmodule(self.__class__)
                if module and hasattr(module, "config"):
                    self.settings = module.config
                else:
                    self.settings = None

        # Extract EEG system from task settings before validation
        config["eeg_system"] = self._extract_eeg_system()

        # Propagate task-level move_flagged_files setting (default True)
        if self.settings and "move_flagged_files" in self.settings:
            config.setdefault("move_flagged_files", self.settings["move_flagged_files"])
        else:
            config.setdefault("move_flagged_files", True)

        # Configuration must be validated first as other initializations depend on it
        self.config = self.validate_config(config)

        # Initialize MNE data containers to None
        # These will be populated during the processing pipeline
        self.raw: Optional[mne.io.Raw] = None  # Holds continuous EEG data
        self.original_raw: Optional[mne.io.Raw] = None
        self.epochs: Optional[mne.Epochs] = None  # Holds epoched data segments
        self.flagged = False
        self.flagged_reasons = []
        self.fast_ica: Optional[mne.ICA] = None
        self.final_ica: Optional[mne.ICA] = None
        self.ica_flags = None

    def _extract_eeg_system(self) -> str:
        """Extract EEG system/montage from task settings.

        Returns
        -------
        str
            The montage name from task config, or "auto" as fallback
        """
        if (
            self.settings
            and "montage" in self.settings
            and self.settings["montage"].get("enabled", False)
        ):
            return self.settings["montage"]["value"]
        return "auto"

    def _check_manual_ica_resume(self) -> Optional[Path]:
        """Check for manual ICA files to resume from.

        Looks for files in the manual_ica folder that match the current run_id.
        This allows users to manually correct ICA classifications and resume
        processing from that point.

        Returns
        -------
        Optional[Path]
            Path to manual ICA file if found, None otherwise
        """
        if not hasattr(self, 'config') or not self.config:
            return None

        run_id = self.config.get("run_id")
        if not run_id:
            return None

        # Check for manual_ica folder in output directory
        output_dir = Path(self.config.get("output_dir", ""))
        manual_ica_dir = output_dir / "manual_ica"

        if not manual_ica_dir.exists():
            return None

        # Look for files matching the run_id pattern
        run_pattern = f"*{run_id}*"
        matching_files = list(manual_ica_dir.glob(run_pattern))

        if matching_files:
            # Return the most recently modified file
            latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
            return latest_file

        return None

    def import_raw(self) -> None:
        """Import the raw EEG data from file or resume from manual ICA.

        Notes
        -----
        Imports data using the configured import function and flags files with
        duration less than 60 seconds. For resume operations, checks for manual
        ICA files in the manual_ica folder and loads those instead.
        Saves the imported data as a post-import stage file.

        """

        # Check for manual ICA resume file
        manual_file = self._check_manual_ica_resume()
        if manual_file:
            message("info", f"Resuming from manual ICA file: {manual_file}")
            # Load the manual ICA file directly
            import mne
            self.raw = mne.io.read_raw(str(manual_file), preload=True)
            self._resume_mode = True
            self._resume_stage = "post_basic_steps"  # Skip import + basic steps
        else:
            # Normal import workflow
            self.raw = import_eeg(self.config)
        if self.raw.duration < 60:
            self.flagged = True
            self.flagged_reasons = [
                f"WARNING: Initial duration ({float(self.raw.duration):.1f}s) less than 1 minute"
            ]

        self.create_bids_path()

        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_import",
            flagged=self.flagged,
        )

    def import_epochs(self) -> None:
        """Import the epochs from file.

        Notes
        -----
        Imports data using the configured import function and saves the imported
        data as a post-import stage file.

        """

        self.epochs = import_eeg(self.config)

        self.create_bids_path(use_epochs=True)

        save_epochs_to_set(
            epochs=self.epochs,
            autoclean_dict=self.config,
            stage="post_import",
            flagged=self.flagged,
        )

    @abstractmethod
    @require_authentication
    def run(self) -> None:
        """Run the standard EEG preprocessing pipeline.

        Notes
        -----
        Defines interface for MNE-based preprocessing operations including filtering,
        resampling, and artifact detection. Maintains processing state through
        self.raw modifications.

        The specific parameters for each preprocessing step should be
        defined in the task configuration and validated before use.
        """

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the complete task configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary to validate.
            See __init__ docstring for required fields.

        Returns
        -------
        Dict[str, Any]
            The validated configuration dictionary.
            May contain additional fields added during validation.

        Notes
        -----
        Implements two-stage validation pattern with base validation followed by
        task-specific checks. Uses type annotations and runtime checks to ensure
        configuration integrity before processing begins.

        Examples
        --------
        >>> config = {...}  # Your configuration dictionary
        >>> validated_config = task.validate_config(config)
        >>> print(f"Validation successful: {validated_config['task']}")
        """
        # Schema definition for base configuration requirements
        # All tasks must provide these fields with exact types
        required_fields = {
            "run_id": str,  # Unique identifier for tracking
            "unprocessed_file": Path,  # Input file path
            "task": str,  # Task identifier
        }

        # Two-stage validation: first check existence, then type
        for field, field_type in required_fields.items():
            # Stage 1: Check field existence
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

            # Stage 2: Validate field type using isinstance for safety
            if not isinstance(config[field], field_type):
                raise TypeError(
                    f"Field '{field}' must be of type {field_type.__name__}, "
                    f"got {type(config[field]).__name__} instead"
                )

        # No longer validate required_stages - stages are created dynamically when export=True is used

        return config

    def get_flagged_status(self) -> tuple[bool, list[str]]:
        """Get the flagged status of the task.

        Returns
        -------
        tuple of (bool, list of str)
            A tuple containing a boolean flag and a list of reasons for flagging.
        """
        return self.flagged, self.flagged_reasons

    def get_raw(self) -> Optional[mne.io.Raw]:
        """Get the raw data of the task.

        Returns
        -------
        mne.io.Raw
            The raw data of the task.

        """
        if self.raw is None:
            raise ValueError("Raw data is not available.")
        return self.raw

    def get_epochs(self) -> Optional[mne.Epochs]:
        """Get the epochs of the task.

        Returns
        -------
        mne.Epochs
            The epochs of the task.

        """
        if self.epochs is None:
            raise ValueError("Epochs are not available.")
        return self.epochs

    # -------------------------
    # LLM Reporting Integration
    # -------------------------
    def emit_llm_reports(self, out_dir: Optional[Path] = None) -> Optional[Path]:
        """Create LLM-backed textual reports using always-present outputs.

        Uses the per-file processing log CSV and the generated PDF report
        to build a minimal RunContext and write deterministic methods text
        plus optional LLM summaries.

        Returns the reports directory path on success, otherwise None.
        """
        # Respect task configuration flag; default is OFF
        try:
            if not (hasattr(self, "settings") and isinstance(self.settings, dict)):
                return None
            if not self.settings.get("ai_reporting", False):
                return None
        except Exception:
            return None

        try:
            from autoclean.reporting.llm_reporting import (
                FilterParams,
                ICAStats,
                EpochStats,
                RunContext,
                create_reports,
            )
            from autoclean import __version__ as ac_version
        except Exception:
            # Reporting module not available; skip silently
            return None

        cfg = self.config
        try:
            metadata_dir: Path = cfg["metadata_dir"]
            input_file: Path = cfg["unprocessed_file"]
            run_id: str = cfg["run_id"]
        except Exception:
            return None

        # Derive paths
        derivatives_root = metadata_dir.parent
        subj_basename = Path(input_file).stem
        per_file_csv = derivatives_root / f"{subj_basename}_processing_log.csv"
        report_pdf = metadata_dir / f"{subj_basename}_autoclean_report.pdf"

        if not per_file_csv.exists():
            # Also check final_files copy as fallback
            final_files_dir = Path(cfg.get("final_files_dir", metadata_dir))
            alt_csv = final_files_dir / per_file_csv.name
            if alt_csv.exists():
                per_file_csv = alt_csv
            else:
                return None

        # Parse one-row CSV into dict
        row: Dict[str, Any]
        try:
            import csv

            with per_file_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader)
        except Exception:
            return None

        # Helpers to parse values robustly
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        def _to_int(x):
            try:
                return int(float(x))
            except Exception:
                return None

        def _to_list_of_floats(x):
            if x is None or x == "":
                return []
            try:
                import ast

                v = ast.literal_eval(x)
                if isinstance(v, (list, tuple)):
                    return [float(y) for y in v]
                return [float(v)]
            except Exception:
                # Fallback: comma/space separated
                parts = [p for p in str(x).replace("[", "").replace("]", "").split(",") if p.strip()]
                out = []
                for p in parts:
                    try:
                        out.append(float(p))
                    except Exception:
                        pass
                return out

        def _to_list_of_ints(x):
            try:
                import ast

                v = ast.literal_eval(x)
                if isinstance(v, (list, tuple)):
                    return [int(float(y)) for y in v]
                return []
            except Exception:
                return []

        # Build dataclasses from CSV
        fp = FilterParams(
            l_freq=_to_float(row.get("proc_filt_lowcutoff")),
            h_freq=_to_float(row.get("proc_filt_highcutoff")),
            notch_freqs=_to_list_of_floats(row.get("proc_filt_notch")),
            notch_widths=_to_float(row.get("proc_filt_notch_width")),
        )

        # ICA details are limited in CSV; provide best-effort mapping
        ica_removed = _to_list_of_ints(row.get("proc_removeComps"))
        ica_stats = (
            ICAStats(
                method=str(row.get("ica_method") or "unspecified"),
                n_components=_to_int(row.get("proc_nComps")),
                removed_indices=ica_removed,
                labels_histogram={},
                classifier=str(row.get("classification_method") or None),
            )
            if (row.get("proc_nComps") or row.get("proc_removeComps"))
            else None
        )

        # Epoch stats
        epoch_limits = None
        try:
            import ast

            v = ast.literal_eval(row.get("epoch_limits", ""))
            if isinstance(v, (list, tuple)) and len(v) == 2:
                epoch_limits = (
                    float(v[0]) if v[0] is not None else None,
                    float(v[1]) if v[1] is not None else None,
                )
        except Exception:
            epoch_limits = None

        kept = _to_int(row.get("epoch_trials"))
        rejected = _to_int(row.get("epoch_badtrials"))
        total = None
        if kept is not None and rejected is not None:
            total = kept + rejected

        epochs = EpochStats(
            tmin=epoch_limits[0] if epoch_limits else None,
            tmax=epoch_limits[1] if epoch_limits else None,
            baseline=None,
            total_epochs=total,
            kept_epochs=kept,
            rejected_epochs=rejected,
            rejection_rules={},
        )

        # Assemble context
        try:
            import mne as _mne

            mne_version = getattr(_mne, "__version__", None)
        except Exception:
            mne_version = None

        notes = []
        if row.get("flags"):
            notes.append(f"flags: {row['flags']}")

        figures = {}
        if report_pdf.exists():
            figures["autoclean_report_pdf"] = str(report_pdf)

        context = RunContext(
            run_id=str(run_id),
            dataset_name=None,
            input_file=str(input_file),
            montage=None,
            resample_hz=_to_float(row.get("proc_sRate1")),
            reference=None,
            filter_params=fp,
            ica=ica_stats,
            epochs=epochs,
            durations_s=_to_float(row.get("proc_xmax_post")),
            n_channels=_to_int(row.get("net_nbchan_post")),
            bids_root=str(cfg.get("bids_dir")) if cfg.get("bids_dir") else None,
            bids_subject_id=None,
            pipeline_version=str(ac_version),
            mne_version=mne_version,
            compliance_user=None,
            notes=notes,
            figures=figures,
        )

        # Determine output directory for reports
        reports_dir = out_dir or (metadata_dir / "llm_reports")
        create_reports(context, reports_dir)
        return reports_dir
