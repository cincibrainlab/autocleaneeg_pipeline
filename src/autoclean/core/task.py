"""Base class for all EEG processing tasks."""

# Standard library imports
import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports
import mne  # Core EEG processing library for data containers and processing

from autoclean.io.export import save_epochs_to_set, save_raw_to_set
from autoclean.io.import_ import import_eeg
from autoclean.utils.exclusion_list import evaluate_exclusion_list
from autoclean.utils.logging import message

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

from autoclean.configkit.schema import (
    format_task_config_error,
    validate_task_module_config,
)
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
            # Get the module where this class was defined
            module = inspect.getmodule(self.__class__)
            if module and hasattr(module, "config"):
                self.settings = module.config
                # Validate python task module config (raises on mismatch)
                try:
                    self.settings = validate_task_module_config(self.settings)
                except Exception as exc:
                    task_file = getattr(module, "__file__", None)
                    message_text = format_task_config_error(
                        exc,
                        getattr(exc, "task_config", self.settings),
                        task_name=self.__class__.__name__,
                        task_file=task_file,
                    )
                    raise ValueError(message_text) from exc
            else:
                self.settings = None

        # Extract EEG system from task settings before validation
        config["eeg_system"] = self._extract_eeg_system()

        # Propagate task-level move_flagged_files setting (default True)
        if self.settings and "move_flagged_files" in self.settings:
            config.setdefault("move_flagged_files", self.settings["move_flagged_files"])
        else:
            config.setdefault("move_flagged_files", True)

        # Propagate task-level incremental_cleanup setting (default: absent = disabled).
        # Without this lift from settings -> runtime config, save_raw_to_set /
        # save_epochs_to_set never see the flag and intermediate stages are never pruned.
        if self.settings and "incremental_cleanup" in self.settings:
            config.setdefault(
                "incremental_cleanup", self.settings["incremental_cleanup"]
            )

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

    def import_raw(self) -> None:
        """Import the raw EEG data from file.

        Notes
        -----
        Imports data using the configured import function and flags files with
        duration less than 60 seconds. Saves the imported data as a post-import
        stage file.

        """

        self.raw = import_eeg(self.config)
        self._apply_exclusion_list_tag()
        if self.raw.duration < 60:
            self._update_flagged_status(
                flagged=True,
                reason=f"WARNING: Initial duration ({float(self.raw.duration):.1f}s) less than 1 minute",
            )

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
        self._apply_exclusion_list_tag()

        self.create_bids_path(use_epochs=True)

        save_epochs_to_set(
            epochs=self.epochs,
            autoclean_dict=self.config,
            stage="post_import",
            flagged=self.flagged,
        )

    def _apply_exclusion_list_tag(self) -> None:
        """Tag recordings listed in a configured user exclusion table."""

        is_enabled, config_value = self._check_step_enabled("exclusion_list")
        if not is_enabled:
            return

        value = (config_value or {}).get("value") or {}
        result = evaluate_exclusion_list(value, self.config.get("unprocessed_file", ""))
        if result.mode != "tag":
            return

        self._update_metadata("step_exclusion_list", result.metadata)

        if result.warning:
            message("warning", result.warning)

        if result.excluded:
            reason = result.reason or "Recording matched exclusion list"
            self._update_flagged_status(
                flagged=True,
                reason=f"EXCLUSION_LIST: {reason}",
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

    def finalize_run(self) -> None:
        """Hook called after a successful run completes.

        Override in subclasses to convert final outputs, prune intermediate stages,
        or perform other post-run housekeeping. Exceptions raised here are logged
        but do not mark the run as failed.
        """

    def run_postprocessing_analysis(self) -> list[dict[str, Any]]:
        """Run enabled post-processing analysis blocks from task configuration."""

        enabled, config_value = self._check_step_enabled("postprocessing_analysis")
        if not enabled:
            return []

        block_configs = (config_value or {}).get("value") or {}
        if not isinstance(block_configs, dict):
            raise ValueError("postprocessing_analysis.value must be a dictionary")

        results: list[dict[str, Any]] = []
        self._postprocessing_outputs: dict[str, Any] = {}
        for block_name in (
            "sensor_psd",
            "source_localization",
            "source_psd",
            "fooof",
        ):
            block_config = block_configs.get(block_name)
            if not isinstance(block_config, dict) or not block_config.get("enabled"):
                continue

            resolved = self._resolve_postprocessing_settings(block_name, block_config)
            result = self._run_postprocessing_block(block_name, resolved)
            results.append(result)
            self._register_postprocessing_output(block_name, resolved)

        if results:
            self._update_metadata(
                "step_postprocessing_analysis",
                {
                    "blocks": results,
                    "documented_order": [
                        "sensor_psd",
                        "source_localization",
                        "source_psd",
                        "fooof",
                    ],
                },
            )
            self._write_postprocessing_metadata(results)

        return results

    def _resolve_postprocessing_settings(
        self, block_name: str, block_config: dict[str, Any]
    ) -> dict[str, Any]:
        settings = {k: v for k, v in block_config.items() if k != "enabled"}
        settings.setdefault("input", self._default_postprocessing_input(block_name))
        settings["block"] = block_name
        return settings

    @staticmethod
    def _default_postprocessing_input(block_name: str) -> str:
        defaults = {
            "sensor_psd": "clean_epochs",
            "source_localization": "clean_epochs",
            "source_psd": "source_epochs",
            "fooof": "source_psd",
        }
        return defaults[block_name]

    def _run_postprocessing_block(
        self, block_name: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        input_name = str(settings.get("input"))
        data_object = self._resolve_postprocessing_input(input_name)
        common = {
            "block": block_name,
            "input": input_name,
            "settings": settings,
            "status": "completed",
        }

        if block_name == "sensor_psd":
            method = getattr(self, "apply_sensor_psd", None)
            if method is None:
                raise ValueError(
                    "sensor_psd requested but apply_sensor_psd is unavailable"
                )
            freq_range = settings.get("freq_range", [1.0, 45.0])
            psd_kwargs = {
                "data": data_object,
                "method": settings.get("method", "welch"),
                "fmin": settings.get("fmin", freq_range[0]),
                "fmax": settings.get("fmax", freq_range[1]),
                "picks": settings.get("picks", "eeg"),
                "time_windows": settings.get("time_windows"),
                "baseline": settings.get("baseline"),
                "stage_name": "postprocessing_sensor_psd",
            }
            if "freq_bands" in settings:
                psd_kwargs["freq_bands"] = settings["freq_bands"]
            psd_df, band_df, artifacts = self._call_postprocessing_method(
                method,
                "apply_sensor_psd",
                settings,
                **psd_kwargs,
            )
            self.sensor_psd_result = {
                "spectra": psd_df,
                "bands": band_df,
                "artifacts": artifacts,
            }
            common["artifacts"] = artifacts
            return common

        if block_name == "source_localization":
            method = getattr(self, "apply_source_localization", None)
            if method is None:
                raise ValueError(
                    "source_localization requested but apply_source_localization is unavailable"
                )
            source_data = self._call_postprocessing_method(
                method,
                "apply_source_localization",
                settings,
                data=data_object,
                method=settings.get("method", "MNE"),
                lambda2=settings.get("lambda2", 1.0 / 9.0),
                montage=settings.get("montage"),
                resample_freq=settings.get("resample_freq"),
                max_memory_gb=settings.get("max_memory_gb", 8.0),
                stage_name="postprocessing_source_localization",
            )
            self.source_eeg = source_data
            common["output"] = "source_eeg"
            return common

        if block_name == "source_psd":
            method = getattr(self, "apply_source_psd", None)
            if method is None:
                raise ValueError(
                    "source_psd requested but apply_source_psd is unavailable"
                )
            psd_df, file_path = self._call_postprocessing_method(
                method,
                "apply_source_psd",
                settings,
                stc_list=data_object,
                segment_duration=settings.get("segment_duration", 80),
                n_jobs=settings.get("n_jobs", 4),
                generate_plots=settings.get("generate_plots", True),
                stage_name="postprocessing_source_psd",
            )
            self.source_psd_df = psd_df
            common["output_file"] = str(file_path)
            common["rows"] = int(len(psd_df)) if psd_df is not None else 0
            return common

        if block_name == "fooof":
            psd_df = self._postprocessing_psd_dataframe(data_object)
            if psd_df is None:
                raise ValueError(
                    "postprocessing fooof requires a PSD table input; configure "
                    "fooof.input to a sensor_psd or source_psd output"
                )
            (
                aperiodic_df,
                aperiodic_file,
                periodic_df,
                periodic_file,
            ) = self._run_postprocessing_tabular_fooof(
                psd_df=psd_df,
                settings=settings,
                input_name=input_name,
            )
            common["aperiodic_file"] = str(aperiodic_file)
            common["aperiodic_rows"] = int(len(aperiodic_df))
            if periodic_file is not None:
                common["periodic_file"] = str(periodic_file)
                common["periodic_rows"] = int(len(periodic_df))
            common["method"] = "tabular_psd_parameterization"
            return common

        raise ValueError(f"Unsupported postprocessing analysis block: {block_name}")

    def _call_postprocessing_method(
        self,
        analysis_method: Any,
        legacy_step_name: str,
        settings: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Call legacy analysis methods from the new postprocessing block config.

        Temporarily stages `settings` under `legacy_step_name` in
        `self.settings` so the legacy mixin's own `_check_step_enabled`-based
        config parsing sees it as enabled, without duplicating that parsing
        here. Safe under sequential block execution; do not call concurrently.
        """

        legacy_value = {
            key: value
            for key, value in settings.items()
            if key not in {"block", "input", "enabled"}
        }
        previous_settings = getattr(self, "settings", None)
        created_settings = previous_settings is None
        if created_settings:
            self.settings = {}
        original_step = self.settings.get(legacy_step_name)
        self.settings[legacy_step_name] = {"enabled": True, "value": legacy_value}
        try:
            return analysis_method(**kwargs)
        finally:
            if original_step is None:
                self.settings.pop(legacy_step_name, None)
            else:
                self.settings[legacy_step_name] = original_step
            if created_settings:
                self.settings = previous_settings

    @staticmethod
    def _postprocessing_psd_dataframe(data_object: Any) -> Any:
        """Return a PSD DataFrame from supported postprocessing inputs."""

        try:
            import pandas as pd
        except Exception:
            return None

        if isinstance(data_object, pd.DataFrame):
            return data_object
        if isinstance(data_object, dict):
            for key in ("spectra", "psd", "data"):
                value = data_object.get(key)
                if isinstance(value, pd.DataFrame):
                    return value
        return None

    def _run_postprocessing_tabular_fooof(
        self,
        psd_df: Any,
        settings: dict[str, Any],
        input_name: str,
    ) -> tuple[Any, Path, Any, Optional[Path]]:
        """Parameterize spectra already represented as a PSD table.

        This lightweight log-log parameterization estimates peak center
        frequency and power but not bandwidth, so peak rows always report
        ``bandwidth: None`` (unlike ``apply_fooof_periodic``, which fits full
        Gaussian peaks and can estimate bandwidth directly).
        """

        import numpy as np
        import pandas as pd

        required = {"frequency", "psd"}
        missing = required - set(psd_df.columns)
        if missing:
            raise ValueError(
                "fooof input must include PSD table columns: "
                f"{sorted(required)}; missing {sorted(missing)}"
            )

        aperiodic_mode = str(settings.get("aperiodic_mode", "fixed")).lower()
        if aperiodic_mode != "fixed":
            raise ValueError(
                "tabular fooof supports only aperiodic_mode='fixed'; "
                f"got {aperiodic_mode!r}"
            )

        freq_range = settings.get("freq_range", [1.0, 45.0])
        low, high = float(freq_range[0]), float(freq_range[1])
        if low >= high:
            raise ValueError("fooof freq_range start must be less than stop")

        group_columns = [
            column
            for column in ("subject", "channel", "roi", "time_window")
            if column in psd_df.columns
        ]
        if not group_columns:
            group_columns = ["_spectrum"]
            psd_df = psd_df.copy()
            psd_df["_spectrum"] = "all"

        rows = []
        peak_rows = []
        grouped = psd_df.groupby(group_columns, dropna=False)
        for group_key, group in grouped:
            spectrum = group[(group["frequency"] >= low) & (group["frequency"] <= high)]
            spectrum = spectrum.sort_values("frequency")
            spectrum = spectrum[(spectrum["frequency"] > 0) & (spectrum["psd"] > 0)]
            if len(spectrum) < 2:
                status = "INSUFFICIENT_DATA"
                offset = exponent = r_squared = error = None
            else:
                x = np.log10(spectrum["frequency"].to_numpy(dtype=float))
                y = np.log10(spectrum["psd"].to_numpy(dtype=float))
                slope, intercept = np.polyfit(x, y, deg=1)
                predicted = slope * x + intercept
                residual = y - predicted
                ss_res = float(np.sum(residual**2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                offset = float(intercept)
                exponent = float(-slope)
                r_squared = float(1 - ss_res / ss_tot) if ss_tot else 1.0
                error = float(np.sqrt(np.mean(residual**2)))
                status = "SUCCESS"

            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = {column: value for column, value in zip(group_columns, group_key)}
            row.update(
                {
                    "input": input_name,
                    "freq_min": low,
                    "freq_max": high,
                    "aperiodic_mode": aperiodic_mode,
                    "offset": offset,
                    "exponent": exponent,
                    "r_squared": r_squared,
                    "error": error,
                    "status": status,
                }
            )
            rows.append(row)

            if settings.get("run_periodic", True) and status == "SUCCESS":
                residual_df = spectrum.assign(_residual=residual)
                threshold = float(settings.get("peak_residual_threshold", 0.0))
                candidates = residual_df[residual_df["_residual"] > threshold]
                if candidates.empty:
                    candidates = residual_df.nlargest(1, "_residual")
                max_peaks = int(settings.get("max_n_peaks", 6))
                for _, peak in candidates.nlargest(max_peaks, "_residual").iterrows():
                    peak_row = dict(row)
                    peak_row.update(
                        {
                            "input": input_name,
                            "center_frequency": float(peak["frequency"]),
                            "power": float(peak["psd"]),
                            "residual": float(peak["_residual"]),
                            "bandwidth": None,
                            "status": "SUCCESS",
                        }
                    )
                    peak_rows.append(peak_row)

        result_df = pd.DataFrame(rows)
        output_dir = self._resolve_report_path("fooof")
        subject_id = "unknown_subject"
        if hasattr(self, "config") and self.config.get("unprocessed_file"):
            subject_id = Path(self.config["unprocessed_file"]).stem
        output_file = output_dir / f"{subject_id}_postprocessing_fooof_aperiodic.csv"
        result_df.to_csv(output_file, index=False)
        self.fooof_aperiodic_df = result_df
        self.fooof_aperiodic_file = str(output_file)
        periodic_df = pd.DataFrame(peak_rows)
        periodic_file = None
        if settings.get("run_periodic", True):
            periodic_file = (
                output_dir / f"{subject_id}_postprocessing_fooof_periodic.csv"
            )
            periodic_df.to_csv(periodic_file, index=False)
            self.fooof_periodic_df = periodic_df
            self.fooof_periodic_file = str(periodic_file)
        self._update_metadata(
            "step_postprocessing_fooof",
            {
                "input": input_name,
                "freq_range": [low, high],
                "aperiodic_mode": aperiodic_mode,
                "output_file": str(self._report_relative_path(output_file)),
                "periodic_output_file": (
                    str(self._report_relative_path(periodic_file))
                    if periodic_file is not None
                    else None
                ),
                "n_spectra": int(len(result_df)),
                "n_peaks": int(len(periodic_df)),
            },
        )
        return result_df, output_file, periodic_df, periodic_file

    def _postprocessing_output_value(self, block_name: str) -> Any:
        outputs = {
            "sensor_psd": getattr(self, "sensor_psd_result", None),
            "source_localization": getattr(self, "source_eeg", None),
            "source_psd": getattr(self, "source_psd_df", None),
            "fooof": {
                "aperiodic": getattr(self, "fooof_aperiodic_df", None),
                "periodic": getattr(self, "fooof_periodic_df", None),
            },
        }
        return outputs.get(block_name)

    def _register_postprocessing_output(
        self, block_name: str, settings: dict[str, Any]
    ) -> None:
        value = self._postprocessing_output_value(block_name)
        if value is None:
            return
        output_map = getattr(self, "_postprocessing_outputs", None)
        if output_map is None:
            self._postprocessing_outputs = {}
            output_map = self._postprocessing_outputs
        output_map[block_name] = value
        for alias_key in ("output", "output_name"):
            alias = settings.get(alias_key)
            if isinstance(alias, str) and alias:
                output_map[alias] = value
        aliases = settings.get("outputs")
        if isinstance(aliases, dict):
            alias_values = aliases.values()
        elif isinstance(aliases, (list, tuple, set)):
            alias_values = aliases
        else:
            alias_values = []
        for alias in alias_values:
            if isinstance(alias, str) and alias:
                output_map[alias] = value

    def _resolve_postprocessing_input(self, input_name: str) -> Any:
        output_map = getattr(self, "_postprocessing_outputs", {})
        if input_name in output_map:
            return output_map[input_name]
        sensor_psd = getattr(self, "sensor_psd_result", None)
        if sensor_psd is None:
            sensor_psd = getattr(self, "sensor_psd_df", None)
        imported_raw = getattr(self, "original_raw", None)
        if imported_raw is None:
            imported_raw = getattr(self, "raw", None)
        source_epochs = getattr(self, "source_eeg", None)
        if source_epochs is None:
            source_epochs = getattr(self, "source_epochs", None)
        input_map = {
            "imported_raw": imported_raw,
            "clean_raw": getattr(self, "raw", None),
            "clean_epochs": getattr(self, "epochs", None),
            "source_epochs": source_epochs,
            "sensor_psd": sensor_psd,
            "source_psd": getattr(self, "source_psd_df", None),
        }
        if input_name not in input_map:
            raise ValueError(
                f"Unsupported postprocessing input '{input_name}'. "
                f"Supported inputs: {sorted(input_map)}"
            )
        data_object = input_map[input_name]
        if data_object is None:
            raise ValueError(
                f"Postprocessing input '{input_name}' is not available for this task. "
                "Run or configure the prerequisite step before this analysis block."
            )
        return data_object

    def _write_postprocessing_metadata(self, results: list[dict[str, Any]]) -> None:
        reports_dir = None
        if hasattr(self, "config"):
            reports_dir = self.config.get("reports_dir")
        if not reports_dir:
            return

        output_dir = Path(reports_dir) / "postprocessing_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "resolved_settings.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=str)

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
            from autoclean import __version__ as ac_version
            from autoclean.reporting.llm_reporting import (
                EpochStats,
                FilterParams,
                ICAStats,
                RunContext,
                create_reports,
            )
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
        derivatives_root = Path(cfg.get("derivatives_dir") or metadata_dir.parent)
        subj_basename = Path(input_file).stem
        logs_root = Path(cfg.get("logs_dir") or derivatives_root)
        reports_root = Path(cfg.get("reports_dir") or (metadata_dir.parent / "reports"))
        run_reports_dir = reports_root / "run_reports"
        pdf_name = f"{subj_basename}_autoclean_report.pdf"
        pdf_candidates = [
            reports_root / "run_reports" / pdf_name,
            reports_root / pdf_name,
            metadata_dir / pdf_name,
        ]
        report_pdf = next((p for p in pdf_candidates if p.exists()), pdf_candidates[0])

        per_file_csv = None
        for base_dir in [run_reports_dir, reports_root, derivatives_root, logs_root]:
            candidate = base_dir / f"{subj_basename}_processing_log.csv"
            if candidate.exists() and not candidate.name.startswith("._"):
                per_file_csv = candidate
                break

        if per_file_csv is None:
            # Also check exports copy as fallback
            final_files_dir = Path(cfg.get("final_files_dir", metadata_dir))
            alt_csv = final_files_dir / f"{subj_basename}_processing_log.csv"
            if alt_csv.exists() and not alt_csv.name.startswith("._"):
                per_file_csv = alt_csv
            else:
                return None

        # Parse one-row CSV into dict
        row: Dict[str, Any]
        try:
            import csv

            with per_file_csv.open("r", encoding="utf-8", errors="strict") as f:
                reader = csv.DictReader(f)
                row = next(reader)
        except UnicodeDecodeError:
            message(
                "warning",
                f"Skipping per-file log with invalid encoding: {per_file_csv.name}",
            )
            return None
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
                parts = [
                    p
                    for p in str(x).replace("[", "").replace("]", "").split(",")
                    if p.strip()
                ]
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
        if out_dir:
            reports_dir = Path(out_dir)
        else:
            reports_root = cfg.get("reports_dir")
            if reports_root:
                reports_root = Path(reports_root)
            else:
                reports_root = metadata_dir.parent / "reports"
            reports_dir = reports_root / "llm" / subj_basename

        reports_dir.mkdir(parents=True, exist_ok=True)

        llm_settings = None
        if isinstance(self.settings, dict):
            maybe_llm_settings = self.settings.get("llm_reporting")
            if isinstance(maybe_llm_settings, dict):
                llm_settings = maybe_llm_settings

        create_reports(context, reports_dir, llm_settings=llm_settings)
        return reports_dir
