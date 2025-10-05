"""Source localization mixin using autocleaneeg-eeg2source package.

This mixin provides a seamless interface to the autocleaneeg-eeg2source PyPI package
for EEG source localization. All processing is delegated to the standalone package,
ensuring consistent results and easier maintenance.

The mixin always outputs 68-channel EEG data (Desikan-Killiany atlas regions).
"""

from pathlib import Path
from typing import Optional, Union
import tempfile
import shutil
import os

import warnings

import mne

# Import from PyPI package
try:
    from autoclean_eeg2source.core.converter import SequentialProcessor
    from autoclean_eeg2source.core.memory_manager import MemoryManager
except ImportError:
    raise ImportError(
        "autocleaneeg-eeg2source package not found. Install with:\n"
        "  pip install autocleaneeg-eeg2source"
    )


class SourceLocalizationMixin:
    """Mixin class for EEG source localization using autocleaneeg-eeg2source.

    This mixin provides a streamlined interface to source localization that always
    produces 68-channel EEG data representing Desikan-Killiany atlas regions.

    All source localization is performed by the autocleaneeg-eeg2source package,
    ensuring consistent, well-tested results across different use cases.

    Outputs
    -------
    - {subject}_dk_regions.set: EEGLAB file with 68 ROI time courses
    - {subject}_region_info.csv: ROI metadata (names, hemispheres, coordinates)

    The converted data is stored in:
    - self.source_eeg: MNE Raw or Epochs object with 68 channels
    - self.source_eeg_file: Path to saved .set file
    """

    def apply_source_localization(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        method: str = "MNE",
        lambda2: float = 1.0 / 9.0,
        montage: Optional[str] = None,
        resample_freq: Optional[float] = None,
        max_memory_gb: float = 8.0,
        stage_name: str = "apply_source_localization",
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Apply MNE source localization and convert to 68 DK atlas regions.

        This method uses the autocleaneeg-eeg2source package to:
        1. Apply minimum norm estimation (MNE) source localization
        2. Extract time courses for 68 Desikan-Killiany atlas regions
        3. Return as standard 68-channel EEG data

        The method automatically handles both Raw and Epochs data, preserving
        event structure for epoched data.

        Args:
            data: Optional Raw or Epochs object. If None, uses self.raw or self.epochs
            method: Source estimation method (currently only "MNE" supported in package)
            lambda2: Regularization parameter (default: 1/9 = 0.111)
            montage: EEG montage name (default: None, auto-detect from data)
            resample_freq: Target sampling frequency (default: keep original)
            max_memory_gb: Maximum memory usage in GB (default: 8.0)
            stage_name: Name for saving and metadata tracking

        Returns:
            Raw or Epochs: 68-channel EEG data with DK atlas regions as channels

        Raises:
            AttributeError: If no input data found (no self.raw or self.epochs)
            TypeError: If input is not Raw or Epochs
            RuntimeError: If source localization fails

        Example:
            ```python
            # In a task
            self.import_raw()
            self.resample_data()
            self.filter_data()
            self.create_regular_epochs()

            # Apply source localization - returns 68-channel epochs
            source_epochs = self.apply_source_localization()

            # source_epochs now has 68 channels (DK regions)
            # self.source_eeg_file points to saved .set file
            ```

        Notes:
            - First run downloads fsaverage data (~200MB)
            - Processing time: ~2-5 minutes per subject
            - Automatically removes EOG channels before processing
            - Preserves event structure for epoched data
            - Outputs saved to derivatives/source_localization/
        """
        # Check if this step is enabled in configuration
        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled(
                "apply_source_localization"
            )

            if not is_enabled:
                if hasattr(self, "message"):
                    self.message("info", "Source localization step is disabled")
                else:
                    print("INFO: Source localization step is disabled")
                return None

            # Get parameters from config if available
            if config_value and isinstance(config_value, dict):
                method = config_value.get("method", method)
                lambda2 = config_value.get("lambda2", lambda2)
                montage = config_value.get("montage", montage)
                resample_freq = config_value.get("resample_freq", resample_freq)
                max_memory_gb = config_value.get("max_memory_gb", max_memory_gb)

        # Determine which data to use
        if data is None:
            # Try to get data from task object
            if hasattr(self, "epochs") and self.epochs is not None:
                data = self.epochs
            elif hasattr(self, "raw") and self.raw is not None:
                data = self.raw
            else:
                raise AttributeError(
                    "No input data found. Provide data parameter or ensure "
                    "self.raw or self.epochs exists."
                )

        # Type checking
        is_raw = isinstance(data, mne.io.BaseRaw)
        is_epochs = isinstance(data, mne.BaseEpochs)

        if not (is_raw or is_epochs):
            raise TypeError(
                f"Data must be mne.io.Raw or mne.Epochs, got {type(data)}"
            )

        # Auto-detect montage from data if not specified
        if montage is None:
            # Check if data has montage set
            if data.get_montage() is not None:
                detected_montage = data.get_montage()
                # Get montage name if available, otherwise use 'unknown'
                montage_name = getattr(detected_montage, 'kind', 'unknown')
                if hasattr(self, "message"):
                    self.message("info", f"Auto-detected montage from data: {montage_name}")
                # Note: SequentialProcessor requires montage, but data already has positions
                # We'll pass the detected name, but positions come from exported .set file
                montage = montage_name if montage_name != 'unknown' else "standard_1020"
            else:
                # No montage on data, use default
                montage = "standard_1020"
                if hasattr(self, "message"):
                    self.message("warning", "No montage detected, using default: standard_1020")

        try:
            # Log start
            if hasattr(self, "message"):
                self.message("header", "Applying source localization")
                self.message("info", f"Using autocleaneeg-eeg2source package")
                self.message("info", f"Method: {method}, lambda2: {lambda2}")
                self.message(
                    "info",
                    f"Input: {'Raw' if is_raw else 'Epochs'}, montage: {montage}",
                )
            else:
                print("=== Applying Source Localization ===")
                print(f"Using autocleaneeg-eeg2source package")
                print(f"Method: {method}, lambda2: {lambda2}")

            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as tmpdir:
                # Export data to temporary EEGLAB file
                tmp_input = os.path.join(tmpdir, "temp_input.set")

                if hasattr(self, "message"):
                    self.message("info", "Exporting to temporary file for processing...")

                data.export(tmp_input, fmt='eeglab', overwrite=True)

                # Initialize processor from package
                memory_manager = MemoryManager(max_memory_gb=max_memory_gb)

                processor = SequentialProcessor(
                    memory_manager=memory_manager,
                    montage=montage,
                    resample_freq=resample_freq or data.info['sfreq'],
                    lambda2=lambda2
                )

                if hasattr(self, "message"):
                    self.message("info", "Processing with source localization...")

                # Process file - this does everything:
                # 1. Validates file
                # 2. Applies source localization
                # 3. Converts to 68 DK regions
                # 4. Saves output
                result = processor.process_file(tmp_input, tmpdir)

                if result['status'] != 'success':
                    raise RuntimeError(
                        f"Source localization failed: {result.get('error', 'Unknown error')}"
                    )

                # Load the 68-region output
                output_file = result['output_file']

                if hasattr(self, "message"):
                    self.message("info", "Loading 68-region output...")

                # Read back the source-localized data (68 channels)
                if is_raw:
                    source_data = mne.io.read_raw_eeglab(output_file, preload=True)
                else:
                    source_data = mne.read_epochs_eeglab(output_file)

                # Determine output directory
                if hasattr(self, "config") and "derivatives_dir" in self.config:
                    base_dir = Path(self.config["derivatives_dir"])
                elif hasattr(self, "file_path"):
                    base_dir = Path(self.file_path).parent / "derivatives"
                else:
                    base_dir = Path.cwd() / "derivatives"

                output_dir = base_dir / "source_localization"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Determine subject ID
                subject_id = None
                if hasattr(self, "config"):
                    config = self.config
                    if "unprocessed_file" in config:
                        subject_id = Path(config["unprocessed_file"]).stem
                    elif "subject_id" in config:
                        subject_id = config["subject_id"]
                    elif "base_fname" in config:
                        subject_id = config["base_fname"]
                    elif "original_fname" in config:
                        subject_id = Path(config["original_fname"]).stem

                if subject_id is None and hasattr(self, "file_path"):
                    subject_id = Path(self.file_path).stem

                if subject_id is None:
                    subject_id = "unknown"

                # Copy output files to derivatives directory
                final_file = output_dir / f"{subject_id}_dk_regions.set"
                final_fdt = output_dir / f"{subject_id}_dk_regions.fdt"
                region_info_src = Path(tmpdir) / "temp_input_region_info.csv"
                region_info_dst = output_dir / f"{subject_id}_region_info.csv"

                # Copy .set file
                shutil.copy2(output_file, final_file)

                # Copy .fdt file if it exists
                fdt_file = output_file.replace('.set', '.fdt')
                if os.path.exists(fdt_file):
                    shutil.copy2(fdt_file, final_fdt)

                # Copy region info CSV
                if region_info_src.exists():
                    shutil.copy2(region_info_src, region_info_dst)

                # Store results in task object
                self.source_eeg = source_data
                self.source_eeg_file = str(final_file)

                # Legacy attributes: make absence explicit for downstream callers
                self.stc = None
                self.stc_list = None
                if hasattr(self, "message"):
                    self.message(
                        "warning",
                        (
                            "Legacy STC outputs are no longer generated. "
                            "Use self.source_eeg (68 DK ROIs)."
                        ),
                    )
                else:
                    warnings.warn(
                        "Legacy STC outputs are no longer generated. "
                        "Use source-localized ROI EEG data instead.",
                        stacklevel=2,
                    )

                if hasattr(self, "message"):
                    self.message(
                        "success",
                        f"Source localization complete: 68 DK regions"
                    )
                    self.message("info", f"Saved to: {final_file}")

            # Update metadata
            if hasattr(self, "_update_metadata"):
                metadata = {
                    "method": method,
                    "lambda2": lambda2,
                    "montage": montage,
                    "n_roi_channels": 68,
                    "atlas": "Desikan-Killiany (aparc)",
                    "output_file": str(final_file),
                    "region_info_file": str(region_info_dst),
                    "package": "autocleaneeg-eeg2source",
                    "data_type": "raw" if is_raw else "epochs",
                }

                if is_raw:
                    metadata["duration_sec"] = source_data.times[-1]
                    metadata["sfreq"] = source_data.info['sfreq']
                else:
                    metadata["n_epochs"] = len(source_data)
                    metadata["epoch_duration_sec"] = source_data.times[-1] - source_data.times[0]
                    metadata["sfreq"] = source_data.info['sfreq']

                self._update_metadata("step_apply_source_localization", metadata)

            return source_data

        except Exception as e:
            error_msg = f"Error during source localization: {str(e)}"
            if hasattr(self, "message"):
                self.message("error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(f"Failed to apply source localization: {str(e)}") from e
