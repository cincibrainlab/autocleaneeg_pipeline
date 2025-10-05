"""Source PSD mixin for autoclean tasks.

This module provides functionality for calculating power spectral density (PSD) from
source-localized EEG data with region-of-interest (ROI) averaging using the
Desikan-Killiany atlas.

The SourcePSDMixin class implements methods for computing PSD from source estimates
(STCs) produced by source localization. The method uses Welch's method with adaptive
windowing and parallel processing for efficiency. Results are parcellated into 68
anatomical ROIs and saved as both frequency-resolved data and frequency band summaries.

Source-level PSD is fundamental for:
- Spectral analysis of brain rhythms in anatomical regions
- Quantifying frequency band power (delta, theta, alpha, beta, gamma)
- Comparing spectral patterns across brain regions
- Group-level statistical analyses of regional power
"""

from pathlib import Path
from typing import Optional, Union

import mne
import pandas as pd

# Import algorithm functions
from .algorithm import calculate_roi_psd, calculate_source_psd_list, visualize_psd_results


class SourcePSDMixin:
    """Mixin class providing functionality for source-level PSD calculation.

    This mixin provides methods for calculating power spectral density (PSD) from
    source estimates (STCs) with ROI averaging. The implementation uses Welch's method
    for PSD estimation and parcellates results into 68 cortical ROIs using the
    Desikan-Killiany atlas.

    The mixin automatically retrieves source estimates from the task object (self.stc
    or self.stc_list) and computes frequency-resolved PSD for each anatomical region.
    Results are saved as both parquet files (full frequency resolution) and CSV files
    (band power summaries).

    The mixin respects configuration settings from the task config, allowing users to
    customize parameters such as segment duration, number of parallel jobs, and
    plotting options.

    References
    ----------
    Welch P (1967). The use of fast Fourier transform for the estimation of power
    spectra: A method based on time averaging over short, modified periodograms.
    IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.

    Desikan RS, et al. (2006). An automated labeling system for subdividing the human
    cerebral cortex on MRI scans into gyral based regions of interest. NeuroImage,
    31(3), 968-980.
    """

    def apply_source_psd(
        self,
        stc_list: Union[mne.SourceEstimate, list, None] = None,
        segment_duration: float = 80,
        n_jobs: int = 4,
        generate_plots: bool = True,
        stage_name: str = "apply_source_psd",
    ) -> tuple:
        """Calculate power spectral density from source-localized data with ROI averaging.

        This method supports two modes:

        **ROI-Optimized Mode (v2.0.0+):**
        - For data from source localization v2.0.1+ (self.source_eeg exists)
        - Processes 68 DK atlas ROI channels directly
        - Much faster (~10-30 seconds vs 2-5 minutes)
        - Uses MNE's built-in PSD computation

        **Vertex-Level Mode (v1.0.0 compatible):**
        - For data from source localization v1.0.0 (self.stc/self.stc_list exists)
        - Processes 20,484 cortical vertices with parcellation
        - Uses parallel batch processing for efficiency
        - Backward compatible with existing workflows

        Both modes produce identical output formats for downstream compatibility.

        The PSD calculation uses Welch's method with adaptive windowing (4s windows,
        50% overlap). Results include:
        - Full frequency-resolved PSD (0.5-45 Hz) for each ROI
        - Band power summaries (delta, theta, alpha, beta, gamma)
        - Diagnostic visualizations (4-panel plots)

        Args:
            stc_list: Optional SourceEstimate or list. If None, auto-detects from
                self.source_eeg (ROI mode) or self.stc/self.stc_list (vertex mode)
            segment_duration: Duration in seconds to process (default: 80s for optimal
                balance of accuracy and performance). If None, uses entire data.
            n_jobs: Number of parallel jobs for computation (default: 4)
                Note: Only used in vertex-level mode
            generate_plots: Whether to generate diagnostic PSD plots (default: True)
            stage_name: Name for saving and metadata tracking

        Returns:
            tuple: (psd_df, file_path) where:
                - psd_df: DataFrame with columns [subject, roi, hemisphere, frequency, psd]
                - file_path: Path to saved parquet file

        Raises:
            AttributeError: If no source data found (no self.source_eeg, self.stc, or self.stc_list)
            TypeError: If input is not valid source data
            RuntimeError: If PSD calculation fails

        Example:
            ```python
            # Apply source PSD with default parameters (auto-detects mode)
            psd_df, file_path = task.apply_source_psd()

            # Apply with custom parameters
            psd_df, file_path = task.apply_source_psd(
                segment_duration=60,
                n_jobs=8,
                generate_plots=False
            )

            # Access results (same format regardless of mode)
            print(psd_df.head())  # View ROI PSDs
            ```

        Notes
        -----
        - Requires prior source localization (v2.0.1 or v1.0.0)
        - Uses Desikan-Killiany atlas (68 cortical ROIs)
        - Saves three files:
            * {subject}_roi_psd.parquet: Full frequency-resolved data
            * {subject}_roi_bands.csv: Band power summaries
            * {subject}_psd_visualization.png: Diagnostic plots (if generate_plots=True)
        - Frequency bands: delta (1-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-45)
        - Processes middle epochs/segments for better stationarity
        """
        # Check if this step is enabled in the configuration
        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled("apply_source_psd")

            if not is_enabled:
                if hasattr(self, "message"):
                    self.message("info", "Source PSD step is disabled")
                else:
                    print("INFO: Source PSD step is disabled")
                return None, None

            # Get parameters from config if available
            if config_value and isinstance(config_value, dict):
                segment_duration = config_value.get("segment_duration", segment_duration)
                n_jobs = config_value.get("n_jobs", n_jobs)
                generate_plots = config_value.get("generate_plots", generate_plots)

        # Determine which data to use and which mode to run
        # Priority: explicit parameter > self.source_eeg (v2.0+) > self.stc/stc_list (v1.0)
        use_roi_mode = False
        roi_data = None

        if stc_list is None:
            # Check for ROI-optimized data first (v2.0.1+ source localization)
            if hasattr(self, "source_eeg") and self.source_eeg is not None:
                use_roi_mode = True
                roi_data = self.source_eeg
                if hasattr(self, "message"):
                    self.message("info", "Detected ROI-optimized source data (source localization v2.0.1+)")
                    self.message("info", f"Using fast 68-channel mode (vs 20,484-vertex mode)")
                else:
                    print("INFO: Detected ROI-optimized source data (source localization v2.0.1+)")
                    print("INFO: Using fast 68-channel mode (vs 20,484-vertex mode)")
            # Fall back to vertex-level data (v1.0.0 source localization)
            elif hasattr(self, "stc_list") and self.stc_list is not None:
                stc_list = self.stc_list
                use_roi_mode = False
                if hasattr(self, "message"):
                    self.message("info", "Detected vertex-level source data (source localization v1.0.0)")
                    self.message("info", "Using vertex-level mode with parcellation")
                else:
                    print("INFO: Detected vertex-level source data (source localization v1.0.0)")
                    print("INFO: Using vertex-level mode with parcellation")
            elif hasattr(self, "stc") and self.stc is not None:
                stc_list = self.stc
                use_roi_mode = False
                if hasattr(self, "message"):
                    self.message("info", "Detected vertex-level source data (source localization v1.0.0)")
                    self.message("info", "Using vertex-level mode with parcellation")
                else:
                    print("INFO: Detected vertex-level source data (source localization v1.0.0)")
                    print("INFO: Using vertex-level mode with parcellation")
            else:
                raise AttributeError(
                    "No source data found. Apply source localization first "
                    "(v2.0.1: self.source_eeg, or v1.0.0: self.stc/self.stc_list must exist)."
                )
        else:
            # Explicit parameter provided - determine type
            # Check if it's ROI data (Raw/Epochs) or vertex data (SourceEstimate)
            if isinstance(stc_list, (mne.io.BaseRaw, mne.BaseEpochs)):
                use_roi_mode = True
                roi_data = stc_list
                if hasattr(self, "message"):
                    self.message("info", "Using ROI-optimized mode (68 channels)")
                else:
                    print("INFO: Using ROI-optimized mode (68 channels)")
            elif isinstance(stc_list, (mne.SourceEstimate, list)):
                use_roi_mode = False
                if hasattr(self, "message"):
                    self.message("info", "Using vertex-level mode (20,484 vertices)")
                else:
                    print("INFO: Using vertex-level mode (20,484 vertices)")
            else:
                raise TypeError(
                    f"Data must be Raw, Epochs, SourceEstimate, or list, got {type(stc_list)}"
                )

        # Type checking for vertex mode
        if not use_roi_mode:
            is_single_stc = isinstance(stc_list, mne.SourceEstimate)
            is_list = isinstance(stc_list, list)

            if not (is_single_stc or is_list):
                raise TypeError(
                    f"Data must be mne.SourceEstimate or list, got {type(stc_list)}"
                )

        try:
            # Log start
            if hasattr(self, "message"):
                self.message("header", "Calculating source-level PSD")
                self.message(
                    "info",
                    f"Segment duration: {segment_duration}s, n_jobs: {n_jobs}",
                )
                if use_roi_mode:
                    data_type = "Raw" if isinstance(roi_data, mne.io.BaseRaw) else "Epochs"
                    self.message("info", f"Input: {data_type} with {len(roi_data.ch_names)} ROI channels")
                    self.message("info", "Mode: ROI-optimized (fast)")
                else:
                    if is_single_stc:
                        self.message("info", "Input: Single SourceEstimate")
                    else:
                        self.message("info", f"Input: {len(stc_list)} SourceEstimates")
                    self.message("info", "Mode: Vertex-level with parcellation")
            else:
                print("=== Calculating Source-Level PSD ===")
                print(f"Segment duration: {segment_duration}s, n_jobs: {n_jobs}")
                if use_roi_mode:
                    data_type = "Raw" if isinstance(roi_data, mne.io.BaseRaw) else "Epochs"
                    print(f"Input: {data_type} with {len(roi_data.ch_names)} ROI channels")
                    print("Mode: ROI-optimized (fast)")
                else:
                    if is_single_stc:
                        print("Input: Single SourceEstimate")
                    else:
                        print(f"Input: {len(stc_list)} SourceEstimates")
                    print("Mode: Vertex-level with parcellation")

            # Prepare parameters
            output_dir = None
            subject_id = None  # Leave as None to let algorithm function handle default

            # Get output directory and subject ID from task config
            if hasattr(self, "config"):
                config = self.config
                # Use derivatives_dir if available (BIDS structure)
                if "derivatives_dir" in config:
                    output_dir = Path(config["derivatives_dir"]) / "psd"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(output_dir)
                elif "output_dir" in config:
                    output_dir = config["output_dir"]

                # Get subject ID from config - use same method as export functions
                if "unprocessed_file" in config:
                    subject_id = Path(config["unprocessed_file"]).stem
                elif "subject_id" in config:
                    subject_id = config["subject_id"]
                elif "base_fname" in config:
                    subject_id = config["base_fname"]
                elif "original_fname" in config:
                    # Extract just the stem (no extension)
                    subject_id = Path(config["original_fname"]).stem

            # If output_dir not in config, try to construct from file paths
            if output_dir is None and hasattr(self, "file_path"):
                output_dir = Path(self.file_path).parent / "derivatives" / "source_psd"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_dir)

            # If subject_id not in config, try to extract from filename
            if subject_id is None and hasattr(self, "file_path"):
                subject_id = Path(self.file_path).stem

            # Call appropriate algorithm function based on mode
            if use_roi_mode:
                # ROI-optimized mode: Process 68 channels directly
                psd_df, file_path = calculate_roi_psd(
                    data=roi_data,
                    segment_duration=segment_duration,
                    n_jobs=n_jobs,  # Not currently used but kept for API consistency
                    output_dir=output_dir,
                    subject_id=subject_id,
                    generate_plots=generate_plots,
                )
            else:
                # Vertex-level mode: Process 20,484 vertices with parcellation
                psd_df, file_path = calculate_source_psd_list(
                    stc_list=stc_list,
                    subjects_dir=None,  # Uses MNE environment variable
                    subject="fsaverage",
                    n_jobs=n_jobs,
                    output_dir=output_dir,
                    subject_id=subject_id,
                    generate_plots=generate_plots,
                    segment_duration=segment_duration,
                )

            # Generate additional visualization if requested
            if generate_plots:
                try:
                    visualize_psd_results(
                        psd_df=psd_df, output_dir=output_dir, subject_id=subject_id
                    )
                except Exception as e:
                    if hasattr(self, "message"):
                        self.message("warning", f"Could not generate PSD visualization: {str(e)}")
                    else:
                        print(f"WARNING: Could not generate PSD visualization: {str(e)}")

            # Log completion
            n_rois = len(psd_df["roi"].unique())
            n_freqs = len(psd_df["frequency"].unique())

            if hasattr(self, "message"):
                self.message(
                    "success",
                    f"Source PSD complete: {n_rois} ROIs, {n_freqs} frequencies",
                )
                self.message("info", f"Saved to: {file_path}")
            else:
                print(
                    f"SUCCESS: Source PSD complete: {n_rois} ROIs, {n_freqs} frequencies"
                )
                print(f"Saved to: {file_path}")

            # Update metadata
            if hasattr(self, "_update_metadata"):
                metadata = {
                    "segment_duration": segment_duration,
                    "n_jobs": n_jobs,
                    "generate_plots": generate_plots,
                    "n_rois": n_rois,
                    "n_frequencies": n_freqs,
                    "freq_min": float(psd_df["frequency"].min()),
                    "freq_max": float(psd_df["frequency"].max()),
                    "output_file": file_path,
                    "psd_mode": "roi_optimized" if use_roi_mode else "vertex_level",
                }

                if use_roi_mode:
                    metadata["data_type"] = "raw" if isinstance(roi_data, mne.io.BaseRaw) else "epochs"
                    metadata["n_channels"] = len(roi_data.ch_names)
                    metadata["processing_mode"] = "68_channel_direct"
                else:
                    if is_single_stc:
                        metadata["stc_type"] = "single"
                    else:
                        metadata["stc_type"] = "list"
                        metadata["n_stcs"] = len(stc_list)
                    metadata["processing_mode"] = "vertex_parcellation"

                self._update_metadata("step_apply_source_psd", metadata)

            # Store in task object for downstream use
            self.source_psd_df = psd_df
            self.source_psd_file = file_path

            return psd_df, file_path

        except Exception as e:
            error_msg = f"Error during source PSD calculation: {str(e)}"
            if hasattr(self, "message"):
                self.message("error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(f"Failed to calculate source PSD: {str(e)}") from e