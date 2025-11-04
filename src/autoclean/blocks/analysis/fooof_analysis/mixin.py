"""specparam analysis mixin for autoclean tasks.

This module provides functionality for spectral parameterization of source-localized
EEG data using the specparam (Fitting Oscillations & One Over F) algorithm.

The specparamAnalysisMixin class implements methods for calculating vertex-level PSDs
and extracting both aperiodic (1/f) and periodic (oscillatory) parameters from
source estimates produced by source localization.

specparam separates neural power spectra into:
1. Aperiodic component: 1/f background activity (offset, knee, exponent)
2. Periodic component: Oscillatory peaks (center frequency, power, bandwidth)

References
----------
Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and
aperiodic components. Nature Neuroscience, 23(12), 1655-1665.
"""

import importlib.util
from pathlib import Path
from typing import Optional

import mne

# Import algorithm functions dynamically (for bundled block compatibility)
_algorithm_path = Path(__file__).parent / "algorithm.py"
_spec = importlib.util.spec_from_file_location("_fooof_algorithm", _algorithm_path)
_algorithm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algorithm_module)
calculate_fooof_aperiodic = _algorithm_module.calculate_fooof_aperiodic
calculate_fooof_periodic = _algorithm_module.calculate_fooof_periodic
calculate_vertex_psd_for_fooof = _algorithm_module.calculate_vertex_psd_for_fooof


class specparamAnalysisMixin:
    """Mixin class providing specparam spectral parameterization functionality.

    This mixin provides methods for decomposing neural power spectra into aperiodic
    and periodic components using the specparam algorithm. The implementation operates
    at the vertex level, providing high spatial resolution parameterization across
    the cortical surface.

    The mixin automatically retrieves source estimates from the task object (self.stc)
    and computes specparam parameters for each vertex. Results are saved as parquet and
    CSV files for easy analysis.

    References
    ----------
    Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and
    aperiodic components. Nature Neuroscience, 23(12), 1655-1665.
    """

    def apply_fooof_aperiodic(
        self,
        stc=None,
        fmin: float = 1.0,
        fmax: float = 45.0,
        n_jobs: int = 10,
        aperiodic_mode: str = "knee",
        stage_name: str = "apply_fooof_aperiodic",
    ) -> tuple:
        """Calculate specparam aperiodic parameters from source estimates.

        This method applies the specparam algorithm to extract aperiodic (1/f) parameters
        from source-localized EEG data. The method first calculates vertex-level PSD
        using Welch's method, then fits the specparam model to extract:
        - Offset: Overall power level
        - Exponent: Slope of 1/f decay
        - Knee (optional): Bend point in 1/f curve

        Args:
            stc: Optional SourceEstimate. If None, uses self.stc
            fmin: Minimum frequency for analysis (default: 1.0 Hz)
            fmax: Maximum frequency for analysis (default: 45.0 Hz)
            n_jobs: Number of parallel jobs (default: 10)
            aperiodic_mode: 'fixed' or 'knee' (default: 'knee')
            stage_name: Name for saving and metadata tracking

        Returns:
            tuple: (aperiodic_df, file_path) where:
                - aperiodic_df: DataFrame with columns [subject, vertex, offset,
                  knee, exponent, r_squared, error, status]
                - file_path: Path to saved parquet file

        Raises:
            AttributeError: If no source estimates found (no self.stc)
            ImportError: If specparam library not available
            RuntimeError: If specparam fitting fails

        Example:
            ```python
            # Apply specparam aperiodic analysis with default parameters
            df, file_path = task.apply_fooof_aperiodic()

            # Apply with custom parameters
            df, file_path = task.apply_fooof_aperiodic(
                fmin=0.5,
                fmax=40.0,
                aperiodic_mode='fixed',
                n_jobs=8
            )

            # Access results
            print(df.head())  # View aperiodic parameters
            ```

        Notes
        -----
        - Requires prior source localization (self.stc must exist)
        - Uses fsaverage brain for visualization
        - Saves two files:
            * {subject}_fooof_aperiodic.parquet: Full parameters
            * {subject}_fooof_aperiodic.csv: Same data in CSV format
        - 'knee' mode better for broadband data, 'fixed' for narrow bands
        - Batch processing with robust error handling
        """
        # Check if this step is enabled in the configuration
        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled("apply_fooof_aperiodic")

            if not is_enabled:
                if hasattr(self, "message"):
                    self.message("info", "specparam aperiodic step is disabled")
                else:
                    print("INFO: specparam aperiodic step is disabled")
                return None, None

            # Get parameters from config if available
            if config_value and isinstance(config_value, dict):
                params = config_value.get("value", config_value)
                fmin = params.get("fmin", fmin)
                fmax = params.get("fmax", fmax)
                n_jobs = params.get("n_jobs", n_jobs)
                aperiodic_mode = params.get("aperiodic_mode", aperiodic_mode)

        # Determine which data to use
        if stc is None:
            if hasattr(self, "stc") and self.stc is not None:
                stc = self.stc
            else:
                raise AttributeError(
                    "No source estimates found. Apply source localization first "
                    "(self.stc must exist)."
                )

        # Type checking
        if not isinstance(stc, mne.SourceEstimate):
            raise TypeError(f"Data must be mne.SourceEstimate, got {type(stc)}")

        try:
            # Log start
            if hasattr(self, "message"):
                self.message("header", "Calculating specparam aperiodic parameters")
                self.message(
                    "info",
                    f"Frequency range: {fmin}-{fmax} Hz, mode: {aperiodic_mode}, n_jobs: {n_jobs}",
                )
            else:
                print("=== Calculating specparam Aperiodic Parameters ===")
                print(
                    f"Frequency range: {fmin}-{fmax} Hz, mode: {aperiodic_mode}, n_jobs: {n_jobs}"
                )

            # Prepare parameters - get from task config
            output_dir = None
            subject_id = None  # Leave as None to let algorithm function handle default

            # Get output directory and subject ID from task config
            if hasattr(self, "config"):
                config = self.config
                # Use derivatives_dir if available (BIDS structure)
                if "derivatives_dir" in config:
                    output_dir = Path(config["derivatives_dir"]) / "fooof"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(output_dir)
                elif "output_dir" in config:
                    output_dir = Path(config["output_dir"]) / "fooof"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(output_dir)

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

            # Fallback: construct from file_path if needed
            if output_dir is None and hasattr(self, "file_path"):
                output_dir = Path(self.file_path).parent / "derivatives" / "fooof"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_dir)

            # Fallback: extract subject_id from filename
            if subject_id is None and hasattr(self, "file_path"):
                subject_id = Path(self.file_path).stem

            # Step 1: Calculate vertex-level PSD
            if hasattr(self, "message"):
                self.message("info", "Step 1: Calculating vertex-level PSD...")
            else:
                print("Step 1: Calculating vertex-level PSD...")

            stc_psd, psd_path = calculate_vertex_psd_for_fooof(
                stc=stc,
                fmin=fmin,
                fmax=fmax,
                n_jobs=n_jobs,
                output_dir=output_dir,
                subject_id=subject_id,
            )

            # Step 2: Calculate specparam aperiodic parameters
            if hasattr(self, "message"):
                self.message("info", "Step 2: Fitting specparam models...")
            else:
                print("Step 2: Fitting specparam models...")

            aperiodic_df, file_path = calculate_fooof_aperiodic(
                stc_psd=stc_psd,
                subject_id=subject_id,
                output_dir=output_dir,
                n_jobs=n_jobs,
                aperiodic_mode=aperiodic_mode,
            )

            # Log completion
            n_vertices = len(aperiodic_df)
            success_count = (aperiodic_df["status"] == "SUCCESS").sum()
            success_rate = (success_count / n_vertices) * 100

            if hasattr(self, "message"):
                self.message(
                    "success",
                    f"specparam aperiodic complete: {n_vertices} vertices, {success_rate:.1f}% success",
                )
                self.message("info", f"Saved to: {file_path}")
            else:
                print(
                    f"SUCCESS: specparam aperiodic complete: {n_vertices} vertices, {success_rate:.1f}% success"
                )
                print(f"Saved to: {file_path}")

            # Update metadata
            if hasattr(self, "_update_metadata"):
                # Get block info for reproducibility
                block_info = self._get_block_info("fooof_analysis")

                metadata = {
                    "block_name": "fooof_analysis",
                    "fmin": fmin,
                    "fmax": fmax,
                    "n_jobs": n_jobs,
                    "aperiodic_mode": aperiodic_mode,
                    "n_vertices": n_vertices,
                    "success_count": int(success_count),
                    "success_rate": float(success_rate),
                    "output_file": file_path,
                }

                # Add block version/commit info for reproducibility
                if block_info:
                    metadata.update(block_info)

                self._update_metadata("step_apply_fooof_aperiodic", metadata)

            # Store in task object for downstream use
            self.fooof_aperiodic_df = aperiodic_df
            self.fooof_aperiodic_file = file_path
            self.stc_psd = stc_psd  # Store PSD STC for potential reuse

            return aperiodic_df, file_path

        except Exception as e:
            error_msg = f"Error during specparam aperiodic analysis: {str(e)}"
            if hasattr(self, "message"):
                self.message("error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(
                f"Failed to calculate specparam aperiodic: {str(e)}"
            ) from e

    def apply_fooof_periodic(
        self,
        stc_psd=None,
        freq_bands: Optional[dict] = None,
        n_jobs: int = 10,
        aperiodic_mode: str = "knee",
        stage_name: str = "apply_fooof_periodic",
    ) -> tuple:
        """Calculate specparam periodic (oscillatory) parameters from source estimates.

        This method extracts oscillatory peaks from source-localized EEG data using
        specparam. For each frequency band, it identifies the dominant peak and returns:
        - Center frequency: Peak frequency in Hz
        - Power: Peak power/amplitude
        - Bandwidth: Width of the peak

        Args:
            stc_psd: Optional SourceEstimate with PSD data. If None, uses self.stc_psd
                (must have run apply_fooof_aperiodic first)
            freq_bands: Dictionary of bands, e.g., {'alpha': (8, 13)}. If None,
                uses defaults: delta, theta, alpha, beta, gamma
            n_jobs: Number of parallel jobs (default: 10)
            aperiodic_mode: 'fixed' or 'knee' (default: 'knee')
            stage_name: Name for saving and metadata tracking

        Returns:
            tuple: (periodic_df, file_path) where:
                - periodic_df: DataFrame with columns [subject, vertex, band,
                  center_frequency, power, bandwidth]
                - file_path: Path to saved parquet file

        Raises:
            AttributeError: If no PSD source estimates found
            ImportError: If specparam library not available
            RuntimeError: If specparam fitting fails

        Example:
            ```python
            # Apply specparam periodic analysis (after aperiodic)
            df, file_path = task.apply_fooof_periodic()

            # Apply with custom frequency bands
            custom_bands = {
                'slow_alpha': (8, 10),
                'fast_alpha': (10, 13),
                'beta': (13, 30)
            }
            df, file_path = task.apply_fooof_periodic(freq_bands=custom_bands)

            # Access results
            print(df.groupby('band')['center_frequency'].mean())
            ```

        Notes
        -----
        - Requires either stc_psd or prior call to apply_fooof_aperiodic()
        - Default bands: delta (1-4), theta (4-8), alpha (8-13), beta (13-30),
          gamma (30-45)
        - Saves two files:
            * {subject}_fooof_periodic.parquet: Full parameters
            * {subject}_fooof_periodic.csv: Same data in CSV format
        - Returns NaN for bands with no detected peaks
        """
        # Check if this step is enabled in the configuration
        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled("apply_fooof_periodic")

            if not is_enabled:
                if hasattr(self, "message"):
                    self.message("info", "specparam periodic step is disabled")
                else:
                    print("INFO: specparam periodic step is disabled")
                return None, None

            # Get parameters from config if available
            if config_value and isinstance(config_value, dict):
                params = config_value.get("value", config_value)
                freq_bands = params.get("freq_bands", freq_bands)
                n_jobs = params.get("n_jobs", n_jobs)
                aperiodic_mode = params.get("aperiodic_mode", aperiodic_mode)

        # Determine which data to use
        if stc_psd is None:
            if hasattr(self, "stc_psd") and self.stc_psd is not None:
                stc_psd = self.stc_psd
            else:
                raise AttributeError(
                    "No PSD source estimates found. Run apply_fooof_aperiodic first "
                    "(self.stc_psd must exist)."
                )

        try:
            # Log start
            if hasattr(self, "message"):
                self.message("header", "Calculating specparam periodic parameters")
                self.message(
                    "info",
                    f"Mode: {aperiodic_mode}, n_jobs: {n_jobs}",
                )
            else:
                print("=== Calculating specparam Periodic Parameters ===")
                print(f"Mode: {aperiodic_mode}, n_jobs: {n_jobs}")

            # Prepare parameters - get from task config
            output_dir = None
            subject_id = None  # Leave as None to let algorithm function handle default

            # Get output directory and subject ID from task config
            if hasattr(self, "config"):
                config = self.config
                # Use derivatives_dir if available (BIDS structure)
                if "derivatives_dir" in config:
                    output_dir = Path(config["derivatives_dir"]) / "fooof"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(output_dir)
                elif "output_dir" in config:
                    output_dir = Path(config["output_dir"]) / "fooof"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(output_dir)

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

            # Fallback: construct from file_path if needed
            if output_dir is None and hasattr(self, "file_path"):
                output_dir = Path(self.file_path).parent / "derivatives" / "fooof"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_dir)

            # Fallback: extract subject_id from filename
            if subject_id is None and hasattr(self, "file_path"):
                subject_id = Path(self.file_path).stem

            # Calculate specparam periodic parameters
            periodic_df, file_path = calculate_fooof_periodic(
                stc=stc_psd,
                freq_bands=freq_bands,
                n_jobs=n_jobs,
                output_dir=output_dir,
                subject_id=subject_id,
                aperiodic_mode=aperiodic_mode,
            )

            # Log completion
            n_vertices = len(periodic_df["vertex"].unique())
            n_bands = len(periodic_df["band"].unique())

            if hasattr(self, "message"):
                self.message(
                    "success",
                    f"specparam periodic complete: {n_vertices} vertices, {n_bands} bands",
                )
                self.message("info", f"Saved to: {file_path}")
            else:
                print(
                    f"SUCCESS: specparam periodic complete: {n_vertices} vertices, {n_bands} bands"
                )
                print(f"Saved to: {file_path}")

            # Update metadata
            if hasattr(self, "_update_metadata"):
                # Get block info for reproducibility
                block_info = self._get_block_info("fooof_analysis")

                metadata = {
                    "block_name": "fooof_analysis",
                    "n_jobs": n_jobs,
                    "aperiodic_mode": aperiodic_mode,
                    "n_vertices": n_vertices,
                    "n_bands": n_bands,
                    "bands": list(periodic_df["band"].unique()),
                    "output_file": file_path,
                }

                # Add block version/commit info for reproducibility
                if block_info:
                    metadata.update(block_info)

                self._update_metadata("step_apply_fooof_periodic", metadata)

            # Store in task object for downstream use
            self.fooof_periodic_df = periodic_df
            self.fooof_periodic_file = file_path

            return periodic_df, file_path

        except Exception as e:
            error_msg = f"Error during specparam periodic analysis: {str(e)}"
            if hasattr(self, "message"):
                self.message("error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(
                f"Failed to calculate specparam periodic: {str(e)}"
            ) from e
