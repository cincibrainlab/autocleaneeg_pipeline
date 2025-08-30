"""Inter-trial coherence mixin for autoclean tasks.

This module provides functionality for computing inter-trial coherence (ITC)
analysis on epoched EEG data. The mixin wraps the standalone ITC analysis
function while providing configuration handling, metadata tracking, and
result saving capabilities within the AutoClean pipeline.

ITC analysis is particularly useful for statistical learning paradigms where
neural entrainment to rhythmic stimuli is expected.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from autoclean.functions.analysis.statistical_learning import (
    analyze_itc_bands,
    calculate_word_learning_index,
    compute_statistical_learning_itc,
    extract_itc_at_frequencies,
)
from autoclean.utils.logging import message


class InterTrialCoherenceMixin:
    """Mixin class for computing inter-trial coherence (ITC) from epoched EEG data."""

    def compute_itc_analysis(
        self,
        epochs: Union[mne.Epochs, None] = None,
        freqs: Optional[np.ndarray] = None,
        n_cycles: Union[float, np.ndarray] = 7.0,
        time_bandwidth: float = 4.0,
        use_multitaper: bool = False,
        decim: int = 1,
        n_jobs: int = 1,
        picks: Optional[Union[str, List[str]]] = None,
        baseline: Optional[Tuple[float, float]] = None,
        mode: str = "mean",
        stage_name: str = "itc_analysis",
        save_results: bool = True,
        analyze_bands: bool = True,
        time_window: Optional[Tuple[float, float]] = None,
        calculate_wli: bool = False,
    ) -> Tuple[
        mne.time_frequency.AverageTFR,
        mne.time_frequency.AverageTFR,
        Optional[Dict[str, float]],
    ]:
        """Compute inter-trial coherence (ITC) analysis on epoched data.

        This is a mixin wrapper that handles configuration, metadata tracking,
        and result saving. The core ITC computation is implemented in the
        standalone function compute_statistical_learning_itc().

        Parameters
        ----------
        epochs : mne.Epochs, optional
            The epoched data. If None, uses self.epochs.
        freqs : np.ndarray, optional
            Frequencies of interest. If None, uses 0.5-20 Hz range.
        n_cycles : float or np.ndarray, optional
            Number of cycles for Morlet wavelets. Default is 7.0.
        time_bandwidth : float, optional
            Time-bandwidth product for multitaper method. Default is 4.0.
        use_multitaper : bool, optional
            Whether to use multitaper method. Default is False.
        decim : int, optional
            Decimation factor. Default is 1.
        n_jobs : int, optional
            Number of jobs for parallel processing. Default is 1.
        picks : str or list of str, optional
            Channels to include. Default is None (all EEG).
        baseline : tuple of float, optional
            Baseline period for correction. Default is None.
        mode : str, optional
            Baseline correction mode. Default is 'mean'.
        stage_name : str, optional
            Name for saving and metadata tracking. Default is 'itc_analysis'.
        save_results : bool, optional
            Whether to save results to files. Default is True.
        analyze_bands : bool, optional
            Whether to compute frequency band summaries. Default is True.
        time_window : tuple of float, optional
            Time window for band analysis. Default is None (entire epoch).
        calculate_wli : bool, optional
            Whether to calculate Word Learning Index. Default is False.

        Returns
        -------
        power : mne.time_frequency.AverageTFR
            Time-frequency representation of power.
        itc : mne.time_frequency.AverageTFR
            Time-frequency representation of inter-trial coherence.
        band_results : dict or None
            Frequency band analysis results if analyze_bands=True.
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("itc_analysis")

        if not is_enabled:
            message("info", "ITC analysis step is disabled in configuration")
            return None, None, None

        # Get parameters from config if available
        if config_value and isinstance(config_value, dict):
            itc_value = config_value.get("value", {})
            if isinstance(itc_value, dict):
                freqs = itc_value.get("freqs", freqs)
                n_cycles = itc_value.get("n_cycles", n_cycles)
                use_multitaper = itc_value.get("use_multitaper", use_multitaper)
                decim = itc_value.get("decim", decim)
                baseline = itc_value.get("baseline", baseline)
                mode = itc_value.get("mode", mode)
                picks = itc_value.get("picks", picks)
                analyze_bands = itc_value.get("analyze_bands", analyze_bands)
                time_window = itc_value.get("time_window", time_window)
                calculate_wli = itc_value.get("calculate_wli", calculate_wli)

        # Determine which epochs to use
        if epochs is None:
            if hasattr(self, "epochs") and self.epochs is not None:
                epochs = self.epochs
            else:
                raise ValueError(
                    "No epochs available. Run epoching first or provide epochs parameter."
                )

        if not isinstance(epochs, mne.Epochs):
            raise TypeError("epochs must be an MNE Epochs object")

        try:
            message(
                "header",
                f"Computing inter-trial coherence analysis on {len(epochs)} epochs...",
            )

            # Call the standalone ITC analysis function
            power, itc = compute_statistical_learning_itc(
                epochs=epochs,
                freqs=freqs,
                n_cycles=n_cycles,
                time_bandwidth=time_bandwidth,
                use_multitaper=use_multitaper,
                decim=decim,
                n_jobs=n_jobs,
                picks=picks,
                baseline=baseline,
                mode=mode,
                verbose=True,  # Use autoclean logging
            )

            # Compute frequency band analysis if requested
            band_results = None
            if analyze_bands:
                message("info", "Computing frequency band analysis...")
                band_results = analyze_itc_bands(
                    itc=itc,
                    time_window=time_window,
                    picks=picks,
                    verbose=True,
                )

                # Log band results
                message("info", "Frequency band ITC results:")
                for band, value in band_results.items():
                    if not np.isnan(value):
                        message("debug", f"  {band}: {value:.3f}")

            # Calculate Word Learning Index if requested
            wli_results = None
            if calculate_wli:
                message("info", "Computing Word Learning Index (WLI)...")
                wli_results = calculate_word_learning_index(
                    itc=itc,
                    time_window=time_window,
                    picks=picks,
                    verbose=True,
                )

                # Log WLI results
                wli_mean = wli_results.get("wli_mean", np.nan)
                message("info", f"Word Learning Index: {wli_mean:.4f}")
                message("debug", f"  Word ITC: {np.mean(wli_results['itc_word']):.4f}")
                message(
                    "debug",
                    f"  Syllable ITC: {np.mean(wli_results['itc_syllable']):.4f}",
                )

            # Save results if requested
            if save_results:
                self._save_itc_results(
                    power, itc, band_results, stage_name, wli_results
                )

            # Update metadata
            self._update_itc_metadata(power, itc, band_results, stage_name, epochs)

            # Store results in object if it has the capability
            if hasattr(self, "itc_power"):
                self.itc_power = power
            if hasattr(self, "itc_coherence"):
                self.itc_coherence = itc
            if hasattr(self, "itc_bands") and band_results is not None:
                self.itc_bands = band_results

            return power, itc, band_results

        except Exception as e:
            message("error", f"Error during ITC analysis: {str(e)}")
            raise RuntimeError(f"Failed to compute ITC analysis: {str(e)}") from e

    def _save_itc_results(
        self,
        power: mne.time_frequency.AverageTFR,
        itc: mne.time_frequency.AverageTFR,
        band_results: Optional[Dict[str, float]],
        stage_name: str,
        wli_results: Optional[Dict[str, Union[float, np.ndarray]]] = None,
    ) -> None:
        """Save ITC analysis results to files."""
        try:
            # Save power results
            power_filename = f"{stage_name}_power-tfr.h5"
            if hasattr(self, "_save_analysis_result"):
                self._save_analysis_result(
                    result_data=power,
                    stage_name=f"{stage_name}_power",
                    filename=power_filename,
                )
            else:
                # Fallback: save directly if no pipeline save method
                power.save(power_filename, overwrite=True)
                message("debug", f"Saved power TFR to {power_filename}")

            # Save ITC results
            itc_filename = f"{stage_name}_itc-tfr.h5"
            if hasattr(self, "_save_analysis_result"):
                self._save_analysis_result(
                    result_data=itc,
                    stage_name=f"{stage_name}_itc",
                    filename=itc_filename,
                )
            else:
                # Fallback: save directly if no pipeline save method
                itc.save(itc_filename, overwrite=True)
                message("debug", f"Saved ITC TFR to {itc_filename}")

            # Save band results as JSON if available
            if band_results is not None:
                import json

                band_filename = f"{stage_name}_bands.json"

                # Convert numpy values to float for JSON serialization
                json_bands = {}
                for key, value in band_results.items():
                    if isinstance(value, np.ndarray):
                        json_bands[key] = (
                            float(value) if value.size == 1 else value.tolist()
                        )
                    else:
                        json_bands[key] = float(value) if not np.isnan(value) else None

                if hasattr(self, "output_dir"):
                    full_path = os.path.join(self.output_dir, band_filename)
                    with open(full_path, "w") as f:
                        json.dump(json_bands, f, indent=2)
                    message("debug", f"Saved frequency band results to {band_filename}")

            # Save WLI results if available
            if wli_results is not None:
                import json

                wli_filename = f"{stage_name}_wli.json"

                # Convert numpy values to serializable format
                json_wli = {}
                for key, value in wli_results.items():
                    if isinstance(value, np.ndarray):
                        json_wli[key] = (
                            value.tolist() if value.ndim > 0 else float(value)
                        )
                    elif isinstance(value, (int, float, np.integer, np.floating)):
                        json_wli[key] = float(value)
                    elif isinstance(value, list):
                        json_wli[key] = value
                    else:
                        json_wli[key] = str(value)

                if hasattr(self, "output_dir"):
                    full_path = os.path.join(self.output_dir, wli_filename)
                    with open(full_path, "w") as f:
                        json.dump(json_wli, f, indent=2)
                    message(
                        "debug", f"Saved Word Learning Index results to {wli_filename}"
                    )

        except Exception as e:
            message("warning", f"Could not save ITC results: {str(e)}")

    def _update_itc_metadata(
        self,
        power: mne.time_frequency.AverageTFR,
        itc: mne.time_frequency.AverageTFR,
        band_results: Optional[Dict[str, float]],
        stage_name: str,
        epochs: mne.Epochs,
    ) -> None:
        """Update metadata with ITC analysis information."""
        try:
            metadata = {
                "power_shape": power.data.shape,
                "itc_shape": itc.data.shape,
                "frequency_range": [float(power.freqs[0]), float(power.freqs[-1])],
                "n_frequencies": len(power.freqs),
                "time_range": [float(power.times[0]), float(power.times[-1])],
                "n_timepoints": len(power.times),
                "n_channels": power.data.shape[0],
                "n_epochs_analyzed": len(epochs),
                "channel_names": power.ch_names,
                "power_mean": float(np.mean(power.data)),
                "itc_mean": float(np.mean(itc.data)),
                "itc_max": float(np.max(itc.data)),
                "stage_name": stage_name,
            }

            # Add band results to metadata
            if band_results is not None:
                metadata["frequency_bands"] = {}
                for band, value in band_results.items():
                    metadata["frequency_bands"][band] = (
                        float(value) if not np.isnan(value) else None
                    )

            # Update metadata using pipeline method if available
            if hasattr(self, "_update_metadata"):
                self._update_metadata("step_itc_analysis", metadata)

            message("debug", f"Updated metadata for ITC analysis: {stage_name}")

        except Exception as e:
            message("warning", f"Could not update ITC metadata: {str(e)}")
