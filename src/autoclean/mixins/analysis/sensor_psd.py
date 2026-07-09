"""Sensor-level PSD analysis mixin for epoched EEG data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd

from autoclean.utils.logging import message

DEFAULT_SENSOR_PSD_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "lowalpha": (8.0, 10.0),
    "highalpha": (10.0, 13.0),
    "lowbeta": (13.0, 20.0),
    "highbeta": (20.0, 30.0),
    "gamma": (30.0, 45.0),
}


class SensorPSDMixin:
    """Mixin providing electrode-level PSD summaries for epoched EEG data."""

    def apply_sensor_psd(
        self,
        epochs: Optional[mne.BaseEpochs] = None,
        method: str = "welch",
        fmin: float = 1.0,
        fmax: float = 45.0,
        picks: Optional[Union[str, List[str], Tuple[str, ...]]] = "eeg",
        n_jobs: int = 1,
        n_fft: Optional[int] = None,
        n_overlap: Optional[int] = None,
        bandwidth: Optional[float] = None,
        adaptive: bool = False,
        low_bias: bool = True,
        normalization: str = "length",
        stage_name: str = "apply_sensor_psd",
    ) -> tuple:
        """Calculate scalp-electrode PSD from epoched EEG data.

        The output is intentionally averaged across epochs so it stays usable for
        routine task pipelines and large batch runs. Frequency-resolved PSD is
        saved per electrode, alongside a band-power summary table.
        """

        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled("apply_sensor_psd")
            if not is_enabled:
                message("info", "Sensor PSD step is disabled")
                return None, None, {}

            if config_value and isinstance(config_value, dict):
                params = config_value.get("value", config_value)
                method = params.get("method", method)
                fmin = params.get("fmin", fmin)
                fmax = params.get("fmax", fmax)
                picks = params.get("picks", picks)
                n_jobs = params.get("n_jobs", n_jobs)
                n_fft = params.get("n_fft", n_fft)
                n_overlap = params.get("n_overlap", n_overlap)
                bandwidth = params.get("bandwidth", bandwidth)
                adaptive = params.get("adaptive", adaptive)
                low_bias = params.get("low_bias", low_bias)
                normalization = params.get("normalization", normalization)

        if epochs is None:
            if hasattr(self, "epochs") and self.epochs is not None:
                epochs = self.epochs
            else:
                raise ValueError(
                    "No epochs available. Run epoching first or provide epochs."
                )

        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError(f"epochs must be an MNE Epochs object, got {type(epochs)}")

        method = str(method).lower()
        if method not in {"welch", "multitaper"}:
            raise ValueError(
                f"Unsupported PSD method '{method}'. Use 'welch' or 'multitaper'."
            )

        epochs_to_use = epochs.copy()
        if picks is not None:
            epochs_to_use.pick(picks)

        if not epochs_to_use.ch_names:
            raise ValueError("No channels selected for sensor PSD analysis.")

        sfreq = float(epochs_to_use.info["sfreq"])
        epoch_samples = len(epochs_to_use.times)

        psd_kwargs = {
            "method": method,
            "fmin": float(fmin),
            "fmax": float(fmax),
            "n_jobs": int(n_jobs),
            "verbose": False,
        }

        if method == "welch":
            if n_fft is None:
                n_fft = max(2, min(int(round(4 * sfreq)), epoch_samples))
            if n_overlap is None:
                n_overlap = min(max(0, n_fft // 2), max(0, n_fft - 1))
            psd_kwargs.update(
                {
                    "n_fft": int(n_fft),
                    "n_overlap": int(n_overlap),
                    "average": "mean",
                }
            )
        else:
            if bandwidth is not None:
                psd_kwargs["bandwidth"] = float(bandwidth)
            psd_kwargs["adaptive"] = bool(adaptive)
            psd_kwargs["low_bias"] = bool(low_bias)
            psd_kwargs["normalization"] = normalization

        message(
            "header",
            f"Calculating sensor PSD on {len(epochs_to_use)} epochs and "
            f"{len(epochs_to_use.ch_names)} channels",
        )
        message(
            "info",
            f"Method: {method}, range: {float(fmin):.1f}-{float(fmax):.1f} Hz",
        )

        psd_spectrum = epochs_to_use.compute_psd(**psd_kwargs)
        freqs = psd_spectrum.freqs
        psd_values = psd_spectrum.get_data()

        # EpochsSpectrum returns (n_epochs, n_channels, n_freqs). Collapse to one
        # PSD per channel so the saved artifact remains compact and analysis-ready.
        mean_psd = np.mean(psd_values, axis=0)

        subject_id = "unknown_subject"
        if hasattr(self, "config") and self.config.get("unprocessed_file"):
            subject_id = Path(self.config["unprocessed_file"]).stem

        psd_rows = []
        band_rows = []
        for ch_idx, ch_name in enumerate(epochs_to_use.ch_names):
            channel_psd = mean_psd[ch_idx]
            for freq_idx, freq in enumerate(freqs):
                psd_rows.append(
                    {
                        "subject": subject_id,
                        "channel": ch_name,
                        "frequency": float(freq),
                        "psd": float(channel_psd[freq_idx]),
                    }
                )

            for band_name, (band_min, band_max) in DEFAULT_SENSOR_PSD_BANDS.items():
                band_mask = (freqs >= band_min) & (freqs < band_max)
                band_power = (
                    float(np.mean(channel_psd[band_mask])) if np.any(band_mask) else 0.0
                )
                band_rows.append(
                    {
                        "subject": subject_id,
                        "channel": ch_name,
                        "band": band_name,
                        "band_start_hz": float(band_min),
                        "band_end_hz": float(band_max),
                        "power": band_power,
                    }
                )

        psd_df = pd.DataFrame(psd_rows)
        band_df = pd.DataFrame(band_rows)
        artifact_paths = self._save_sensor_psd_tables(
            psd_df=psd_df,
            band_df=band_df,
            subject_id=subject_id,
            stage_name=stage_name,
        )

        metadata = {
            "stage_name": stage_name,
            "method": method,
            "frequency_range": [float(freqs[0]), float(freqs[-1])],
            "n_frequencies": int(len(freqs)),
            "n_epochs_analyzed": int(len(epochs_to_use)),
            "n_channels": int(len(epochs_to_use.ch_names)),
            "channel_names": list(epochs_to_use.ch_names),
            "sfreq": sfreq,
            "artifact_reports": artifact_paths,
        }
        if method == "welch":
            metadata["n_fft"] = int(n_fft)
            metadata["n_overlap"] = int(n_overlap)
        else:
            metadata["bandwidth"] = float(bandwidth) if bandwidth is not None else None
            metadata["adaptive"] = bool(adaptive)
            metadata["low_bias"] = bool(low_bias)
            metadata["normalization"] = normalization

        self._update_metadata("step_apply_sensor_psd", metadata)

        self.sensor_psd_df = psd_df
        self.sensor_bandpower_df = band_df

        return psd_df, band_df, artifact_paths

    def _save_sensor_psd_tables(
        self,
        psd_df: pd.DataFrame,
        band_df: pd.DataFrame,
        subject_id: str,
        stage_name: str,
    ) -> Dict[str, str]:
        """Persist sensor PSD outputs and return portable artifact references."""

        output_dir = self._resolve_report_path("sensor_psd")
        psd_parquet_path = output_dir / f"{subject_id}_{stage_name}_spectra.parquet"
        psd_csv_path = output_dir / f"{subject_id}_{stage_name}_spectra.csv"
        band_csv_path = output_dir / f"{subject_id}_{stage_name}_bands.csv"

        saved_psd_path = psd_parquet_path
        try:
            psd_df.to_parquet(psd_parquet_path, index=False)
        except Exception:
            psd_df.to_csv(psd_csv_path, index=False)
            saved_psd_path = psd_csv_path

        band_df.to_csv(band_csv_path, index=False)

        message("info", f"Saved sensor PSD spectra to {saved_psd_path}")
        message("info", f"Saved sensor PSD band powers to {band_csv_path}")

        return {
            "sensor_psd_spectra": str(self._report_relative_path(saved_psd_path)),
            "sensor_psd_bands": str(self._report_relative_path(band_csv_path)),
        }
