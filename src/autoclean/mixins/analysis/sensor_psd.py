"""Sensor-level PSD analysis mixin for epoched EEG data."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        freq_bands: Optional[
            Union[str, Dict[str, Optional[Tuple[float, float]]]]
        ] = "default",
        time_windows: Optional[
            Union[List[float], Tuple[float, float], Dict[str, Any]]
        ] = None,
        baseline: Optional[
            Union[List[Optional[float]], Tuple[Optional[float], Optional[float]]]
        ] = None,
        stage_name: str = "apply_sensor_psd",
        data: Optional[Union[mne.io.BaseRaw, mne.BaseEpochs]] = None,
    ) -> tuple:
        """Calculate scalp-electrode PSD summaries from Raw or epoched EEG data."""

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
                freq_bands = params.get("freq_bands", freq_bands)
                time_windows = params.get("time_windows", time_windows)
                baseline = params.get("baseline", baseline)

        if data is None:
            data = epochs
        if data is None:
            if hasattr(self, "epochs") and self.epochs is not None:
                data = self.epochs
            elif hasattr(self, "raw") and self.raw is not None:
                data = self.raw
            else:
                raise ValueError(
                    "No epochs available and no raw data available for sensor PSD."
                )

        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError(
                f"data must be an MNE Raw or Epochs object, got {type(data)}"
            )

        method = str(method).lower()
        if method not in {"welch", "multitaper"}:
            raise ValueError(
                f"Unsupported PSD method '{method}'. Use 'welch' or 'multitaper'."
            )

        windows = self._normalize_sensor_psd_time_windows(time_windows)
        if freq_bands == "default":
            requested_bands = DEFAULT_SENSOR_PSD_BANDS
        elif freq_bands is None:
            requested_bands = {}
        else:
            requested_bands = freq_bands

        subject_id = "unknown_subject"
        if hasattr(self, "config") and self.config.get("unprocessed_file"):
            subject_id = Path(self.config["unprocessed_file"]).stem

        psd_rows = []
        band_rows = []
        metadata_windows = {}
        window_details: Dict[str, Dict[str, Any]] = {}
        last_freqs = None
        last_psd_kwargs = {}
        n_channels = 0
        n_observations = 0

        if baseline is not None and isinstance(data, mne.io.BaseRaw):
            message(
                "warning", "Sensor PSD baseline is ignored for continuous Raw input."
            )

        for window_name, window_limits in windows.items():
            data_to_use = data.copy()
            if picks is not None:
                data_to_use.pick(picks)

            if not data_to_use.ch_names:
                raise ValueError("No channels selected for sensor PSD analysis.")

            if baseline is not None and isinstance(data_to_use, mne.BaseEpochs):
                data_to_use.apply_baseline(tuple(baseline))

            if window_limits is not None:
                start, stop = self._validate_sensor_psd_time_window(
                    data_to_use, window_limits
                )
                data_to_use.crop(tmin=start, tmax=stop)
                metadata_windows[window_name] = [float(start), float(stop)]
            else:
                metadata_windows[window_name] = None

            sfreq = float(data_to_use.info["sfreq"])
            sample_count = len(data_to_use.times)
            psd_kwargs = {
                "method": method,
                "fmin": float(fmin),
                "fmax": float(fmax),
                "n_jobs": int(n_jobs),
                "verbose": False,
            }

            if method == "welch":
                local_n_fft = n_fft
                local_n_overlap = n_overlap
                if local_n_fft is None:
                    local_n_fft = max(2, min(int(round(4 * sfreq)), sample_count))
                if local_n_overlap is None:
                    local_n_overlap = min(
                        max(0, local_n_fft // 2), max(0, local_n_fft - 1)
                    )
                psd_kwargs.update(
                    {
                        "n_fft": int(local_n_fft),
                        "n_overlap": int(local_n_overlap),
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
                f"Calculating sensor PSD on {type(data_to_use).__name__} window '{window_name}'",
            )
            message(
                "info",
                f"Method: {method}, range: {float(fmin):.1f}-{float(fmax):.1f} Hz",
            )

            psd_spectrum = data_to_use.compute_psd(**psd_kwargs)
            freqs = psd_spectrum.freqs
            psd_values = psd_spectrum.get_data()

            if isinstance(data_to_use, mne.BaseEpochs):
                mean_psd = np.mean(psd_values, axis=0)
                window_n_observations = len(data_to_use)
            else:
                mean_psd = psd_values
                window_n_observations = len(data_to_use.times)
            n_observations += window_n_observations

            n_channels = len(data_to_use.ch_names)
            last_freqs = freqs
            last_psd_kwargs = psd_kwargs
            valid_bands = self._validate_sensor_psd_freq_bands(
                requested_bands, freqs, fmin=float(fmin), fmax=float(fmax)
            )

            window_detail = {
                "frequency_range": (
                    [float(freqs[0]), float(freqs[-1])] if len(freqs) else []
                ),
                "n_frequencies": int(len(freqs)),
                "n_observations_analyzed": int(window_n_observations),
                "n_channels": int(len(data_to_use.ch_names)),
            }
            if method == "welch":
                window_detail["n_fft"] = int(psd_kwargs.get("n_fft", 0))
                window_detail["n_overlap"] = int(psd_kwargs.get("n_overlap", 0))
            else:
                window_detail["bandwidth"] = (
                    float(psd_kwargs["bandwidth"])
                    if "bandwidth" in psd_kwargs
                    else None
                )
                window_detail["adaptive"] = bool(psd_kwargs.get("adaptive", False))
                window_detail["low_bias"] = bool(psd_kwargs.get("low_bias", True))
                window_detail["normalization"] = psd_kwargs.get(
                    "normalization", normalization
                )
            window_details[window_name] = window_detail

            for ch_idx, ch_name in enumerate(data_to_use.ch_names):
                channel_psd = mean_psd[ch_idx]
                for freq_idx, freq in enumerate(freqs):
                    psd_rows.append(
                        {
                            "subject": subject_id,
                            "time_window": window_name,
                            "channel": ch_name,
                            "frequency": float(freq),
                            "psd": float(channel_psd[freq_idx]),
                        }
                    )

                for band_name, (band_min, band_max) in valid_bands.items():
                    band_mask = (freqs >= band_min) & (freqs < band_max)
                    band_power = (
                        float(np.mean(channel_psd[band_mask]))
                        if np.any(band_mask)
                        else 0.0
                    )
                    band_rows.append(
                        {
                            "subject": subject_id,
                            "time_window": window_name,
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

        freqs = last_freqs if last_freqs is not None else np.array([])
        metadata = {
            "stage_name": stage_name,
            "method": method,
            "input_type": "epochs" if isinstance(data, mne.BaseEpochs) else "raw",
            # Per-window breakdown (authoritative when multiple time windows are
            # requested, since window length/cropping can change frequency
            # resolution between windows).
            "windows": window_details,
            # Convenience top-level fields mirroring the last window processed;
            # only reliable when a single time window was requested.
            "frequency_range": (
                [float(freqs[0]), float(freqs[-1])] if len(freqs) else []
            ),
            "n_frequencies": int(len(freqs)),
            "n_observations_analyzed": int(n_observations),
            "n_channels": int(n_channels),
            "time_windows": metadata_windows,
            "baseline": list(baseline) if baseline is not None else None,
            "freq_bands": {
                name: list(value) if value is not None else None
                for name, value in requested_bands.items()
            },
            "artifact_reports": artifact_paths,
        }
        if method == "welch":
            metadata["n_fft"] = int(last_psd_kwargs.get("n_fft", 0))
            metadata["n_overlap"] = int(last_psd_kwargs.get("n_overlap", 0))
        else:
            metadata["bandwidth"] = float(bandwidth) if bandwidth is not None else None
            metadata["adaptive"] = bool(adaptive)
            metadata["low_bias"] = bool(low_bias)
            metadata["normalization"] = normalization

        self._update_metadata("step_apply_sensor_psd", metadata)

        self.sensor_psd_df = psd_df
        self.sensor_bandpower_df = band_df

        return psd_df, band_df, artifact_paths

    @staticmethod
    def _normalize_sensor_psd_time_windows(
        time_windows: Optional[Union[List[float], Tuple[float, float], Dict[str, Any]]],
    ) -> Dict[str, Optional[Tuple[float, float]]]:
        if time_windows is None:
            return {"all": None}
        if isinstance(time_windows, (list, tuple)) and len(time_windows) == 2:
            return {"window": (float(time_windows[0]), float(time_windows[1]))}
        if isinstance(time_windows, dict):
            normalized = {}
            for name, limits in time_windows.items():
                if limits is None:
                    normalized[str(name)] = None
                elif isinstance(limits, (list, tuple)) and len(limits) == 2:
                    normalized[str(name)] = (float(limits[0]), float(limits[1]))
                else:
                    raise ValueError(
                        "time_windows values must be None or [start, stop] seconds"
                    )
            return normalized
        raise ValueError("time_windows must be None, [start, stop], or a mapping")

    @staticmethod
    def _validate_sensor_psd_time_window(
        data, limits: Tuple[float, float]
    ) -> Tuple[float, float]:
        start, stop = float(limits[0]), float(limits[1])
        if start >= stop:
            raise ValueError("time window start must be less than stop")
        min_time = float(data.times[0])
        max_time = float(data.times[-1])
        if start < min_time or stop > max_time:
            raise ValueError(
                f"time window [{start}, {stop}] is outside available data range "
                f"[{min_time}, {max_time}] seconds"
            )
        return start, stop

    @staticmethod
    def _validate_sensor_psd_freq_bands(
        freq_bands: Dict[str, Optional[Tuple[float, float]]],
        freqs: np.ndarray,
        fmin: float,
        fmax: float,
    ) -> Dict[str, Tuple[float, float]]:
        valid = {}
        min_freq = float(fmin)
        max_freq = float(fmax)
        for name, limits in freq_bands.items():
            if limits is None:
                continue
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise ValueError("freq_bands values must be None or [low, high] Hz")
            low, high = float(limits[0]), float(limits[1])
            if low >= high:
                raise ValueError(
                    f"frequency band '{name}' start must be less than stop"
                )
            if low < min_freq or high > max_freq:
                raise ValueError(
                    f"frequency band '{name}' [{low}, {high}] is outside PSD range "
                    f"[{min_freq}, {max_freq}] Hz"
                )
            valid[str(name)] = (low, high)
        return valid

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
