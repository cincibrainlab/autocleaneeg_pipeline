import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

DEFAULT_ASSR_FREQ_BANDS = {
    "alpha": (8, 13),
    "theta": (4, 7),
    "gamma1": (30, 55),
    "gamma2": (65, 80),
    "itc40": (35, 45),
    "itc80": (75, 85),
    "itc_onset": (2, 13),
}

DEFAULT_ASSR_TIME_WINDOWS = {
    "all": (0, 3.0),
    "itc_onset": (0.092, 0.308),
    "itc_offset": (2.8, 3.0),
}

DEFAULT_ASSR_COMBINED_BANDS = {
    "gamma": (30, 80),
}

LEGACY_ASSR_METRIC_COLUMNS = [
    "eegid",
    "trials",
    "chan",
    "rejtrials",
    "stp_gamma",
    "stp_gamma1",
    "stp_gamma2",
    "stp_alpha",
    "stp_theta",
    "ersp_gamma",
    "ersp_gamma1",
    "ersp_gamma2",
    "ersp_alpha",
    "ersp_theta",
    "itc40",
    "itc80",
    "itconset",
    "itcoffset",
]

LEGACY_ASSR_SETTINGS = {
    "profile": None,
    "baseline": (-0.5, 0),
    "time_windows": DEFAULT_ASSR_TIME_WINDOWS,
    "freq_bands": DEFAULT_ASSR_FREQ_BANDS,
    "combined_bands": DEFAULT_ASSR_COMBINED_BANDS,
    "exclude_channel_types": [],
    "save_tfr": False,
}

ASSR_ANALYSIS_PROFILES = {
    "assr_epochs": {
        "profile": "assr_epochs",
        "baseline": (-0.2, 0.0),
        "time_windows": {
            "all": (0.0, 0.7),
            "itc_onset": (0.092, 0.308),
            "itc_offset": (0.5, 0.7),
        },
        "freq_bands": DEFAULT_ASSR_FREQ_BANDS,
        "combined_bands": {
            "gamma_combined": [(30, 55), (65, 80)],
        },
        "exclude_channel_types": ["eog", "ecg", "misc"],
        "save_tfr": False,
    }
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_assr_analysis_settings(
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve ASSR analysis profile settings plus task/runner overrides."""
    if profile is None and overrides is None:
        return copy.deepcopy(LEGACY_ASSR_SETTINGS)

    if profile is None:
        settings = copy.deepcopy(LEGACY_ASSR_SETTINGS)
    else:
        if profile not in ASSR_ANALYSIS_PROFILES:
            available = ", ".join(sorted(ASSR_ANALYSIS_PROFILES))
            raise ValueError(
                f"Unknown ASSR analysis profile '{profile}'. Available: {available}"
            )
        settings = copy.deepcopy(ASSR_ANALYSIS_PROFILES[profile])

    if overrides:
        settings = _deep_merge(settings, overrides)
    return settings


def resolve_assr_analysis_config(
    analysis_profile: str | None = None,
    analysis_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve settings from explicit args or a task-style config mapping."""
    config = analysis_config or {}
    profile = analysis_profile or config.get("profile")
    overrides = {key: value for key, value in config.items() if key != "profile"}
    return resolve_assr_analysis_settings(profile=profile, overrides=overrides or None)


def load_epochs(file_path):
    """
    Load epochs from an EEGLAB .set file

    Parameters:
    -----------
    file_path : str or Path
        Path to the EEGLAB .set file

    Returns:
    --------
    epochs : mne.Epochs
        The loaded epochs object
    """
    file_path = Path(file_path)
    epochs = mne.io.read_epochs_eeglab(file_path)
    print(f"Loaded {len(epochs)} epochs with {len(epochs.ch_names)} channels")
    print(f"Epoch duration: {epochs.times[0]:.3f}s to {epochs.times[-1]:.3f}s")
    return epochs


def compute_time_frequency(epochs, freqs=None, n_cycles=None, baseline=(-0.5, 0)):
    """
    Compute time-frequency representations (power, ITC, ERSP, single trial power)

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to analyze
    freqs : array, optional
        Frequencies to analyze. If None, will use optimized frequency array for 40 Hz
    n_cycles : array, optional
        Number of cycles for Morlet wavelets. If None, will optimize for 40 Hz
    baseline : tuple, optional
        Baseline period for ERSP calculation

    Returns:
    --------
    dict : Dictionary containing time-frequency results
        - 'power': Average power
        - 'itc': ITC values (tuple with power and itc)
        - 'ersp': Event-related spectral perturbation
        - 'single_trial_power': Single trial power
        - 'freqs': Frequency array used
    """
    # Create optimized frequency array if not provided
    if freqs is None:
        freqs_low = np.arange(1, 30, 2)  # Lower frequencies with coarser resolution
        freqs_mid = np.arange(30, 50, 0.5)  # Finer resolution around 40 Hz
        freqs_high = np.arange(50, 101, 2)  # Higher frequencies with coarser resolution
        freqs = np.concatenate([freqs_low, freqs_mid, freqs_high])

    # Optimize wavelet cycles if not provided
    if n_cycles is None:
        n_cycles_base = freqs / 2.0  # Base cycles
        # Increase cycles around 40 Hz for better frequency resolution
        n_cycles = n_cycles_base.copy()
        freq_mask = (freqs >= 35) & (freqs <= 45)
        n_cycles[freq_mask] = freqs[freq_mask] / 1.5  # More cycles around 40 Hz

    # Compute average power
    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=1,
        average=True,
    )

    # Compute ITC (inter-trial coherence)
    itc = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=True,
        decim=3,
        n_jobs=1,
        average=True,
    )

    # Compute single trial power (non-baseline corrected)
    single_trial_power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=1,
        average=False,
    )

    # Compute ERSP (baseline corrected power)
    ersp = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        decim=3,
        n_jobs=1,
        average=True,
    )
    # Apply baseline correction after computing TFR
    ersp.apply_baseline(baseline, mode="mean")

    return {
        "power": power,
        "itc": itc,
        "ersp": ersp,
        "single_trial_power": single_trial_power,
        "freqs": freqs,
    }


def _record_skip(audit: dict[str, Any], category: str, name: str, reason: str) -> None:
    audit.setdefault(category, []).append({"name": name, "reason": reason})
    warnings.warn(
        f"Skipping ASSR {category.removeprefix('skipped_')} '{name}': {reason}"
    )


def _normalize_segments(value: Any) -> list[tuple[float, float]] | None:
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return [(float(value[0]), float(value[1]))]

    segments = []
    if isinstance(value, (list, tuple)):
        for segment in value:
            if not (
                isinstance(segment, (list, tuple))
                and len(segment) == 2
                and all(isinstance(item, (int, float)) for item in segment)
            ):
                return []
            segments.append((float(segment[0]), float(segment[1])))
        return segments
    return []


def _validate_time_windows(
    time_windows: dict[str, Any] | None,
    times: np.ndarray,
    audit: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    valid = {}
    if not time_windows:
        return valid

    time_min = float(np.min(times))
    time_max = float(np.max(times))
    for name, window in time_windows.items():
        if not (
            isinstance(window, (list, tuple))
            and len(window) == 2
            and all(isinstance(item, (int, float)) for item in window)
        ):
            _record_skip(audit, "skipped_time_windows", name, "invalid_shape")
            continue
        tmin = float(window[0])
        tmax = float(window[1])
        if tmin > tmax:
            _record_skip(audit, "skipped_time_windows", name, "min_greater_than_max")
            continue
        if tmax < time_min or tmin > time_max:
            _record_skip(audit, "skipped_time_windows", name, "outside_epoch_range")
            continue
        indices = np.where((times >= tmin) & (times <= tmax))[0]
        if len(indices) == 0:
            _record_skip(audit, "skipped_time_windows", name, "no_time_bins")
            continue
        valid[name] = (tmin, tmax)
    return valid


def _validate_freq_bands(
    freq_bands: dict[str, Any] | None,
    freqs: np.ndarray,
    audit: dict[str, Any],
) -> dict[str, list[tuple[float, float]]]:
    valid = {}
    if freq_bands is None:
        _record_skip(audit, "skipped_freq_bands", "freq_bands", "not_configured")
        return valid

    freq_min = float(np.min(freqs))
    freq_max = float(np.max(freqs))
    for name, band in freq_bands.items():
        segments = _normalize_segments(band)
        if segments is None:
            _record_skip(audit, "skipped_freq_bands", name, "not_configured")
            continue
        if not segments:
            _record_skip(audit, "skipped_freq_bands", name, "invalid_shape")
            continue

        valid_segments = []
        for fmin, fmax in segments:
            if fmin > fmax:
                _record_skip(audit, "skipped_freq_bands", name, "min_greater_than_max")
                valid_segments = []
                break
            if fmax < freq_min or fmin > freq_max:
                continue
            indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            if len(indices) > 0:
                valid_segments.append((fmin, fmax))
        if valid_segments:
            valid[name] = valid_segments
        else:
            _record_skip(
                audit, "skipped_freq_bands", name, "outside_tfr_frequency_range"
            )
    return valid


def _segment_freq_indices(
    freqs: np.ndarray,
    segments: list[tuple[float, float]],
) -> np.ndarray:
    index_parts = [
        np.where((freqs >= fmin) & (freqs <= fmax))[0] for fmin, fmax in segments
    ]
    if not index_parts:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(index_parts)).astype(int)


def _time_indices(times: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    return np.where((times >= window[0]) & (times <= window[1]))[0]


def _channel_indices_for_metrics(
    epochs: Any,
    exclude_channel_types: list[str] | None,
    audit: dict[str, Any],
) -> list[tuple[int, str]]:
    excluded_types = {item.lower() for item in (exclude_channel_types or [])}
    if not excluded_types:
        return list(enumerate(epochs.ch_names))

    channel_types = []
    if hasattr(epochs, "get_channel_types"):
        channel_types = [item.lower() for item in epochs.get_channel_types()]
    else:
        channel_types = ["eeg"] * len(epochs.ch_names)

    included = []
    excluded_channels = []
    for index, (channel, channel_type) in enumerate(
        zip(epochs.ch_names, channel_types)
    ):
        if channel_type in excluded_types:
            excluded_channels.append({"channel": channel, "type": channel_type})
        else:
            included.append((index, channel))
    audit["excluded_channels"] = excluded_channels
    return included


def compute_metrics(
    tf_data,
    epochs,
    freq_bands=None,
    time_windows=None,
    combined_bands=None,
    exclude_channel_types=None,
    audit=None,
):
    """
    Compute ASSR metrics from time-frequency data.

    Invalid or unavailable configured bands/windows are skipped and recorded in
    ``audit`` instead of producing all-NaN metric columns silently.
    """
    audit = audit if audit is not None else {}
    itc = tf_data["itc"]
    single_trial_power = tf_data["single_trial_power"]
    ersp = tf_data["ersp"]
    freqs = np.asarray(tf_data["freqs"])

    if freq_bands is None and combined_bands is None:
        freq_bands = DEFAULT_ASSR_FREQ_BANDS
        combined_bands = DEFAULT_ASSR_COMBINED_BANDS
    if time_windows is None:
        time_windows = DEFAULT_ASSR_TIME_WINDOWS
    if combined_bands is None:
        combined_bands = {}

    valid_windows = _validate_time_windows(time_windows, itc[1].times, audit)
    valid_bands = _validate_freq_bands(freq_bands, freqs, audit)
    valid_combined_bands = _validate_freq_bands(combined_bands, freqs, audit)
    all_window = valid_windows.get("all")

    if hasattr(epochs, "filename") and epochs.filename is not None:
        file_path = Path(epochs.filename)
        file_basename = file_path.stem
    else:
        file_basename = "synthetic_epochs"

    n_total_trials = len(epochs.drop_log)
    n_rejected_trials = sum(1 for log in epochs.drop_log if log)
    channel_indices = _channel_indices_for_metrics(epochs, exclude_channel_types, audit)

    results = []
    for ch_idx, ch_name in channel_indices:
        row = {
            "eegid": file_basename,
            "trials": n_total_trials,
            "chan": ch_name,
            "rejtrials": n_rejected_trials,
        }

        if all_window is not None:
            time_all_idx = _time_indices(itc[1].times, all_window)
            if "itc40" in valid_bands:
                indices = _segment_freq_indices(freqs, valid_bands["itc40"])
                row["itc40"] = np.mean(itc[1].data[ch_idx, indices][:, time_all_idx])
            if "itc80" in valid_bands:
                indices = _segment_freq_indices(freqs, valid_bands["itc80"])
                row["itc80"] = np.mean(itc[1].data[ch_idx, indices][:, time_all_idx])

        if "itc_onset" in valid_bands and "itc_onset" in valid_windows:
            freq_idx = _segment_freq_indices(freqs, valid_bands["itc_onset"])
            time_idx = _time_indices(itc[1].times, valid_windows["itc_onset"])
            row["itconset"] = np.mean(itc[1].data[ch_idx, freq_idx][:, time_idx])

        if "itc_onset" in valid_bands and "itc_offset" in valid_windows:
            freq_idx = _segment_freq_indices(freqs, valid_bands["itc_onset"])
            time_idx = _time_indices(itc[1].times, valid_windows["itc_offset"])
            row["itcoffset"] = np.mean(itc[1].data[ch_idx, freq_idx][:, time_idx])

        for band_name, segments in valid_bands.items():
            if band_name.startswith("itc"):
                continue
            freq_idx = _segment_freq_indices(freqs, segments)
            row[f"stp_{band_name}"] = np.mean(
                single_trial_power.data[:, ch_idx, freq_idx, :]
            )
            row[f"ersp_{band_name}"] = np.mean(ersp.data[ch_idx, freq_idx, :])

        for band_name, segments in valid_combined_bands.items():
            freq_idx = _segment_freq_indices(freqs, segments)
            row[f"stp_{band_name}"] = np.mean(
                single_trial_power.data[:, ch_idx, freq_idx, :]
            )
            row[f"ersp_{band_name}"] = np.mean(ersp.data[ch_idx, freq_idx, :])

        results.append(row)

    results_df = pd.DataFrame(results)
    ordered_columns = [
        column for column in LEGACY_ASSR_METRIC_COLUMNS if column in results_df.columns
    ]
    ordered_columns.extend(
        column for column in results_df.columns if column not in ordered_columns
    )
    return results_df.loc[:, ordered_columns]


def _resolve_file_basename(
    file_path: Any, epochs: Any, file_basename: str | None
) -> str:
    if file_basename is not None:
        return file_basename
    if file_path is not None:
        return Path(file_path).stem
    if hasattr(epochs, "filename") and epochs.filename is not None:
        return Path(epochs.filename).stem
    return "assr_analysis"


def _save_tfr_artifacts(
    tf_data: dict[str, Any],
    data_dir: Path,
    file_basename: str,
    audit: dict[str, Any],
) -> None:
    saved = []
    for name in ("power", "ersp", "single_trial_power"):
        tfr = tf_data.get(name)
        if tfr is None or not hasattr(tfr, "save"):
            continue
        output_file = data_dir / f"{file_basename}_{name}_assr-tfr.h5"
        tfr.save(output_file, overwrite=True)
        saved.append(str(output_file))

    itc = tf_data.get("itc")
    if isinstance(itc, tuple) and len(itc) > 1 and hasattr(itc[1], "save"):
        output_file = data_dir / f"{file_basename}_itc_assr-tfr.h5"
        itc[1].save(output_file, overwrite=True)
        saved.append(str(output_file))
    audit["saved_tfr_artifacts"] = saved


def _write_analysis_metadata(
    data_dir: Path,
    file_basename: str,
    settings: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    payload = {
        "analysis_settings": _json_ready(settings),
        "analysis_audit": _json_ready(audit),
    }
    metadata_file = data_dir / f"{file_basename}_assr_analysis_settings.json"
    log_file = data_dir / f"{file_basename}_assr_analysis_log.json"
    with metadata_file.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)
    with log_file.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2)


def analyze_assr(
    file_path=None,
    output_dir=None,
    save_results=True,
    epochs=None,
    file_basename=None,
    analysis_profile=None,
    analysis_config=None,
):
    """
    Main function to analyze ASSR data.

    ``analysis_profile`` may be set to ``"assr_epochs"``. ``analysis_config``
    can supply task/runner overrides such as ``time_windows``, ``freq_bands``,
    ``combined_bands``, ``baseline``, ``exclude_channel_types``, and
    ``save_tfr``. When neither is supplied, legacy defaults are preserved.
    """
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if epochs is None:
        if file_path is None:
            raise ValueError("Either file_path or epochs must be provided")
        epochs = load_epochs(file_path)

    analysis_settings = resolve_assr_analysis_config(
        analysis_profile=analysis_profile,
        analysis_config=analysis_config,
    )
    audit = {
        "profile": analysis_settings.get("profile"),
        "skipped_freq_bands": [],
        "skipped_time_windows": [],
        "excluded_channels": [],
    }

    tf_data = compute_time_frequency(epochs, baseline=analysis_settings["baseline"])
    results_df = compute_metrics(
        tf_data,
        epochs,
        freq_bands=analysis_settings.get("freq_bands"),
        time_windows=analysis_settings.get("time_windows"),
        combined_bands=analysis_settings.get("combined_bands"),
        exclude_channel_types=analysis_settings.get("exclude_channel_types"),
        audit=audit,
    )

    freq_idx = np.argmin(np.abs(tf_data["freqs"] - 40))
    print(
        f"Maximum ITC value at 40 Hz: {np.max(tf_data['itc'][1].data[:, freq_idx, :]):.3f}"
    )

    freqs_mid = [f for f in tf_data["freqs"] if 30 <= f <= 50]
    if len(freqs_mid) > 1:
        print(
            f"Frequency resolution around 40 Hz: {freqs_mid[1] - freqs_mid[0]:.2f} Hz"
        )

    resolved_basename = _resolve_file_basename(file_path, epochs, file_basename)
    if save_results:
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        output_filename = data_dir / f"{resolved_basename}_metrics_assr.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"Saved analysis results to {output_filename}")

        if analysis_settings.get("save_tfr"):
            _save_tfr_artifacts(tf_data, data_dir, resolved_basename, audit)
        _write_analysis_metadata(data_dir, resolved_basename, analysis_settings, audit)

    return {
        "results_df": results_df,
        "tf_data": tf_data,
        "epochs": epochs,
        "file_basename": resolved_basename,
        "analysis_settings": analysis_settings,
        "analysis_audit": audit,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze ASSR data from EEGLAB .set files"
    )
    parser.add_argument("file_path", type=str, help="Path to the EEGLAB .set file")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--no_save_results",
        action="store_false",
        dest="save_results",
        help="Do not save results to disk",
    )
    parser.add_argument(
        "--analysis_profile",
        type=str,
        default=None,
        help="Optional ASSR analysis profile, e.g. assr_epochs",
    )
    parser.add_argument(
        "--analysis_config",
        type=str,
        default=None,
        help="Optional JSON object or JSON file path with ASSR analysis overrides",
    )

    args = parser.parse_args()

    analysis_config = None
    if args.analysis_config:
        config_arg = Path(args.analysis_config)
        if config_arg.exists():
            analysis_config = json.loads(config_arg.read_text(encoding="utf8"))
        else:
            analysis_config = json.loads(args.analysis_config)

    analyze_assr(
        args.file_path,
        args.output_dir,
        args.save_results,
        analysis_profile=args.analysis_profile,
        analysis_config=analysis_config,
    )
