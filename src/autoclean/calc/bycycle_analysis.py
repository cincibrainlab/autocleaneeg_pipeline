"""Cycle-by-cycle EEG analysis helpers built on the bycycle package."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "amp_fraction": 0.3,
    "amp_consistency": 0.4,
    "period_consistency": 0.5,
    "min_n_cycles": 3,
}
DEFAULT_NARROWBAND_KWARGS: dict[str, float | None] = {
    "n_seconds": 0.5,
    "n_cycles": None,
}
_RESULT_COLUMNS = [
    "channel",
    "freq_range",
    "f_range_low",
    "f_range_high",
    "sfreq",
    "source_type",
]


def classify_frequency_band(f_range: Sequence[float]) -> str:
    """Return a stable band label for a frequency range."""
    fmin, fmax = _normalize_f_range(f_range)
    if fmin >= 8.0 and fmax <= 12.5:
        return "alpha"
    if fmin >= 30.0 and fmax <= 80.0:
        return "gamma"
    return "custom"


def build_bycycle_output_filename(subject_id: str, f_range: Sequence[float]) -> str:
    """Build the parquet filename requested in issue #119."""
    band = classify_frequency_band(f_range)
    prefix = {"alpha": "AlphaFilt", "gamma": "GammaFilt"}.get(band, "CustomFilt")
    return f"bycycle_results_{prefix}_{subject_id}.parquet"


def resolve_bycycle_thresholds(
    f_range: Sequence[float],
    thresholds: Mapping[str, float | int] | None = None,
) -> dict[str, float | int]:
    """Merge explicit thresholds with band-aware defaults."""
    band = classify_frequency_band(f_range)
    merged: dict[str, float | int] = dict(DEFAULT_THRESHOLDS)
    merged["monotonicity"] = 0.65 if band == "gamma" else 0.55
    if thresholds:
        merged.update(dict(thresholds))
    return merged


def save_bycycle_results(
    results_df: pd.DataFrame,
    output_dir: str | Path,
    subject_id: str,
    f_range: Sequence[float],
    compression: str = "snappy",
) -> Path:
    """Save cycle-by-cycle features to the standard parquet filename."""
    output_path = Path(output_dir) / build_bycycle_output_filename(subject_id, f_range)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, compression=compression, index=False)
    return output_path


def compute_bycycle_analysis(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    f_range: Sequence[float] = (8.0, 12.5),
    thresholds: Mapping[str, float | int] | None = None,
    picks: str | Sequence[str] | Sequence[int] | None = None,
    metadata: Mapping[str, Any] | None = None,
    limit_duration_s: float | None = None,
    center_extrema: str = "trough",
    burst_method: str = "cycles",
    narrowband_kwargs: Mapping[str, float | None] | None = None,
) -> pd.DataFrame:
    """Compute bycycle features for Raw or Epochs data and return a DataFrame."""
    if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError("data must be an MNE Raw or Epochs object")

    fmin, fmax = _normalize_f_range(f_range)
    resolved_thresholds = resolve_bycycle_thresholds((fmin, fmax), thresholds)
    channel_indices = _resolve_pick_indices(data, picks)
    band_label = classify_frequency_band((fmin, fmax))
    source_type = "epochs" if isinstance(data, mne.BaseEpochs) else "raw"
    sfreq = float(data.info["sfreq"])
    bycycle_module = _load_bycycle_module()
    bycycle_cls = getattr(bycycle_module, "Bycycle")
    features: list[pd.DataFrame] = []

    for channel_index in channel_indices:
        channel_name = data.ch_names[channel_index]
        signal = _extract_channel_signal(data, channel_index, limit_duration_s)
        if signal.size == 0:
            continue

        model = bycycle_cls(
            center_extrema=center_extrema,
            burst_method=burst_method,
            thresholds=resolved_thresholds,
            find_extrema_kwargs={
                "filter_kwargs": dict(
                    narrowband_kwargs or DEFAULT_NARROWBAND_KWARGS,
                )
            },
        )
        model.fit(signal, sfreq, (fmin, fmax))
        df_features = getattr(model, "df_features", None)
        if df_features is None or len(df_features) == 0:
            continue

        channel_df = df_features.copy()
        channel_df["channel"] = channel_name
        channel_df["freq_range"] = band_label
        channel_df["f_range_low"] = fmin
        channel_df["f_range_high"] = fmax
        channel_df["sfreq"] = sfreq
        channel_df["source_type"] = source_type
        if metadata:
            for key, value in metadata.items():
                channel_df[key] = value
        features.append(channel_df)

    if not features:
        empty_columns = list(_RESULT_COLUMNS)
        if metadata:
            empty_columns.extend(metadata.keys())
        return pd.DataFrame(columns=empty_columns)

    return pd.concat(features, ignore_index=True)


def _normalize_f_range(f_range: Sequence[float]) -> tuple[float, float]:
    if len(f_range) != 2:
        raise ValueError("f_range must contain exactly two values")
    fmin = float(f_range[0])
    fmax = float(f_range[1])
    if fmin <= 0 or fmax <= 0 or fmin >= fmax:
        raise ValueError("f_range must be a positive (fmin, fmax) pair")
    return fmin, fmax


def _resolve_pick_indices(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    picks: str | Sequence[str] | Sequence[int] | None,
) -> list[int]:
    if picks is None:
        return list(range(len(data.ch_names)))
    if isinstance(picks, str):
        return list(mne.pick_channels(data.ch_names, include=[picks], ordered=True))

    picks_list = list(picks)
    if not picks_list:
        return []
    if all(isinstance(pick, int) for pick in picks_list):
        return [int(pick) for pick in picks_list]
    if all(isinstance(pick, str) for pick in picks_list):
        return list(mne.pick_channels(data.ch_names, include=picks_list, ordered=True))
    raise TypeError("picks must be None, a channel name, or a sequence of names/indices")


def _extract_channel_signal(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    channel_index: int,
    limit_duration_s: float | None,
) -> np.ndarray:
    sfreq = float(data.info["sfreq"])
    if isinstance(data, mne.BaseEpochs):
        epoch_data = data.get_data(picks=[channel_index])[:, 0, :]
        signal = epoch_data.reshape(-1)
    else:
        stop = None
        if limit_duration_s is not None:
            stop = int(limit_duration_s * sfreq)
        signal = data.get_data(picks=[channel_index], stop=stop)[0]
    if limit_duration_s is not None and isinstance(data, mne.BaseEpochs):
        max_samples = int(limit_duration_s * sfreq)
        signal = signal[:max_samples]
    return np.asarray(signal, dtype=float)


def _load_bycycle_module() -> ModuleType:
    try:
        return importlib.import_module("bycycle")
    except ImportError as exc:
        raise ImportError(
            "bycycle is required for compute_bycycle_analysis; install bycycle>=1.1.0"
        ) from exc
