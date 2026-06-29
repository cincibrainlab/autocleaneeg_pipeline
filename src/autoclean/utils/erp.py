"""Helpers for pre-epoched ERP task templates."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import mne

from autoclean.utils.logging import message


def validate_erp_input(
    epochs: mne.BaseEpochs,
    *,
    required_conditions: list[str] | None = None,
    analysis_window: tuple[float, float] | list[float] | None = None,
) -> dict[str, Any]:
    """Validate condition labels and analysis-window coverage for ERP epochs."""

    event_id = dict(getattr(epochs, "event_id", {}) or {})
    required_conditions = list(required_conditions or [])
    missing = [name for name in required_conditions if name not in event_id]
    if missing:
        available = ", ".join(sorted(event_id)) or "none"
        raise ValueError(
            "Pre-epoched ERP input is missing required condition(s): "
            f"{', '.join(missing)}. Available conditions: {available}"
        )

    warnings: list[str] = []
    analysis_bounds = _analysis_window_bounds(analysis_window)
    if analysis_bounds is not None:
        start, end = analysis_bounds
        if start < float(epochs.tmin) or end > float(epochs.tmax):
            warning = (
                "ERP analysis window "
                f"[{start:g}, {end:g}]s is outside epoch coverage "
                f"[{float(epochs.tmin):g}, {float(epochs.tmax):g}]s."
            )
            warnings.append(warning)
            message("warning", warning)

    counts = _condition_counts(epochs)
    return {
        "n_epochs": int(len(epochs)),
        "event_id": event_id,
        "condition_counts": counts,
        "analysis_window": (
            list(analysis_bounds) if analysis_bounds is not None else None
        ),
        "warnings": warnings,
    }


def generate_erp_outputs(
    epochs: mne.BaseEpochs,
    output_dir: str | Path,
    *,
    conditions: list[str] | None = None,
    difference_waves: list[dict[str, str]] | None = None,
    analysis_window: tuple[float, float] | list[float] | None = None,
    save_evokeds: bool = True,
    save_amplitudes: bool = True,
) -> dict[str, Any]:
    """Write compact ERP summaries for pre-epoched data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    event_id = dict(getattr(epochs, "event_id", {}) or {})
    selected_conditions = list(conditions or sorted(event_id))
    missing = [
        condition for condition in selected_conditions if condition not in event_id
    ]
    if missing:
        available = ", ".join(sorted(event_id)) or "none"
        raise ValueError(
            f"Cannot generate ERP outputs for missing condition(s): {', '.join(missing)}. "
            f"Available conditions: {available}"
        )

    analysis_bounds = _analysis_window_bounds(analysis_window)
    counts = _condition_counts(epochs)
    counts_path = output_path / "erp_condition_counts.csv"
    with counts_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["condition", "event_code", "n_epochs"]
        )
        writer.writeheader()
        for condition in selected_conditions:
            writer.writerow(
                {
                    "condition": condition,
                    "event_code": event_id[condition],
                    "n_epochs": counts.get(condition, 0),
                }
            )

    evoked_paths: dict[str, str] = {}
    evokeds: dict[str, mne.Evoked] = {}
    for condition in selected_conditions:
        evoked = epochs[condition].average()
        evokeds[condition] = evoked
        if save_evokeds:
            evoked_path = output_path / f"erp_average_{_safe_name(condition)}-ave.fif"
            mne.write_evokeds(evoked_path, evoked, overwrite=True, verbose=False)
            evoked_paths[condition] = str(evoked_path)

    difference_paths: dict[str, str] = {}
    difference_evokeds: dict[str, mne.Evoked] = {}
    for spec in difference_waves or []:
        name = (
            spec.get("name")
            or f"{spec.get('positive', '')}_minus_{spec.get('negative', '')}"
        )
        positive = spec.get("positive")
        negative = spec.get("negative")
        if positive not in evokeds or negative not in evokeds:
            raise ValueError(
                "Difference wave requires generated conditions "
                f"positive={positive!r}, negative={negative!r}"
            )
        diff = mne.combine_evoked([evokeds[positive], evokeds[negative]], [1, -1])
        diff.comment = name
        diff_path = output_path / f"erp_difference_{_safe_name(name)}-ave.fif"
        mne.write_evokeds(diff_path, diff, overwrite=True, verbose=False)
        difference_paths[name] = str(diff_path)
        difference_evokeds[name] = diff

    amplitude_path = None
    if save_amplitudes:
        amplitude_path = output_path / "erp_amplitude_summary.csv"
        _write_amplitude_summary(
            amplitude_path,
            {**evokeds, **difference_evokeds},
            analysis_window=analysis_bounds,
        )

    return {
        "counts_file": str(counts_path),
        "evoked_files": evoked_paths,
        "difference_files": difference_paths,
        "amplitude_summary_file": str(amplitude_path) if amplitude_path else None,
    }


def _condition_counts(epochs: mne.BaseEpochs) -> dict[str, int]:
    counts: dict[str, int] = {}
    for condition in sorted((getattr(epochs, "event_id", {}) or {}).keys()):
        try:
            counts[condition] = int(len(epochs[condition]))
        except Exception as exc:
            message(
                "warning", f"Could not count epochs for condition {condition!r}: {exc}"
            )
            counts[condition] = 0
    return counts


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
    return safe.strip("_") or "condition"


def _analysis_window_bounds(
    analysis_window: tuple[float, float] | list[float] | None,
) -> tuple[float, float] | None:
    if analysis_window is None:
        return None
    if len(analysis_window) < 2:
        raise ValueError("ERP analysis_window must contain start and end times.")
    return float(analysis_window[0]), float(analysis_window[1])


def _write_amplitude_summary(
    path: Path,
    evokeds: dict[str, mne.Evoked],
    *,
    analysis_window: tuple[float, float] | list[float] | None,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "window_start",
                "window_end",
                "mean_amplitude",
                "peak_amplitude",
                "peak_latency",
            ],
        )
        writer.writeheader()
        for condition, evoked in evokeds.items():
            cropped = evoked.copy()
            if analysis_window is not None:
                start = max(float(analysis_window[0]), float(cropped.times[0]))
                end = min(float(analysis_window[1]), float(cropped.times[-1]))
                if start <= end:
                    cropped.crop(start, end)
            data = cropped.data
            channel_mean = data.mean(axis=0)
            peak_idx = int(abs(channel_mean).argmax())
            writer.writerow(
                {
                    "condition": condition,
                    "window_start": float(cropped.times[0]),
                    "window_end": float(cropped.times[-1]),
                    "mean_amplitude": float(data.mean()),
                    "peak_amplitude": float(channel_mean[peak_idx]),
                    "peak_latency": float(cropped.times[peak_idx]),
                }
            )
