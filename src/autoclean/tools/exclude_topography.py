"""Time-to-sample mapping helpers for Exclude topography review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EpochSample:
    """An epoched-data location selected in the browser."""

    epoch_index: int
    sample_index: int


def raw_sample_from_time(time_seconds: float, *, sfreq: float, n_samples: int) -> int:
    """Map a raw-browser time to a valid sample index."""
    if n_samples <= 0:
        raise ValueError("Raw data has no samples")
    if sfreq <= 0:
        raise ValueError("Sampling frequency must be positive")
    index = int(round(time_seconds * sfreq))
    return max(0, min(n_samples - 1, index))


def epoch_sample_from_time(
    time_seconds: float, *, epoch_times: Sequence[float], n_epochs: int
) -> EpochSample:
    """Map an epoched-browser time to its epoch and within-epoch sample.

    The browser display shows epochs consecutively, with each epoch retaining
    its ``epoch_times`` offset so negative-latency clicks map to real samples.
    """
    if n_epochs <= 0:
        raise ValueError("Epoch data has no epochs")
    if len(epoch_times) == 0:
        raise ValueError("Epoch data has no samples")
    if len(epoch_times) == 1:
        return EpochSample(epoch_index=0, sample_index=0)

    sample_interval = epoch_times[1] - epoch_times[0]
    if sample_interval <= 0:
        raise ValueError("Epoch times must be increasing")
    epoch_duration = epoch_times[-1] - epoch_times[0] + sample_interval
    first_epoch_start = epoch_times[0]
    epoch_index = int((time_seconds - first_epoch_start) // epoch_duration)
    epoch_index = max(0, min(n_epochs - 1, epoch_index))
    within_epoch = time_seconds - (first_epoch_start + epoch_index * epoch_duration)
    sample_index = int(round(within_epoch / sample_interval))
    sample_index = max(0, min(len(epoch_times) - 1, sample_index))
    return EpochSample(epoch_index=epoch_index, sample_index=sample_index)


def should_open_topography_from_click(
    *,
    is_target: bool,
    is_mouse_release: bool,
    is_secondary_button: bool,
    is_control_click: bool,
    widget_width: int,
    widget_height: int,
) -> bool:
    """Return whether a browser mouse event should open a topography dialog."""
    return (
        is_target
        and is_mouse_release
        and (is_secondary_button or is_control_click)
        and widget_width >= 300
        and widget_height >= 150
    )


def review_shortcut_action(
    *, text: str, key: int, up_key: int, down_key: int, text_input_has_focus: bool
) -> str | None:
    """Return the review action for a global shortcut key press."""
    if text_input_has_focus:
        return None
    if key == up_key:
        return "UP"
    if key == down_key:
        return "DOWN"
    return {
        "P": "PASS",
        "F": "FAIL",
        "R": "REVIEW",
        "C": "UNSET",
    }.get(text.upper())
