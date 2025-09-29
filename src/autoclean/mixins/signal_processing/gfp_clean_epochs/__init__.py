"""GFP plugin exposing mixin helpers and manifest metadata."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import mne

from autoclean.utils.logging import message

from .processing import (
    DEFAULT_NON_SCALP_CHANNELS,
    GFPCleaningResult,
    clean_epochs_by_gfp,
    render_gfp_plots,
)

with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as _manifest_handle:
    PLUGIN_MANIFEST = json.load(_manifest_handle)


class GFPCleanEpochsMixin:
    """Expose GFP epoch cleaning as a modular plugin block."""

    def gfp_clean_epochs(
        self,
        epochs: Optional[mne.BaseEpochs] = None,
        *,
        gfp_threshold: float = 3.0,
        number_of_epochs: Optional[int] = None,
        random_seed: Optional[int] = None,
        stage_name: str = "post_gfp_clean",
        export: bool = False,
    ) -> mne.BaseEpochs:
        """Clean epochs using GFP-driven outlier rejection."""

        data = self._get_data_object(epochs, use_epochs=True)
        if not isinstance(data, mne.BaseEpochs):
            raise TypeError("Data must be an MNE Epochs object for GFP cleaning")

        message("header", "Cleaning epochs based on Global Field Power (GFP)")

        try:
            result = clean_epochs_by_gfp(
                data,
                gfp_threshold=gfp_threshold,
                number_of_epochs=number_of_epochs,
                random_seed=random_seed,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            message("error", f"Error during GFP epoch cleaning: {exc}")
            raise RuntimeError(f"Failed to clean epochs using GFP: {exc}") from exc

        annotation_types = _summarize_drop_log(data.drop_log)

        metadata = _build_metadata(
            original=data,
            result=result,
            gfp_threshold=gfp_threshold,
            annotation_types=annotation_types,
        )

        if hasattr(self, "_update_metadata"):
            try:
                self._update_metadata("step_gfp_clean_epochs", metadata)
            except Exception:  # pragma: no cover - defensive
                message("debug", "GFP metadata update skipped")

        if hasattr(self, "_update_instance_data"):
            try:
                self._update_instance_data(data, result.epochs, use_epochs=True)
            except Exception:  # pragma: no cover - defensive
                message("debug", "GFP instance update skipped")

        if hasattr(self, "_save_epochs_result"):
            try:
                self._save_epochs_result(result.epochs, stage_name)
            except Exception:  # pragma: no cover - defensive
                message("debug", "GFP epochs export skipped")

        if hasattr(self, "_auto_export_if_enabled"):
            try:
                self._auto_export_if_enabled(result.epochs, stage_name, export)
            except Exception:  # pragma: no cover - defensive
                message("debug", "GFP auto-export skipped")

        config = getattr(self, "config", None)
        if isinstance(config, dict):
            derivatives_dir = config.get("derivatives_dir")
        elif hasattr(config, "get"):
            derivatives_dir = config.get("derivatives_dir", None)
        else:
            derivatives_dir = None

        if derivatives_dir is not None and hasattr(derivatives_dir, "copy"):
            try:
                render_gfp_plots(derivatives_dir, result.stats, result.cleaned_stats)
            except Exception as exc:  # pragma: no cover - optional plotting
                message("warning", f"Could not create GFP plots: {exc}")

        message("success", "Epoch GFP cleaning process completed")
        return result.epochs


def _summarize_drop_log(drop_log) -> dict:
    """Aggregate annotation counts from an MNE drop log."""

    annotation_types: dict[str, int] = {}
    good_epochs = 0
    for entry in drop_log:
        if len(entry) == 0:
            good_epochs += 1
            continue
        for annotation in entry:
            key = str(annotation)
            annotation_types[key] = annotation_types.get(key, 0) + 1

    annotation_types["KEEP"] = good_epochs
    annotation_types["TOTAL"] = len(drop_log)
    return annotation_types


def _build_metadata(
    *,
    original: mne.BaseEpochs,
    result: GFPCleaningResult,
    gfp_threshold: float,
    annotation_types: dict[str, int],
) -> dict:
    """Compose a metadata payload mirroring the legacy mixin output."""

    cleaned = result.cleaned_stats
    mean_amp = float(cleaned["mean_amplitude"].mean()) if not cleaned.empty else float("nan")
    max_amp = float(cleaned["max_amplitude"].max()) if not cleaned.empty else float("nan")
    min_amp = float(cleaned["min_amplitude"].min()) if not cleaned.empty else float("nan")
    std_amp = float(cleaned["std_amplitude"].mean()) if not cleaned.empty else float("nan")
    mean_gfp = float(cleaned["gfp"].mean()) if not cleaned.empty else float("nan")

    epoch_duration = float(original.times[-1] - original.times[0]) if original.times.size else 0.0
    samples_per_epoch = int(original.times.size)

    metadata = {
        "initial_epochs": len(original),
        "final_epochs": len(result.epochs),
        "removed_by_gfp": int(result.removed_count),
        "mean_amplitude": mean_amp,
        "max_amplitude": max_amp,
        "min_amplitude": min_amp,
        "std_amplitude": std_amp,
        "mean_gfp": mean_gfp,
        "gfp_threshold": float(gfp_threshold),
        "removed_total": int(result.removed_count),
        "annotation_types": annotation_types,
        "epoch_duration": epoch_duration,
        "samples_per_epoch": samples_per_epoch,
        "total_duration_sec": epoch_duration * len(result.epochs),
        "total_samples": samples_per_epoch * len(result.epochs),
        "channel_count": len(original.ch_names),
        "scalp_channels_used": result.scalp_channels,
        "requested_epochs_exceeded": bool(result.requested_epochs_exceeded),
    }

    return metadata


__all__ = [
    "DEFAULT_NON_SCALP_CHANNELS",
    "GFPCleanEpochsMixin",
    "GFPCleaningResult",
    "PLUGIN_MANIFEST",
    "clean_epochs_by_gfp",
    "render_gfp_plots",
]
