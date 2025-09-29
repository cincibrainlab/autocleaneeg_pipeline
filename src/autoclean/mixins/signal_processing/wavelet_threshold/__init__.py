"""Wavelet thresholding mixin for AutoClean tasks."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import pywt

from .processing import generate_wavelet_report, wavelet_threshold
from .toolkit import apply_asr, run_autoreject, run_autoreject_raw, run_star_cleaning, run_zapline

with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as _manifest_handle:
    PLUGIN_MANIFEST = json.load(_manifest_handle)
from autoclean.utils.logging import message


class WaveletThresholdMixin:
    """Mixin providing wavelet thresholding via the shared preprocessing helper."""

    def apply_wavelet_threshold(
        self,
        data: Optional[mne.io.BaseRaw] = None,
        stage_name: str = "post_wavelet_threshold",
    ) -> Optional[mne.io.BaseRaw]:
        """Apply wavelet thresholding if enabled in configuration.

        Parameters
        ----------
        data
            Optional raw object to denoise. If not provided, uses ``self.raw``.
        stage_name
            Stage identifier used when exporting the wavelet-cleaned data.

        Returns
        -------
        Optional[mne.io.BaseRaw]
            The cleaned raw instance when wavelet thresholding ran, otherwise the
            original object.
        """

        inst = data if data is not None else getattr(self, "raw", None)
        if inst is None:
            message("warning", "Wavelet thresholding skipped: no raw data available")
            return inst

        is_enabled, settings = self._check_step_enabled("wavelet_threshold")
        if not is_enabled:
            message("info", "Wavelet thresholding disabled in configuration")
            return inst

        params = (settings or {}).get("value", {})
        wavelet_name = params.get("wavelet", "sym4")
        level_cfg = params.get("level", 5)
        if isinstance(level_cfg, str):
            if level_cfg.lower() != "auto":
                raise ValueError(
                    "wavelet_threshold level must be a non-negative integer or 'auto'"
                )
            level: Union[int, str] = "auto"
        else:
            level = int(level_cfg)
            if level < 0:
                raise ValueError("wavelet_threshold level must be non-negative")
        threshold_mode = params.get("threshold_mode", "soft")
        is_erp = bool(params.get("is_erp", False))
        bandpass_cfg = params.get("bandpass", (1.0, 30.0))
        filter_kwargs_cfg = params.get("filter_kwargs")
        threshold_scale_cfg = params.get("threshold_scale", 1.0)
        picks_cfg = params.get("picks")
        psd_fmax_cfg = params.get("psd_fmax")

        try:
            threshold_scale = float(threshold_scale_cfg)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "wavelet_threshold threshold_scale must be a numeric value"
            ) from exc
        if threshold_scale <= 0:
            raise ValueError("wavelet_threshold threshold_scale must be positive")

        psd_fmax_value: Optional[float]
        if psd_fmax_cfg is None:
            psd_fmax_value = None
        else:
            try:
                psd_fmax_value = float(psd_fmax_cfg)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "wavelet_threshold psd_fmax must be a numeric value when provided"
                ) from exc
            if psd_fmax_value <= 0:
                message(
                    "warning",
                    "wavelet_threshold psd_fmax must be positive; ignoring provided value",
                )
                psd_fmax_value = None

        if bandpass_cfg is None:
            bandpass_tuple: Optional[Tuple[float, float]] = None
        else:
            try:
                low, high = bandpass_cfg
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "wavelet_threshold bandpass must be a two-element sequence"
                ) from exc
            bandpass_tuple = (float(low), float(high))
            if bandpass_tuple[0] >= bandpass_tuple[1]:
                raise ValueError(
                    "wavelet_threshold bandpass low frequency must be less than high frequency"
                )

        if filter_kwargs_cfg is None:
            filter_kwargs = None
        elif isinstance(filter_kwargs_cfg, Mapping):
            filter_kwargs = dict(filter_kwargs_cfg)
        else:
            raise TypeError("wavelet_threshold filter_kwargs must be a mapping if provided")

        baseline = inst.get_data()
        message("header", "Applying wavelet thresholding...")
        cleaned = wavelet_threshold(
            inst,
            wavelet=wavelet_name,
            level=level,
            threshold_mode=threshold_mode,
            is_erp=is_erp,
            bandpass=bandpass_tuple,
            filter_kwargs=filter_kwargs,
            threshold_scale=threshold_scale,
            picks=picks_cfg,
        )

        cleaned_data = cleaned.get_data()
        diff = baseline - cleaned_data
        mean_abs_diff_uv = float(np.mean(np.abs(diff)) * 1e6)
        baseline_ptp = np.ptp(baseline, axis=1)
        cleaned_ptp = np.ptp(cleaned_data, axis=1)
        reduction_pct = np.zeros_like(baseline_ptp)
        np.divide(
            baseline_ptp - cleaned_ptp,
            baseline_ptp,
            out=reduction_pct,
            where=baseline_ptp != 0,
        )
        reduction_pct *= 100.0
        ptp_mean_pct = float(np.mean(reduction_pct))

        try:
            wavelet_obj = pywt.Wavelet(wavelet_name)
            max_level = pywt.dwt_max_level(baseline.shape[1], wavelet_obj.dec_len)
            if isinstance(level, str):
                requested_level = max_level
            else:
                requested_level = level
            effective_level = int(max(0, min(requested_level, max_level)))
        except Exception:
            effective_level = 0 if isinstance(level, str) else level

        report_path: Optional[Path] = None
        report_relative: Optional[Path] = None
        try:
            base_token: Optional[str] = None
            if hasattr(self, "config"):
                bids_path = self.config.get("bids_path")
                if bids_path is not None and hasattr(bids_path, "basename"):
                    base_token = bids_path.basename
            if base_token:
                base_name = base_token.replace("_eeg", "")
            else:
                raw_source = "wavelet"
                if hasattr(self, "config"):
                    raw_source = str(self.config.get("unprocessed_file", raw_source))
                base_name = Path(raw_source).stem
            filename = f"{base_name}_wavelet_threshold.pdf"

            if hasattr(self, "_resolve_report_path"):
                report_path = self._resolve_report_path("wavelet_threshold", filename)
            else:
                derivatives_dir: Optional[Path] = None
                if hasattr(self, "config"):
                    dir_value = self.config.get("derivatives_dir")
                    if dir_value:
                        derivatives_dir = Path(dir_value)
                if derivatives_dir is None:
                    raise ValueError("No derivatives directory available for wavelet report")
                derivatives_dir.mkdir(parents=True, exist_ok=True)
                report_path = derivatives_dir / filename

            generate_wavelet_report(
                source=inst.copy(),
                output_pdf=report_path,
                wavelet=wavelet_name,
                level=level,
                threshold_mode=threshold_mode,
                is_erp=is_erp,
                bandpass=bandpass_tuple,
                filter_kwargs=filter_kwargs,
                psd_fmax=psd_fmax_value,
                threshold_scale=threshold_scale,
                picks=picks_cfg,
            )

            if hasattr(self, "_report_relative_path"):
                try:
                    report_relative = self._report_relative_path(report_path)
                except Exception:
                    report_relative = None
        except Exception as exc:
            message("warning", f"Wavelet report generation skipped: {exc}")
            report_path = None
            report_relative = None

        original_data = inst
        self._update_instance_data(original_data, cleaned)
        self._save_raw_result(cleaned, stage_name)

        metadata = {
            "wavelet": wavelet_name,
            "level_requested": level if isinstance(level, str) else int(level),
            "level_effective": effective_level,
            "threshold_mode": threshold_mode,
            "erp_mode": is_erp,
            "bandpass_low": bandpass_tuple[0] if bandpass_tuple else None,
            "bandpass_high": bandpass_tuple[1] if bandpass_tuple else None,
            "mean_abs_diff_uv": mean_abs_diff_uv,
            "mean_ptp_reduction_pct": ptp_mean_pct,
            "n_channels": int(cleaned_data.shape[0]),
            "report_path": str(report_relative or report_path) if report_path else None,
        }
        if psd_fmax_value is not None:
            metadata["psd_fmax"] = psd_fmax_value
        metadata["threshold_scale"] = threshold_scale
        if picks_cfg is not None:
            if isinstance(picks_cfg, str):
                metadata["picks"] = picks_cfg
            elif isinstance(picks_cfg, Sequence):
                metadata["picks"] = list(picks_cfg)
            else:
                metadata["picks"] = picks_cfg
        if filter_kwargs:
            metadata["filter_kwargs"] = filter_kwargs

        self._update_metadata("step_wavelet_threshold", metadata)
        message("success", "Wavelet thresholding complete")
        return cleaned


__all__ = [
    "WaveletThresholdMixin",
    "wavelet_threshold",
    "generate_wavelet_report",
    "PLUGIN_MANIFEST",
    "run_autoreject",
    "run_autoreject_raw",
    "run_star_cleaning",
    "apply_asr",
    "run_zapline",
]
