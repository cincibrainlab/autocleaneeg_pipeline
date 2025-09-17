"""Wavelet thresholding mixin for AutoClean tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple

import mne
import numpy as np
import pywt

from autoclean.functions.preprocessing.wavelet_thresholding import (
    generate_wavelet_report,
    wavelet_threshold,
)
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
        level = int(params.get("level", 5))
        if level < 0:
            raise ValueError("wavelet_threshold level must be non-negative")
        threshold_mode = params.get("threshold_mode", "soft")
        is_erp = bool(params.get("is_erp", False))
        bandpass_cfg = params.get("bandpass", (1.0, 30.0))
        filter_kwargs_cfg = params.get("filter_kwargs")

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
            effective_level = int(max(0, min(level, max_level)))
        except Exception:
            effective_level = level

        report_path: Optional[Path] = None
        try:
            derivatives_dir: Optional[Path] = None
            if hasattr(self, "config"):
                dir_value = self.config.get("derivatives_dir")
                if dir_value:
                    derivatives_dir = Path(dir_value)
            if derivatives_dir is not None:
                derivatives_dir.mkdir(parents=True, exist_ok=True)
                bids_path = self.config.get("bids_path") if hasattr(self, "config") else None
                if bids_path and hasattr(bids_path, "basename"):
                    base_name = bids_path.basename.replace("_eeg", "_wavelet_threshold")
                else:
                    raw_source = "wavelet"
                    if hasattr(self, "config"):
                        raw_source = str(self.config.get("unprocessed_file", raw_source))
                    source_name = Path(raw_source)
                    base_name = f"{source_name.stem}_wavelet_threshold"
                report_path = (derivatives_dir / base_name).with_suffix(".pdf")
                generate_wavelet_report(
                    source=inst.copy(),
                    output_pdf=report_path,
                    wavelet=wavelet_name,
                    level=level,
                    threshold_mode=threshold_mode,
                    is_erp=is_erp,
                    bandpass=bandpass_tuple,
                    filter_kwargs=filter_kwargs,
                )
        except Exception as exc:
            message("warning", f"Wavelet report generation skipped: {exc}")
            report_path = None

        original_data = inst
        self._update_instance_data(original_data, cleaned)
        self._save_raw_result(cleaned, stage_name)

        metadata = {
            "wavelet": wavelet_name,
            "level_requested": level,
            "level_effective": effective_level,
            "threshold_mode": threshold_mode,
            "erp_mode": is_erp,
            "bandpass_low": bandpass_tuple[0] if bandpass_tuple else None,
            "bandpass_high": bandpass_tuple[1] if bandpass_tuple else None,
            "mean_abs_diff_uv": mean_abs_diff_uv,
            "mean_ptp_reduction_pct": ptp_mean_pct,
            "n_channels": int(cleaned_data.shape[0]),
            "report_path": str(report_path) if report_path else None,
        }
        if filter_kwargs:
            metadata["filter_kwargs"] = filter_kwargs

        self._update_metadata("step_wavelet_threshold", metadata)
        message("success", "Wavelet thresholding complete")
        return cleaned
