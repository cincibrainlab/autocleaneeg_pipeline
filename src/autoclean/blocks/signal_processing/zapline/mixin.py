"""Zapline DSS-based line noise removal mixin for AutoClean tasks."""

from __future__ import annotations

from typing import Optional

import mne

from .algorithm import apply_zapline_dss, compute_line_noise_power
from autoclean.utils.logging import message


class ZaplineMixin:
    """Mixin providing Zapline DSS-based line noise removal."""

    def apply_zapline(
        self,
        data: Optional[mne.io.BaseRaw] = None,
        stage_name: str = "post_zapline",
    ) -> Optional[mne.io.BaseRaw]:
        """Apply Zapline DSS-based line noise removal if enabled in configuration.

        Uses Denoising Source Separation (DSS) to remove power line artifacts
        (50 or 60 Hz) and their harmonics from EEG/MEG data.

        Parameters
        ----------
        data : mne.io.BaseRaw, optional
            Raw object to clean. If not provided, uses ``self.raw``.
        stage_name : str, default="post_zapline"
            Stage identifier used when exporting the cleaned data.

        Returns
        -------
        mne.io.BaseRaw or None
            The cleaned raw instance when Zapline ran, otherwise the original object.

        Notes
        -----
        Configuration parameters (in task config):
        - fline : float
            Line noise frequency in Hz (60 for US/Americas, 50 for Europe/Asia)
        - nkeep : int
            Number of noise components to remove (typically 1)
        - use_iter : bool
            Use iterative removal for thorough noise reduction
        - max_iter : int
            Maximum iterations for iterative mode

        Examples
        --------
        In task config:
        >>> config = {
        ...     "apply_zapline": {
        ...         "enabled": True,
        ...         "value": {"fline": 60, "nkeep": 1, "use_iter": False}
        ...     }
        ... }

        References
        ----------
        .. [1] de Cheveigné, A. (2020). ZapLine: A simple and effective method to
           remove power line artifacts. NeuroImage, 207, 116356.
        """
        inst = data if data is not None else getattr(self, "raw", None)
        if inst is None:
            message("warning", "Zapline skipped: no raw data available")
            return inst

        # Check if step is enabled in configuration
        is_enabled, settings = self._check_step_enabled("apply_zapline")
        if not is_enabled:
            message("info", "Zapline disabled in configuration")
            return inst

        # Extract parameters from config
        params = (settings or {}).get("value", {})
        fline = float(params.get("fline", 60.0))
        nkeep = int(params.get("nkeep", 1))
        use_iter = bool(params.get("use_iter", False))
        max_iter = int(params.get("max_iter", 10))

        # Validate parameters
        if fline not in [50.0, 60.0]:
            message(
                "warning",
                f"Unusual line frequency {fline} Hz (typically 50 or 60 Hz)",
            )

        nyquist = inst.info["sfreq"] / 2
        if fline >= nyquist:
            message(
                "error",
                f"Line frequency ({fline} Hz) must be below Nyquist ({nyquist} Hz)",
            )
            return inst

        if nkeep < 1 or nkeep > 5:
            message("warning", f"nkeep={nkeep} is unusual (typically 1-2)")

        n_channels = inst.info["nchan"]
        if n_channels < 2:
            message(
                "error",
                f"Zapline requires at least 2 channels, found {n_channels}",
            )
            return inst

        if n_channels < 32:
            message(
                "warning",
                f"Zapline works best with >32 channels (found {n_channels})",
            )

        # Measure line noise before removal
        try:
            power_before, snr_before = compute_line_noise_power(inst, fline=fline)
            message(
                "info",
                f"Pre-Zapline: {power_before:.1f} dB at {fline} Hz (SNR: {snr_before:.2f})",
            )
        except Exception as exc:
            message("warning", f"Could not compute pre-Zapline power: {exc}")
            power_before = None
            snr_before = None

        # Apply Zapline
        message(
            "header",
            f"Applying Zapline (fline={fline} Hz, nkeep={nkeep}, "
            f"iter={use_iter})...",
        )

        try:
            cleaned, info = apply_zapline_dss(
                inst,
                fline=fline,
                nkeep=nkeep,
                use_iter=use_iter,
                max_iter=max_iter,
            )
        except ImportError as exc:
            message("error", f"Zapline requires meegkit: {exc}")
            return inst
        except Exception as exc:
            message("error", f"Zapline failed: {exc}")
            return inst

        # Measure line noise after removal
        try:
            power_after, snr_after = compute_line_noise_power(cleaned, fline=fline)
            reduction_db = power_before - power_after if power_before else None

            if reduction_db is not None:
                message(
                    "info",
                    f"Post-Zapline: {power_after:.1f} dB at {fline} Hz "
                    f"(reduction: {reduction_db:.1f} dB, SNR: {snr_after:.2f})",
                )

                if reduction_db < 10:
                    message(
                        "warning",
                        f"Line noise reduction is modest ({reduction_db:.1f} dB). "
                        "Consider increasing nkeep or using iterative mode.",
                    )
            else:
                message(
                    "info",
                    f"Post-Zapline: {power_after:.1f} dB at {fline} Hz (SNR: {snr_after:.2f})",
                )
        except Exception as exc:
            message("warning", f"Could not compute post-Zapline power: {exc}")
            power_after = None
            snr_after = None
            reduction_db = None

        # Update instance data and save result
        self._update_instance_data(inst, cleaned)
        self._save_raw_result(cleaned, stage_name)

        # Store metadata
        metadata = {
            "fline": fline,
            "nkeep": nkeep,
            "use_iter": use_iter,
            "max_iter": max_iter,
            "method": info.get("method"),
            "iterations": info.get("iterations"),
            "n_channels": n_channels,
        }

        if power_before is not None:
            metadata["power_before_db"] = float(power_before)
        if power_after is not None:
            metadata["power_after_db"] = float(power_after)
        if reduction_db is not None:
            metadata["reduction_db"] = float(reduction_db)
        if snr_before is not None:
            metadata["snr_before"] = float(snr_before)
        if snr_after is not None:
            metadata["snr_after"] = float(snr_after)

        self._update_metadata("step_zapline", metadata)

        message("success", f"Zapline complete ({info.get('iterations')} iterations)")
        return cleaned
