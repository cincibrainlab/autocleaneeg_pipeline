"""ASR plugin manifest and mixin helpers."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import mne

from autoclean.utils.logging import message

from .processing import apply_asr

with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as _manifest_handle:
    PLUGIN_MANIFEST = json.load(_manifest_handle)


class ASRMixin:
    """Expose ASR cleaning as a Lego-style mixin block."""

    def apply_asr(
        self,
        data: Optional[mne.io.BaseRaw] = None,
        *,
        stage_name: str = "post_asr",
        method: str = "euclid",
        cutoff: float = 20.0,
        train_duration: int = 20,
    ) -> mne.io.BaseRaw:
        """Apply ASR to Raw data supplied explicitly or stored on ``self``."""

        inst = data if data is not None else getattr(self, "raw", None)
        if inst is None:
            message("warning", "ASR skipped: no Raw data available")
            raise ValueError("ASR requires an MNE Raw object")

        message("header", "Applying ASR artifact reconstruction")
        cleaned = apply_asr(
            inst,
            method=method,
            cutoff=cutoff,
            train_duration=train_duration,
        )

        metadata = {
            "method": method,
            "cutoff": cutoff,
            "train_duration": train_duration,
            "n_channels": int(cleaned.info.get("nchan", 0)),
        }

        if hasattr(self, "_update_instance_data"):
            try:
                self._update_instance_data(inst, cleaned)
            except Exception:  # pragma: no cover - defensive
                message("debug", "ASR instance update skipped")

        if hasattr(self, "_update_metadata"):
            try:
                self._update_metadata("step_asr", metadata)
            except Exception:  # pragma: no cover - defensive
                message("debug", "ASR metadata update skipped")

        if hasattr(self, "_save_raw_result"):
            try:
                self._save_raw_result(cleaned, stage_name)
            except Exception:  # pragma: no cover - defensive
                message("debug", "ASR export skipped")

        message("success", "ASR cleaning complete")
        return cleaned


__all__ = ["ASRMixin", "PLUGIN_MANIFEST", "apply_asr"]
