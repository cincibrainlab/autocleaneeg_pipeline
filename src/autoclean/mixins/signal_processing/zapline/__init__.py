"""Zapline plugin manifest and mixin wrapper."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import mne

from autoclean.utils.logging import message

from .processing import run_zapline

with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as _manifest_handle:
    PLUGIN_MANIFEST = json.load(_manifest_handle)


class ZaplineMixin:
    """Provide Zapline line-noise removal as a reusable block."""

    def apply_zapline(
        self,
        data: Optional[mne.io.BaseRaw] = None,
        *,
        stage_name: str = "post_zapline",
        line_freq: float = 60.0,
        nkeep: int = 1,
    ) -> mne.io.BaseRaw:
        """Apply Zapline on provided Raw data or fall back to ``self.raw``."""

        inst = data if data is not None else getattr(self, "raw", None)
        if inst is None:
            message("warning", "Zapline skipped: no Raw data available")
            raise ValueError("Zapline requires an MNE Raw object")

        message("header", f"Removing {line_freq} Hz line noise with Zapline")
        cleaned = run_zapline(inst, line_freq=line_freq, nkeep=nkeep)

        if hasattr(self, "_update_instance_data"):
            try:
                self._update_instance_data(inst, cleaned)
            except Exception:  # pragma: no cover - defensive
                message("debug", "Zapline instance update skipped")

        if hasattr(self, "_update_metadata"):
            try:
                self._update_metadata(
                    "step_zapline",
                    {
                        "line_freq": line_freq,
                        "nkeep": nkeep,
                        "n_channels": int(cleaned.info.get("nchan", 0)),
                    },
                )
            except Exception:  # pragma: no cover - defensive
                message("debug", "Zapline metadata update skipped")

        if hasattr(self, "_save_raw_result"):
            try:
                self._save_raw_result(cleaned, stage_name)
            except Exception:  # pragma: no cover - defensive
                message("debug", "Zapline export skipped")

        message("success", "Zapline cleaning complete")
        return cleaned


__all__ = ["ZaplineMixin", "PLUGIN_MANIFEST", "run_zapline"]
