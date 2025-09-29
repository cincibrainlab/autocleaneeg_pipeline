"""STAR cleaning plugin entry point."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import mne

from autoclean.utils.logging import message

from .processing import run_star_cleaning

with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as _manifest_handle:
    PLUGIN_MANIFEST = json.load(_manifest_handle)


class StarCleaningMixin:
    """Mixin wiring in the STAR spatial filtering block."""

    def apply_star_cleaning(
        self,
        data: Optional[mne.io.BaseRaw] = None,
        *,
        stage_name: str = "post_star_cleaning",
        lmbda: float = 2.0,
    ) -> mne.io.BaseRaw:
        """Run STAR cleaning on provided Raw data or ``self.raw``."""

        inst = data if data is not None else getattr(self, "raw", None)
        if inst is None:
            message("warning", "STAR cleaning skipped: no Raw data available")
            raise ValueError("STAR cleaning requires an MNE Raw object")

        message("header", "Applying STAR spatial cleaning")
        cleaned = run_star_cleaning(inst, lmbda=lmbda)

        if hasattr(self, "_update_instance_data"):
            try:
                self._update_instance_data(inst, cleaned)
            except Exception:  # pragma: no cover - defensive
                message("debug", "STAR instance update skipped")

        if hasattr(self, "_update_metadata"):
            try:
                self._update_metadata(
                    "step_star_cleaning",
                    {"lambda": lmbda, "n_channels": int(cleaned.info.get("nchan", 0))},
                )
            except Exception:  # pragma: no cover - defensive
                message("debug", "STAR metadata update skipped")

        if hasattr(self, "_save_raw_result"):
            try:
                self._save_raw_result(cleaned, stage_name)
            except Exception:  # pragma: no cover - defensive
                message("debug", "STAR export skipped")

        message("success", "STAR cleaning complete")
        return cleaned


__all__ = ["StarCleaningMixin", "PLUGIN_MANIFEST", "run_star_cleaning"]
