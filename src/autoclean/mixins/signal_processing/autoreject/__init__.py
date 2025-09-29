"""AutoReject plugin exposing mixin helpers and manifest metadata."""

from __future__ import annotations

import json
from importlib import resources
from typing import Optional

import mne

from autoclean.utils.logging import message

from .processing import AutorejectResult, RawBuilder, run_autoreject, run_autoreject_raw

with resources.files(__package__).joinpath("manifest.json").open("r", encoding="utf-8") as _manifest_handle:
    PLUGIN_MANIFEST = json.load(_manifest_handle)


class AutorejectMixin:
    """Provide convenience wrappers for running AutoReject inside tasks."""

    def apply_autoreject(
        self,
        data: Optional[mne.BaseEpochs | mne.io.BaseRaw] = None,
        *,
        stage_name: str = "post_autoreject",
        epoch_builder: Optional[RawBuilder] = None,
        **autoreject_kwargs,
    ) -> AutorejectResult:
        """Run AutoReject on the provided data or fall back to ``self`` attributes."""

        inst = data
        if inst is None:
            inst = getattr(self, "epochs", None)
        if inst is None:
            inst = getattr(self, "raw", None)

        if inst is None:
            message("warning", "AutoReject skipped: no Raw or Epochs data available")
            raise ValueError("AutoReject requires Raw or Epochs data")

        message("header", "Running AutoReject clean-up")

        if isinstance(inst, mne.BaseEpochs):
            result = run_autoreject(inst, **autoreject_kwargs)
        else:
            result = run_autoreject_raw(
                inst, epoch_builder=epoch_builder, **autoreject_kwargs
            )

        bad_epochs = getattr(result.reject_log, "bad_epochs", [])
        metadata = {
            "n_channels": int(result.epochs.info.get("nchan", 0)),
            "n_epochs": len(result.epochs),
            "reject_log_len": len(bad_epochs),
        }

        if hasattr(self, "_update_metadata"):
            try:
                self._update_metadata("step_autoreject", metadata)
            except Exception:  # pragma: no cover - defensive
                message("debug", "AutoReject metadata update skipped")

        if hasattr(self, "_save_epochs_result"):
            try:
                self._save_epochs_result(result.epochs, stage_name)
            except Exception:  # pragma: no cover - defensive
                message("debug", "AutoReject epochs export skipped")

        message("success", "AutoReject complete")
        return result


__all__ = [
    "AutorejectMixin",
    "AutorejectResult",
    "PLUGIN_MANIFEST",
    "RawBuilder",
    "run_autoreject",
    "run_autoreject_raw",
]
