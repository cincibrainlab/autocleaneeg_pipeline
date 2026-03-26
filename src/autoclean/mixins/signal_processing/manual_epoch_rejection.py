"""Manual epoch rejection mixin."""

from __future__ import annotations

from typing import Optional

import mne

from autoclean.utils.logging import message


class ManualEpochRejectionMixin:
    """Apply manual bad-epoch overrides to an epochs object."""

    def drop_manual_bad_epochs(
        self,
        data: Optional[mne.Epochs] = None,
        manual_bad_epoch_indices: Optional[list[int]] = None,
        manual_bad_epoch_times: Optional[list[str]] = None,
        manual_bad_epoch_events: Optional[list[str]] = None,
        export: bool = False,
        stage_name: str = "post_manual_epoch_exclusions",
    ) -> mne.Epochs:
        """Drop user-selected bad epochs, mapping stored epoch numbers via selection."""
        epochs = self._get_data_object(data, use_epochs=True)
        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError("Data must be an MNE Epochs object for manual epoch rejection")

        requested_indices = sorted(
            {int(idx) for idx in (manual_bad_epoch_indices or []) if int(idx) >= 0}
        )
        requested_times = [str(v) for v in (manual_bad_epoch_times or []) if str(v)]
        requested_events = [str(v) for v in (manual_bad_epoch_events or []) if str(v)]

        if not requested_indices:
            return epochs

        selection = (
            epochs.selection.tolist()
            if hasattr(epochs.selection, "tolist")
            else list(epochs.selection)
        )
        bad_selection_indices = [
            selection.index(bad_num) for bad_num in requested_indices if bad_num in selection
        ]
        applied_bad_epoch_indices = [selection[idx] for idx in bad_selection_indices]
        skipped_bad_epoch_indices = [
            bad_num for bad_num in requested_indices if bad_num not in selection
        ]

        epochs_clean = epochs.copy()
        if bad_selection_indices:
            message(
                "info",
                f"Applying manual epoch exclusions: {applied_bad_epoch_indices}",
            )
            epochs_clean.drop(bad_selection_indices, reason="USER", verbose=False)

        self._update_metadata(
            "step_manual_epoch_exclusion",
            {
                "requested_bad_epoch_indices": requested_indices,
                "requested_bad_epoch_times": requested_times,
                "requested_bad_epoch_events": requested_events,
                "requested_bad_epoch_count": len(requested_indices),
                "applied_bad_epoch_indices": applied_bad_epoch_indices,
                "applied_bad_epoch_count": len(applied_bad_epoch_indices),
                "skipped_bad_epoch_indices": skipped_bad_epoch_indices,
            },
        )
        self._update_instance_data(data, epochs_clean, use_epochs=True)
        self._auto_export_if_enabled(epochs_clean, stage_name, export)
        return epochs_clean
