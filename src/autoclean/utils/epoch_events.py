"""Centralized epoch event extraction shared by EEGLAB export and BIDS mock-raw conversion.

``epochs.events`` / ``epochs.event_id`` is the authoritative source of truth for
condition codes on any :class:`mne.BaseEpochs` object — it always covers every
epoch exactly once. Some pipeline steps additionally attach a richer per-epoch
``additional_events`` metadata column (see
``autoclean.mixins.signal_processing.eventid_epochs``) that can carry more than
one intra-epoch event. That column is only trustworthy when every epoch is
covered and every label maps back onto ``epochs.event_id``; otherwise we fall
back to the primary source rather than inventing new codes.
"""

from typing import Dict, Optional, Tuple

import mne
import numpy as np

from autoclean.utils.logging import message

__all__ = ["extract_epoch_events"]


def _primary_events(epochs: mne.BaseEpochs) -> Tuple[np.ndarray, Dict[str, int], np.ndarray]:
    """Build one event per epoch directly from epochs.events / epochs.event_id."""
    n_epochs = len(epochs)
    event_id = dict(epochs.event_id) if epochs.event_id else {}

    if n_epochs == 0:
        return np.empty((0, 3), dtype=int), event_id, np.empty(0, dtype=int)

    n_samples = len(epochs.times)
    trigger_offset = int(round(-epochs.tmin * epochs.info["sfreq"]))

    # Only valid when the epoch window actually contains t=0 (where the
    # trigger sits). For windows that don't (e.g. tmin > 0 or tmax < 0),
    # clamp into range and warn rather than silently placing the "trigger"
    # inside a neighboring epoch's data block.
    if not (0 <= trigger_offset < n_samples):
        message(
            "warning",
            f"Epoch window [{epochs.tmin}, {epochs.tmax}] does not contain t=0; "
            "condition-code placement within each epoch's block is approximate.",
        )
        trigger_offset = max(0, min(trigger_offset, n_samples - 1))

    events = np.array(
        [
            [i * n_samples + trigger_offset, 0, int(code)]
            for i, code in enumerate(epochs.events[:, 2])
        ],
        dtype=int,
    )
    epoch_indices = np.arange(n_epochs, dtype=int)
    return events, event_id, epoch_indices


def _rebuild_from_metadata(
    epochs: mne.BaseEpochs,
) -> Optional[Tuple[np.ndarray, Dict[str, int], np.ndarray]]:
    """Attempt to rebuild a per-epoch event list from the additional_events metadata column.

    Returns None if the metadata cannot be trusted to preserve the original
    condition codes (missing labels, unparsable timings, etc.) so the caller
    can fall back to the primary epochs.events/event_id source.
    """
    metadata = epochs.metadata
    sfreq = epochs.info["sfreq"]
    n_samples = len(epochs.times)

    if len(metadata) != len(epochs.events):
        message(
            "warning",
            "additional_events metadata length "
            f"({len(metadata)}) does not match epochs.events length "
            f"({len(epochs.events)}); truncating to align.",
        )

    if not epochs.event_id:
        message(
            "warning",
            "epochs.event_id is empty; cannot validate additional_events labels "
            "against original condition codes.",
        )
        return None

    # Collect every label referenced in the metadata and confirm each one maps
    # back to an existing code in epochs.event_id. We never invent new codes.
    all_labels = set()
    for entry in metadata["additional_events"].dropna():
        if not isinstance(entry, list):
            continue
        for event_tuple in entry:
            if isinstance(event_tuple, (list, tuple)) and len(event_tuple) >= 1:
                all_labels.add(str(event_tuple[0]))

    event_id_rebuilt: Dict[str, int] = {}
    for label in all_labels:
        match = next(
            (code for orig_label, code in epochs.event_id.items() if str(orig_label) == label),
            None,
        )
        if match is None:
            message(
                "warning",
                f"Label '{label}' found in additional_events metadata has no matching "
                "code in epochs.event_id; cannot trust metadata reconstruction.",
            )
            return None
        event_id_rebuilt[label] = match

    events = []
    epoch_indices = []
    used_samples = set()
    n_rows = min(len(metadata), len(epochs.events))

    for i, entry in enumerate(metadata["additional_events"].head(n_rows)):
        if not isinstance(entry, list):
            continue
        for label, rel_time in entry:
            try:
                sample_in_epoch = int(round((float(rel_time) - epochs.tmin) * sfreq))
            except (TypeError, ValueError):
                message(
                    "warning",
                    f"Epoch {i}, event '{label}': could not convert rel_time "
                    f"'{rel_time}' to float. Skipping this event.",
                )
                continue

            if not (0 <= sample_in_epoch < n_samples):
                message(
                    "warning",
                    f"Epoch {i}, event '{label}': sample {sample_in_epoch} is outside "
                    f"epoch data window [0, {n_samples - 1}]. Skipping.",
                )
                continue

            # Collisions are nudged forward one sample at a time, but never
            # past this epoch's own block — drifting into the next epoch's
            # samples would silently mismatch the epoch_indices mapping.
            epoch_start = i * n_samples
            epoch_end = epoch_start + n_samples - 1
            global_sample = epoch_start + sample_in_epoch
            while global_sample in used_samples and global_sample < epoch_end:
                global_sample += 1
            if global_sample in used_samples:
                message(
                    "warning",
                    f"Epoch {i}, event '{label}': too many colliding events in "
                    "this epoch's window; dropping overflow event rather than "
                    "let it drift into a neighboring epoch's data block.",
                )
                continue
            used_samples.add(global_sample)

            events.append([global_sample, 0, event_id_rebuilt[str(label)]])
            epoch_indices.append(i)

    if not events:
        return None

    return (
        np.array(events, dtype=int),
        event_id_rebuilt,
        np.array(epoch_indices, dtype=int),
    )


def extract_epoch_events(
    epochs: mne.BaseEpochs,
) -> Tuple[np.ndarray, Dict[str, int], np.ndarray, str]:
    """Extract a per-epoch event list, preferring metadata but never losing condition codes.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        The epochs object to extract events from.

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
        Events expressed as global sample positions in the concatenated
        (n_epochs * n_times) timeline, in the same [sample, 0, code] format
        as MNE event arrays.
    event_id : dict
        Mapping of condition label -> code.
    epoch_indices : np.ndarray
        0-based epoch index that each row in `events` belongs to.
    source : str
        ``"metadata"`` when the richer additional_events reconstruction was
        used, ``"primary"`` when this fell back to epochs.events/event_id
        directly (also the value whenever no additional_events metadata is
        present at all).

    Notes
    -----
    ``epochs.metadata["additional_events"]`` is only used when it is present
    and reconstructs at least one event for every epoch with labels that all
    resolve to codes already present in ``epochs.event_id``. Otherwise this
    falls back to ``epochs.events``/``epochs.event_id`` — which always covers
    every epoch exactly once — and logs a warning explaining why.
    """
    n_epochs = len(epochs)
    primary_events, primary_event_id, primary_epoch_indices = _primary_events(epochs)

    if epochs.metadata is None or "additional_events" not in epochs.metadata.columns:
        return primary_events, primary_event_id, primary_epoch_indices, "primary"

    rebuilt = _rebuild_from_metadata(epochs)
    if rebuilt is None:
        message(
            "warning",
            "additional_events metadata unusable; falling back to epochs.events/"
            "epochs.event_id to preserve original condition codes.",
        )
        return primary_events, primary_event_id, primary_epoch_indices, "primary"

    events, event_id, epoch_indices = rebuilt
    n_covered = len(np.unique(epoch_indices))
    if n_covered < n_epochs:
        message(
            "warning",
            f"additional_events metadata only covers {n_covered}/{n_epochs} epochs; "
            "falling back to epochs.events/epochs.event_id to preserve original "
            "condition codes.",
        )
        return primary_events, primary_event_id, primary_epoch_indices, "primary"

    return events, event_id, epoch_indices, "metadata"
