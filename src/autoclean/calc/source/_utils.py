"""Shared helpers for handling source estimate inputs."""
from __future__ import annotations

from collections.abc import Sequence

import mne
from mne import SourceEstimate


def ensure_stc_list(
    stc_or_stcs: SourceEstimate | Sequence[SourceEstimate],
) -> tuple[list[SourceEstimate], bool]:
    """Return a list of :class:`~mne.SourceEstimate` objects and a flag for plurality.

    Parameters
    ----------
    stc_or_stcs
        Either a single source estimate or an iterable of them.

    Returns
    -------
    stcs, is_sequence
        ``stcs`` is a new list containing the provided source estimates and
        ``is_sequence`` is ``True`` when the caller supplied more than one.

    Raises
    ------
    TypeError
        If *stc_or_stcs* is neither a ``SourceEstimate`` nor a sequence of them.
    ValueError
        If an empty sequence is provided.
    """

    if isinstance(stc_or_stcs, SourceEstimate):
        return [stc_or_stcs], False

    if isinstance(stc_or_stcs, Sequence):
        stc_list = list(stc_or_stcs)
        if not stc_list:
            raise ValueError("Expected at least one source estimate, received none.")
        if not all(isinstance(stc, SourceEstimate) for stc in stc_list):
            raise TypeError("All items must be instances of mne.SourceEstimate.")
        return stc_list, True

    raise TypeError(
        "Expected a SourceEstimate or a sequence of SourceEstimate objects, "
        f"got {type(stc_or_stcs)!r}."
    )


def coerce_stc_to_single(
    stc_or_stcs: SourceEstimate | Sequence[SourceEstimate],
) -> tuple[SourceEstimate, bool]:
    """Return a single source estimate, concatenating sequences when necessary."""

    stc_list, was_sequence = ensure_stc_list(stc_or_stcs)
    if not was_sequence:
        return stc_list[0], False

    combined = mne.concatenate_stcs(stc_list)
    # Preserve the sampling frequency metadata when present (custom attribute).
    if hasattr(stc_list[0], "sfreq") and not hasattr(combined, "sfreq"):
        combined.sfreq = getattr(stc_list[0], "sfreq")
    return combined, True


__all__ = ["ensure_stc_list", "coerce_stc_to_single"]
