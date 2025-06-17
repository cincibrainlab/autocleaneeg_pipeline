"""Referencing functions for EEG data.

This module provides standalone functions for applying different referencing
schemes to EEG data, including average reference, specific channel reference,
and REST (Reference Electrode Standardization Technique).
"""

from typing import List, Optional, Union

import mne


def rereference_data(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    ref_channels: Union[str, List[str]] = "average",
    projection: bool = False,
    ch_type: str = "auto",
    forward: Optional[mne.Forward] = None,
    verbose: Optional[bool] = None,
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Apply referencing scheme to EEG data.

    This function applies various referencing schemes to continuous (Raw) or epoched
    EEG data. Referencing is a crucial preprocessing step that determines the voltage
    reference point for all EEG measurements, significantly affecting signal interpretation
    and analysis results.

    The function supports common referencing schemes including average reference,
    single-channel reference, multi-channel reference, and REST (Reference Electrode
    Standardization Technique) when appropriate forward model is provided.

    Parameters
    ----------
    data : mne.io.BaseRaw or mne.Epochs
        The EEG data to rereference. Can be any MNE Raw object (e.g., RawFIF,
        RawEEGLAB, etc.) or Epochs object.
    ref_channels : str or list of str, default 'average'
        Reference channel(s) to use. Options:
        - 'average': Compute average reference across all EEG channels
        - 'REST': Apply Reference Electrode Standardization Technique (requires forward model)
        - str: Name of single channel to use as reference (e.g., 'Cz', 'TP9')
        - list of str: Names of multiple channels to average for reference
        - []: Empty list creates an artificial reference channel
    projection : bool, default False
        Whether to apply referencing as a projection. If True, referencing is
        applied as a reversible projection operator. If False, data is modified
        directly (recommended for most use cases).
    ch_type : str, default 'auto'
        Channel type to apply referencing to. 'auto' automatically selects EEG
        channels. Can also specify 'eeg', 'ecog', 'seeg', or 'dbs'.
    forward : mne.Forward or None, default None
        Forward model for REST referencing. Required only when ref_channels='REST'.
        Should be computed for the same electrode montage as the data.
    verbose : bool or None, default None
        Control verbosity of output. If None, uses MNE default.

    Returns
    -------
    rereferenced_data : mne.io.BaseRaw or mne.Epochs
        The rereferenced data object, same type as input. Contains identical
        structure and metadata but with modified voltage references.

    Notes
    -----
    Common referencing schemes: average reference, single channel, linked mastoids, or REST.

    Examples
    --------
    >>> avg_ref_data = rereference_data(raw, ref_channels='average')
    >>> mastoid_ref_data = rereference_data(raw, ref_channels='TP9')
    >>> rest_ref_data = rereference_data(raw, ref_channels='REST', forward=forward_model)

    See Also
    --------
    mne.io.Raw.set_eeg_reference : MNE's raw data referencing method
    mne.Epochs.set_eeg_reference : MNE's epochs referencing method
    mne.set_eeg_reference : General referencing function
    mne.make_forward_solution : For creating forward models for REST
    """
    # Input validation
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError(
            f"Data must be an MNE Raw or Epochs object, got {type(data).__name__}"
        )

    # Validate reference channels parameter
    if isinstance(ref_channels, str):
        if (
            ref_channels not in ["average", "REST"]
            and ref_channels not in data.ch_names
        ):
            if ref_channels != "average" and ref_channels != "REST":
                raise ValueError(
                    f"Reference channel '{ref_channels}' not found in data. "
                    f"Available channels: {data.ch_names}"
                )
    elif isinstance(ref_channels, (list, tuple)):
        if ref_channels:  # Non-empty list
            missing_channels = [ch for ch in ref_channels if ch not in data.ch_names]
            if missing_channels:
                raise ValueError(
                    f"Reference channels {missing_channels} not found in data. "
                    f"Available channels: {data.ch_names}"
                )
    else:
        raise TypeError(
            f"ref_channels must be str, list, or tuple, got {type(ref_channels).__name__}"
        )

    # Validate REST requirements
    if ref_channels == "REST" and forward is None:
        raise ValueError("Forward model is required for REST referencing")

    # Create a copy to avoid modifying the original
    rereferenced_data = data.copy()

    try:
        # Apply referencing using MNE's built-in method
        if ref_channels == "average":
            # Special handling for average reference
            rereferenced_data.set_eeg_reference(
                ref_channels="average",
                projection=projection,
                ch_type=ch_type,
                verbose=verbose,
            )
        elif ref_channels == "REST":
            # REST referencing with forward model
            rereferenced_data.set_eeg_reference(
                ref_channels="REST",
                projection=projection,
                ch_type=ch_type,
                forward=forward,
                verbose=verbose,
            )
        else:
            # Single channel or multi-channel reference
            rereferenced_data.set_eeg_reference(
                ref_channels=ref_channels,
                projection=projection,
                ch_type=ch_type,
                verbose=verbose,
            )

        return rereferenced_data

    except Exception as e:
        ref_str = (
            str(ref_channels)
            if not isinstance(ref_channels, list)
            else f"[{', '.join(ref_channels)}]"
        )
        raise RuntimeError(f"Failed to apply {ref_str} reference: {str(e)}") from e
