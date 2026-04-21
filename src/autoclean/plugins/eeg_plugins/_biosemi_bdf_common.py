"""Shared BioSemi BDF import helpers."""

from __future__ import annotations

from pathlib import Path

import mne

from autoclean.utils.logging import message

_REFERENCE_CHANNELS = ("LM", "RM")
_EOG_CHANNELS = ("LVE", "RVE", "LHE", "RHE")
_MISC_CHANNELS = ("EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8")


def _biosemi_import_options(autoclean_dict: dict) -> tuple[bool, bool, dict[str, str]]:
    """Resolve optional BioSemi import settings from the task config."""
    options = autoclean_dict.get("biosemi_import", {})
    if not isinstance(options, dict):
        return False, False, {}

    channel_types = options.get("channel_types", {})
    if not isinstance(channel_types, dict):
        channel_types = {}

    return (
        bool(options.get("keep_reference_channels", False)),
        bool(options.get("keep_external_channels", False)),
        {str(name): str(kind) for name, kind in channel_types.items()},
    )


def import_biosemi_bdf(
    file_path: Path,
    autoclean_dict: dict,
    preload: bool,
    montage_name: str,
) -> mne.io.Raw:
    """Import a BioSemi BDF file and apply the shared configuration flow."""
    keep_reference_channels, keep_external_channels, channel_type_overrides = (
        _biosemi_import_options(autoclean_dict)
    )

    message("info", f"Loading BioSemi BDF file with {montage_name} montage: {file_path}")
    message(
        "info",
        "BioSemi BDF import immediately re-references EEG to LM/RM mastoids when both channels are available.",
    )

    raw = mne.io.read_raw_bdf(
        input_fname=file_path,
        preload=preload,
        stim_channel="auto",
        exclude=[],
        verbose=True,
    )
    message("success", "Successfully loaded BDF file with status channel")
    message(
        "debug",
        f"Loaded {len(raw.ch_names)} channels: {', '.join(raw.ch_names[:10])}...",
    )

    rename_mapping: dict[str, str] = {}
    for ch_name in raw.ch_names:
        if ch_name == "Status":
            continue
        if "_" in ch_name:
            _, standard_name = ch_name.split("_", 1)
            if standard_name == "Afz":
                standard_name = "AFz"
            rename_mapping[ch_name] = standard_name

    if rename_mapping:
        raw.rename_channels(rename_mapping)
        message("debug", f"Renamed {len(rename_mapping)} channels to standard names")

    channel_type_mapping = {
        "Status": "stim",
        **{ch: "eog" for ch in _EOG_CHANNELS},
        **{ch: "misc" for ch in _MISC_CHANNELS},
    }
    channel_type_mapping.update(channel_type_overrides)
    available_channel_types = {
        ch: kind for ch, kind in channel_type_mapping.items() if ch in raw.ch_names
    }
    if available_channel_types:
        raw.set_channel_types(available_channel_types)
        message(
            "debug",
            f"Set channel types for {len(available_channel_types)} external/status channels",
        )

    message("info", f"Configuring {montage_name} montage")
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")
    message("success", f"Successfully applied {montage_name} montage")

    ref_channels = [ch for ch in _REFERENCE_CHANNELS if ch in raw.ch_names]
    if len(ref_channels) == 2:
        raw.set_eeg_reference(
            ref_channels=ref_channels,
            projection=False,
            verbose=False,
        )
        message("success", f"Re-referenced EEG to mastoid channels: {ref_channels}")
    else:
        message(
            "warning",
            "Could not re-reference to [LM, RM] because one or both channels are missing",
        )

    if not keep_reference_channels:
        ref_to_drop = [ch for ch in _REFERENCE_CHANNELS if ch in raw.ch_names]
        if ref_to_drop:
            raw.drop_channels(ref_to_drop)
            message(
                "debug",
                f"Dropped reference channels after rereferencing: {ref_to_drop}",
            )

    if not keep_external_channels:
        external_to_drop = [
            ch for ch in (*_EOG_CHANNELS, *_MISC_CHANNELS) if ch in raw.ch_names
        ]
        if external_to_drop:
            raw.drop_channels(external_to_drop)
            message("debug", f"Dropped external channels: {external_to_drop}")

    raw.pick_types(
        eeg=True,
        eog=keep_external_channels,
        stim=True,
        misc=keep_external_channels,
        exclude=[],
    )

    message("info", f"Selected {len(raw.ch_names)} channels after configuration")
    return raw
