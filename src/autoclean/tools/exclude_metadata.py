"""Pure metadata helpers for the Exclude review tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def unique_channels(channels: List[str]) -> List[str]:
    """Return channel names once each, preserving first-seen order."""
    return list(dict.fromkeys(channels))


def bad_channels_from_metadata(metadata: Dict) -> List[str]:
    """Extract the unique operational bad-channel list from run metadata."""
    channel_removals = metadata.get("channel_removals", [])
    if channel_removals:
        return unique_channels([removal["channel"] for removal in channel_removals])

    legacy_bad_channels = metadata.get("step_clean_bad_channels", {}).get("bads", [])
    return unique_channels(legacy_bad_channels)


def parse_metadata_json(json_path: Path) -> dict[str, list]:
    """Parse metadata JSON for bad channels and rejected ICA components."""
    result = {"bad_channels": [], "rejected_ica": []}

    if not json_path or not json_path.exists():
        return result

    try:
        data = json.loads(json_path.read_text())
        metadata_section = data.get("metadata", {})

        channel_removals = metadata_section.get("channel_removals", [])
        if channel_removals and isinstance(channel_removals, list):
            result["channel_removals"] = channel_removals

        result["bad_channels"] = bad_channels_from_metadata(metadata_section)

        ica_rejection = metadata_section.get("step_apply_ica_component_rejection", {})
        rejected_comps = ica_rejection.get("ica", {}).get("final_excluded_indices", [])
        if isinstance(rejected_comps, list):
            result["rejected_ica"] = rejected_comps

    except Exception as e:
        print(f"Warning: Could not parse metadata JSON {json_path}: {e}")

    return result
