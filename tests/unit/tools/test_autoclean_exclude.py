"""Focused tests for exclusion GUI metadata loading."""

import json
from copy import deepcopy

from autoclean.tools.autoclean_exclude import (
    ExclusionFileSelector,
    _bad_channels_from_metadata,
    _unique_channels,
)


def test_bad_channels_from_duplicate_removal_reasons_is_unique_and_stable():
    metadata = {
        "channel_removals": [
            {"channel": "E2", "reason": "UNCORRELATED"},
            {"channel": "E1", "reason": "RANSAC"},
            {"channel": "E2", "reason": "RANSAC"},
        ]
    }
    original_metadata = deepcopy(metadata)

    assert _bad_channels_from_metadata(metadata) == ["E2", "E1"]
    assert metadata == original_metadata


def test_unique_channels_keeps_saved_selection_unique_and_stable():
    assert _unique_channels(["E2", "E1", "E2", "E3", "E1"]) == [
        "E2",
        "E1",
        "E3",
    ]


def test_bad_channels_from_legacy_metadata_preserves_nonduplicate_behavior():
    metadata = {"step_clean_bad_channels": {"bads": ["E3", "E1"]}}

    assert _bad_channels_from_metadata(metadata) == ["E3", "E1"]


def test_parse_metadata_json_deduplicates_channels_but_preserves_removals(tmp_path):
    channel_removals = [
        {"channel": "E2", "reason": "UNCORRELATED"},
        {"channel": "E1", "reason": "RANSAC"},
        {"channel": "E2", "reason": "RANSAC"},
    ]
    json_path = tmp_path / "metadata.json"
    json_path.write_text(
        json.dumps({"metadata": {"channel_removals": channel_removals}}),
        encoding="utf-8",
    )

    result = ExclusionFileSelector._parse_metadata_json(None, json_path)

    assert result["bad_channels"] == ["E2", "E1"]
    assert result["channel_removals"] == channel_removals


def test_parse_metadata_json_preserves_legacy_bad_channels(tmp_path):
    json_path = tmp_path / "metadata.json"
    json_path.write_text(
        json.dumps(
            {"metadata": {"step_clean_bad_channels": {"bads": ["E3", "E1", "E3"]}}}
        ),
        encoding="utf-8",
    )

    result = ExclusionFileSelector._parse_metadata_json(None, json_path)

    assert result["bad_channels"] == ["E3", "E1"]
