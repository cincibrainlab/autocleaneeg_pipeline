"""Round-trip tests: pre-epoched condition codes must survive EEGLAB export.

Covers the bug where a synthetic (or pre-epoched) EpochsArray with multiple
event IDs could lose its condition structure on export if
epochs.metadata["additional_events"] was missing, empty, or misaligned.
"""

from collections import Counter
from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np

from autoclean.io.export import save_epochs_to_set


def _make_epochs(n_epochs=8, n_channels=4, n_times=50, sfreq=100.0):
    event_id = {"FREQ": 1, "RARE": 2}
    codes = [1, 2]
    data = np.random.randn(n_epochs, n_channels, n_times).astype("float64") * 1e-6
    ch_names = [f"Ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, "eeg")
    events = np.array(
        [[i * n_times, 0, codes[i % len(codes)]] for i in range(n_epochs)]
    )
    return mne.EpochsArray(
        data, info, events=events, tmin=-0.1, event_id=event_id, verbose=False
    )


def _autoclean_dict(tmp_path: Path) -> dict:
    return {
        "run_id": "run-001",
        "unprocessed_file": str(tmp_path / "sample.set"),
        "stage_dir": tmp_path / "stage",
        "move_flagged_files": True,
    }


@patch("autoclean.io.export.manage_database_conditionally")
def test_export_reopen_preserves_event_labels_and_codes(
    mock_manage_database, tmp_path: Path
) -> None:
    """No additional_events metadata: epochs.event_id must survive export/reopen."""
    epochs = _make_epochs()

    result = save_epochs_to_set(
        epochs, _autoclean_dict(tmp_path), stage="post_import"
    )
    reopened = mne.io.read_epochs_eeglab(str(result), verbose=False)

    assert reopened.event_id == {"FREQ": 1, "RARE": 2}
    assert Counter(reopened.events[:, 2].tolist()) == Counter(
        epochs.events[:, 2].tolist()
    )


@patch("autoclean.io.export.manage_database_conditionally")
def test_export_reopen_preserves_counts_per_condition(
    mock_manage_database, tmp_path: Path
) -> None:
    epochs = _make_epochs(n_epochs=12)

    result = save_epochs_to_set(
        epochs, _autoclean_dict(tmp_path), stage="post_import"
    )
    reopened = mne.io.read_epochs_eeglab(str(result), verbose=False)

    for label, code in epochs.event_id.items():
        orig_count = int((epochs.events[:, 2] == code).sum())
        new_count = int((reopened.events[:, 2] == reopened.event_id[label]).sum())
        assert new_count == orig_count


@patch("autoclean.io.export.manage_database_conditionally")
def test_export_with_valid_metadata_still_preserves_codes(
    mock_manage_database, tmp_path: Path
) -> None:
    """additional_events metadata covering every epoch should also round-trip cleanly."""
    import pandas as pd

    epochs = _make_epochs(n_epochs=6)
    labels = ["FREQ", "RARE"]
    epochs.metadata = pd.DataFrame(
        {
            "additional_events": [
                [(labels[i % 2], 0.0)] for i in range(len(epochs))
            ]
        }
    )

    result = save_epochs_to_set(
        epochs, _autoclean_dict(tmp_path), stage="post_import"
    )
    reopened = mne.io.read_epochs_eeglab(str(result), verbose=False)

    assert reopened.event_id == {"FREQ": 1, "RARE": 2}
    assert Counter(reopened.events[:, 2].tolist()) == Counter(
        epochs.events[:, 2].tolist()
    )
