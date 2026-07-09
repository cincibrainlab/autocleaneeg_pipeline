"""Tests for the centralized epoch event extraction helper."""

from typing import Dict

import mne
import numpy as np
import pandas as pd

from autoclean.utils.epoch_events import extract_epoch_events


def _make_epochs(
    n_epochs: int = 6,
    n_channels: int = 3,
    n_times: int = 50,
    sfreq: float = 100.0,
    tmin: float = -0.1,
    event_id: Dict[str, int] = None,
) -> mne.EpochsArray:
    event_id = event_id or {"FREQ": 1, "RARE": 2}
    codes = list(event_id.values())
    data = np.random.randn(n_epochs, n_channels, n_times).astype("float64") * 1e-6
    ch_names = [f"Ch{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names, sfreq, "eeg")
    events = np.array(
        [[i * n_times, 0, codes[i % len(codes)]] for i in range(n_epochs)]
    )
    return mne.EpochsArray(
        data, info, events=events, tmin=tmin, event_id=event_id, verbose=False
    )


class TestPrimarySource:
    def test_used_when_metadata_is_none(self):
        epochs = _make_epochs()
        events, event_id, epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        assert event_id == {"FREQ": 1, "RARE": 2}
        assert list(events[:, 2]) == list(epochs.events[:, 2])
        assert list(epoch_indices) == list(range(len(epochs)))

    def test_used_when_additional_events_column_absent(self):
        epochs = _make_epochs()
        epochs.metadata = pd.DataFrame({"other_col": [1] * len(epochs)})

        _events, event_id, _epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        assert event_id == {"FREQ": 1, "RARE": 2}

    def test_covers_every_epoch_exactly_once(self):
        epochs = _make_epochs(n_epochs=8)
        events, _event_id, epoch_indices, _source = extract_epoch_events(epochs)

        assert len(events) == len(epochs)
        assert sorted(epoch_indices.tolist()) == list(range(len(epochs)))

    def test_condition_counts_match_original(self):
        epochs = _make_epochs(n_epochs=10)
        events, _event_id, _epoch_indices, _source = extract_epoch_events(epochs)

        orig_counts = {
            code: int((epochs.events[:, 2] == code).sum())
            for code in epochs.event_id.values()
        }
        new_counts = {
            code: int((events[:, 2] == code).sum()) for code in epochs.event_id.values()
        }
        assert new_counts == orig_counts


class TestMetadataSource:
    def test_used_when_metadata_valid_and_covers_every_epoch(self):
        epochs = _make_epochs(n_epochs=4)
        epochs.metadata = pd.DataFrame(
            {
                "additional_events": [
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                ]
            }
        )

        _events, event_id, epoch_indices, source = extract_epoch_events(epochs)

        assert source == "metadata"
        assert event_id == {"FREQ": 1, "RARE": 2}
        assert sorted(np.unique(epoch_indices).tolist()) == list(range(4))

    def test_falls_back_to_primary_when_metadata_missing_an_epoch(self):
        epochs = _make_epochs(n_epochs=4)
        epochs.metadata = pd.DataFrame(
            {
                "additional_events": [
                    [("FREQ", 0.0)],
                    None,
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                ]
            }
        )

        _events, event_id, _epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        assert event_id == {"FREQ": 1, "RARE": 2}

    def test_falls_back_to_primary_when_label_unknown_to_event_id(self):
        epochs = _make_epochs(n_epochs=4)
        epochs.metadata = pd.DataFrame(
            {
                "additional_events": [
                    [("BOGUS", 0.0)],
                    [("RARE", 0.0)],
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                ]
            }
        )

        _events, event_id, _epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        # Original condition codes must not be silently replaced/renumbered.
        assert event_id == {"FREQ": 1, "RARE": 2}

    def test_falls_back_to_primary_when_metadata_entry_malformed(self):
        epochs = _make_epochs(n_epochs=4)
        epochs.metadata = pd.DataFrame(
            {
                "additional_events": [
                    [("FREQ", 0.0)],
                    [("RARE",)],
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                ]
            }
        )

        _events, event_id, _epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        assert event_id == {"FREQ": 1, "RARE": 2}

    def test_never_invents_codes_outside_original_event_id(self):
        epochs = _make_epochs(n_epochs=4)
        epochs.metadata = pd.DataFrame(
            {
                "additional_events": [
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                    [("FREQ", 0.0)],
                    [("RARE", 0.0)],
                ]
            }
        )

        _events, event_id, _epoch_indices, _source = extract_epoch_events(epochs)

        assert set(event_id.values()).issubset(set(epochs.event_id.values()))

    def test_colliding_events_do_not_drift_into_next_epochs_block(self):
        """Many same-instant additional_events in one epoch must not spill over
        into the sample range owned by the next epoch."""
        n_times = 10
        epochs = _make_epochs(n_epochs=3, n_times=n_times)
        # 15 events all at rel_time=0.0 for epoch 0 -- more than n_times slots.
        epochs.metadata = pd.DataFrame(
            {
                "additional_events": [
                    [("FREQ", 0.0)] * 15,
                    [("RARE", 0.0)],
                    [("FREQ", 0.0)],
                ]
            }
        )

        events, _event_id, epoch_indices, _source = extract_epoch_events(epochs)

        epoch0_samples = events[epoch_indices == 0][:, 0]
        assert epoch0_samples.max() < n_times  # never crosses into epoch 1's block


class TestPrimarySourceEdgeCases:
    def test_handles_zero_epochs_without_crashing(self):
        epochs = _make_epochs(n_epochs=2)
        epochs.drop(list(range(len(epochs))))

        events, event_id, epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        assert events.shape == (0, 3)
        assert epoch_indices.shape == (0,)
        assert event_id == {"FREQ": 1, "RARE": 2}

    def test_handles_epoch_window_that_excludes_t_zero(self):
        """tmin > 0 (e.g. response-locked, post-stimulus-only epochs) must not
        place a "trigger" sample inside a neighboring epoch's data block."""
        n_times = 50
        epochs = _make_epochs(n_epochs=4, n_times=n_times, tmin=0.2)

        events, _event_id, _epoch_indices, source = extract_epoch_events(epochs)

        assert source == "primary"
        # Every event's sample offset within its own epoch block must stay
        # inside [0, n_times), never landing in an adjacent epoch's block.
        offsets_within_epoch = events[:, 0] % n_times
        assert np.all((offsets_within_epoch >= 0) & (offsets_within_epoch < n_times))
