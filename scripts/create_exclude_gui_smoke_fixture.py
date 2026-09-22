#!/usr/bin/env python3
"""Create a deterministic epoched EEGLAB file for Exclude GUI smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np


def create_fixture(output_dir: Path) -> Path:
    """Write a small epoched EEG fixture and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "exclude_gui_smoke_epo.set"

    sfreq = 100.0
    ch_names = ["Fp1", "Fp2", "Cz", "Oz"]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    times = np.arange(200) / sfreq

    data = []
    for epoch_idx in range(12):
        epoch = []
        for channel_idx, _channel in enumerate(ch_names):
            frequency = channel_idx + 1
            amplitude = (10 + channel_idx * 2) * 1e-6
            drift = epoch_idx * 0.25e-6
            epoch.append(amplitude * np.sin(2 * np.pi * frequency * times) + drift)
        data.append(epoch)

    events = np.column_stack(
        [
            np.arange(12, dtype=int) * len(times),
            np.zeros(12, dtype=int),
            np.ones(12, dtype=int),
        ]
    )
    epochs = mne.EpochsArray(
        np.asarray(data),
        info,
        events=events,
        event_id={"smoke": 1},
        tmin=0.0,
        verbose=False,
    )
    epochs.export(str(output_path), fmt="eeglab", overwrite=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(create_fixture(args.output_dir))


if __name__ == "__main__":
    main()
