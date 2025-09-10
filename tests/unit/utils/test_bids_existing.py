import numpy as np
import mne
from pathlib import Path

from autoclean.utils.bids import prepare_existing_bids


def test_prepare_existing_bids(tmp_path):
    root = tmp_path / "source"
    bids_dir = root / "sub-01" / "eeg"
    bids_dir.mkdir(parents=True)

    info = mne.create_info(["Fp1"], sfreq=100, ch_types="eeg")
    raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose=False)
    raw_path = bids_dir / "sub-01_task-test_eeg.fif"
    raw.save(raw_path, overwrite=True)

    (root / "dataset_description.json").write_text("{}")

    out_dir = tmp_path / "out"
    bids_path, derivatives_dir = prepare_existing_bids(raw_path, out_dir)

    assert bids_path.root == out_dir
    assert bids_path.subject == "01"
    assert derivatives_dir.exists()
    assert (out_dir / "sub-01" / "eeg" / raw_path.name).exists()
