"""Unit tests for configurable ROI source PSD outputs."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from autoclean.blocks.analysis.source_psd.algorithm import calculate_roi_psd


def _make_roi_epochs(sfreq: float = 250.0) -> mne.BaseEpochs:
    """Create synthetic ROI epochs with source-style channel names."""
    ch_names = ["lh_precentral", "rh_precentral"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    n_epochs = 4
    n_samples = int(2.0 * sfreq)
    times = np.arange(n_samples) / sfreq

    alpha = np.sin(2 * np.pi * 10 * times)
    highgamma = 0.5 * np.sin(2 * np.pi * 70 * times)
    template = alpha + highgamma

    data = np.stack(
        [np.vstack([template, template * 0.8]) for _ in range(n_epochs)],
        axis=0,
    )

    events = np.column_stack(
        [
            np.arange(n_epochs),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    )
    return mne.EpochsArray(data, info, events=events, tmin=0.0, verbose=False)


def test_calculate_roi_psd_supports_custom_fmax_and_bands(tmp_path: Path):
    """ROI source PSD should emit configured high-gamma summaries."""
    epochs = _make_roi_epochs()
    bands = {
        "alpha": (8.0, 13.0),
        "highgamma": (65.0, 80.0),
    }

    psd_df, output_path = calculate_roi_psd(
        data=epochs,
        segment_duration=None,
        output_dir=str(tmp_path),
        subject_id="subject01",
        generate_plots=False,
        fmin=1.0,
        fmax=80.0,
        bands=bands,
    )

    assert Path(output_path).exists()
    assert psd_df["frequency"].max() <= 80.0

    band_csv = tmp_path / "subject01_roi_bands.csv"
    assert band_csv.exists()

    band_df = pd.read_csv(band_csv)
    assert set(band_df["band"].unique()) == {"alpha", "highgamma"}
