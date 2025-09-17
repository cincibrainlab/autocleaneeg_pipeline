"""Tests for wavelet reporting utilities."""

from __future__ import annotations

import numpy as np

from autoclean.functions.preprocessing.wavelet_thresholding import (
    generate_wavelet_report,
)
from tests.fixtures.synthetic_data import create_synthetic_raw


def test_generate_wavelet_report_from_raw(tmp_path, monkeypatch):
    """Generating a report from Raw data produces a PDF and metrics."""

    home_dir = tmp_path / "home"
    mpl_dir = home_dir / "matplotlib"
    home_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    raw = create_synthetic_raw(n_channels=4, sfreq=200, duration=2)

    output_pdf = tmp_path / "wavelet_report.pdf"
    result = generate_wavelet_report(
        raw,
        output_pdf,
        snippet_duration=0.5,
        top_n_channels=3,
    )

    assert output_pdf.exists()
    assert result.metrics.shape[0] == len(raw.ch_names)
    assert result.psd_metrics["band"].nunique() > 0
    assert np.isfinite(result.metrics["ptp_reduction_pct"]).all()
    assert np.isfinite(result.psd_metrics["power_reduction_pct"]).all()
    expected_keys = {
        "channels",
        "sfreq",
        "duration_sec",
        "effective_level",
        "requested_level",
        "ptp_mean",
        "ptp_median",
        "ptp_max",
        "ptp_max_channel",
        "band_reductions",
    }
    assert expected_keys.issubset(result.summary.keys())
    assert "alpha" in result.summary["band_reductions"]
