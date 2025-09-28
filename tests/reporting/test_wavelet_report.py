"""Tests for wavelet reporting utilities."""

from __future__ import annotations

import numpy as np
import pytest

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
        "psd_fmax",
        "threshold_scale",
        "ptp_mean",
        "ptp_median",
        "ptp_max",
        "ptp_max_channel",
        "band_reductions",
        "picks",
    }
    assert expected_keys.issubset(result.summary.keys())
    assert "alpha" in result.summary["band_reductions"]


def test_generate_wavelet_report_with_custom_params(tmp_path, monkeypatch):
    """Custom wavelet settings should flow into the report summary."""

    home_dir = tmp_path / "home"
    mpl_dir = home_dir / "matplotlib"
    home_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("MPLCONFIGDIR", str(mpl_dir))
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")

    raw = create_synthetic_raw(montage="standard_1020", n_channels=8, sfreq=200, duration=2)

    output_pdf = tmp_path / "wavelet_report_custom.pdf"
    result = generate_wavelet_report(
        raw,
        output_pdf,
        level="auto",
        picks=["Fp1", "Fz"],
        psd_fmax=30.0,
        threshold_scale=0.75,
    )

    assert result.summary["psd_fmax"] == 30.0
    assert result.summary["threshold_scale"] == pytest.approx(0.75)
    assert result.summary["requested_level"] == "auto"
    assert set(result.summary.get("picks", [])) == {"Fp1", "Fz"}
