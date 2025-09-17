"""Tests for wavelet reporting utilities."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw


@pytest.fixture(scope="module")
def wavelet_report_module():
    """Load the wavelet report module without importing autoclean package."""

    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "autoclean"
        / "reporting"
        / "wavelet_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "autoclean.reporting.wavelet_report_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_wavelet_report_from_raw(tmp_path, monkeypatch, wavelet_report_module):
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
    result = wavelet_report_module.generate_wavelet_report(
        raw,
        output_pdf,
        snippet_duration=0.5,
        top_n_channels=3,
    )

    assert output_pdf.exists()
    assert result.metrics.shape[0] == len(raw.ch_names)
    assert np.isfinite(result.metrics["ptp_reduction_pct"]).all()
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
    }
    assert expected_keys.issubset(result.summary.keys())
