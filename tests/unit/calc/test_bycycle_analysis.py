from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib as stdlib_importlib

import mne
import pandas as pd
import pytest

from autoclean.calc.bycycle_analysis import (
    build_bycycle_output_filename,
    classify_frequency_band,
    compute_bycycle_analysis,
    resolve_bycycle_thresholds,
)
from tests.fixtures.synthetic_data import create_synthetic_raw


def _make_epochs() -> mne.Epochs:
    raw = create_synthetic_raw(
        montage="standard_1020",
        n_channels=4,
        duration=6.0,
        sfreq=250.0,
    )
    events = mne.make_fixed_length_events(raw, duration=2.0)
    return mne.Epochs(
        raw, events, tmin=0.0, tmax=2.0, baseline=None, preload=True, verbose=False
    )


class _FakeBycycle:
    created: list[dict] = []

    def __init__(
        self,
        *,
        center_extrema: str,
        burst_method: str,
        thresholds: dict,
        find_extrema_kwargs: dict,
    ) -> None:
        self.df_features = pd.DataFrame({"sample_last_trough": [1], "period": [12.5]})
        _FakeBycycle.created.append(
            {
                "center_extrema": center_extrema,
                "burst_method": burst_method,
                "thresholds": dict(thresholds),
                "find_extrema_kwargs": find_extrema_kwargs,
            }
        )

    def fit(self, signal, sfreq, f_range) -> None:
        self.fit_signal_len = len(signal)
        self.fit_sfreq = sfreq
        self.fit_f_range = f_range


class _EmptyBycycle(_FakeBycycle):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.df_features = pd.DataFrame()


@pytest.fixture(autouse=True)
def _reset_fake_state() -> None:
    _FakeBycycle.created.clear()


def test_classify_frequency_band_supports_issue_ranges() -> None:
    assert classify_frequency_band((8, 12.5)) == "alpha"
    assert classify_frequency_band((30, 80)) == "gamma"
    assert classify_frequency_band((12, 20)) == "custom"


def test_resolve_thresholds_uses_band_defaults_and_overrides() -> None:
    assert resolve_bycycle_thresholds((8, 12.5))["monotonicity"] == 0.55
    assert resolve_bycycle_thresholds((30, 80))["monotonicity"] == 0.65
    assert resolve_bycycle_thresholds((30, 80), {"monotonicity": 0.8})[
        "monotonicity"
    ] == pytest.approx(0.8)


def test_compute_bycycle_analysis_adds_alpha_metadata(monkeypatch) -> None:
    raw = create_synthetic_raw(
        montage="standard_1020",
        n_channels=4,
        duration=5.0,
        sfreq=250.0,
    )
    monkeypatch.setattr(
        "autoclean.calc.bycycle_analysis._load_bycycle_module",
        lambda: SimpleNamespace(Bycycle=_FakeBycycle),
    )

    result = compute_bycycle_analysis(
        raw,
        f_range=(8, 12.5),
        picks=["Fp1", "Fp2"],
        metadata={"subject_id": "sub-01"},
    )

    assert list(result["channel"]) == ["Fp1", "Fp2"]
    assert set(result["freq_range"]) == {"alpha"}
    assert set(result["subject_id"]) == {"sub-01"}
    assert _FakeBycycle.created[0]["thresholds"]["monotonicity"] == pytest.approx(0.55)


def test_compute_bycycle_analysis_supports_epochs_and_gamma_defaults(
    monkeypatch,
) -> None:
    epochs = _make_epochs()
    monkeypatch.setattr(
        "autoclean.calc.bycycle_analysis._load_bycycle_module",
        lambda: SimpleNamespace(Bycycle=_FakeBycycle),
    )

    result = compute_bycycle_analysis(
        epochs,
        f_range=(30, 80),
        picks=[0],
        limit_duration_s=3.0,
    )

    assert result["source_type"].iloc[0] == "epochs"
    assert result["channel"].iloc[0] == epochs.ch_names[0]
    assert result["freq_range"].iloc[0] == "gamma"
    assert _FakeBycycle.created[0]["thresholds"]["monotonicity"] == pytest.approx(0.65)


def test_compute_bycycle_analysis_returns_empty_dataframe_when_no_features(
    monkeypatch,
) -> None:
    raw = create_synthetic_raw(
        montage="standard_1020",
        n_channels=2,
        duration=4.0,
        sfreq=250.0,
    )
    monkeypatch.setattr(
        "autoclean.calc.bycycle_analysis._load_bycycle_module",
        lambda: SimpleNamespace(Bycycle=_EmptyBycycle),
    )

    result = compute_bycycle_analysis(raw, metadata={"subject_id": "sub-02"})

    assert result.empty
    assert "channel" in result.columns
    assert "subject_id" in result.columns


def test_compute_bycycle_analysis_requires_bycycle_dependency(monkeypatch) -> None:
    raw = create_synthetic_raw(
        montage="standard_1020",
        n_channels=2,
        duration=2.0,
        sfreq=250.0,
    )
    original_import_module = stdlib_importlib.import_module

    def _patched_import_module(name: str):
        if name == "bycycle":
            raise ImportError("missing")
        return original_import_module(name)

    monkeypatch.setattr(
        "autoclean.calc.bycycle_analysis.importlib.import_module",
        _patched_import_module,
    )

    with pytest.raises(ImportError, match="bycycle"):
        compute_bycycle_analysis(raw)


def test_build_bycycle_output_filename_matches_issue_comment() -> None:
    assert build_bycycle_output_filename("sub-01", (8, 12.5)).endswith(
        "AlphaFilt_sub-01.parquet"
    )
    assert build_bycycle_output_filename("sub-01", (30, 80)).endswith(
        "GammaFilt_sub-01.parquet"
    )
