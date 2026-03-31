"""Regression tests for ICA visualization cache compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import mne
import numpy as np
import pytest
from mne.preprocessing import ICA

from autoclean.functions.visualization._ica_cache_utils import get_ica_mixing_matrix
from autoclean.functions.visualization._ica_psd_cache import ICAPSDCache
from autoclean.functions.visualization._ica_topography_cache import ICATopographyCache


def _fit_test_ica() -> tuple[ICA, mne.io.RawArray]:
    sfreq = 100.0
    info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, ch_types="eeg")
    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1000))
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore")

    ica = ICA(n_components=2, random_state=0, max_iter="auto")
    ica.fit(raw, verbose=False)
    return ica, raw


def test_cache_helpers_support_mne_mixing_matrix_attr() -> None:
    ica, raw = _fit_test_ica()

    assert hasattr(ica, "mixing_matrix_")
    assert not hasattr(ica, "mixing_")

    topo_key = ICATopographyCache()._generate_cache_key(ica)
    psd_key = ICAPSDCache()._generate_cache_key(
        ica,
        raw,
        {"fmin": 1.0, "fmax": 40.0, "n_fft": 256, "sfreq": raw.info["sfreq"]},
    )

    assert topo_key.startswith("ica_topo_")
    assert psd_key.startswith("ica_psd_")


def test_get_ica_mixing_matrix_falls_back_to_legacy_attr() -> None:
    legacy = SimpleNamespace(mixing_=np.array([[1.0, 2.0], [3.0, 4.0]]))

    mixing = get_ica_mixing_matrix(legacy)  # type: ignore[arg-type]

    np.testing.assert_array_equal(mixing, legacy.mixing_)


def test_get_ica_mixing_matrix_requires_fitted_ica() -> None:
    with pytest.raises(AttributeError, match="mixing matrix"):
        get_ica_mixing_matrix(SimpleNamespace())  # type: ignore[arg-type]
