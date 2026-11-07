"""Tests for channel-level connectivity and the shared connectivity core."""

import mne
import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, sosfiltfilt

from autoclean.calc.source_connectivity import (
    _connectivity_from_epoched_array,
    calculate_source_connectivity,
    compute_sensor_connectivity,
)

SFREQ = 1000.0
BANDS = {"beta": (15, 25)}


def _make_raw(seed=0, duration=60.0):
    rng = np.random.default_rng(seed)
    n = int(duration * SFREQ)
    sos = butter(4, [15, 25], btype="bandpass", fs=SFREQ, output="sos")
    shared = sosfiltfilt(sos, rng.standard_normal(n + 50))
    shared /= shared.std()
    delay = 8  # 8 ms at 1000 Hz
    common = rng.standard_normal(n) * 1.5
    data = rng.standard_normal((4, n)) * 0.3
    data[0] += shared[50 : 50 + n]
    data[1] += shared[50 - delay : 50 - delay + n]
    data += common  # zero-lag common noise on all channels
    info = mne.create_info(["c0", "c1", "c2", "c3"], SFREQ, "eeg")
    return mne.io.RawArray(data * 1e-6, info, verbose=False)


@pytest.fixture(scope="module")
def raw():
    return _make_raw()


@pytest.fixture(scope="module")
def table(raw):
    return compute_sensor_connectivity(raw, BANDS)


def _val(df, a, b, method):
    row = df[(df.channel_a == a) & (df.channel_b == b) & (df.method == method)]
    assert len(row) == 1
    return float(row.value.iloc[0])


def test_table_shape_and_columns(table):
    assert list(table.columns) == [
        "channel_a",
        "channel_b",
        "method",
        "band",
        "band_low",
        "band_high",
        "value",
        "n_epochs",
        "epoch_length_s",
    ]
    assert len(table) == 6 * 6  # 6 pairs x 6 methods x 1 band
    assert (table.n_epochs == 30).all()
    assert (table.epoch_length_s == 2.0).all()


def test_wpli_ranks_connected_pair_first(table):
    w = table[table.method == "wpli"].sort_values("value", ascending=False)
    top = w.iloc[0]
    assert (top.channel_a, top.channel_b) == ("c0", "c1")
    assert top.value > 0.5
    assert (w.iloc[1:].value < 0.3).all()


def test_zero_lag_noise_inflates_coh_over_wpli(table):
    assert _val(table, "c2", "c3", "wpli") <= _val(table, "c2", "c3", "coh")
    assert {"plv", "coh"} <= set(table.method)


def test_deterministic(raw, table):
    again = compute_sensor_connectivity(raw, BANDS)
    assert table.equals(again)


def test_epochs_input(raw):
    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, preload=True, verbose=False
    )
    df = compute_sensor_connectivity(epochs, BANDS, methods=("wpli", "aec"))
    assert len(df) == 12
    assert df.n_epochs.iloc[0] == 30
    assert df.epoch_length_s.iloc[0] == pytest.approx(2.0)
    top = df[df.method == "wpli"].sort_values("value", ascending=False).iloc[0]
    assert (top.channel_a, top.channel_b) == ("c0", "c1")


def test_picks_and_n_epochs(raw):
    df = compute_sensor_connectivity(
        raw, BANDS, methods=("wpli",), picks=["c0", "c1", "c2"], n_epochs=10
    )
    assert len(df) == 3
    assert (df.n_epochs == 10).all()


def test_narrow_band_high_sfreq():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((3, 2000 * 20))
    info = mne.create_info(["a", "b", "c"], 2000.0, "eeg")
    r = mne.io.RawArray(data * 1e-6, info, verbose=False)
    df = compute_sensor_connectivity(r, {"b": (15, 25)})
    assert len(df) == 3 * 6
    assert np.isfinite(df.value).all()


def test_validation_errors(raw):
    with pytest.raises(ValueError, match="Unknown connectivity method"):
        compute_sensor_connectivity(raw, BANDS, methods=("nope",))
    with pytest.raises(ValueError, match="spectral resolution"):
        compute_sensor_connectivity(raw, {"x": (20, 21)})
    with pytest.raises(ValueError, match="below high"):
        compute_sensor_connectivity(raw, {"x": (20, 10)})
    with pytest.raises(ValueError, match="Raise or disable resample_step"):
        compute_sensor_connectivity(raw, {"x": (20, 500)})
    with pytest.raises(ValueError, match="at least 2 channels"):
        compute_sensor_connectivity(raw, BANDS, picks=["c0"])
    with pytest.raises(ValueError, match="at least 2 epochs"):
        compute_sensor_connectivity(raw, BANDS, epoch_length=40.0)
    bad = raw.copy()
    bad._data[0, 5] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        compute_sensor_connectivity(bad, BANDS)


def test_method_failure_raises_runtime_error(monkeypatch):
    import autoclean.calc.source_connectivity as sc

    def boom(*a, **k):
        raise KeyError("bad")

    monkeypatch.setattr(sc, "spectral_connectivity_epochs", boom)
    rng = np.random.default_rng(0)
    with pytest.raises(
        RuntimeError, match="Connectivity method wpli failed for band beta"
    ):
        _connectivity_from_epoched_array(
            rng.standard_normal((4, 3, 2000)), ["a", "b", "c"], 1000.0, BANDS, ["wpli"]
        )


def test_source_path_runs_with_fake_stc(tmp_path):
    # calculate_source_connectivity looks up this fixed ROI list by name
    rois = [
        "precentral-lh",
        "precentral-rh",
        "postcentral-lh",
        "postcentral-rh",
        "paracentral-lh",
        "paracentral-rh",
        "caudalmiddlefrontal-lh",
        "caudalmiddlefrontal-rh",
    ]
    rng = np.random.default_rng(2)
    sfreq = 250.0
    n_times = int(sfreq * 40)
    vertices = [np.arange(6), np.arange(6)]
    data = rng.standard_normal((12, n_times))
    stc = mne.SourceEstimate(data, vertices, tmin=0.0, tstep=1 / sfreq, subject="fake")
    labels = [
        mne.Label(
            vertices=np.arange(k * 2, k * 2 + 2) % 6,
            hemi="lh" if name.endswith("lh") else "rh",
            name=name,
            subject="fake",
        )
        for k, name in enumerate(rois)
    ]
    conn_df, summary_path = calculate_source_connectivity(
        stc,
        labels=labels,
        n_jobs=1,
        output_dir=str(tmp_path),
        subject_id="s1",
        epoch_length=2.0,
        n_epochs=5,
    )
    assert summary_path is not None
    assert set(conn_df.columns) == {
        "subject",
        "method",
        "band",
        "roi1",
        "roi2",
        "connectivity",
    }
    assert set(conn_df.method) == {"wpli", "plv", "coh", "pli", "aec"}
    assert len(conn_df) == 28 * 5 * 5  # 8 ROIs -> 28 pairs x 5 methods x 5 bands
    assert (tmp_path / "s1_wpli_beta_matrix.csv").exists()
    metrics = pd.read_csv(tmp_path / "s1_graph_metrics.csv")
    assert {"clustering", "global_efficiency", "modularity"} <= set(metrics.columns)
    assert len(metrics) > 0


def _noise_raw(n_ch=6, sfreq=500.0, duration=120.0, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info([f"n{i}" for i in range(n_ch)], sfreq, "eeg")
    data = rng.standard_normal((n_ch, int(duration * sfreq))) * 1e-6
    return mne.io.RawArray(data, info, verbose=False)


def test_band_specificity():
    sfreq = 500.0
    n = int(120 * sfreq)
    rng = np.random.default_rng(1)
    sos = butter(4, [15, 25], btype="bandpass", fs=sfreq, output="sos")
    shared = sosfiltfilt(sos, rng.standard_normal(n + 50))
    shared /= shared.std()
    data = rng.standard_normal((4, n)) * 0.5
    data[0] += shared[50 : 50 + n]
    data[1] += shared[45 : 45 + n]  # 10 ms lag
    info = mne.create_info(["c0", "c1", "c2", "c3"], sfreq, "eeg")
    r = mne.io.RawArray(data * 1e-6, info, verbose=False)
    bands = {"beta": (15, 25), "gamma": (30, 45), "hgamma": (60, 90)}
    df = compute_sensor_connectivity(r, bands, methods=("wpli",))
    vals = {}
    for band in bands:
        sub = df[df.band == band]
        pair = (sub.channel_a == "c0") & (sub.channel_b == "c1")
        vals[band] = (float(sub[pair].value.iloc[0]), float(sub[~pair].value.max()))
    assert vals["beta"][0] > 0.5
    assert vals["beta"][0] > vals["beta"][1] + 0.3
    for band in ("gamma", "hgamma"):
        connected, other = vals[band]
        assert connected < other + 0.1, (band, vals)
        assert connected < vals["beta"][0] - 0.3


def test_noise_floor_aec_and_phase():
    # Delta needs 5-second epochs (5 cycles at 1 Hz); check AEC there.
    slow = compute_sensor_connectivity(
        _noise_raw(), {"delta": (1, 4), "theta": (4, 8)}, epoch_length=5.0
    )
    aec = slow[slow.method == "aec"]
    assert aec.value.abs().max() < 0.15
    bands = {"beta": (15, 25), "gamma": (30, 45)}
    df = compute_sensor_connectivity(_noise_raw(), bands)
    for band in ("beta", "gamma"):
        wpli = df[(df.method == "wpli") & (df.band == band)]
        assert wpli.value.max() < 0.3, band
    debiased = df[df.method == "wpli2_debiased"]
    assert debiased.value.abs().max() < 0.1


def test_bad_annotation_excluded():
    base = _make_raw(duration=60.0)
    clean = compute_sensor_connectivity(base, BANDS, methods=("wpli", "aec"))
    dirty = base.copy()
    onset, dur = 20.0, 12.0
    dirty._data[:, int(onset * SFREQ) : int((onset + dur) * SFREQ)] += 500e-6 * (
        np.random.default_rng(9).standard_normal(int(dur * SFREQ))
    )
    dirty.set_annotations(mne.Annotations(onset, dur, "BAD_artifact"))
    out = compute_sensor_connectivity(dirty, BANDS, methods=("wpli", "aec"))
    assert out.n_epochs.iloc[0] < clean.n_epochs.iloc[0]
    assert out.n_epochs.iloc[0] == 24  # 30 epochs minus 6 inside the BAD span
    merged = clean.merge(
        out, on=["channel_a", "channel_b", "method"], suffixes=("_c", "_d")
    )
    for method, tol in (("wpli", 0.2), ("aec", 0.2)):
        m = merged[merged.method == method]
        assert (m.value_c - m.value_d).abs().max() < tol


def test_bad_channels_excluded_and_picks_eeg(raw):
    r = raw.copy()
    r.info["bads"] = ["c3"]
    df = compute_sensor_connectivity(r, BANDS, methods=("wpli",))
    assert set(df.channel_a) | set(df.channel_b) == {"c0", "c1", "c2"}
    named = compute_sensor_connectivity(
        r, BANDS, methods=("wpli",), picks=["c0", "c1", "c3"]
    )
    assert set(named.channel_a) | set(named.channel_b) == {"c0", "c1"}
    eeg = compute_sensor_connectivity(r, BANDS, methods=("wpli",), picks="eeg")
    assert set(eeg.channel_a) | set(eeg.channel_b) == {"c0", "c1", "c2"}


def test_block_algorithm_delegates_to_calc():
    import importlib.util
    from pathlib import Path

    import autoclean

    path = (
        Path(autoclean.__file__).parent
        / "blocks/analysis/source_connectivity/algorithm.py"
    )
    spec = importlib.util.spec_from_file_location("_alg_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.calculate_source_connectivity is calculate_source_connectivity


def test_lowpass_below_band_raises():
    rng = np.random.default_rng(3)
    info = mne.create_info(["a", "b", "c"], 500.0, "eeg")
    r = mne.io.RawArray(rng.standard_normal((3, 500 * 20)) * 1e-6, info, verbose=False)
    r.filter(None, 40.0, verbose=False)
    with pytest.raises(ValueError, match="h_freq"):
        compute_sensor_connectivity(r, {"hi": (60, 90)})
    # core with explicit lowpass
    with pytest.raises(ValueError, match="h_freq"):
        _connectivity_from_epoched_array(
            rng.standard_normal((4, 3, 1000)),
            ["a", "b", "c"],
            500.0,
            {"hi": (60, 90)},
            ["wpli"],
            lowpass=40.0,
        )


def test_low_band_needs_five_cycles_per_epoch(raw):
    with pytest.raises(ValueError, match=r"epoch_length >= 5 s"):
        compute_sensor_connectivity(raw, {"delta": (1, 4)}, epoch_length=2.0)


def test_source_path_partial_roi_match(tmp_path):
    rng = np.random.default_rng(4)
    sfreq = 250.0
    stc = mne.SourceEstimate(
        rng.standard_normal((12, int(sfreq * 40))),
        [np.arange(6), np.arange(6)],
        tmin=0.0,
        tstep=1 / sfreq,
        subject="fake",
    )
    names = ["precentral-lh", "postcentral-rh", "other-lh"]  # only two are hard-coded
    labels = [
        mne.Label(
            vertices=np.arange(k * 2, k * 2 + 2) % 6,
            hemi="lh" if n.endswith("lh") else "rh",
            name=n,
            subject="fake",
        )
        for k, n in enumerate(names)
    ]
    conn_df, _ = calculate_source_connectivity(
        stc,
        labels=labels,
        n_jobs=1,
        output_dir=str(tmp_path),
        subject_id="s2",
        epoch_length=2.0,
        n_epochs=5,
    )
    assert set(conn_df.roi1) | set(conn_df.roi2) == {"precentral-lh", "postcentral-rh"}


def _ring(w, n=4):
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, (i + 1) % n] = mat[(i + 1) % n, i] = w
    return mat


def test_characteristic_path_length_uses_inverse_weight_distance():
    from autoclean.calc.source_connectivity import _characteristic_path_length

    # 4-node ring, edge length 1/w: per node distances are 1, 1, 2 (x 1/w),
    # so the mean over the 12 ordered pairs is (4/3) / w.
    assert _characteristic_path_length(_ring(0.5)) == pytest.approx(8.0 / 3.0)
    assert _characteristic_path_length(_ring(2.0)) == pytest.approx(2.0 / 3.0)
    # Stronger connectivity -> shorter path length.
    assert _characteristic_path_length(_ring(0.9)) < _characteristic_path_length(
        _ring(0.3)
    )


def test_characteristic_path_length_all_zero_graph_is_nan_without_warning():
    import warnings

    from autoclean.calc.source_connectivity import _characteristic_path_length

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(_characteristic_path_length(np.zeros((4, 4))))
