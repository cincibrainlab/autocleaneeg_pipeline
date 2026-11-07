from __future__ import annotations

import time

import mne
import numpy as np
import pandas as pd
import pytest
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from autoclean.calc.oscillation_analysis import (
    compute_phase_amplitude_coupling,
    compute_power_power_coupling,
    detect_hfos,
)

mne.set_log_level("ERROR")


def _raw(data: np.ndarray, sfreq: float, ch_names: list[str] | None = None):
    names = ch_names or [f"C{i}" for i in range(data.shape[0])]
    info = mne.create_info(names, sfreq, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


def _bandpass(x: np.ndarray, band: tuple[float, float], sfreq: float) -> np.ndarray:
    sos = signal.butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, x)


def _slow_envelope(rng: np.random.Generator, n: int, sfreq: float) -> np.ndarray:
    s = gaussian_filter1d(rng.standard_normal(n), sigma=1.0 * sfreq)
    return np.exp(1.0 * s / s.std())


def _pink(rng: np.random.Generator, n: int) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n=n)
    return x / x.std()


# ---------------------------------------------------------------- PPC


def test_ppc_shared_envelope_correlates_and_independent_does_not() -> None:
    sfreq, n = 1000.0, 120_000
    rng = np.random.default_rng(1)
    shared = _slow_envelope(rng, n, sfreq)
    carriers = [_bandpass(rng.standard_normal(n), (30, 45), sfreq) for _ in range(3)]
    other = _slow_envelope(rng, n, sfreq)
    raw = _raw(
        np.vstack([carriers[0] * shared, carriers[1] * shared, carriers[2] * other]),
        sfreq,
        ["A", "B", "C"],
    )
    pairs = [
        {"channel_a": "A", "band_a": [30, 45], "channel_b": "B", "band_b": [30, 45]},
        {"channel_a": "A", "band_a": [30, 45], "channel_b": "C", "band_b": [30, 45]},
    ]
    df = compute_power_power_coupling(raw, pairs=pairs)
    assert list(df.columns) == [
        "channel_a",
        "band_a_low",
        "band_a_high",
        "channel_b",
        "band_b_low",
        "band_b_high",
        "n_windows",
        "pearson_r",
        "spearman_r",
        "p_value",
        "window_s",
        "status",
    ]
    assert (df["status"] == "ok").all()
    assert df.loc[0, "n_windows"] == 118  # 120 s minus 1 s filter edge each side
    assert df.loc[0, "pearson_r"] > 0.5
    assert df.loc[0, "p_value"] < 0.05
    assert abs(df.loc[1, "pearson_r"]) < 0.2


def test_ppc_default_pairs_use_bands_and_require_them() -> None:
    rng = np.random.default_rng(2)
    raw = _raw(rng.standard_normal((3, 10_000)), 1000.0)
    df = compute_power_power_coupling(
        raw, bands={"gamma": [30, 45], "beta": [15, 25]}, picks=["C0", "C1", "C2"]
    )
    assert len(df) == 6  # 3 unordered pairs x 2 bands
    with pytest.raises(ValueError, match="bands is required"):
        compute_power_power_coupling(raw)


def test_ppc_too_short_raises() -> None:
    raw = _raw(np.random.default_rng(3).standard_normal((2, 2500)), 1000.0)
    with pytest.raises(ValueError, match="at least 3"):
        compute_power_power_coupling(raw, bands={"g": [30, 45]})


def test_ppc_epochs_windows_do_not_straddle_epochs() -> None:
    rng = np.random.default_rng(4)
    # 1.5 s epochs, 1 s windows -> 1 window per epoch, not 1.5
    epochs = mne.EpochsArray(
        rng.standard_normal((6, 2, 1500)), mne.create_info(["A", "B"], 1000.0, "eeg")
    )
    df = compute_power_power_coupling(epochs, bands={"g": [30, 45]})
    assert df.loc[0, "n_windows"] == 6


# ---------------------------------------------------------------- PAC


def _pac_signal(modulated: bool, seed: int = 5) -> np.ndarray:
    # Theta is narrowband noise centred near 7 Hz (not a pure tone, so that
    # circular-shift surrogates are not accidentally phase-aligned).
    sfreq, n = 1000.0, 60_000
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sfreq
    theta = _bandpass(rng.standard_normal(n), (5, 9), sfreq)
    theta /= theta.std()
    phase = np.angle(signal.hilbert(theta))
    amp = 0.5 * (1 + np.cos(phase)) if modulated else np.full(n, 0.5)
    x = theta + amp * np.sin(2 * np.pi * 80 * t) + 0.3 * _pink(rng, n)
    return x[np.newaxis]


def test_pac_detects_modulation() -> None:
    raw = _raw(_pac_signal(True), 1000.0)
    df = compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]])
    row = df.iloc[0]
    assert list(df.columns) == [
        "phase_channel",
        "amp_channel",
        "phase_low",
        "phase_high",
        "amp_low",
        "amp_high",
        "mi",
        "surrogate_mean",
        "surrogate_std",
        "z_score",
        "p_value",
        "n_surrogates",
        "preferred_phase_rad",
        "seed",
        "status",
    ]
    assert row["status"] == "ok"
    assert row["z_score"] > 3
    assert row["p_value"] < 0.05
    assert row["mi"] > row["surrogate_mean"]
    # amplitude is maximal at the theta peak (phase 0)
    assert abs(row["preferred_phase_rad"]) < 0.5


def test_pac_unmodulated_not_significant() -> None:
    raw = _raw(_pac_signal(False), 1000.0)
    df = compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]])
    assert df.loc[0, "z_score"] < 2


def test_pac_wrong_amp_band_not_significant() -> None:
    raw = _raw(_pac_signal(True), 1000.0)
    df = compute_phase_amplitude_coupling(raw, [[5, 9]], [[150, 200]])
    assert df.loc[0, "z_score"] < 2
    assert df.loc[0, "p_value"] > 0.05


def test_pac_is_deterministic_for_seed() -> None:
    raw = _raw(_pac_signal(True), 1000.0)
    a = compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]], seed=3)
    b = compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]], seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_pac_zero_surrogates_and_channel_pairs() -> None:
    x = _pac_signal(True)
    raw = _raw(np.vstack([x, x]), 1000.0, ["P", "Q"])
    df = compute_phase_amplitude_coupling(
        raw, [[5, 9]], [[70, 90]], channel_pairs=[("P", "Q")], n_surrogates=0
    )
    assert len(df) == 1
    assert df.loc[0, "phase_channel"] == "P" and df.loc[0, "amp_channel"] == "Q"
    assert df.loc[0, "mi"] > 0
    assert np.isnan(df.loc[0, "z_score"]) and np.isnan(df.loc[0, "p_value"])
    assert df.loc[0, "n_surrogates"] == 0


def test_pac_requires_amp_above_phase() -> None:
    raw = _raw(_pac_signal(True), 1000.0)
    with pytest.raises(ValueError, match="above the phase band"):
        compute_phase_amplitude_coupling(raw, [[5, 9]], [[8, 20]])


def test_pac_too_short_raises() -> None:
    raw = _raw(np.random.default_rng(6).standard_normal((1, 2500)), 1000.0)
    with pytest.raises(ValueError, match="too short"):
        compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]])


def test_pac_epochs_input() -> None:
    x = _pac_signal(True)[0].reshape(6, 10_000)[:, np.newaxis, :]
    epochs = mne.EpochsArray(x, mne.create_info(["A"], 1000.0, "eeg"))
    df = compute_phase_amplitude_coupling(epochs, [[5, 9]], [[70, 90]])
    assert df.loc[0, "z_score"] > 3


def test_pac_performance_120s_2khz() -> None:
    sfreq, n = 2000.0, 240_000
    rng = np.random.default_rng(7)
    raw = _raw(rng.standard_normal((1, n)), sfreq)
    start = time.perf_counter()
    compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]], n_surrogates=200)
    assert time.perf_counter() - start < 5.0


# ---------------------------------------------------------------- HFO


def _hfo_signal(
    sfreq: float = 2000.0, seconds: float = 60.0, n_bursts: int = 10, seed: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(sfreq * seconds)
    x = _pink(rng, n)
    sigma = _bandpass(x, (120, 250), sfreq).std()
    onsets = np.linspace(4.0, seconds - 4.0, n_bursts)
    burst_n = int(0.05 * sfreq)
    win = signal.windows.tukey(burst_n, alpha=0.3)
    tb = np.arange(burst_n) / sfreq
    for onset in onsets:
        i = int(onset * sfreq)
        x[i : i + burst_n] += 8 * sigma * win * np.sin(2 * np.pi * 150 * tb)
    return x[np.newaxis], onsets


def test_hfo_detects_injected_ripples() -> None:
    data, onsets = _hfo_signal()
    events, summary = detect_hfos(_raw(data, 2000.0, ["LFP"]))
    assert list(events.columns) == [
        "channel",
        "onset_s",
        "offset_s",
        "duration_s",
        "peak_freq_hz",
        "peak_amplitude",
        "band_low",
        "band_high",
    ]
    hits = sum(np.any(np.abs(events["onset_s"].to_numpy() - o) <= 0.02) for o in onsets)
    assert hits >= 9
    assert len(events) - hits <= 1
    assert np.all(np.abs(events["peak_freq_hz"] - 150) < 20)
    assert summary.loc[0, "n_events"] == len(events)
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(60.0)
    assert summary.loc[0, "rate_per_min"] == pytest.approx(len(events))


def test_hfo_pure_noise_is_empty_with_schema() -> None:
    data, _ = _hfo_signal(n_bursts=0, seed=9)
    events, summary = detect_hfos(_raw(data, 2000.0, ["LFP"]))
    assert len(events) <= 1
    clean, summary = detect_hfos(_raw(np.zeros((1, 20_000)), 2000.0))
    assert len(clean) == 0
    assert "onset_s" in clean.columns and "band_high" in clean.columns
    assert summary.loc[0, "n_events"] == 0


def test_hfo_epochs_input_has_epoch_index() -> None:
    data, onsets = _hfo_signal(seconds=60.0)
    # Contiguous epochs: events must carry real sample positions (EpochsArray
    # defaults to 0, 1, 2, ...), since onset_s now follows epochs.events.
    ev = np.column_stack([np.arange(6) * 20_000, np.zeros(6, int), np.ones(6, int)])
    epochs = mne.EpochsArray(
        data[0].reshape(6, 1, 20_000),
        mne.create_info(["LFP"], 2000.0, "eeg"),
        events=ev,
    )
    events, summary = detect_hfos(epochs)
    assert "epoch_index" in events.columns
    hits = sum(np.any(np.abs(events["onset_s"].to_numpy() - o) <= 0.02) for o in onsets)
    assert hits >= 8
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(60.0)


# ---------------------------------------------------------------- validation


def _short_raw(sfreq: float = 1000.0, **info_kwargs: float) -> mne.io.RawArray:
    raw = _raw(np.random.default_rng(10).standard_normal((2, 30_000)), sfreq)
    if info_kwargs:
        with raw.info._unlock():
            raw.info.update(info_kwargs)
    return raw


@pytest.mark.parametrize(
    "func",
    [
        lambda d: compute_power_power_coupling(d, bands={"g": [30, 45]}),
        lambda d: compute_phase_amplitude_coupling(d, [[5, 9]], [[70, 90]]),
        lambda d: detect_hfos(d),
    ],
)
def test_non_mne_input_raises_type_error(func) -> None:
    with pytest.raises(TypeError):
        func(np.zeros((1, 100)))


def test_missing_channels_named() -> None:
    raw = _short_raw()
    with pytest.raises(ValueError, match="Nope"):
        compute_power_power_coupling(raw, bands={"g": [30, 45]}, picks=["C0", "Nope"])
    with pytest.raises(ValueError, match="Ghost"):
        compute_phase_amplitude_coupling(
            raw, [[5, 9]], [[70, 90]], channel_pairs=[("C0", "Ghost")]
        )
    with pytest.raises(ValueError, match="Missing"):
        detect_hfos(raw, picks=["Missing"])
    with pytest.raises(ValueError, match="Phantom"):
        compute_power_power_coupling(
            raw,
            pairs=[
                {
                    "channel_a": "C0",
                    "band_a": [30, 45],
                    "channel_b": "Phantom",
                    "band_b": [30, 45],
                }
            ],
        )


def test_non_finite_channel_named() -> None:
    raw = _short_raw()
    raw._data[1, 100] = np.nan
    with pytest.raises(ValueError, match="C1"):
        compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]])
    with pytest.raises(ValueError, match="C1"):
        detect_hfos(raw)
    with pytest.raises(ValueError, match="C1"):
        compute_power_power_coupling(raw, bands={"g": [30, 45]})


@pytest.mark.parametrize("band", [(45, 30), (30, 30), (0, 30), (-5, 30)])
def test_bad_band_raises(band) -> None:
    raw = _short_raw()
    with pytest.raises(ValueError, match="0 < low < high"):
        compute_power_power_coupling(raw, bands={"g": band})
    with pytest.raises(ValueError, match="0 < low < high"):
        detect_hfos(raw, band=band)


def test_nyquist_message_mentions_resample_step() -> None:
    raw = _short_raw(sfreq=500.0)
    for func in (
        lambda: detect_hfos(raw, band=(120, 250)),
        lambda: compute_power_power_coupling(raw, bands={"h": [100, 260]}),
        lambda: compute_phase_amplitude_coupling(raw, [[5, 9]], [[100, 300]]),
    ):
        with pytest.raises(ValueError) as exc:
            func()
        msg = str(exc.value)
        assert "resample_step" in msg
        assert "500" in msg and "250" in msg
        assert "sfreq > 2 x" in msg


def test_lowpass_message_mentions_h_freq() -> None:
    raw = _short_raw(sfreq=2000.0, lowpass=100.0)
    with pytest.raises(ValueError) as exc:
        detect_hfos(raw, band=(120, 250))
    assert "low-pass filtered at 100 Hz" in str(exc.value)
    assert "h_freq above 250 Hz" in str(exc.value)


# ---------------------------------------------------------------- review fixes


def _with_bad_transients(
    data: np.ndarray, sfreq: float, starts: list[float], names: list[str] | None = None
):
    """Add 2 mV-scale 20 ms transients, each annotated BAD_artifact."""
    x = data.copy()
    n_t = int(0.02 * sfreq)
    for t0 in starts:
        i = int(t0 * sfreq)
        x[0, i : i + n_t] += (
            2e-3 * np.hanning(n_t) * np.sin(2 * np.pi * 150 * np.arange(n_t) / sfreq)
        )
    raw = _raw(x, sfreq, names or ["LFP"])
    raw.set_annotations(
        mne.Annotations(
            onset=[t - 0.05 for t in starts],
            duration=[0.15] * len(starts),
            description=["BAD_artifact"] * len(starts),
        )
    )
    return raw


def test_hfo_ignores_bad_annotated_transients() -> None:
    sfreq = 2000.0
    data, _ = _hfo_signal(n_bursts=0, seed=11)
    data = data * 1e-5
    starts = [5.0 + 5 * k for k in range(10)]
    clean, _ = detect_hfos(_raw(data, sfreq, ["LFP"]))
    # Without annotations the transients are detected.
    unmarked = _with_bad_transients(data, sfreq, starts)
    unmarked.set_annotations(None)
    assert len(detect_hfos(unmarked)[0]) >= 5
    events, summary = detect_hfos(_with_bad_transients(data, sfreq, starts))
    assert len(events) <= len(clean) + 1
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(60.0 - 10 * 0.15)


def test_hfo_onsets_in_recording_time_after_bad_span() -> None:
    sfreq = 2000.0
    data, onsets = _hfo_signal()
    raw = _raw(data, sfreq, ["LFP"])
    # First burst is at 4 s; mark 0.5-2.5 s bad so every later span is offset.
    raw.set_annotations(mne.Annotations([0.5], [2.0], ["BAD_x"]))
    events, summary = detect_hfos(raw)
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(58.0)
    hits = sum(np.any(np.abs(events["onset_s"].to_numpy() - o) <= 0.02) for o in onsets)
    assert hits >= 9


def test_bad_annotations_respect_first_samp_and_orig_time() -> None:
    sfreq = 1000.0
    x = np.random.default_rng(12).standard_normal((1, 30_000))
    info = mne.create_info(["A"], sfreq, "eeg")
    raw = mne.io.RawArray(x, info, first_samp=5000, verbose=False)
    raw.set_meas_date(1_700_000_000.0)
    # orig_time set: onset is relative to meas_date, so it includes first_samp.
    raw.set_annotations(
        mne.Annotations(
            [5.0 + 10.0],
            [4.0],
            ["BAD_x"],
            orig_time=raw.info["meas_date"],
        )
    )
    _, summary = detect_hfos(raw)
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(26.0)
    # orig_time None: onset is relative to the first sample.
    raw.set_annotations(mne.Annotations([10.0], [4.0], ["BAD_x"]))
    _, summary = detect_hfos(raw)
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(26.0)


def test_all_bad_raises_and_short_spans_counted() -> None:
    x = np.random.default_rng(13).standard_normal((1, 20_000))
    raw = _raw(x, 1000.0)
    raw.set_annotations(mne.Annotations([0.0], [20.0], ["BAD_all"]))
    with pytest.raises(ValueError, match="No usable data"):
        detect_hfos(raw)
    raw.set_annotations(mne.Annotations([0.01], [19.0], ["BAD_most"]))
    _, summary = detect_hfos(raw)  # 10 ms head span is dropped, tail is kept
    assert summary.attrs["n_segments_dropped"] == 1
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(0.99, abs=0.01)


def test_non_bad_annotations_do_not_split() -> None:
    x = np.random.default_rng(14).standard_normal((1, 20_000))
    raw = _raw(x, 1000.0)
    raw.set_annotations(mne.Annotations([5.0], [2.0], ["stimulus"]))
    _, summary = detect_hfos(raw)
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(20.0)


def test_pac_and_ppc_unchanged_by_bad_transients() -> None:
    x = _pac_signal(True)
    starts = [5.0 + 5 * k for k in range(10)]
    dirty = _with_bad_transients(x, 1000.0, starts)
    # Control: same BAD annotations, no transients in the data.
    control = dirty.copy()
    control._data[:] = x
    got = compute_phase_amplitude_coupling(dirty, [[5, 9]], [[70, 90]])
    ref = compute_phase_amplitude_coupling(control, [[5, 9]], [[70, 90]])
    assert got.loc[0, "z_score"] > 3
    assert got.loc[0, "mi"] == pytest.approx(ref.loc[0, "mi"], rel=1e-6)
    bands = {"g": [30, 45], "b": [15, 25]}
    xx = np.vstack([x, x[:, ::-1]])
    dirty2 = _with_bad_transients(xx, 1000.0, starts, ["A", "B"])
    control2 = dirty2.copy()
    control2._data[:] = np.vstack([x, x[:, ::-1]])
    r1 = compute_power_power_coupling(control2, bands=bands)
    r2 = compute_power_power_coupling(dirty2, bands=bands)
    assert np.allclose(r1["pearson_r"], r2["pearson_r"], atol=1e-6)
    # BAD time is excluded: fewer windows than the full 60 s would give.
    assert r2["n_windows"].iloc[0] < 58


def _stim_bad_raw() -> mne.io.RawArray:
    rng = np.random.default_rng(15)
    info = mne.create_info(["A", "B", "S", "C"], 1000.0, ["eeg", "eeg", "stim", "eeg"])
    raw = mne.io.RawArray(rng.standard_normal((4, 20_000)), info, verbose=False)
    raw.info["bads"] = ["B"]
    return raw


@pytest.mark.parametrize("picks", [None, "eeg", "data", ["eeg"], [0, 1, 3]])
def test_picks_resolve_the_mne_way(picks) -> None:
    raw = _stim_bad_raw()
    _, summary = detect_hfos(raw, picks=picks, band=(120, 250))
    assert list(summary["channel"]) == ["A", "C"]
    df = compute_power_power_coupling(raw, bands={"g": [30, 45]}, picks=picks)
    assert list(zip(df["channel_a"], df["channel_b"])) == [("A", "C")]


def test_picks_names_and_bad_pairs() -> None:
    raw = _stim_bad_raw()
    _, summary = detect_hfos(raw, picks=["C", "A"])
    assert list(summary["channel"]) == ["C", "A"]
    with pytest.raises(ValueError, match="marked bad"):
        compute_power_power_coupling(
            raw,
            pairs=[
                {
                    "channel_a": "A",
                    "band_a": [30, 45],
                    "channel_b": "B",
                    "band_b": [30, 45],
                }
            ],
        )
    with pytest.raises(ValueError, match="marked bad"):
        compute_phase_amplitude_coupling(
            raw, [[5, 9]], [[70, 90]], channel_pairs=[("A", "B")]
        )


def test_hfo_epochs_timing_uses_events_tmin_and_dropped() -> None:
    sfreq = 2000.0
    data, _ = _hfo_signal(n_bursts=0, seed=16, seconds=40.0)
    x = data[0].reshape(4, 1, 20_000)
    # Burst at 3.0 s into each epoch (epoch spans tmin=-0.5 .. 9.5 s).
    burst_n = int(0.05 * sfreq)
    sigma = _bandpass(data[0], (120, 250), sfreq).std()
    tb = np.arange(burst_n) / sfreq
    win = signal.windows.tukey(burst_n, alpha=0.3)
    for e in range(4):
        i = int(3.0 * sfreq)
        x[e, 0, i : i + burst_n] += 8 * sigma * win * np.sin(2 * np.pi * 150 * tb)
    ev = np.column_stack(
        [[10_000, 50_000, 90_000, 130_000], np.zeros(4, int), np.ones(4, int)]
    )
    epochs = mne.EpochsArray(
        x, mne.create_info(["LFP"], sfreq, "eeg"), events=ev, tmin=-0.5
    )
    epochs.drop([1])  # drop the second epoch; indices shift
    events, summary = detect_hfos(epochs)
    assert list(events.columns[-3:]) == [
        "epoch_index",
        "epoch_onset_s",
        "onset_in_epoch_s",
    ]
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(30.0)
    kept = np.array([10_000, 90_000, 130_000]) / sfreq - 0.5
    for e, start in enumerate(kept):
        sub = events[events["epoch_index"] == e]
        assert len(sub) >= 1
        assert np.allclose(sub["epoch_onset_s"], start)
        assert np.any(np.abs(sub["onset_in_epoch_s"] - 3.0) <= 0.02)
        assert np.allclose(sub["onset_s"], start + sub["onset_in_epoch_s"])


def test_pac_narrow_amp_band_raises() -> None:
    raw = _raw(_pac_signal(True), 1000.0)
    with pytest.raises(ValueError, match="at least 2 x the phase band high edge"):
        compute_phase_amplitude_coupling(raw, [[6, 9]], [[83, 87]])
    with pytest.raises(ValueError, match=r"83-87 Hz is 4 Hz wide.*\(18 Hz\)"):
        compute_phase_amplitude_coupling(raw, [[6, 9]], [[83, 87]])


def _pac_shifted(shift: float) -> np.ndarray:
    sfreq, n = 1000.0, 60_000
    rng = np.random.default_rng(5)
    t = np.arange(n) / sfreq
    theta = _bandpass(rng.standard_normal(n), (5, 9), sfreq)
    theta /= theta.std()
    phase = np.angle(signal.hilbert(theta))
    amp = 0.5 * (1 + np.cos(phase - shift))
    x = theta + amp * np.sin(2 * np.pi * 80 * t) + 0.3 * _pink(rng, n)
    return x[np.newaxis]


@pytest.mark.parametrize("shift", [0.0, np.pi / 2, -np.pi / 2])
def test_pac_preferred_phase_tracks_true_phase(shift: float) -> None:
    raw = _raw(_pac_shifted(shift), 1000.0)
    df = compute_phase_amplitude_coupling(raw, [[5, 9]], [[70, 90]])
    diff = np.angle(np.exp(1j * (df.loc[0, "preferred_phase_rad"] - shift)))
    assert abs(diff) < 0.5


# ---------------------------------------------------------------- round 2 fixes


def _span_positions(inst: mne.io.BaseRaw, names: list[str]) -> list[tuple[int, int]]:
    from autoclean.calc.oscillation_analysis import _select_channel_data

    segs, _ = _select_channel_data(inst, names)
    return [
        (int(round(s.start_s * 1000)), int(round(s.start_s * 1000)) + s.data.shape[1])
        for s in segs
    ]


@pytest.mark.parametrize(
    "first_samp,meas_date,orig_time_set",
    [(20_000, None, False), (20_000, 1_700_000_000.0, True), (0, None, False)],
)
def test_bad_mask_positions_match_mne(first_samp, meas_date, orig_time_set) -> None:
    x = np.random.default_rng(21).standard_normal((1, 30_000))
    raw = mne.io.RawArray(
        x, mne.create_info(["A"], 1000.0, "eeg"), first_samp=first_samp, verbose=False
    )
    if meas_date is not None:
        raw.set_meas_date(meas_date)
    if orig_time_set:
        # onset relative to meas_date includes first_samp (20 s)
        ann = mne.Annotations(
            [20.0 + 10.0], [2.0], ["BAD_artifact"], orig_time=raw.info["meas_date"]
        )
    else:
        ann = mne.Annotations([10.0], [2.0], ["BAD_artifact"])
    raw.set_annotations(ann)
    expected = ~np.isnan(raw.get_data(reject_by_annotation="NaN")[0])
    assert not expected[10_000:12_000].any() and expected[:10_000].all()
    assert _span_positions(raw, ["A"]) == [(0, 10_000), (12_000, 30_000)]
    # Same positions through the public API: onsets of detected spans.
    _, summary = detect_hfos(raw)
    assert summary.loc[0, "analyzed_duration_s"] == pytest.approx(28.0)


def test_real_nan_still_raises_with_bad_annotations() -> None:
    x = np.random.default_rng(22).standard_normal((1, 20_000))
    x[0, 3000] = np.nan
    raw = _raw(x, 1000.0)
    raw.set_annotations(mne.Annotations([10.0], [2.0], ["BAD_x"]))
    with pytest.raises(ValueError, match="Non-finite"):
        detect_hfos(raw)


def _pac_slow(shift: float, seconds: float = 120.0) -> np.ndarray:
    sfreq = 1000.0
    t = np.arange(int(seconds * sfreq)) / sfreq
    phase = 2 * np.pi * 4.0 * t
    amp = 0.5 * (1 + np.cos(phase - shift))
    rng = np.random.default_rng(23)
    x = (
        np.cos(phase)
        + amp * np.sin(2 * np.pi * 80 * t)
        + 0.1 * rng.standard_normal(t.size)
    )
    return x[np.newaxis]


def _gapped(x: np.ndarray, good_s: float, bad_s: float) -> mne.io.RawArray:
    raw = _raw(x, 1000.0)
    period = good_s + bad_s
    onsets = np.arange(good_s, x.shape[1] / 1000.0 - bad_s, period)
    raw.set_annotations(
        mne.Annotations(onsets, [bad_s] * len(onsets), ["BAD_gap"] * len(onsets))
    )
    return raw


def test_pac_short_good_spans_are_dropped_not_distorted() -> None:
    raw = _gapped(_pac_slow(0.0), good_s=0.1, bad_s=0.1)
    with pytest.raises(ValueError, match="No usable data.*minimum usable length"):
        compute_phase_amplitude_coupling(raw, [[3, 5]], [[60, 100]], n_surrogates=0)


def test_pac_three_second_spans_give_correct_phase() -> None:
    shift = np.pi / 2
    raw = _gapped(_pac_slow(shift), good_s=3.0, bad_s=0.5)
    df = compute_phase_amplitude_coupling(raw, [[3, 5]], [[60, 100]], n_surrogates=0)
    diff = np.angle(np.exp(1j * (df.loc[0, "preferred_phase_rad"] - shift)))
    assert abs(diff) < 0.5


def test_pac_short_epochs_with_low_phase_band_raise() -> None:
    x = _pac_slow(0.0, 60.0)[0].reshape(300, 1, 200)  # 0.2 s epochs
    epochs = mne.EpochsArray(x, mne.create_info(["A"], 1000.0, "eeg"))
    with pytest.raises(ValueError, match="No usable data"):
        compute_phase_amplitude_coupling(epochs, [[3, 5]], [[60, 100]])


def test_ppc_short_spans_dropped_and_counted() -> None:
    rng = np.random.default_rng(24)
    raw = _raw(rng.standard_normal((2, 60_000)), 1000.0)
    # one long good span plus many 0.5 s spans (< 1 window after edge trim)
    onsets = [30.0 + 1.0 * k for k in range(29)]
    raw.set_annotations(mne.Annotations(onsets, [0.5] * 29, ["BAD_gap"] * 29))
    df = compute_power_power_coupling(raw, bands={"g": [30, 45]})
    # 0.5 s spans pass the basic length rule but keep < 1 window after trimming
    assert df.attrs["n_segments_dropped"] == 0
    assert df.attrs["n_segments_dropped_by_band"]["30-45"] == 28


def test_hfo_drops_spans_shorter_than_twice_min_duration() -> None:
    raw = _raw(np.random.default_rng(25).standard_normal((1, 20_000)), 1000.0)
    # 0.02 s head span is < 2 * 0.015 s? no (0.02 < 0.03): dropped; tail kept
    raw.set_annotations(mne.Annotations([0.02], [19.0], ["BAD_most"]))
    _, summary = detect_hfos(raw)
    assert summary.attrs["n_segments_dropped"] == 1


# ------------------------------------------------- per-band trimming, NaN gaps


def _epochs_2s(n_epochs=60, n_ch=2, seed=30):
    x = np.random.default_rng(seed).standard_normal((n_epochs, n_ch, 2000))
    return mne.EpochsArray(
        x, mne.create_info([f"E{i}" for i in range(n_ch)], 1000.0, "eeg")
    )


def test_ppc_slow_band_does_not_trim_fast_band() -> None:
    epochs = _epochs_2s()
    both = compute_power_power_coupling(
        epochs, bands={"delta": (1, 4), "gamma": (30, 80)}
    )
    only = compute_power_power_coupling(epochs, bands={"gamma": (30, 80)})
    gamma = both[both["band_a_low"] == 30].iloc[0]
    delta = both[both["band_a_low"] == 1].iloc[0]
    assert gamma["status"] == "ok" and gamma["n_windows"] == 60
    assert delta["status"] == "insufficient_data" and delta["n_windows"] == 0
    assert np.isnan(delta["pearson_r"])
    pd.testing.assert_frame_equal(
        both[both["band_a_low"] == 30].reset_index(drop=True), only
    )
    assert both.attrs["n_segments_dropped_by_band"] == {"1-4": 60, "30-80": 0}


def test_ppc_all_bands_insufficient_raises() -> None:
    with pytest.raises(ValueError, match="No usable data.*Epochs|epochs shorter"):
        compute_power_power_coupling(_epochs_2s(), bands={"delta": (1, 4)})


def test_pac_slow_phase_band_insufficient_fast_ok() -> None:
    x = _pac_signal(True)[0][:60_000].reshape(30, 1, 2000)
    epochs = mne.EpochsArray(x, mne.create_info(["A"], 1000.0, "eeg"))
    df = compute_phase_amplitude_coupling(
        epochs, [[2, 4], [6, 9]], [[30, 80]], n_surrogates=20, min_shift_s=0.1
    )
    ok = df[df["phase_low"] == 6].iloc[0]
    slow = df[df["phase_low"] == 2].iloc[0]
    assert ok["status"] == "ok" and np.isfinite(ok["mi"])
    assert slow["status"] == "insufficient_data" and np.isnan(slow["mi"])
    alone = compute_phase_amplitude_coupling(
        epochs, [[6, 9]], [[30, 80]], n_surrogates=20, min_shift_s=0.1
    )
    pd.testing.assert_frame_equal(
        df[df["phase_low"] == 6].reset_index(drop=True), alone
    )
    assert df.attrs["n_segments_dropped_by_band"]["2-4"] == 30


def test_epochs_error_wording_is_not_raw_specific() -> None:
    with pytest.raises(ValueError) as exc:
        compute_power_power_coupling(_epochs_2s(), bands={"delta": (1, 4)})
    msg = str(exc.value)
    assert "BAD*" not in msg and "epochs shorter" in msg
    assert "1-4 Hz:" in msg and "minimum usable length" in msg


def _nan_gap_raw(bad_gap: bool) -> mne.io.RawArray:
    x = np.random.default_rng(31).standard_normal((2, 30_000))  # 60 s @ 500 Hz
    x[:, 10_000:12_500] = np.nan  # 20-25 s
    raw = mne.io.RawArray(x, mne.create_info(["a", "b"], 500.0, "eeg"), verbose=False)
    if bad_gap:
        raw.set_annotations(mne.Annotations([20.0], [5.0], ["BAD_gap"]))
    return raw


def test_nan_gap_under_bad_span_is_excluded() -> None:
    from autoclean.calc.oscillation_analysis import _select_channel_data

    segs, dropped = _select_channel_data(_nan_gap_raw(True), ["a", "b"])
    assert dropped == 0
    assert [(s.start_s, s.data.shape[1] / 500.0) for s in segs] == [
        (0.0, 20.0),
        (25.0, 35.0),
    ]
    assert all(np.isfinite(s.data).all() for s in segs)


def test_nan_gap_outside_bad_span_still_raises() -> None:
    from autoclean.calc.oscillation_analysis import _select_channel_data

    with pytest.raises(ValueError, match="Non-finite"):
        _select_channel_data(_nan_gap_raw(False), ["a", "b"])
    raw = _nan_gap_raw(True)
    raw.set_annotations(mne.Annotations([20.0], [2.0], ["BAD_gap"]))  # too short
    with pytest.raises(ValueError, match="Non-finite"):
        _select_channel_data(raw, ["a", "b"])


def _random_bad_raw(rng: np.random.Generator) -> mne.io.RawArray:
    sfreq, n = 100.0, 3000
    first_samp = int(rng.choice([0, 12345]))
    info = mne.create_info(["a", "b"], sfreq, "eeg")
    raw = mne.io.RawArray(
        rng.standard_normal((2, n)), info, first_samp=first_samp, verbose=False
    )
    if rng.random() < 0.5:
        raw.set_meas_date(1_600_000_000.0)
    k = int(rng.integers(0, 6))
    lo = first_samp / sfreq
    onsets = lo + rng.uniform(-5, n / sfreq + 5, k)
    durs = rng.uniform(0, 8, k)
    descs = [
        str(rng.choice(["BAD_x", "bad", "Bad_seg", "BAD_ACQ_SKIP", "good"]))
        for _ in range(k)
    ]
    raw.set_annotations(
        mne.Annotations(onsets, durs, descs, orig_time=raw.info["meas_date"])
    )
    return raw


def test_bad_sample_mask_matches_mne_reject_by_annotation() -> None:
    from autoclean.calc.oscillation_analysis import _bad_sample_mask

    rng = np.random.default_rng(2024)
    for _ in range(30):
        raw = _random_bad_raw(rng)
        expected = np.isnan(raw.get_data(picks=[0], reject_by_annotation="NaN")[0])
        np.testing.assert_array_equal(_bad_sample_mask(raw), expected)


def test_select_channel_data_does_not_copy_the_data() -> None:
    import tracemalloc

    from autoclean.calc.oscillation_analysis import _select_channel_data

    x = np.random.default_rng(0).standard_normal((32, 120_000))
    raw = mne.io.RawArray(x, mne.create_info(32, 1000.0, "eeg"), verbose=False)
    raw.set_annotations(mne.Annotations([10.0], [5.0], ["BAD_x"]))
    full = x.nbytes
    tracemalloc.start()
    try:
        _select_channel_data(raw, ["0"])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 0.5 * full


@pytest.mark.parametrize("other", [[2, 4], [11, 13]])
def test_pac_pair_result_independent_of_other_bands(other) -> None:
    x = _pac_signal(True)[0][:60_000].reshape(30, 1, 2000)
    epochs = mne.EpochsArray(x, mne.create_info(["A"], 1000.0, "eeg"))
    kw = dict(n_surrogates=30, min_shift_s=0.1, seed=4)
    alone = compute_phase_amplitude_coupling(epochs, [[6, 9]], [[30, 80]], **kw)
    for bands in ([other, [6, 9]], [[6, 9], other]):
        df = compute_phase_amplitude_coupling(epochs, bands, [[30, 80]], **kw)
        got = df[df["phase_low"] == 6].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, alone)
