"""Ground-truth acceptance tests for the mouse depth analysis blocks.

The synthetic recording from ``create_synthetic_mouse_depth_raw`` has known
spectral peaks, aperiodic exponents, power-power coupling, theta-gamma PAC,
ripples and a lagged shared source. Every block is run through
``Task.run_postprocessing_analysis`` and compared against that truth.
"""

import hashlib
from pathlib import Path

import mne
import numpy as np
import pytest

from autoclean.core.task import Task
from tests.fixtures.synthetic_data import create_synthetic_mouse_depth_raw

PSD_BIN_HZ = 0.25  # n_fft = 4 * sfreq at 2 kHz -> 0.25 Hz bins
# The fixture peaks are narrowband noise (+-1 Hz), so the PSD argmax wanders by
# more than one bin; allow max(one bin, 0.5 Hz).
PEAK_TOL_HZ = max(PSD_BIN_HZ, 0.5)
ALL_BLOCKS = ("sensor_psd", "fooof", "power_coupling", "pac", "hfo", "connectivity")


class _DepthTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)
        self.calls = []

    def run(self):
        pass

    def _update_metadata(self, *args, **kwargs):
        self.calls.append(args)


def _block_settings(truth, n_surrogates=200):
    regions = truth["regions"]
    pc = truth["power_coupling"]
    return {
        "sensor_psd": {
            "enabled": True,
            "input": "clean_raw",
            "fmin": 1,
            "fmax": 300,
            "n_fft": 8000,
            "freq_bands": None,
        },
        "fooof": {
            "enabled": True,
            "input": "sensor_psd",
            "freq_range": [2, 55],
            "aperiodic_mode": "fixed",
        },
        "power_coupling": {
            "enabled": True,
            "pairs": [pc["coupled"], pc["control"]],
            "window_s": 1.0,
            "regions": regions,
        },
        "pac": {
            "enabled": True,
            "phase_bands": [truth["pac"]["phase_band"]],
            "amp_bands": [truth["pac"]["amp_band"], [150, 200]],
            "n_surrogates": n_surrogates,
            "seed": 1,
            "regions": regions,
        },
        "hfo": {"enabled": True, "band": truth["hfo"]["band"], "regions": regions},
        "connectivity": {
            "enabled": True,
            "bands": {"beta": truth["connectivity"]["band"]},
            "methods": ["wpli", "wpli2_debiased"],
            "regions": regions,
        },
    }


def _run(raw, truth, reports_dir, blocks=ALL_BLOCKS, overrides=None, epochs=None):
    settings = {b: s for b, s in _block_settings(truth).items() if b in blocks}
    for block, extra in (overrides or {}).items():
        settings[block].update(extra)
    task = _DepthTask(
        {
            "run_id": "run",
            "unprocessed_file": Path(reports_dir) / "mouse.set",
            "task": "MouseDepth",
            "reports_dir": Path(reports_dir) / "reports",
        }
    )
    task.raw = raw
    task.epochs = epochs
    task.settings = {"postprocessing_analysis": {"enabled": True, "value": settings}}
    task.results = task.run_postprocessing_analysis()
    return task


@pytest.fixture(scope="module")
def mouse():
    return create_synthetic_mouse_depth_raw()


@pytest.fixture(scope="module")
def full_run(mouse, tmp_path_factory):
    raw, truth = mouse
    return _run(raw, truth, tmp_path_factory.mktemp("run_a"))


def _effect_off(effect, tmp_path, blocks):
    raw, truth = create_synthetic_mouse_depth_raw(effects={effect: False})
    return _run(raw, truth, tmp_path, blocks=blocks), truth


def _pair_row(df, a, b, **match):
    mask = (df["channel_a"] == a) & (df["channel_b"] == b)
    for key, value in match.items():
        mask &= df[key] == value
    rows = df[mask]
    assert len(rows) == 1
    return rows.iloc[0]


def _pac_row(df, channel, amp_low):
    rows = df[(df["phase_channel"] == channel) & (df["amp_low"] == amp_low)]
    assert len(rows) == 1
    return rows.iloc[0]


def _overlaps(events, truth_events, pad=0.05):
    """Return per-event flags: does each detection overlap a truth event?"""
    flags = []
    for _, ev in events.iterrows():
        flags.append(
            any(
                ev["onset_s"] <= t1 + pad and ev["offset_s"] >= t0 - pad
                for t0, t1, _ in truth_events
            )
        )
    return np.array(flags, dtype=bool)


def _hfo_metrics(events, truth_events, duration_s, channel="HPC_2"):
    ev = events[events["channel"] == channel]
    hit = _overlaps(ev, truth_events)
    detected = sum(
        any(
            e["onset_s"] <= t1 + 0.05 and e["offset_s"] >= t0 - 0.05
            for _, e in ev.iterrows()
        )
        for t0, t1, _ in truth_events
    )
    sensitivity = detected / len(truth_events) if truth_events else float("nan")
    fp_per_min = int((~hit).sum()) / duration_s * 60.0
    return sensitivity, fp_per_min


def _psd_peak(psd, channel, lo, hi):
    d = psd[(psd["channel"] == channel) & psd["frequency"].between(lo, hi)]
    return float(d.loc[d["psd"].idxmax(), "frequency"])


# 1. PSD peaks ---------------------------------------------------------------


def test_psd_peaks_match_truth(full_run, mouse):
    _, truth = mouse
    psd = full_run.sensor_psd_result["spectra"]
    theta = _psd_peak(psd, "HPC_1", 5, 10)
    beta = _psd_peak(psd, "CTX_1", 15, 25)
    truth_theta = next(p for p in truth["peaks"] if p["channel"] == "HPC_1")
    truth_beta = next(p for p in truth["peaks"] if p["channel"] == "CTX_1")
    assert abs(theta - truth_theta["freq_hz"]) <= PEAK_TOL_HZ
    assert abs(beta - truth_beta["freq_hz"]) <= PEAK_TOL_HZ


# 2. Aperiodic ---------------------------------------------------------------


def test_aperiodic_exponent_recovered_for_every_channel(full_run, mouse):
    _, truth = mouse
    ap = full_run.fooof_aperiodic_df.set_index("channel")
    assert (ap["status"] == "SUCCESS").all()
    for channel, exponent in truth["aperiodic"]["exponent"].items():
        assert abs(ap.loc[channel, "exponent"] - exponent) <= 0.2, channel


def test_specparam_closer_to_truth_than_polyfit(full_run, mouse):
    _, truth = mouse
    psd = full_run.sensor_psd_result["spectra"]
    d = psd[(psd["channel"] == "HPC_1") & psd["frequency"].between(2, 55)]
    slope = np.polyfit(np.log10(d["frequency"]), np.log10(d["psd"]), 1)[0]
    truth_exp = truth["aperiodic"]["exponent"]["HPC_1"]
    ap = full_run.fooof_aperiodic_df.set_index("channel")
    spec_err = abs(ap.loc["HPC_1", "exponent"] - truth_exp)
    poly_err = abs(-slope - truth_exp)
    assert spec_err < poly_err


# 3. Power-power coupling ----------------------------------------------------


def test_power_power_coupling_coupled_vs_control(full_run, mouse):
    _, truth = mouse
    df = full_run.power_coupling_df
    c = truth["power_coupling"]["coupled"]
    k = truth["power_coupling"]["control"]
    coupled = _pair_row(df, c["channel_a"], c["channel_b"])
    control = _pair_row(df, k["channel_a"], k["channel_b"])
    assert coupled["pearson_r"] > 0.5
    # 120 one-second windows: chance |r| ~ 1.96/sqrt(120) = 0.18, so 0.1 is
    # noise-limited; assert |r| < 0.2 and not significant at 1%.
    assert abs(control["pearson_r"]) < 0.2
    assert control["p_value"] > 0.01


def test_power_power_coupled_pair_ranks_first_of_all_pairs(mouse):
    # A single control pair can hide chance correlations: slow envelopes are
    # autocorrelated, so unrelated pairs reach |r| ~0.35. The coupled pair must
    # beat every one of the 28 channel pairs, not only the named control.
    from autoclean.calc.oscillation_analysis import compute_power_power_coupling

    raw, truth = mouse
    c = truth["power_coupling"]["coupled"]
    df = compute_power_power_coupling(raw, bands={"gamma": c["band_a"]})
    assert len(df) == 28
    ranked = df.reindex(df["pearson_r"].abs().sort_values(ascending=False).index)
    top, runner_up = ranked.iloc[0], ranked.iloc[1]
    assert {top["channel_a"], top["channel_b"]} == {c["channel_a"], c["channel_b"]}
    assert top["pearson_r"] - abs(runner_up["pearson_r"]) > 0.2


# 4. PAC ---------------------------------------------------------------------


def test_pac_theta_gamma_detected_and_controls_clean(full_run, mouse):
    _, truth = mouse
    df = full_run.pac_df
    hit = _pac_row(df, truth["pac"]["channel"], 70.0)
    control = _pac_row(df, truth["pac"]["control_channel"], 70.0)
    high = _pac_row(df, truth["pac"]["channel"], 150.0)
    assert hit["z_score"] > 3 and hit["p_value"] < 0.05
    assert control["z_score"] < 3
    assert high["z_score"] < 3


# 5. HFO ---------------------------------------------------------------------


def test_hfo_sensitivity_and_false_positive_rate(full_run, mouse):
    _, truth = mouse
    events = full_run.hfo_events_df
    sensitivity, fp = _hfo_metrics(events, truth["hfo"]["events"], truth["duration"])
    assert sensitivity >= 0.9
    assert fp <= 1.0
    others = full_run.hfo_summary_df[full_run.hfo_summary_df["channel"] != "HPC_2"]
    assert (others["rate_per_min"] <= 1.0).all()


# 6. Connectivity ------------------------------------------------------------


def test_connectivity_ranks_connected_pair_first(full_run, mouse):
    _, truth = mouse
    df = full_run.connectivity_df
    assert len(df) == 56  # 28 pairs x (wpli, wpli2_debiased)
    df = df[df["method"] == "wpli2_debiased"]
    assert len(df) == 28
    top = df.sort_values("value", ascending=False).iloc[0]
    assert {top["channel_a"], top["channel_b"]} == set(
        truth["connectivity"]["connected"]
    )


# 7. Effect removal ----------------------------------------------------------


def test_removing_peaks_removes_theta_peak(tmp_path):
    task, _ = _effect_off("peaks", tmp_path, ("sensor_psd", "fooof"))
    periodic = task.fooof_periodic_df
    theta = periodic[
        (periodic["channel"] == "HPC_1") & periodic["center_frequency"].between(6, 9)
    ]
    assert theta.empty


def test_removing_power_coupling_removes_correlation(tmp_path):
    task, truth = _effect_off("power_coupling", tmp_path, ("power_coupling",))
    c = truth["power_coupling"]["coupled"]
    row = _pair_row(task.power_coupling_df, c["channel_a"], c["channel_b"])
    assert row["pearson_r"] < 0.3


def test_removing_pac_removes_modulation(tmp_path):
    task, truth = _effect_off("pac", tmp_path, ("pac",))
    assert _pac_row(task.pac_df, truth["pac"]["channel"], 70.0)["z_score"] < 3


def test_removing_hfo_removes_detections(tmp_path):
    task, truth = _effect_off("hfo", tmp_path, ("hfo",))
    n = int((task.hfo_events_df["channel"] == "HPC_2").sum())
    assert n / truth["duration"] * 60.0 <= 1.0


def test_removing_connectivity_removes_top_pair(tmp_path):
    task, truth = _effect_off("connectivity", tmp_path, ("connectivity",))
    df = task.connectivity_df
    df = df[df["method"] == "wpli2_debiased"]
    top = df.sort_values("value", ascending=False).iloc[0]
    pair = {top["channel_a"], top["channel_b"]}
    connected = _pair_row(df, *sorted(truth["connectivity"]["connected"]))
    assert pair != set(truth["connectivity"]["connected"]) or top["value"] < 0.1
    assert connected["value"] < 0.1


# 8. Error paths -------------------------------------------------------------


def test_hfo_rejects_resampled_data_naming_resample_step(mouse, tmp_path):
    raw, truth = mouse
    with pytest.raises(ValueError, match="resample_step"):
        _run(raw.copy().resample(250), truth, tmp_path, blocks=("hfo",))


def test_hfo_rejects_lowpassed_data_naming_h_freq(mouse, tmp_path):
    raw, truth = mouse
    with pytest.raises(ValueError, match="h_freq"):
        _run(raw.copy().filter(None, 100, verbose=False), truth, tmp_path, ("hfo",))


@pytest.mark.parametrize("block", ["power_coupling", "pac", "hfo", "connectivity"])
def test_unmatched_region_prefix_raises(mouse, tmp_path, block):
    raw, truth = mouse
    with pytest.raises(ValueError, match="ZZ_"):
        _run(
            raw,
            truth,
            tmp_path,
            blocks=(block,),
            overrides={block: {"regions": {"nowhere": "ZZ_"}}},
        )


def test_nan_channel_raises(mouse, tmp_path):
    raw, truth = mouse
    bad = raw.copy()
    bad._data[0, 100] = np.nan
    with pytest.raises(ValueError, match="CTX_1"):
        _run(bad, truth, tmp_path, blocks=("hfo",))


def test_table_input_is_rejected(mouse, tmp_path):
    raw, truth = mouse
    with pytest.raises(ValueError, match="Raw or Epochs"):
        _run(
            raw,
            truth,
            tmp_path,
            blocks=("sensor_psd", "hfo"),
            overrides={"hfo": {"input": "sensor_psd"}},
        )


# 9. Integration and determinism ---------------------------------------------


def _csvs(task):
    root = task.config["reports_dir"]
    return {p.relative_to(root): p for p in sorted(root.rglob("*.csv"))}


def test_integration_all_blocks_write_outputs(full_run):
    root = full_run.config["reports_dir"]
    expected = {
        "fooof/mouse_postprocessing_fooof_aperiodic.csv",
        "power_coupling/mouse_postprocessing_power_coupling.csv",
        "pac/mouse_postprocessing_pac.csv",
        "hfo/mouse_postprocessing_hfo_events.csv",
        "hfo/mouse_postprocessing_hfo_summary.csv",
        "connectivity/mouse_postprocessing_connectivity.csv",
    }
    for rel in expected:
        assert (root / rel).stat().st_size > 0, rel

    assert {"region_a", "region_b"} <= set(full_run.power_coupling_df.columns)
    assert {"phase_region", "amp_region"} <= set(full_run.pac_df.columns)
    assert "region" in full_run.hfo_events_df.columns
    assert "region" in full_run.hfo_summary_df.columns
    assert {"region_a", "region_b"} <= set(full_run.connectivity_df.columns)
    assert set(full_run.pac_df["phase_region"]) == {"cortex", "hippocampus"}

    import json

    resolved = json.loads(
        (root / "postprocessing_analysis" / "resolved_settings.json").read_text()
    )
    assert [r["block"] for r in resolved] == list(ALL_BLOCKS)
    by_block = {r["block"]: r for r in resolved}
    assert by_block["hfo"]["method"] == "rms_robust_threshold"
    assert by_block["pac"]["method"] == "tort_modulation_index"
    assert by_block["hfo"]["n_events"] == len(full_run.hfo_events_df)
    assert by_block["connectivity"]["rows"] == 56


def test_outputs_are_deterministic(full_run, mouse, tmp_path):
    raw, truth = mouse
    second = _run(raw, truth, tmp_path)
    first_files, second_files = _csvs(full_run), _csvs(second)
    assert first_files.keys() == second_files.keys()
    assert first_files
    for rel, path in first_files.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashlib.sha256(second_files[rel].read_bytes()).hexdigest(), rel


# 10. Connectivity specificity and noise floor -------------------------------

CONN_BANDS = {"beta": [15, 25], "low_gamma": [30, 45], "high_gamma": [60, 90]}


@pytest.fixture(scope="module")
def band_run(mouse, tmp_path_factory):
    raw, truth = mouse
    return _run(
        raw,
        truth,
        tmp_path_factory.mktemp("run_bands"),
        blocks=("connectivity",),
        overrides={"connectivity": {"bands": CONN_BANDS}},
    )


def _conn(df, band, method="wpli2_debiased"):
    return df[(df["band"] == band) & (df["method"] == method)]


def _is_connected(row, truth):
    return {row["channel_a"], row["channel_b"]} == set(
        truth["connectivity"]["connected"]
    )


def test_connectivity_band_specificity(band_run, mouse):
    _, truth = mouse
    df = band_run.connectivity_df
    for method in ("wpli", "wpli2_debiased"):
        beta = _conn(df, "beta", method).sort_values("value", ascending=False)
        assert _is_connected(beta.iloc[0], truth), method
    for band in ("low_gamma", "high_gamma"):
        d = _conn(df, band)
        mask = d.apply(lambda r: _is_connected(r, truth), axis=1)
        assert mask.sum() == 1
        assert d[mask]["value"].iloc[0] <= d[~mask]["value"].max() + 0.05, band


def test_connectivity_noise_floor_in_beta(full_run, mouse):
    _, truth = mouse
    d = _conn(full_run.connectivity_df, "beta")
    mask = d.apply(lambda r: _is_connected(r, truth), axis=1)
    assert mask.sum() == 1
    assert d[mask]["value"].iloc[0] > 0.5
    assert (d[~mask]["value"] < 0.1).all()


# 11. BAD annotations through the runner -------------------------------------


def _transient_raw(raw, annotate):
    """Copy raw and add 10 large 20 ms transients on HPC_3 at known times."""
    out = raw.copy()
    sfreq = out.info["sfreq"]
    ch = out.ch_names.index("HPC_3")
    times = np.linspace(10.0, 110.0, 10)
    n_t = int(round(0.02 * sfreq))
    window = np.hanning(n_t)
    for t in times:
        i0 = int(round(t * sfreq))
        out._data[ch, i0 : i0 + n_t] += (
            2e-3 * window * np.sin(2 * np.pi * 200.0 * np.arange(n_t) / sfreq)
        )
    if annotate:
        out.set_annotations(
            out.annotations
            + mne.Annotations(
                onset=times - 0.1,
                duration=[0.22] * len(times),
                description=["BAD_artifact"] * len(times),
            )
        )
    return out, times


def _hfo_hits(events, times, channel="HPC_3"):
    ev = events[events["channel"] == channel]
    return sum(
        bool(((ev["onset_s"] <= t + 0.03) & (ev["offset_s"] >= t - 0.01)).any())
        for t in times
    )


def test_bad_annotations_exclude_transients_from_hfo(mouse, tmp_path):
    raw, truth = mouse
    marked, times = _transient_raw(raw, annotate=True)
    task = _run(marked, truth, tmp_path, blocks=("hfo",))
    events = task.hfo_events_df
    ev3 = events[events["channel"] == "HPC_3"]
    for t in times:
        overlap = (ev3["onset_s"] <= t + 0.1) & (ev3["offset_s"] >= t - 0.1)
        assert not overlap.any(), t
    sensitivity, _ = _hfo_metrics(events, truth["hfo"]["events"], truth["duration"])
    assert sensitivity >= 0.9
    summary = task.hfo_summary_df.set_index("channel")
    bad_s = 0.22 * len(times)
    full = truth["duration"]
    got = summary.loc["HPC_3", "analyzed_duration_s"]
    assert abs((full - got) - bad_s) < 0.1


def test_unannotated_transients_are_detected_control(mouse, tmp_path):
    raw, truth = mouse
    unmarked, times = _transient_raw(raw, annotate=False)
    task = _run(unmarked, truth, tmp_path, blocks=("hfo",))
    assert _hfo_hits(task.hfo_events_df, times) >= 8


# 12. Epochs input -----------------------------------------------------------


def test_epochs_input_through_runner(mouse, tmp_path):
    # Ripples that straddle a 2 s epoch edge are cut or lost, so HFO
    # sensitivity is held to 0.8 here instead of 0.9.
    raw, truth = mouse
    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, preload=True, verbose=False
    )
    task = _run(
        raw,
        truth,
        tmp_path,
        blocks=("pac", "hfo", "connectivity"),
        overrides={
            b: {"input": "clean_epochs"} for b in ("pac", "hfo", "connectivity")
        },
        epochs=epochs,
    )
    assert _pac_row(task.pac_df, truth["pac"]["channel"], 70.0)["z_score"] > 3
    sensitivity, _ = _hfo_metrics(
        task.hfo_events_df, truth["hfo"]["events"], truth["duration"]
    )
    assert sensitivity >= 0.8
    assert {"epoch_index", "onset_in_epoch_s"} <= set(task.hfo_events_df.columns)
    d = _conn(task.connectivity_df, "beta")
    top = d.sort_values("value", ascending=False).iloc[0]
    assert _is_connected(top, truth)


# 13. PAC preferred phase ----------------------------------------------------


def test_pac_preferred_phase_matches_truth(full_run, mouse):
    # Fixture: amplitude = 1 + depth * cos(theta phase), so it peaks at the
    # theta peak (0 rad).
    _, truth = mouse
    row = _pac_row(full_run.pac_df, truth["pac"]["channel"], 70.0)
    phase = float(row["preferred_phase_rad"])
    diff = np.angle(np.exp(1j * (phase - 0.0)))
    assert abs(diff) <= 0.6, phase
