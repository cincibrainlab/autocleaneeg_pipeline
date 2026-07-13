import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import autoclean.calc.assr_runner as assr_runner
from autoclean.calc import assr_analysis


class _FakeTFR:
    def __init__(self, data, times=None):
        self.data = np.asarray(data, dtype=float)
        self.times = np.asarray(times if times is not None else [0.0, 0.1, 0.2])
        self.saved_to = []

    def save(self, output_file, overwrite=False):
        self.saved_to.append((Path(output_file), overwrite))
        Path(output_file).write_text("saved")


class _FakeEpochs:
    filename = None

    def __init__(self):
        self.ch_names = ["Cz", "HEOG", "EKG", "Status"]
        self.drop_log = [(), ("BAD",)]

    def get_channel_types(self):
        return ["eeg", "eog", "ecg", "misc"]


def _fake_tf_data():
    freqs = np.array([4, 8, 12, 35, 40, 50, 70, 80, 90], dtype=float)
    times = np.array([0.0, 0.1, 0.2, 0.6], dtype=float)
    avg_data = np.ones((4, len(freqs), len(times)))
    trial_data = np.ones((2, 4, len(freqs), len(times)))
    return {
        "power": _FakeTFR(avg_data, times),
        "itc": (_FakeTFR(avg_data, times), _FakeTFR(avg_data * 2, times)),
        "ersp": _FakeTFR(avg_data * 3, times),
        "single_trial_power": _FakeTFR(trial_data * 4, times),
        "freqs": freqs,
    }


def _legacy_tf_data():
    freqs = np.array([4, 5, 8, 10, 12, 35, 40, 45, 75, 80, 85], dtype=float)
    times = np.array([0.0, 0.1, 0.2, 2.9], dtype=float)
    avg_data = np.arange(len(freqs) * len(times), dtype=float).reshape(
        1, len(freqs), len(times)
    )
    trial_data = np.stack([avg_data + 100, avg_data + 200], axis=0)
    return {
        "power": _FakeTFR(avg_data, times),
        "itc": (_FakeTFR(avg_data, times), _FakeTFR(avg_data + 1000, times)),
        "ersp": _FakeTFR(avg_data + 2000, times),
        "single_trial_power": _FakeTFR(trial_data),
        "freqs": freqs,
    }


class _SingleChannelEpochs:
    filename = None
    ch_names = ["Cz"]
    drop_log = [(), ("BAD",), ()]


def test_compute_metrics_preserves_legacy_default_columns_and_values():
    tf_data = _legacy_tf_data()
    results = assr_analysis.compute_metrics(tf_data, _SingleChannelEpochs())
    row = results.iloc[0]

    assert results.columns.tolist() == [
        "eegid",
        "trials",
        "chan",
        "rejtrials",
        "stp_gamma",
        "stp_gamma1",
        "stp_gamma2",
        "stp_alpha",
        "stp_theta",
        "ersp_gamma",
        "ersp_gamma1",
        "ersp_gamma2",
        "ersp_alpha",
        "ersp_theta",
        "itc40",
        "itc80",
        "itconset",
        "itcoffset",
    ]
    assert row["eegid"] == "synthetic_epochs"
    assert row["trials"] == 3
    assert row["chan"] == "Cz"
    assert row["rejtrials"] == 1

    freqs = tf_data["freqs"]
    times = tf_data["itc"][1].times
    itc_data = tf_data["itc"][1].data[0]
    stp_data = tf_data["single_trial_power"].data[:, 0]
    ersp_data = tf_data["ersp"].data[0]

    def freq_idx(fmin, fmax):
        return np.where((freqs >= fmin) & (freqs <= fmax))[0]

    def time_idx(tmin, tmax):
        return np.where((times >= tmin) & (times <= tmax))[0]

    all_time = time_idx(0, 3.0)
    onset_time = time_idx(0.092, 0.308)
    offset_time = time_idx(2.8, 3.0)

    expected = {
        "itc40": np.mean(itc_data[freq_idx(35, 45)][:, all_time]),
        "itc80": np.mean(itc_data[freq_idx(75, 85)][:, all_time]),
        "itconset": np.mean(itc_data[freq_idx(2, 13)][:, onset_time]),
        "itcoffset": np.mean(itc_data[freq_idx(2, 13)][:, offset_time]),
        "stp_alpha": np.mean(stp_data[:, freq_idx(8, 13), :]),
        "stp_theta": np.mean(stp_data[:, freq_idx(4, 7), :]),
        "stp_gamma1": np.mean(stp_data[:, freq_idx(30, 55), :]),
        "stp_gamma2": np.mean(stp_data[:, freq_idx(65, 80), :]),
        "stp_gamma": np.mean(stp_data[:, freq_idx(30, 80), :]),
        "ersp_alpha": np.mean(ersp_data[freq_idx(8, 13), :]),
        "ersp_theta": np.mean(ersp_data[freq_idx(4, 7), :]),
        "ersp_gamma1": np.mean(ersp_data[freq_idx(30, 55), :]),
        "ersp_gamma2": np.mean(ersp_data[freq_idx(65, 80), :]),
        "ersp_gamma": np.mean(ersp_data[freq_idx(30, 80), :]),
    }
    for column, value in expected.items():
        assert row[column] == value


def test_compute_metrics_skips_combined_band_name_collisions():
    audit = {"skipped_freq_bands": [], "skipped_time_windows": []}

    with pytest.warns(UserWarning, match="name_conflicts_with_freq_bands"):
        results = assr_analysis.compute_metrics(
            _legacy_tf_data(),
            _SingleChannelEpochs(),
            freq_bands={"alpha": (8, 13)},
            combined_bands={"alpha": (75, 85)},
            audit=audit,
        )

    row = results.iloc[0]
    freqs = _legacy_tf_data()["freqs"]
    stp_data = _legacy_tf_data()["single_trial_power"].data[:, 0]
    alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
    combined_idx = np.where((freqs >= 75) & (freqs <= 85))[0]

    assert row["stp_alpha"] == np.mean(stp_data[:, alpha_idx, :])
    assert row["stp_alpha"] != np.mean(stp_data[:, combined_idx, :])
    assert audit["skipped_combined_bands"] == [
        {"name": "alpha", "reason": "name_conflicts_with_freq_bands"}
    ]


def test_resolve_assr_epochs_profile_and_overrides():
    settings = assr_analysis.resolve_assr_analysis_settings(
        profile="assr_epochs",
        overrides={"save_tfr": True, "time_windows": {"all": (0.0, 0.5)}},
    )

    assert settings["profile"] == "assr_epochs"
    assert settings["save_tfr"] is True
    assert settings["time_windows"]["all"] == (0.0, 0.5)
    assert settings["time_windows"]["itc_onset"] == (0.092, 0.308)
    assert settings["combined_bands"]["gamma_combined"] == [(30, 55), (65, 80)]


def test_resolve_assr_analysis_settings_rejects_unknown_profile():
    with pytest.raises(ValueError, match="Unknown ASSR analysis profile 'unknown'"):
        assr_analysis.resolve_assr_analysis_settings(profile="unknown")


def test_resolve_assr_analysis_config_uses_embedded_profile():
    settings = assr_analysis.resolve_assr_analysis_config(
        analysis_config={"profile": "assr_epochs", "save_tfr": True}
    )

    assert settings["profile"] == "assr_epochs"
    assert settings["save_tfr"] is True
    assert settings["time_windows"]["itc_onset"] == (0.092, 0.308)


def test_compute_metrics_excludes_non_eeg_and_records_skips():
    audit = {"skipped_freq_bands": [], "skipped_time_windows": []}

    results = assr_analysis.compute_metrics(
        _fake_tf_data(),
        _FakeEpochs(),
        freq_bands={"alpha": (8, 13), "missing": None, "too_high": (120, 130)},
        time_windows={"all": (0, 0.2), "bad_window": (3, 4)},
        combined_bands={"gamma_combined": [(35, 40), (70, 80)]},
        exclude_channel_types=["eog", "ecg", "misc"],
        audit=audit,
    )

    assert results["chan"].tolist() == ["Cz"]
    assert "stp_gamma_combined" in results.columns
    assert "ersp_gamma_combined" in results.columns
    assert audit["excluded_channels"] == [
        {"channel": "HEOG", "type": "eog"},
        {"channel": "EKG", "type": "ecg"},
        {"channel": "Status", "type": "misc"},
    ]
    assert {item["name"] for item in audit["skipped_freq_bands"]} >= {
        "missing",
        "too_high",
    }
    assert audit["skipped_time_windows"] == [
        {"name": "bad_window", "reason": "outside_epoch_range"}
    ]


def test_write_analysis_metadata_splits_settings_and_log_payloads(tmp_path):
    settings = {
        "profile": "assr_epochs",
        "save_tfr": True,
        "freq_bands": {"alpha": np.array([8, 13])},
    }
    audit = {
        "profile": "assr_epochs",
        "skipped_combined_bands": [
            {"name": "alpha", "reason": "name_conflicts_with_freq_bands"}
        ],
    }

    assr_analysis._write_analysis_metadata(tmp_path, "subject01", settings, audit)

    settings_payload = json.loads(
        (tmp_path / "subject01_assr_analysis_settings.json").read_text(encoding="utf8")
    )
    log_payload = json.loads(
        (tmp_path / "subject01_assr_analysis_log.json").read_text(encoding="utf8")
    )

    assert settings_payload == {
        "analysis_settings": {
            "profile": "assr_epochs",
            "save_tfr": True,
            "freq_bands": {"alpha": [8, 13]},
        }
    }
    assert "analysis_audit" not in settings_payload
    assert log_payload["analysis_audit"] == audit
    assert log_payload["analysis_profile"] == "assr_epochs"
    assert log_payload["saved_tfr"] is True
    assert "analysis_settings" not in log_payload


def test_analyze_assr_writes_settings_and_saves_tfr_when_requested():
    epochs = _FakeEpochs()
    tf_data = _fake_tf_data()

    with (
        patch.object(
            assr_analysis, "compute_time_frequency", return_value=tf_data
        ) as compute,
        patch.object(assr_analysis.Path, "mkdir") as mkdir,
        patch("autoclean.calc.assr_analysis.pd.DataFrame.to_csv") as to_csv,
        patch.object(assr_analysis, "_save_tfr_artifacts") as save_tfr,
        patch.object(assr_analysis, "_write_analysis_metadata") as write_metadata,
    ):
        result = assr_analysis.analyze_assr(
            output_dir="out",
            save_results=True,
            epochs=epochs,
            file_basename="subject01",
            analysis_profile="assr_epochs",
            analysis_config={"save_tfr": True},
        )

    compute.assert_called_once_with(epochs, baseline=(-0.2, 0.0))
    assert mkdir.call_count >= 1
    to_csv.assert_called_once()
    save_tfr.assert_called_once()
    write_metadata.assert_called_once()
    assert result["analysis_settings"]["save_tfr"] is True


def test_runner_main_forwards_analysis_config_json():
    argv = [
        "assr_runner.py",
        "input.set",
        "--output_dir",
        "out",
        "--analysis_type",
        "analysis_only",
        "--analysis_profile",
        "assr_epochs",
        "--analysis_config",
        '{"save_tfr": true}',
    ]

    with (
        patch.object(assr_runner.sys, "argv", argv),
        patch.object(assr_runner.Path, "mkdir"),
        patch.object(assr_runner, "analyze_assr") as analyze,
    ):
        analyze.return_value = {"ok": True}
        assr_runner.main()

    _, kwargs = analyze.call_args
    assert kwargs["analysis_profile"] == "assr_epochs"
    assert kwargs["analysis_config"] == {"save_tfr": True}


def test_runner_passes_analysis_config_to_analyze_assr():
    with (
        patch.object(assr_runner.Path, "mkdir"),
        patch.object(assr_runner, "analyze_assr") as analyze,
    ):
        analyze.return_value = {"ok": True}
        result = assr_runner.run_analysis_only(
            "input.set",
            "out",
            analysis_profile="assr_epochs",
            analysis_config={"save_tfr": True},
        )

    assert result == {"ok": True}
    _, kwargs = analyze.call_args
    assert kwargs["analysis_profile"] == "assr_epochs"
    assert kwargs["analysis_config"] == {"save_tfr": True}
