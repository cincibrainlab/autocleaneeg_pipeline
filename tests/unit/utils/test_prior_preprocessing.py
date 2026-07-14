from __future__ import annotations

import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import autoclean.utils.prior_preprocessing as prior_preprocessing_module
from autoclean.utils.prior_preprocessing import (
    build_prior_preprocessing_dataset_summary,
    detect_prior_preprocessing,
    resolve_prior_preprocessing_provenance,
    write_prior_preprocessing_artifacts,
)


class _RawStub:
    def __init__(self, data, sfreq=256.0, ch_names=None, channel_types=None):
        self._data = np.asarray(data, dtype=float)
        self.info = {"sfreq": sfreq, "custom_ref_applied": "A1/A2"}
        self.ch_names = ch_names or ["Cz", "Pz", "VEOG", "Status"]
        self._channel_types = channel_types or ["eeg", "eeg", "eog", "misc"]
        self.n_times = self._data.shape[-1]
        self.times = np.arange(self.n_times) / sfreq

    def get_data(self):
        return self._data

    def get_channel_types(self):
        return self._channel_types


class _EpochsStub(_RawStub):
    def __init__(self, data, sfreq=128.0):
        super().__init__(data, sfreq=sfreq, ch_names=["Cz", "Pz"])
        self.tmin = -0.2
        self.tmax = 0.8
        self.times = np.linspace(self.tmin, self.tmax, self._data.shape[-1])
        self.events = np.array([[0, 0, 11], [128, 0, 22], [256, 0, 11]])

    def __len__(self):
        return self._data.shape[0]


def _provenance_summary():
    return {
        "documented_provenance": {
            "srate": 500,
            "nbchan": 63,
            "trials": 120,
            "epoch_window": {"xmin": -0.2, "xmax": 0.8},
            "reference": "A1/A2",
            "history": "pop_eegfiltnew highpass; notch 60 Hz; pop_rmbase; runica; pop_epoch",
            "comments": "ICA components manually rejected.",
            "etc_keys": ["interpolated_channels"],
            "channels": {
                "labels": ["Cz", "Pz"],
                "types": {"EEG": 2},
            },
            "events": {
                "labels": ["standard", "target"],
                "codes": ["11", "22"],
                "counts": {"standard": 2, "target": 1},
            },
            "ica": {"icaweights": {"present": True, "shape": [62, 63]}},
            "iclabel": {"present": True},
            "interpolation": {"interpolated_channels": ["Oz"]},
        },
        "summary_row": {
            "source_file": "sub-01.set",
            "srate": 500,
            "nbchan": 63,
            "trials": 120,
            "epoch_window": '{"xmax": 0.8, "xmin": -0.2}',
            "reference": "A1/A2",
            "event_codes": '["11", "22"]',
            "event_counts": '{"standard": 2, "target": 1}',
        },
    }


def _notch_like_data(sfreq=256.0, n_times=1024):
    times = np.arange(n_times) / sfreq
    signal = np.sin(2 * np.pi * 55 * times) + np.sin(2 * np.pi * 65 * times)
    return np.vstack([signal, signal * 0.5])


def _write_dataset_summary(
    output_dir: str, source_file: str, warnings: list[str]
) -> None:
    summary = {
        "summary_row": {"source_file": source_file},
        "warnings": warnings,
        "artifact_paths": {},
    }
    write_prior_preprocessing_artifacts(
        summary, Path(output_dir), Path(source_file).stem
    )


def test_detect_prior_preprocessing_uses_202_documented_schema():
    raw = _RawStub(_notch_like_data())

    summary = detect_prior_preprocessing(
        raw,
        import_metadata={"file_format": "EEGLAB_SET", "plugin_used": "EEGLAB"},
        provenance_summary=_provenance_summary(),
    )

    assert summary["documented_metadata"]["reference"] == "A1/A2"
    assert summary["findings"]["reference"]["confidence"] == "documented"
    assert summary["findings"]["ica_present"]["confidence"] == "documented"
    assert summary["findings"]["notch_filter_60hz"]["confidence"] == "documented"
    assert summary["findings"]["baseline_applied"]["confidence"] == "documented"
    assert summary["summary_row"]["source_file"] == "sub-01.set"


def test_epoch_window_is_canonical_across_provenance_and_local_paths():
    raw = _EpochsStub(np.zeros((3, 2, 64)))
    provenance = _provenance_summary()
    full = detect_prior_preprocessing(raw, provenance_summary=provenance)
    compact = detect_prior_preprocessing(
        raw,
        provenance_summary={
            "summary_row": {
                "source_file": "compact.set",
                "epoch_window": '{"xmax": 0.8, "xmin": -0.2}',
            }
        },
    )
    local = detect_prior_preprocessing(raw)

    expected = {"tmin": -0.2, "tmax": 0.8}
    assert full["documented_metadata"]["epoch_window"] == expected
    assert compact["documented_metadata"]["epoch_window"] == expected
    assert local["documented_metadata"]["epoch_window"] == expected
    assert full["summary_row"]["epoch_window"] == expected
    assert compact["summary_row"]["epoch_window"] == expected
    assert local["summary_row"]["epoch_window"] == expected

    dataset = build_prior_preprocessing_dataset_summary([full, compact, local])
    assert not any("epoch_window" in warning for warning in dataset["warnings"])


def test_documented_filter_cutoffs_do_not_imply_notch_filtering():
    provenance = _provenance_summary()
    provenance["documented_provenance"] = {
        **provenance["documented_provenance"],
        "history": "Applied 1-60Hz bandpass; later 0.1-150Hz filter.",
    }

    summary = detect_prior_preprocessing(
        _RawStub(np.zeros((2, 512)), sfreq=512.0),
        provenance_summary=provenance,
    )

    assert summary["findings"]["notch_filter_60hz"]["confidence"] != "documented"
    assert summary["findings"]["notch_filter_50hz"]["confidence"] != "documented"
    assert summary["findings"]["notch_filter_100hz"]["confidence"] != "documented"
    assert summary["findings"]["notch_filter_120hz"]["confidence"] != "documented"


def test_documented_highpass_history_does_not_imply_lowpass_filtering():
    provenance = _provenance_summary()
    provenance["documented_provenance"] = {
        **provenance["documented_provenance"],
        "history": "pop_eegfiltnew EEG locutoff=1 hicutoff=0; highpass only.",
    }

    summary = detect_prior_preprocessing(
        _RawStub(np.zeros((2, 512)), sfreq=512.0),
        provenance_summary=provenance,
    )

    assert summary["findings"]["highpass_filter"]["confidence"] == "documented"
    assert summary["findings"]["lowpass_filter"]["confidence"] != "documented"


def test_provenance_resolution_prefers_full_plugin_over_compact_extracted():
    full = _provenance_summary()
    compact = {"summary_row": full["summary_row"]}

    resolved = resolve_prior_preprocessing_provenance(
        {"eeglab_provenance": full}, {}, compact
    )

    assert resolved is full


def test_provenance_resolution_uses_extracted_when_plugin_is_absent():
    extracted = _provenance_summary()

    resolved = resolve_prior_preprocessing_provenance({}, {}, extracted)

    assert resolved is extracted


def test_provenance_resolution_uses_full_extracted_over_compact_plugin():
    extracted = _provenance_summary()
    compact = {"summary_row": extracted["summary_row"]}

    resolved = resolve_prior_preprocessing_provenance(
        {"eeglab_provenance": compact}, {}, extracted
    )

    assert resolved is extracted


def test_provenance_resolution_keeps_explicit_caller_precedence():
    direct = {"summary_row": {"source_file": "caller.set"}}

    resolved = resolve_prior_preprocessing_provenance(
        {"eeglab_provenance": _provenance_summary()},
        {"eeglab_provenance": direct},
        _provenance_summary(),
    )

    assert resolved is direct


def test_boolean_custom_reference_flag_is_not_reported_as_label():
    raw = _RawStub(np.zeros((2, 128)))
    raw.info["custom_ref_applied"] = True

    summary = detect_prior_preprocessing(raw)

    assert summary["documented_metadata"]["reference"] == "custom reference applied"
    assert summary["findings"]["reference"]["value"] is not True


def test_info_description_is_not_treated_as_reference_metadata():
    raw = _RawStub(np.zeros((2, 128)))
    raw.info["custom_ref_applied"] = False
    raw.info["description"] = "Participant completed resting-state recording"

    summary = detect_prior_preprocessing(raw)

    assert summary["documented_metadata"]["reference"] == "unavailable"


def test_signal_inference_flags_likely_notch_and_aux_channels():
    raw = _RawStub(_notch_like_data())

    summary = detect_prior_preprocessing(raw)

    assert summary["signal_inference"]["notch_filter_60hz"]["confidence"] == "likely"
    aux = summary["findings"]["eog_ecg_misc_channel_presence"]
    assert aux["confidence"] == "likely"
    assert aux["value"] == {"eog": 1, "misc": 1}


def test_signal_inference_does_not_infer_baseline_from_raw_time_zero():
    data = np.ones((4, 128))
    data[:, 0] = [-1.0, 1.0, -1.0, 1.0]
    raw = _RawStub(data)

    summary = detect_prior_preprocessing(raw)

    baseline = summary["signal_inference"]["baseline_applied"]
    assert baseline["confidence"] == "unknown"
    assert baseline["evidence"] == "no pre-stimulus baseline window"


def test_signal_only_notch_inference_reaches_summary_and_warnings():
    raw = _RawStub(_notch_like_data())
    task_config = {
        "filtering": {
            "enabled": True,
            "value": {"notch_freqs": [60]},
        }
    }

    summary = detect_prior_preprocessing(raw, task_config=task_config)

    assert summary["summary_row"]["notch_filter_60hz"] == "likely"
    assert summary["warnings"] == [
        "Task notch filtering may repeat prior notch at 60 Hz"
    ]


def test_signal_inference_computes_mean_spectrum_once(monkeypatch):
    raw = _RawStub(_notch_like_data())
    original = prior_preprocessing_module._mean_spectrum
    calls = []

    def counting_mean_spectrum(data, sfreq):
        calls.append((data, sfreq))
        return original(data, sfreq)

    monkeypatch.setattr(
        prior_preprocessing_module, "_mean_spectrum", counting_mean_spectrum
    )

    detect_prior_preprocessing(raw)

    assert len(calls) == 1


def test_task_config_warnings_are_non_blocking_unless_strict():
    raw = _EpochsStub(np.zeros((3, 2, 64)))
    task_config = {
        "filtering": {
            "enabled": True,
            "value": {"l_freq": 1.0, "h_freq": 80.0, "notch_freqs": [60]},
        },
        "epoch_settings": {
            "enabled": True,
            "event_id": {"rare": 99},
            "remove_baseline": {"enabled": True, "window": [-0.2, 0.0]},
        },
        "ICA": {"enabled": True, "value": {"method": "infomax"}},
        "reference_step": {"enabled": True, "value": "average"},
    }

    summary = detect_prior_preprocessing(
        raw,
        provenance_summary=_provenance_summary(),
        task_config=task_config,
        strict=False,
    )
    strict_summary = detect_prior_preprocessing(
        raw,
        provenance_summary=_provenance_summary(),
        task_config=task_config,
        strict=True,
    )

    assert any("notch" in warning for warning in summary["warnings"])
    assert any("baseline" in warning for warning in summary["warnings"])
    assert any("ICA" in warning for warning in summary["warnings"])
    assert any("event codes" in warning for warning in summary["warnings"])
    assert summary["strict_violations"] == []
    assert strict_summary["strict_violations"] == strict_summary["warnings"]


def test_dataset_summary_counts_warnings():
    first = detect_prior_preprocessing(
        _RawStub(_notch_like_data()), provenance_summary=_provenance_summary()
    )
    second = detect_prior_preprocessing(
        _RawStub(_notch_like_data(sfreq=512.0), sfreq=512.0),
        import_metadata={"sampleRate": 512},
    )
    first["warnings"] = ["Task ICA may repeat prior ICA decomposition"]
    second["warnings"] = ["Task ICA may repeat prior ICA decomposition"]

    dataset = build_prior_preprocessing_dataset_summary([first, second])

    assert len(dataset["rows"]) == 2
    assert dataset["warning_counts"] == {
        "Task ICA may repeat prior ICA decomposition": 2
    }


def test_dataset_summary_source_replacement_is_idempotent(tmp_path):
    warning = "Task ICA may repeat prior ICA decomposition"

    _write_dataset_summary(str(tmp_path), "sub-01.set", [warning])
    _write_dataset_summary(str(tmp_path), "sub-01.set", [warning])

    dataset_path = tmp_path / "prior_preprocessing_dataset_summary.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["rows"] == [{"source_file": "sub-01.set"}]
    assert dataset["warning_counts"] == {warning: 1}
    assert dataset["warning_counts_by_source"] == {"sub-01.set": {warning: 1}}


def test_dataset_summary_missing_source_replacement_is_idempotent(tmp_path):
    warning = "warning for unknown source"
    summary = {
        "summary_row": {"sampling_rate": 250},
        "warnings": [warning],
        "artifact_paths": {},
    }

    write_prior_preprocessing_artifacts(summary, tmp_path, "unknown")
    write_prior_preprocessing_artifacts(summary, tmp_path, "unknown")

    dataset_path = tmp_path / "prior_preprocessing_dataset_summary.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["rows"] == [{"sampling_rate": 250}]
    assert dataset["warning_counts"] == {warning: 1}
    assert len(dataset["warning_counts_by_source"]) == 1
    contribution_key = next(iter(dataset["warning_counts_by_source"]))
    assert contribution_key.startswith("__missing_source__:")
    assert dataset["warning_counts_by_source"][contribution_key] == {warning: 1}


def test_named_source_cannot_collide_with_anonymous_contribution_key(tmp_path):
    anonymous = {
        "summary_row": {"sampling_rate": 250},
        "warnings": ["anonymous warning"],
        "artifact_paths": {},
    }
    write_prior_preprocessing_artifacts(anonymous, tmp_path, "anonymous")
    dataset_path = tmp_path / "prior_preprocessing_dataset_summary.json"
    initial = json.loads(dataset_path.read_text(encoding="utf-8"))
    anonymous_key = next(iter(initial["warning_counts_by_source"]))

    named = {
        "summary_row": {"source_file": anonymous_key, "sampling_rate": 500},
        "warnings": ["named warning"],
        "artifact_paths": {},
    }
    write_prior_preprocessing_artifacts(named, tmp_path, "named")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert len(dataset["rows"]) == 2
    assert {row.get("source_file") for row in dataset["rows"]} == {
        None,
        anonymous_key,
    }
    assert dataset["warning_counts"] == {
        "anonymous warning": 1,
        "named warning": 1,
    }
    assert dataset["warning_counts_by_source"][anonymous_key] == {
        "anonymous warning": 1
    }
    escaped_named_key = f"__named_source__:{anonymous_key}"
    assert dataset["warning_counts_by_source"][escaped_named_key] == {
        "named warning": 1
    }


def test_dataset_summary_serialization_order_is_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    for source in ("sub-02.set", "sub-01.set"):
        _write_dataset_summary(str(first_dir), source, [f"warning {source}"])
    for source in ("sub-01.set", "sub-02.set"):
        _write_dataset_summary(str(second_dir), source, [f"warning {source}"])

    filename = "prior_preprocessing_dataset_summary.json"
    assert (first_dir / filename).read_text(encoding="utf-8") == (
        second_dir / filename
    ).read_text(encoding="utf-8")


def test_dataset_summary_reprocessing_legacy_source_resets_unattributed_counts(
    tmp_path,
):
    dataset_path = tmp_path / "prior_preprocessing_dataset_summary.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "rows": [
                    {"source_file": "untouched.set"},
                    {"source_file": "legacy.set"},
                ],
                "warning_counts": {"legacy warning": 2},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    _write_dataset_summary(str(tmp_path), "legacy.set", ["current warning"])

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert dataset["rows"] == [
        {"source_file": "legacy.set"},
        {"source_file": "untouched.set"},
    ]
    assert dataset["warning_counts"] == {"current warning": 1}
    assert dataset["warning_counts_by_source"] == {"legacy.set": {"current warning": 1}}
    assert dataset["warning_counts_migration"] == {
        "status": "legacy_aggregate_reset",
        "reason": "per-source warning contributions were unavailable",
        "warning_counts_complete": False,
        "warning_counts_scope": "partial_current_sources_only",
    }


def test_windows_lock_retries_then_succeeds(tmp_path, monkeypatch):
    attempts = []
    times = iter([0.0, 0.0])

    def locking(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise OSError("busy")

    monkeypatch.setattr(
        prior_preprocessing_module.time, "monotonic", lambda: next(times)
    )
    monkeypatch.setattr(prior_preprocessing_module.time, "sleep", lambda _: None)
    with (tmp_path / "summary.lock").open("w+b") as lock_file:
        prior_preprocessing_module._acquire_windows_lock(
            lock_file, locking, 2, timeout=1.0
        )

    assert len(attempts) == 2


def test_windows_lock_timeout_is_clear(tmp_path, monkeypatch):
    times = iter([0.0, 0.0, 1.0])

    def locking(*args):
        raise OSError("busy")

    monkeypatch.setattr(
        prior_preprocessing_module.time, "monotonic", lambda: next(times)
    )
    monkeypatch.setattr(prior_preprocessing_module.time, "sleep", lambda _: None)
    with (
        (tmp_path / "summary.lock").open("w+b") as lock_file,
        pytest.raises(TimeoutError, match="dataset summary lock"),
    ):
        prior_preprocessing_module._acquire_windows_lock(
            lock_file, locking, 2, timeout=0.5
        )


def test_dataset_summary_concurrent_writers_preserve_all_rows(tmp_path):
    sources = [f"sub-{index:02d}.set" for index in range(12)]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                _write_dataset_summary, str(tmp_path), source, ["shared warning"]
            )
            for source in sources
        ]
        for future in futures:
            future.result(timeout=30)

    dataset_path = tmp_path / "prior_preprocessing_dataset_summary.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert {row["source_file"] for row in dataset["rows"]} == set(sources)
    assert dataset["warning_counts"] == {"shared warning": len(sources)}


class _ImportPathPlugin:
    def __init__(self, raw):
        self._raw = raw

    def import_and_configure(self, file_path, autoclean_dict, preload=True):
        return self._raw

    def process_events(self, raw):
        return None, None, None

    def get_metadata(self):
        provenance = _provenance_summary()
        return {
            "plugin_name": self.__class__.__name__,
            "eeglab_provenance": provenance,
        }


def _load_import_module():
    module_path = (
        Path(__file__).resolve().parents[3] / "src" / "autoclean" / "io" / "import_.py"
    )
    spec = importlib.util.spec_from_file_location(
        "autoclean_io_import_for_prior_preprocessing_test", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_eeg_attaches_prior_preprocessing_metadata_and_artifacts(tmp_path):
    import_module = _load_import_module()
    raw = _RawStub(_notch_like_data(), sfreq=500.0, ch_names=["Cz", "Pz"])
    input_file = tmp_path / "sub-01.set"
    input_file.write_text("stub", encoding="utf-8")
    database_updates = []

    import_module.get_plugin_for_combination = lambda format_id, montage: (
        _ImportPathPlugin(raw)
    )
    import_module.manage_database_conditionally = (
        lambda **kwargs: database_updates.append(kwargs)
    )
    import_module.message = lambda *args, **kwargs: None

    result = import_module.import_eeg(
        {
            "unprocessed_file": str(input_file),
            "eeg_system": "standard_1020",
            "run_id": "run-203",
            "reports_dir": str(tmp_path),
            "prior_preprocessing_detection": {"enabled": True, "strict": True},
            "settings": {
                "filtering": {
                    "enabled": True,
                    "value": {"notch_freqs": [60]},
                },
                "reference_step": {"enabled": True, "value": "average"},
            },
        }
    )

    assert result is raw
    import_metadata = database_updates[0]["update_record"]["metadata"]["import_eeg"]
    prior_metadata = import_metadata["prior_preprocessing"]
    assert prior_metadata["available"] is True
    assert prior_metadata["schema_version"] == "1.0"
    assert prior_metadata["summary_row"]["source_file"] == "sub-01.set"
    assert prior_metadata["findings"]["highpass_filter"]["confidence"] == "documented"
    assert prior_metadata["findings"]["ica_present"]["confidence"] == "documented"
    assert prior_metadata["strict_violations"] == prior_metadata["warnings"]
    assert prior_metadata["artifact_paths"].keys() == {
        "json",
        "report",
        "dataset_summary",
    }
    for artifact_path in prior_metadata["artifact_paths"].values():
        assert Path(artifact_path).exists()


def test_import_eeg_does_not_read_signal_when_detection_is_not_enabled(tmp_path):
    import_module = _load_import_module()
    raw = _RawStub(_notch_like_data(), sfreq=500.0, ch_names=["Cz", "Pz"])
    input_file = tmp_path / "sub-01.set"
    input_file.write_text("stub", encoding="utf-8")
    database_updates = []

    def fail_get_data():
        raise AssertionError("disabled detection must not read signal data")

    raw.get_data = fail_get_data
    import_module.get_plugin_for_combination = lambda format_id, montage: (
        _ImportPathPlugin(raw)
    )
    import_module.manage_database_conditionally = (
        lambda **kwargs: database_updates.append(kwargs)
    )
    import_module.message = lambda *args, **kwargs: None

    result = import_module.import_eeg(
        {
            "unprocessed_file": str(input_file),
            "eeg_system": "standard_1020",
            "run_id": "run-203-disabled",
            "reports_dir": str(tmp_path),
        }
    )

    assert result is raw
    import_metadata = database_updates[0]["update_record"]["metadata"]["import_eeg"]
    assert "prior_preprocessing" not in import_metadata
