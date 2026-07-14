from __future__ import annotations

import importlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

import mne
import numpy as np
import pytest

import autoclean.io.eeglab_provenance as eeglab_provenance_module
from autoclean.io.eeglab_provenance import (
    build_eeglab_dataset_summary,
    extract_eeglab_provenance,
    render_eeglab_provenance_report,
    summarize_eeglab_provenance,
    write_eeglab_dataset_artifacts,
    write_eeglab_provenance_artifacts,
)


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_summarize_eeglab_provenance_extracts_documented_fields() -> None:
    eeg = _ns(
        setname="sub-01 task",
        srate=500,
        nbchan=2,
        trials=120,
        pnts=250,
        xmin=-0.2,
        xmax=0.8,
        ref="average",
        history="pop_resample; pop_runica; pop_interp;",
        comments="Imported from collaborator pipeline.",
        chanlocs=np.array(
            [_ns(labels="Cz", type="EEG"), _ns(labels="VEOG", type="EOG")]
        ),
        event=np.array(
            [
                _ns(type="standard", code=11),
                _ns(type="standard", code=11),
                _ns(type="target", code=22),
            ]
        ),
        icaweights=np.zeros((3, 2)),
        icasphere=np.zeros((2, 2)),
        icawinv=np.zeros((2, 3)),
        etc=_ns(
            ic_classification=_ns(
                ICLabel=_ns(
                    classes=np.array(["brain", "muscle"]),
                    classifications=np.zeros((3, 2)),
                )
            ),
            interpolated_channels=np.array(["Pz"]),
            task_trial_counts=_ns(standard=80, target=40),
        ),
    )

    summary = summarize_eeglab_provenance(eeg, "sub-01.set")

    documented = summary["documented_provenance"]
    assert documented["setname"] == "sub-01 task"
    assert documented["events"]["counts"] == {"standard": 2, "target": 1}
    assert documented["channels"]["types"] == {"EEG": 1, "EOG": 1}
    assert documented["ica"]["icaweights"]["shape"] == [3, 2]
    assert documented["iclabel"]["probability_matrix_shape"] == [3, 2]
    assert "resampling" in summary["inferred_preprocessing"]["steps"]
    assert "ica" in summary["inferred_preprocessing"]["steps"]
    assert summary["summary_row"]["source_file"] == "sub-01.set"


def test_write_eeglab_provenance_artifacts_creates_json_and_report(tmp_path) -> None:
    eeg = _ns(setname="minimal", srate=250, nbchan=1, trials=1, pnts=100)
    summary = summarize_eeglab_provenance(eeg, "minimal.set")

    paths = write_eeglab_provenance_artifacts(summary, tmp_path, "minimal")

    json_path = tmp_path / "minimal_eeglab_provenance.json"
    report_path = tmp_path / "minimal_eeglab_provenance.md"
    assert paths == {"json": str(json_path), "report": str(report_path)}
    assert json.loads(json_path.read_text(encoding="utf-8"))["artifact_paths"] == paths
    assert "Documented Provenance" in report_path.read_text(encoding="utf-8")


def test_extract_from_loaded_mne_data_does_not_reparse_set(tmp_path) -> None:
    set_file = tmp_path / "subject.set"
    set_file.write_bytes(b"MATLAB 7.3 MAT-file, Platform: test\\x00")
    info = mne.create_info(["Cz", "VEOG"], 250, ["eeg", "eog"])
    raw = mne.io.RawArray(np.zeros((2, 100)), info, verbose=False)
    raw.set_annotations(mne.Annotations([0.1], [0.0], ["stimulus"]))

    with patch("autoclean.io.eeglab_provenance.sio.loadmat") as loadmat:
        summary = extract_eeglab_provenance(set_file, raw)

    loadmat.assert_not_called()
    documented = summary["documented_provenance"]
    assert documented["srate"] == 250.0
    assert documented["channels"]["labels"] == ["Cz", "VEOG"]
    assert documented["events"]["counts"] == {"stimulus": 1}
    assert documented["history"] == "unavailable"


def test_extract_from_loaded_epochs_preserves_event_codes_without_reparse(
    tmp_path,
) -> None:
    set_file = tmp_path / "epochs.set"
    set_file.write_bytes(b"not parsed")
    info = mne.create_info(["Cz"], 250, ["eeg"])
    events = np.array([[0, 0, 11], [100, 0, 22]])
    epochs = mne.EpochsArray(
        np.zeros((2, 1, 25)),
        info,
        events=events,
        event_id={"standard": 11, "target": 22},
        verbose=False,
    )

    with patch("autoclean.io.eeglab_provenance.sio.loadmat") as loadmat:
        summary = extract_eeglab_provenance(set_file, epochs)

    loadmat.assert_not_called()
    documented_events = summary["documented_provenance"]["events"]
    assert documented_events["labels"] == ["standard", "target"]
    assert documented_events["codes"] == ["11", "22"]
    assert summary["documented_provenance"]["reference"] == "unavailable"


def test_render_eeglab_provenance_report_labels_missing_as_unavailable() -> None:
    summary = summarize_eeglab_provenance(_ns(), "empty.set")

    report = render_eeglab_provenance_report(summary)

    assert "EEG.history: unavailable" in report
    assert "EEG.comments: unavailable" in report
    assert "setname" in summary["unavailable"]


def test_dataset_summary_warns_on_batch_inconsistency() -> None:
    first = summarize_eeglab_provenance(
        _ns(
            srate=250,
            xmin=-0.2,
            xmax=0.8,
            nbchan=1,
            ref="average",
            chanlocs=[_ns(labels="Cz")],
            event=[_ns(type="standard")],
            etc=_ns(
                ic_classification=_ns(
                    ICLabel=_ns(classes=["brain"], classifications=np.zeros((1, 1)))
                )
            ),
        ),
        "a.set",
    )
    second = summarize_eeglab_provenance(
        _ns(
            srate=500,
            xmin=-0.1,
            xmax=0.5,
            nbchan=2,
            ref="Cz",
            chanlocs=[_ns(labels="Pz"), _ns(labels="Fz")],
            event=[_ns(type="target")],
            etc=_ns(
                ic_classification=_ns(
                    ICLabel=_ns(
                        classes=["brain", "muscle"],
                        classifications=np.zeros((2, 2)),
                    )
                )
            ),
        ),
        "b.set",
    )

    dataset_summary = build_eeglab_dataset_summary([first, second])

    warning_text = "\n".join(dataset_summary["warnings"])
    for field in (
        "sampling rate",
        "epoch window",
        "event labels",
        "channel count",
        "channel labels",
        "reference",
        "ICLabel structure",
    ):
        assert field in warning_text


def test_write_dataset_artifacts_persists_multiple_rows_and_excludes_itself(
    tmp_path,
) -> None:
    for stem, rate in (("a", 250), ("b", 500)):
        summary = summarize_eeglab_provenance(_ns(srate=rate, nbchan=1), f"{stem}.set")
        write_eeglab_provenance_artifacts(summary, tmp_path, stem)

    paths = write_eeglab_dataset_artifacts(tmp_path)
    paths = write_eeglab_dataset_artifacts(tmp_path)

    persisted = json.loads(
        (tmp_path / "dataset_eeglab_provenance.json").read_text(encoding="utf-8")
    )
    assert [row["source_file"] for row in persisted["rows"]] == ["a.set", "b.set"]
    assert persisted["artifact_paths"] == paths
    table = (tmp_path / "dataset_eeglab_provenance.csv").read_text(encoding="utf-8")
    assert "a.set" in table
    assert "b.set" in table
    assert not list(tmp_path.glob(".*.tmp"))


def test_dataset_artifact_updates_serialize_scan_and_replace(
    tmp_path, monkeypatch
) -> None:
    first_summary = summarize_eeglab_provenance(_ns(srate=250, nbchan=1), "a.set")
    write_eeglab_provenance_artifacts(first_summary, tmp_path, "a")

    original_build = eeglab_provenance_module.build_eeglab_dataset_summary
    first_build_started = Event()
    release_first_build = Event()
    second_build_started = Event()
    second_call_started = Event()
    call_guard = Lock()
    call_count = 0

    def coordinated_build(summaries):
        nonlocal call_count
        with call_guard:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_build_started.set()
            assert release_first_build.wait(timeout=5)
        else:
            second_build_started.set()
        return original_build(summaries)

    monkeypatch.setattr(
        eeglab_provenance_module,
        "build_eeglab_dataset_summary",
        coordinated_build,
    )

    def second_update():
        second_call_started.set()
        return write_eeglab_dataset_artifacts(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_update = executor.submit(write_eeglab_dataset_artifacts, tmp_path)
        assert first_build_started.wait(timeout=5)

        second_summary = summarize_eeglab_provenance(_ns(srate=500, nbchan=2), "b.set")
        write_eeglab_provenance_artifacts(second_summary, tmp_path, "b")
        second_update_future = executor.submit(second_update)
        assert second_call_started.wait(timeout=5)

        entered_while_first_locked = second_build_started.wait(timeout=0.5)
        release_first_build.set()
        first_update.result(timeout=5)
        second_update_future.result(timeout=5)

    assert entered_while_first_locked is False
    assert second_build_started.is_set()
    persisted = json.loads(
        (tmp_path / "dataset_eeglab_provenance.json").read_text(encoding="utf-8")
    )
    assert [row["source_file"] for row in persisted["rows"]] == [
        "a.set",
        "b.set",
    ]
    table = (tmp_path / "dataset_eeglab_provenance.csv").read_text(encoding="utf-8")
    assert "a.set" in table
    assert "b.set" in table
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("dataset_error", [None, RuntimeError("dataset failed")])
def test_import_eeg_records_eeglab_provenance_metadata(tmp_path, dataset_error) -> None:
    input_file = tmp_path / "subject.set"
    input_file.write_text("stub", encoding="utf-8")
    artifact_paths = {
        "json": str(tmp_path / "subject_eeglab_provenance.json"),
        "report": str(tmp_path / "subject_eeglab_provenance.md"),
    }
    dataset_paths = {
        "json": str(tmp_path / "dataset_eeglab_provenance.json"),
        "table": str(tmp_path / "dataset_eeglab_provenance.csv"),
    }
    provenance_summary = {
        "schema_version": "1.0",
        "summary_row": {"source_file": "subject.set", "srate": 500},
        "artifact_paths": {},
    }

    class _RawLike:
        info = {"sfreq": 500.0}
        ch_names = ["Cz", "Pz"]
        n_times = 1000

    class _Plugin:
        def import_and_configure(self, file_path, autoclean_dict, preload=True):
            return _RawLike()

        def process_events(self, eeg_data):
            return None, None, None

        def get_metadata(self):
            return {"plugin_note": "ok"}

    def _write_artifacts(summary, output_dir, stem):
        summary["artifact_paths"] = artifact_paths
        return artifact_paths

    autoclean_dict = {
        "run_id": "run-001",
        "unprocessed_file": str(input_file),
        "eeg_system": "GSN-HydroCel-129",
    }

    fake_database = _ns(manage_database_conditionally=lambda *args, **kwargs: None)
    fake_logging = _ns(message=lambda *args, **kwargs: None)
    with patch.dict(
        sys.modules,
        {
            "autoclean.utils.database": fake_database,
            "autoclean.utils.logging": fake_logging,
        },
    ):
        sys.modules.pop("autoclean.io.import_", None)
        import_module = importlib.import_module("autoclean.io.import_")

    with (
        patch.object(
            import_module, "get_plugin_for_combination", return_value=_Plugin()
        ),
        patch.object(
            import_module,
            "extract_eeglab_provenance",
            return_value=provenance_summary,
        ) as extract,
        patch.object(
            import_module,
            "write_eeglab_provenance_artifacts",
            side_effect=_write_artifacts,
        ) as write_artifacts,
        patch.object(
            import_module,
            "write_eeglab_dataset_artifacts",
            return_value=dataset_paths,
            side_effect=dataset_error,
        ) as write_dataset_artifacts,
        patch.object(
            import_module, "resolve_eeglab_provenance_dir", return_value=tmp_path
        ),
        patch.object(import_module, "manage_database_conditionally") as manage_db,
        patch.object(import_module, "message") as log_message,
    ):
        result = import_module.import_eeg(autoclean_dict)

    assert isinstance(result, _RawLike)
    extract.assert_called_once_with(input_file, result)
    write_artifacts.assert_called_once_with(provenance_summary, tmp_path, "subject")
    write_dataset_artifacts.assert_called_once_with(tmp_path)
    update_record = manage_db.call_args.kwargs["update_record"]
    eeglab_metadata = update_record["metadata"]["import_eeg"]["eeglab_provenance"]
    assert eeglab_metadata == {
        "available": True,
        "schema_version": "1.0",
        "artifact_paths": artifact_paths,
        "dataset_artifact_paths": {} if dataset_error else dataset_paths,
        "summary_row": {"source_file": "subject.set", "srate": 500},
    }
    assert update_record["metadata"]["import_eeg"]["artifact_reports"] == (
        {
            "eeglab_provenance_json": artifact_paths["json"],
            "eeglab_provenance_report": artifact_paths["report"],
        }
        if dataset_error
        else {
            "eeglab_provenance_json": artifact_paths["json"],
            "eeglab_provenance_report": artifact_paths["report"],
            "eeglab_dataset_provenance_json": dataset_paths["json"],
            "eeglab_dataset_provenance_table": dataset_paths["table"],
        }
    )
    if dataset_error:
        assert any(
            "dataset provenance summary unavailable" in str(call)
            for call in log_message.call_args_list
        )
