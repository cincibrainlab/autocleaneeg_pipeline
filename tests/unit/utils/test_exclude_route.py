from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from autoclean.api.routes import exclude
from autoclean.api.routes.exclude import (
    EpochReviewUpdate,
    OverridesUpdate,
    QaExportRequest,
    _create_qa_preprocessing_log,
    _default_record,
    _load_decisions,
    _resolve_reprocess_fix_type_with_epochs,
    _save_decisions,
    export_to_qa,
    get_eeg_epochs,
    get_eeg_manifest,
    get_epoch_review,
    get_exclude_file_detail,
    get_exclude_ica_summary,
    get_exclude_root,
    get_reprocess_status,
    list_exclude_files,
    save_epoch_review,
    save_overrides,
    start_reprocess,
)
from autoclean.api.state import api_state


@pytest.fixture
def workspace(tmp_path: Path):
    workspace_dir = tmp_path / "workspace"
    exports_dir = workspace_dir / "task" / "exports"
    exports_dir.mkdir(parents=True)
    old_workspace = api_state.workspace_dir
    api_state.workspace_dir = workspace_dir
    try:
        yield workspace_dir, exports_dir
    finally:
        api_state.workspace_dir = old_workspace


def test_save_decisions_writes_extended_csv_fields(tmp_path: Path):
    root = tmp_path / "exports"
    root.mkdir()
    decisions = {
        "subj/file_comp_epo": {
            **_default_record("subj/file_comp_epo.set"),
            "qa_export_hash": "abc123",
            "qa_export_timestamp": "2026-03-16T10:00:00",
            "qa_export_path": "qa/file_comp_epo.set",
            "reprocess_modified": True,
            "reprocess_fix_type": "ica",
            "reprocess_timestamp": "2026-03-16 10:00:00",
        }
    }

    _save_decisions(root, decisions)

    csv_path = root / "autoclean_exclusion_decisions.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["qa_export_hash"] == "abc123"
    assert row["qa_export_path"] == "qa/file_comp_epo.set"
    assert row["reprocess_modified"] == "True"
    assert row["reprocess_fix_type"] == "ica"
    assert row["modified_source"] == "web"


def test_create_qa_preprocessing_log_merges_manual_review_data(tmp_path: Path):
    task_root = tmp_path / "task"
    task_root.mkdir()
    qa_dir = task_root / "qa"
    qa_dir.mkdir()
    preprocessing_log = task_root / "preprocessing_log.csv"
    preprocessing_log.write_text(
        "subj_basename,epoch_trials,epoch_badtrials,epoch_percent\n"
        "subject01,10,1,0.9\n",
        encoding="utf-8",
    )
    decisions = {
        "subject01_comp_epo": {
            **_default_record("subject01_comp_epo.set"),
            "status": "REVIEW",
            "bad_epochs_count": 2,
            "bad_epoch_indices": "1,4",
            "last_updated": "2026-03-16 10:00:00",
            "notes": "manual review note",
            "qa_export_timestamp": "2026-03-16T10:05:00",
        }
    }

    qa_log_path = _create_qa_preprocessing_log(task_root, decisions)

    assert qa_log_path == qa_dir / "qa_preprocessing_log.csv"
    rows = list(csv.DictReader(qa_log_path.open("r", newline="", encoding="utf-8")))
    assert rows[0]["qa_status"] == "REVIEW"
    assert rows[0]["manual_bad_epochs"] == "2"
    assert rows[0]["manual_bad_epoch_indices"] == "1,4"
    assert rows[0]["manual_review_notes"] == "manual review note"
    assert rows[0]["epoch_badtrials"] in {"3", "3.0"}


def test_resolve_reprocess_fix_type_with_epochs_returns_epoch_for_epoch_only():
    result = _resolve_reprocess_fix_type_with_epochs(
        existing_bad_channels=[],
        existing_rejected_ica=[],
        next_bad_channels=[],
        next_rejected_ica=[],
        manual_bad_epoch_count=2,
    )

    assert result == "epoch"


@pytest.mark.asyncio
async def test_save_overrides_sets_reprocess_metadata(workspace, monkeypatch):
    _workspace_dir, exports_dir = workspace
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(exclude, "_load_epochs", lambda _path: _FakeEpochs())

    result = await save_overrides(
        "subject01_comp_epo",
        OverridesUpdate(manual_bad_channels=["Fp1"], manual_rejected_ica=[]),
        route_id=None,
    )

    assert result["saved"] is True
    decisions = _load_decisions(exports_dir)
    record = decisions["subject01_comp_epo"]
    assert record["manual_bad_channels"] == ["FP1"]
    assert record["manual_rejected_ica"] == []
    assert record["reprocess_modified"] is True
    assert record["reprocess_fix_type"] == "channel"
    assert record["reprocess_timestamp"]


class _FakeEpochs:
    def __init__(self):
        self.selection = [0, 1, 2, 3]
        self.dropped: list[int] = []
        self.ch_names = ["FP1", "FP2"]
        self.info = {"sfreq": 250.0}
        self.times = np.array([0.0, 0.5, 1.0])
        self.events = np.array(
            [
                [0, 0, 101],
                [250, 0, 102],
                [500, 0, 103],
                [750, 0, 104],
            ]
        )

    def drop(self, indices, reason="USER", verbose=False):
        self.dropped.extend(indices)

    def export(self, path: str, fmt: str, overwrite: bool):
        Path(path).write_text(f"exported:{fmt}:{self.dropped}", encoding="utf-8")

    def __len__(self):
        return 4

    def get_data(self, picks=None):
        if picks is None:
            picks = self.ch_names
        n_channels = len(picks)
        return np.array(
            [[[0.0, 0.000001, 0.0] for _ in range(n_channels)] for _ in range(4)],
            dtype=float,
        )


@pytest.mark.asyncio
async def test_export_to_qa_updates_record_and_creates_log(workspace, monkeypatch):
    workspace_dir, exports_dir = workspace
    task_root = exports_dir.parent
    (task_root / "qa").mkdir()
    (task_root / "preprocessing_log.csv").write_text(
        "subj_basename,epoch_trials,epoch_badtrials\nsubject01,10,1\n",
        encoding="utf-8",
    )
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")

    decisions = {
        "subject01_comp_epo": {
            **_default_record("subject01_comp_epo.set"),
            "bad_epochs_count": 1,
            "bad_epoch_indices": "2",
            "total_epochs": 4,
        }
    }
    _save_decisions(exports_dir, decisions)
    monkeypatch.setattr(exclude, "_load_epochs", lambda _path: _FakeEpochs())

    result = await export_to_qa(QaExportRequest(file_keys=["subject01_comp_epo"]))

    assert result["exported"] == 1
    assert result["skipped"] == 0
    assert result["errors"] == []
    assert (task_root / "qa" / "subject01_comp_epo.set").exists()
    assert (task_root / "qa" / "qa_preprocessing_log.csv").exists()

    updated = json.loads(
        (exports_dir / "autoclean_exclusion_decisions.json").read_text()
    )
    record = updated["subject01_comp_epo"]
    assert record["qa_export_hash"]
    assert record["qa_export_timestamp"]
    assert record["qa_export_path"] == "qa/subject01_comp_epo.set"


@pytest.mark.asyncio
async def test_epoch_review_save_and_load(workspace, monkeypatch):
    _workspace_dir, exports_dir = workspace
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(exclude, "_load_epochs", lambda _path: _FakeEpochs())

    save_result = await save_epoch_review(
        "subject01_comp_epo",
        EpochReviewUpdate(bad_epoch_indices=[1, 3]),
        route_id=None,
    )
    load_result = await get_epoch_review("subject01_comp_epo", route_id=None)

    assert save_result["saved"] is True
    assert load_result["bad_epoch_indices"] == [1, 3]
    assert load_result["bad_epochs_count"] == 2
    assert load_result["total_epochs"] == 4
    assert (exports_dir.parent / "postedit" / "subject01_postedit.set").exists()


@pytest.mark.asyncio
async def test_eeg_manifest_and_epoch_window(workspace, monkeypatch):
    _workspace_dir, exports_dir = workspace
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(exclude, "_load_epochs", lambda _path: _FakeEpochs())

    manifest = await get_eeg_manifest("subject01_comp_epo", route_id=None)
    window = await get_eeg_epochs("subject01_comp_epo", start=0, count=2, route_id=None)

    assert manifest.n_epochs == 4
    assert manifest.n_channels == 2
    assert manifest.existing_bad_epoch_indices == []
    assert window.count == 2
    assert window.epochs[0]["epoch_index"] == 0


@pytest.mark.asyncio
async def test_get_file_detail_includes_postedit_artifact(workspace):
    _workspace_dir, exports_dir = workspace
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    decisions = {
        "subject01_comp_epo": {
            **_default_record("subject01_comp_epo.set"),
            "bad_epoch_indices": "1",
            "bad_epochs_count": 1,
        }
    }
    _save_decisions(exports_dir, decisions)
    postedit_dir = exports_dir.parent / "postedit"
    postedit_dir.mkdir()
    (postedit_dir / "subject01_postedit.set").write_text("edited", encoding="utf-8")

    detail = await get_exclude_file_detail("subject01_comp_epo", route_id=None)

    assert (
        detail.artifacts["postedit"]
        == "/api/exclude/files/subject01_comp_epo/artifacts/postedit"
    )


@pytest.mark.asyncio
async def test_get_exclude_ica_summary_uses_pdf_extractor(workspace, monkeypatch):
    _workspace_dir, exports_dir = workspace
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    ica_dir = exports_dir.parent / "reports" / "ica_components"
    ica_dir.mkdir(parents=True)
    (ica_dir / "subject01_ica_components_all.pdf").write_text("pdf", encoding="utf-8")
    monkeypatch.setattr(
        exclude,
        "extract_ica_full",
        lambda _path: {
            "components": [{"component": "IC1"}],
            "structure": {"detail_page_map": {"IC1": 3}},
        },
    )

    result = await get_exclude_ica_summary("subject01_comp_epo", route_id=None)

    assert result["components"][0]["component"] == "IC1"
    assert result["structure"]["detail_page_map"]["IC1"] == 3


class _FakeProcess:
    def __init__(self):
        self._polled = False

    def poll(self):
        if not self._polled:
            self._polled = True
            return None
        return 0


class _FakeSubprocessModule:
    PIPE = None
    STDOUT = None

    def __init__(self):
        self.calls: list[list[str]] = []

    def Popen(self, cmd, **_kwargs):
        self.calls.append(list(cmd))
        return _FakeProcess()


@pytest.mark.asyncio
async def test_reprocess_start_and_status(workspace, monkeypatch):
    _workspace_dir, exports_dir = workspace
    task_root = exports_dir.parent
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    reports_dir = task_root / "reports" / "run_reports"
    reports_dir.mkdir(parents=True)
    metadata_path = reports_dir / "subject01_autoclean_metadata.json"
    raw_file = task_root / "subject01_raw.set"
    raw_file.write_text("", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "unprocessed_file": str(raw_file),
                "metadata": {
                    "import_eeg": {"originalChannelNames": ["FP1", "FP2"]},
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = task_root / "status"
    status_dir.mkdir()
    task_file = status_dir / "ExampleTask.py"
    task_file.write_text(
        "class ExampleTask:\n    config = {}\n    def run(self):\n        self.clean_bad_channels()\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(exclude, "extract_ica_full", lambda _path: {})
    fake_subprocess = _FakeSubprocessModule()
    monkeypatch.setattr(exclude, "subprocess", fake_subprocess)

    decisions = {
        "subject01_comp_epo": {
            **_default_record("subject01_comp_epo.set"),
            "bad_epochs_count": 2,
            "bad_epoch_indices": "1,3",
            "bad_epoch_times": "0.500,1.500",
            "bad_epoch_events": "102,104",
        }
    }
    _save_decisions(exports_dir, decisions)

    response = await start_reprocess(
        "subject01_comp_epo",
        exclude.ReprocessRequest(manual_bad_channels=["FP1"], manual_rejected_ica=[]),
        route_id=None,
    )
    status = await get_reprocess_status(response.job_id)
    payload = json.loads(
        (task_root / "qa" / "manual_fixes" / "subject01_manual_fix.json").read_text(
            encoding="utf-8"
        )
    )

    assert response.status == "running"
    assert status["job_id"] == response.job_id
    assert payload["modifications"]["epoch_review"]["indices"] == [1, 3]
    assert payload["modifications"]["epoch_review"]["count"] == 2
    assert fake_subprocess.calls[0][:3] == [exclude.sys.executable, "-m", "autoclean"]
    reprocess_folder_name = Path(
        exclude._REPROCESS_JOBS[response.job_id]["reprocess_folder"]
    ).name
    assert reprocess_folder_name.startswith("subject01_")
    assert not reprocess_folder_name.startswith(f"{task_root.name}_")


@pytest.mark.asyncio
async def test_post_epoch_ica_reprocess_records_metadata_and_warning(
    workspace, monkeypatch
):
    _workspace_dir, exports_dir = workspace
    task_root = exports_dir.parent
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    reports_dir = task_root / "reports" / "run_reports"
    reports_dir.mkdir(parents=True)
    metadata_path = reports_dir / "subject01_autoclean_metadata.json"
    raw_file = task_root / "subject01_raw.set"
    raw_file.write_text("", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "unprocessed_file": str(raw_file),
                "metadata": {
                    "import_eeg": {"originalChannelNames": ["FP1", "FP2"]},
                    "step_run_ica": {
                        "ica": {
                            "ica_components": 2,
                            "ica_kwargs": {
                                "method": "fastica",
                                "n_components": None,
                                "random_state": 97,
                            },
                            "ica_fit_data_type": "raw",
                        }
                    },
                    "classify_ica_components": {
                        "ica": {
                            "classification_method": "iclabel",
                            "ica_components": 2,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = task_root / "status"
    status_dir.mkdir()
    task_file = status_dir / "ExampleTask.py"
    task_file.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components(method='iclabel')\n"
            "        self.create_regular_epochs(export=True)\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(exclude, "_load_epochs", lambda _path: _FakeEpochs())
    monkeypatch.setattr(exclude, "extract_ica_full", lambda _path: {})
    fake_subprocess = _FakeSubprocessModule()
    monkeypatch.setattr(exclude, "subprocess", fake_subprocess)

    decisions = {
        "subject01_comp_epo": {
            **_default_record("subject01_comp_epo.set"),
            "bad_epochs_count": 3,
            "bad_epoch_indices": "0,1,2",
            "bad_epoch_times": "0.000,0.500,1.000",
            "bad_epoch_events": "101,102,103",
            "total_epochs": 4,
        }
    }
    _save_decisions(exports_dir, decisions)

    response = await start_reprocess(
        "subject01_comp_epo",
        exclude.ReprocessRequest(
            manual_bad_channels=[],
            manual_rejected_ica=[],
            action="post_epoch_ica",
        ),
        route_id=None,
    )
    payload = json.loads(
        (task_root / "qa" / "manual_fixes" / "subject01_manual_fix.json").read_text(
            encoding="utf-8"
        )
    )
    generated_task = (status_dir / "subject01_Reprocess.py").read_text(encoding="utf-8")

    assert response.action == "post_epoch_ica"
    assert response.warning
    assert payload["fix_type"] == "post_epoch_ica"
    assert payload["action"] == "post_epoch_ica"
    assert payload["post_epoch_rejection_ica"]["epochs_before_rejection"] == 4
    assert payload["post_epoch_rejection_ica"]["epochs_after_rejection"] == 1
    assert payload["post_epoch_rejection_ica"]["iclabel_rerun"] is True
    assert payload["post_epoch_rejection_ica"]["source_ica_settings"] == {
        "method": "fastica",
        "n_components": None,
        "random_state": 97,
        "ica_components": 2,
        "ica_fit_data_type": "raw",
    }
    assert payload["post_epoch_rejection_ica"]["source_classifier_settings"] == {
        "classification_method": "iclabel",
        "ica_components": 2,
    }
    assert "run_ica(use_epochs=True" in generated_task
    assert "classify_ica_components(method='iclabel', reject=False" in generated_task


@pytest.mark.asyncio
async def test_second_ica_reprocess_keeps_existing_bad_channels(workspace, monkeypatch):
    _workspace_dir, exports_dir = workspace
    task_root = exports_dir.parent
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    reports_dir = task_root / "reports" / "run_reports"
    reports_dir.mkdir(parents=True)
    metadata_path = reports_dir / "subject01_autoclean_metadata.json"
    raw_file = task_root / "subject01_raw.set"
    raw_file.write_text("", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "unprocessed_file": str(raw_file),
                "metadata": {
                    "import_eeg": {"originalChannelNames": ["FP1", "FP2"]},
                    "step_run_ica": {"ica": {"ica_components": 3}},
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = task_root / "status"
    status_dir.mkdir()
    task_file = status_dir / "ExampleTask.py"
    task_file.write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.clean_bad_channels()\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components()\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        exclude,
        "extract_ica_full",
        lambda _path: {
            "components": [
                {"component": "IC0"},
                {"component": "IC1"},
                {"component": "IC2"},
            ]
        },
    )
    fake_subprocess = _FakeSubprocessModule()
    monkeypatch.setattr(exclude, "subprocess", fake_subprocess)

    _save_decisions(
        exports_dir,
        {"subject01_comp_epo": {**_default_record("subject01_comp_epo.set")}},
    )

    await start_reprocess(
        "subject01_comp_epo",
        exclude.ReprocessRequest(manual_bad_channels=["FP1"], manual_rejected_ica=[]),
        route_id=None,
    )
    await start_reprocess(
        "subject01_comp_epo",
        exclude.ReprocessRequest(manual_bad_channels=["FP1"], manual_rejected_ica=[1]),
        route_id=None,
    )

    generated_task = (status_dir / "subject01_Reprocess.py").read_text(encoding="utf-8")

    assert "manual_bad_channels=['FP1']" in generated_task
    assert "classify_ica_components(reject=False)" in generated_task
    assert (
        "apply_ica_component_rejection(manual_rejected_components=[1])"
        in generated_task
    )


@pytest.mark.asyncio
async def test_reprocess_prefers_original_task_over_existing_reprocess_file(
    workspace, monkeypatch
):
    _workspace_dir, exports_dir = workspace
    task_root = exports_dir.parent
    set_file = exports_dir / "subject01_comp_epo.set"
    set_file.write_text("", encoding="utf-8")
    reports_dir = task_root / "reports" / "run_reports"
    reports_dir.mkdir(parents=True)
    metadata_path = reports_dir / "subject01_autoclean_metadata.json"
    raw_file = task_root / "subject01_raw.set"
    raw_file.write_text("", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "unprocessed_file": str(raw_file),
                "metadata": {
                    "import_eeg": {"originalChannelNames": ["FP1", "FP2"]},
                    "step_run_ica": {"ica": {"ica_components": 3}},
                },
            }
        ),
        encoding="utf-8",
    )
    status_dir = task_root / "status"
    status_dir.mkdir()
    (status_dir / "ExampleTask.py").write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class ExampleTask(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.clean_bad_channels()\n"
            "        self.run_ica()\n"
            "        self.classify_ica_components()\n"
        ),
        encoding="utf-8",
    )
    (status_dir / "subject01_Reprocess.py").write_text(
        (
            "from autoclean.core.task import Task\n\n"
            "config = {}\n\n"
            "class Subject01_Reprocess(Task):\n"
            "    def run(self):\n"
            "        self.import_raw()\n"
            "        self.classify_ica_components(reject=False)\n"
            "        self.apply_ica_component_rejection(manual_rejected_components=[6, 9])\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        exclude,
        "extract_ica_full",
        lambda _path: {
            "components": [
                {"component": "IC0"},
                {"component": "IC1"},
                {"component": "IC2"},
            ]
        },
    )
    fake_subprocess = _FakeSubprocessModule()
    monkeypatch.setattr(exclude, "subprocess", fake_subprocess)

    _save_decisions(
        exports_dir,
        {"subject01_comp_epo": {**_default_record("subject01_comp_epo.set")}},
    )

    await start_reprocess(
        "subject01_comp_epo",
        exclude.ReprocessRequest(manual_bad_channels=[], manual_rejected_ica=[1]),
        route_id=None,
    )

    generated_task = (status_dir / "subject01_Reprocess.py").read_text(encoding="utf-8")

    assert "manual_rejected_components=[1]" in generated_task
    assert "manual_rejected_components=[6, 9]" not in generated_task


@pytest.mark.asyncio
async def test_missing_workspace_edge_cases_raise_http_exception(monkeypatch):
    old_workspace = api_state.workspace_dir
    api_state.workspace_dir = None
    try:
        with pytest.raises(HTTPException):
            await get_exclude_root(route_id=None)
        with pytest.raises(HTTPException):
            await list_exclude_files(route_id=None)
    finally:
        api_state.workspace_dir = old_workspace
