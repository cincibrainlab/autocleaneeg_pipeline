"""Integration tests for current quality-control metadata and channel audit trails."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

pytest.importorskip("autoclean.core.pipeline")

from autoclean.core.pipeline import Pipeline


class TestQualityControlIntegration:
    """Validate QC-related metadata in the current Python-task workflow."""

    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="autoclean_qc_integration_")
        workspace = Path(temp_dir)
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "tasks").mkdir()
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @pytest.fixture
    def synthetic_eeg_file(self, temp_workspace: Path) -> Path:
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=32,
            duration=10.0,
            sfreq=250.0,
        )
        data_file = temp_workspace / "input" / "quality_control_raw.fif"
        raw.save(data_file, overwrite=True, verbose=False)
        return data_file

    def _write_qc_task(self, workspace: Path) -> Path:
        task_file = workspace / "tasks" / "quality_control_task.py"
        task_file.write_text(
            """
from typing import Any, Dict

import mne

from autoclean.core.task import Task

config = {
    "resample_step": {"enabled": False, "value": None},
    "filtering": {"enabled": False, "value": {}},
    "reference_step": {"enabled": False, "value": None},
    "ICA": {"enabled": False, "value": {"method": "infomax"}},
    "move_flagged_files": False,
}


class QualityControlTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()["config"]
        super().__init__(config)

    def run(self) -> None:
        self.raw = mne.io.read_raw_fif(
            self.config["unprocessed_file"], preload=True, verbose=False
        )
        self.original_raw = self.raw.copy()
        self.clean_bad_channels(
            cleaning_method=None,
            manual_bad_channels=self.raw.ch_names[:5],
        )
""",
            encoding="utf-8",
        )
        return task_file

    def _run_qc_task(self, workspace: Path, input_file: Path) -> tuple[Path, dict]:
        task_file = self._write_qc_task(workspace)
        pipeline = Pipeline(output_dir=workspace / "output", verbose="ERROR")
        pipeline.add_task(str(task_file))
        pipeline.process_file(file_path=input_file, task="qualitycontroltask")
        task_root = workspace / "output" / "qualitycontroltask"
        metadata_path = next((task_root / "reports" / "run_reports").glob("*_autoclean_metadata.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return task_root, metadata

    def test_manual_channel_overrides_recorded_in_step_metadata(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        _task_root, metadata = self._run_qc_task(temp_workspace, synthetic_eeg_file)

        step_metadata = metadata["metadata"]["step_clean_bad_channels"]
        assert step_metadata["method"] == "ManualOverride"
        assert sorted(step_metadata["options"]["manual_bad_channels"]) == sorted(
            step_metadata["bads"]
        )
        assert len(step_metadata["bads"]) == 5

    def test_channel_removal_audit_trail_preserved(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        _task_root, metadata = self._run_qc_task(temp_workspace, synthetic_eeg_file)

        removals = metadata["metadata"]["channel_removals"]
        assert len(removals) == 5
        assert all(entry["reason"] == "MANUAL_OVERRIDE" for entry in removals)
        assert all(entry["source_step"] == "clean_bad_channels" for entry in removals)

    def test_qc_run_still_generates_report_and_export(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        task_root, metadata = self._run_qc_task(temp_workspace, synthetic_eeg_file)

        report_file = next((task_root / "reports" / "run_reports").glob("*_autoclean_report.pdf"))
        export_file = next((task_root / "exports").glob("*.set"))

        assert metadata["success"] == 1
        assert metadata["status"] == "completed"
        assert report_file.stat().st_size > 0
        assert export_file.stat().st_size > 0
