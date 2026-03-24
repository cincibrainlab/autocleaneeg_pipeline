"""Integration tests for current pipeline output artifacts."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

pytest.importorskip("autoclean.core.pipeline")

from autoclean.core.pipeline import Pipeline


class TestOutputValidation:
    """Validate output artifacts produced by the current Python-task workflow."""

    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="autoclean_output_validation_")
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
        data_file = temp_workspace / "input" / "output_validation_raw.fif"
        raw.save(data_file, overwrite=True, verbose=False)
        return data_file

    def _write_output_task(self, workspace: Path) -> Path:
        task_file = workspace / "tasks" / "output_validation_task.py"
        task_file.write_text(
            """
from typing import Any, Dict

import mne

from autoclean.core.task import Task

config = {
    "resample_step": {"enabled": True, "value": 125},
    "filtering": {"enabled": False, "value": {}},
    "reference_step": {"enabled": False, "value": None},
    "ICA": {"enabled": False, "value": {"method": "infomax"}},
    "epoch_settings": {"enabled": True, "value": {"tmin": -0.5, "tmax": 0.5}},
}


class OutputValidationTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()["config"]
        super().__init__(config)

    def run(self) -> None:
        self.raw = mne.io.read_raw_fif(
            self.config["unprocessed_file"], preload=True, verbose=False
        )
        self.resample_data()
        self.create_regular_epochs(export=True)
""",
            encoding="utf-8",
        )
        return task_file

    def _run_output_task(self, workspace: Path, input_file: Path) -> Path:
        task_file = self._write_output_task(workspace)
        pipeline = Pipeline(output_dir=workspace / "output", verbose="ERROR")
        pipeline.add_task(str(task_file))
        pipeline.process_file(file_path=input_file, task="outputvalidationtask")
        return workspace / "output" / "outputvalidationtask"

    def test_python_task_workflow_creates_output_tree(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        task_root = self._run_output_task(temp_workspace, synthetic_eeg_file)

        assert task_root.exists()
        assert (task_root / "exports").exists()
        assert (task_root / "reports" / "run_reports").exists()
        assert (task_root / "status" / "pipeline.log").exists()
        assert (task_root / "status" / "output_validation_task.py").exists()

    def test_stage_derivatives_include_exported_set_files(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        task_root = self._run_output_task(temp_workspace, synthetic_eeg_file)

        stage_files = sorted((task_root / "bids" / "derivatives").rglob("*.set"))
        assert stage_files, "Expected derivative stage .set files"
        stage_dirs = {path.parent.name for path in stage_files}
        assert "01_resample" in stage_dirs
        assert "02_epochs" in stage_dirs or "04_epochs" in stage_dirs

    def test_run_metadata_json_records_output_steps(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        task_root = self._run_output_task(temp_workspace, synthetic_eeg_file)

        metadata_path = next((task_root / "reports" / "run_reports").glob("*_autoclean_metadata.json"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert metadata["success"] == 1
        assert metadata["status"] == "completed"
        assert "save_raw_to_set" in metadata["metadata"] or "save_epochs_to_set" in metadata["metadata"]

    def test_report_pdf_and_exported_companion_file_exist(
        self, temp_workspace: Path, synthetic_eeg_file: Path
    ) -> None:
        task_root = self._run_output_task(temp_workspace, synthetic_eeg_file)

        report_file = next((task_root / "reports" / "run_reports").glob("*_autoclean_report.pdf"))
        export_file = next((task_root / "exports").glob("*_comp_epo.set"))

        assert report_file.stat().st_size > 0
        assert export_file.stat().st_size > 0
