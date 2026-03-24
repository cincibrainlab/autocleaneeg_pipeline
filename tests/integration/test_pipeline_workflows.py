"""Integration tests for complete current-workflow pipeline scenarios."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

pytest.importorskip("autoclean.core.pipeline")

from autoclean.core.pipeline import Pipeline
from autoclean.utils.logging import configure_logger


def _write_pipeline_task(
    workspace: Path,
    task_name: str,
    *,
    resample: int | None = 125,
    annotate_events: bool = False,
    filter_value: dict | None = None,
) -> Path:
    task_file = workspace / "tasks" / f"{task_name.lower()}.py"
    task_file.parent.mkdir(parents=True, exist_ok=True)
    task_file.write_text(
        f"""
from typing import Any, Dict

import mne

from autoclean.core.task import Task

config = {{
    "resample_step": {{"enabled": {resample is not None}, "value": {resample!r}}},
    "filtering": {{"enabled": {filter_value is not None}, "value": {filter_value!r}}},
    "reference_step": {{"enabled": False, "value": None}},
    "ICA": {{"enabled": False, "value": {{"method": "infomax"}}}},
    "epoch_settings": {{"enabled": {annotate_events}, "value": {{"tmin": -0.2, "tmax": 0.8}}}},
}}


class {task_name}(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()["config"]
        super().__init__(config)

    def run(self) -> None:
        self.raw = mne.io.read_raw_fif(
            self.config["unprocessed_file"], preload=True, verbose=False
        )
        if {resample is not None}:
            self.resample_data()
        if {filter_value is not None}:
            self.filter_data()
        if {annotate_events}:
            self.create_regular_epochs(export=True)
""",
        encoding="utf-8",
    )
    return task_file


class TestPipelineWorkflows:
    """Test end-to-end processing using the current Python-task workflow."""

    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="autoclean_integration_")
        workspace = Path(temp_dir)
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "tasks").mkdir()
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @pytest.fixture
    def synthetic_input_file(self, temp_workspace: Path):
        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=32,
            duration=20.0,
            sfreq=250.0,
            seed=42,
        )
        input_file = temp_workspace / "input" / "test_subject_001_raw.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        return input_file

    def test_single_file_resting_processing(
        self,
        temp_workspace: Path,
        synthetic_input_file: Path,
    ):
        """A single direct-FIF Python task should process end to end."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        task_name = "SingleFileTask"
        pipeline = Pipeline(output_dir=temp_workspace / "output", verbose="ERROR")
        pipeline.add_task(str(_write_pipeline_task(temp_workspace, task_name, resample=125)))

        pipeline.process_file(file_path=synthetic_input_file, task=task_name.lower())
        assert any((temp_workspace / "output").rglob("*_autoclean_metadata.json"))

    def test_batch_processing_workflow(self, temp_workspace: Path):
        """Multiple FIF files should process successfully in one pipeline session."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        task_name = "BatchTask"
        pipeline = Pipeline(output_dir=temp_workspace / "output", verbose="ERROR")
        pipeline.add_task(str(_write_pipeline_task(temp_workspace, task_name, resample=125)))

        results = []
        for i in range(3):
            raw = create_synthetic_raw(
                montage="standard_1020",
                n_channels=32,
                duration=12.0,
                sfreq=250.0,
                seed=42 + i,
            )
            input_file = temp_workspace / "input" / f"subject_{i+1:03d}.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(file_path=input_file, task=task_name.lower())
            results.append(input_file)

        assert len(results) == 3
        assert len(list((temp_workspace / "output").rglob("*_autoclean_metadata.json"))) == 3

    def test_different_task_types(self, temp_workspace: Path):
        """Different Python tasks with different settings should both succeed."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        task_configs = [
            ("RestingDraftTask", {"resample": 125, "annotate_events": False, "filter_value": None}),
            ("EpochingTask", {"resample": 125, "annotate_events": True, "filter_value": None}),
        ]
        pipeline = Pipeline(output_dir=temp_workspace / "output", verbose="ERROR")
        successful_tasks: list[str] = []

        for idx, (task_name, config) in enumerate(task_configs):
            pipeline.add_task(
                str(
                    _write_pipeline_task(
                        temp_workspace,
                        task_name,
                        resample=config["resample"],
                        annotate_events=config["annotate_events"],
                        filter_value=config["filter_value"],
                    )
                )
            )
            raw = create_synthetic_raw(
                montage="standard_1020",
                n_channels=32,
                duration=20.0,
                sfreq=250.0,
                seed=100 + idx,
            )
            if config["annotate_events"]:
                raw.annotations.append(
                    onset=[2.0, 5.0, 8.0],
                    duration=[0.1, 0.1, 0.1],
                    description=["stimulus"] * 3,
                )
            input_file = temp_workspace / "input" / f"{task_name.lower()}_test.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(file_path=input_file, task=task_name.lower())
            if any((temp_workspace / "output" / task_name.lower()).rglob("*_autoclean_metadata.json")):
                successful_tasks.append(task_name)

        assert successful_tasks == [name for name, _ in task_configs]

    def test_pipeline_error_handling(self, temp_workspace: Path):
        """The pipeline should raise on missing files and invalid task names."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        task_name = "ErrorHandlingTask"
        pipeline = Pipeline(output_dir=temp_workspace / "output", verbose="ERROR")
        pipeline.add_task(str(_write_pipeline_task(temp_workspace, task_name, resample=125)))

        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            pipeline.process_file(
                file_path=temp_workspace / "input" / "nonexistent.fif",
                task=task_name.lower(),
            )

        raw = create_synthetic_raw(montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0)
        input_file = temp_workspace / "input" / "test_invalid_task.fif"
        raw.save(input_file, overwrite=True, verbose=False)

        with pytest.raises((ValueError, KeyError, AttributeError)):
            pipeline.process_file(file_path=input_file, task="nonexistenttask")

    def test_pipeline_configuration_variations(self, temp_workspace: Path):
        """Different embedded task configurations should both work."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        configs = {
            "high_quality": {"resample": 125, "filter_value": {"l_freq": 0.5, "h_freq": 40.0}},
            "permissive": {"resample": None, "filter_value": {"l_freq": 0.1, "h_freq": 50.0}},
        }

        raw = create_synthetic_raw(
            montage="standard_1020",
            n_channels=32,
            duration=12.0,
            sfreq=250.0,
        )
        input_file = temp_workspace / "input" / "config_test.fif"
        raw.save(input_file, overwrite=True, verbose=False)

        successful_configs = []
        for name, config in configs.items():
            pipeline = Pipeline(output_dir=temp_workspace / "output" / name, verbose="ERROR")
            pipeline.add_task(
                str(
                    _write_pipeline_task(
                        temp_workspace,
                        f"{name.title()}ConfigTask",
                        resample=config["resample"],
                        annotate_events=False,
                        filter_value=config["filter_value"],
                    )
                )
            )
            pipeline.process_file(
                file_path=input_file,
                task=f"{name.title()}ConfigTask".lower(),
            )
            if any((temp_workspace / "output" / name).rglob("*_autoclean_metadata.json")):
                successful_configs.append(name)

        assert sorted(successful_configs) == sorted(configs.keys())


class TestPipelineMemoryAndPerformance:
    """Test current workflow performance and memory characteristics."""

    def test_memory_usage_tracking(self, tmp_path):
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        configure_logger(verbose="ERROR", output_dir=tmp_path)
        task_name = "MemoryTask"
        pipeline = Pipeline(output_dir=tmp_path / "output", verbose="ERROR")
        pipeline.add_task(str(_write_pipeline_task(tmp_path, task_name, resample=125)))

        raw = create_synthetic_raw(montage="standard_1020", n_channels=32, duration=10.0, sfreq=250.0)
        input_file = tmp_path / "memory_test_raw.fif"
        raw.save(input_file, overwrite=True, verbose=False)

        pipeline.process_file(file_path=input_file, task=task_name.lower())
        final_memory = process.memory_info().rss / 1024 / 1024
        assert final_memory - initial_memory < 500

    @pytest.mark.timeout(300)
    def test_processing_time_reasonable(self, tmp_path):
        configure_logger(verbose="ERROR", output_dir=tmp_path)
        task_name = "TimingTask"
        pipeline = Pipeline(output_dir=tmp_path / "output", verbose="ERROR")
        pipeline.add_task(str(_write_pipeline_task(tmp_path, task_name, resample=125)))

        raw = create_synthetic_raw(montage="standard_1020", n_channels=32, duration=15.0, sfreq=250.0)
        input_file = tmp_path / "timing_test_raw.fif"
        raw.save(input_file, overwrite=True, verbose=False)

        import time

        start_time = time.time()
        pipeline.process_file(file_path=input_file, task=task_name.lower())
        assert time.time() - start_time < 60
