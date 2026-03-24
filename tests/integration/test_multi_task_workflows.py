"""Integration tests for current multi-task workflow scenarios."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

pytest.importorskip("autoclean.core.pipeline")

from autoclean.core.pipeline import Pipeline
from autoclean.utils.logging import configure_logger


def _write_python_task(
    workspace: Path,
    task_name: str,
    *,
    resample: int | None = None,
    filter_value: dict | None = None,
    annotate_events: bool = False,
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


class TestMultiTaskWorkflows:
    """Test workflows involving multiple current Python task types."""

    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="autoclean_multitask_")
        workspace = Path(temp_dir)
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "tasks").mkdir()
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_sequential_different_tasks(self, temp_workspace: Path):
        """Different Python tasks should run sequentially in one pipeline session."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        pipeline = Pipeline(output_dir=temp_workspace / "output", verbose="ERROR")

        task_specs = [
            ("RestingDraftTask", {"resample": 125, "filter_value": None, "annotate_events": False}),
            ("FilteredTask", {"resample": None, "filter_value": {"l_freq": 1.0, "h_freq": 40.0}, "annotate_events": False}),
            ("EpochTask", {"resample": 125, "filter_value": None, "annotate_events": True}),
        ]

        successful_tasks: list[str] = []

        for idx, (task_name, spec) in enumerate(task_specs):
            pipeline.add_task(
                str(
                    _write_python_task(
                        temp_workspace,
                        task_name,
                        resample=spec["resample"],
                        filter_value=spec["filter_value"],
                        annotate_events=spec["annotate_events"],
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
            if spec["annotate_events"]:
                raw.annotations.append(
                    onset=[2.0, 6.0, 10.0],
                    duration=[0.1, 0.1, 0.1],
                    description=["stimulus", "stimulus", "stimulus"],
                )
            input_file = temp_workspace / "input" / f"{task_name.lower()}_raw.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(file_path=input_file, task=task_name.lower())
            task_root = temp_workspace / "output" / task_name.lower()
            if any(task_root.rglob("*_autoclean_metadata.json")):
                successful_tasks.append(task_name)

        assert successful_tasks == [name for name, _ in task_specs]

    def test_task_switching_performance(self, temp_workspace: Path):
        """Switching between Python tasks should stay fast with synthetic FIF inputs."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        pipeline = Pipeline(output_dir=temp_workspace / "output", verbose="ERROR")

        task_names = ["TaskAlpha", "TaskBeta", "TaskGamma"]
        for task_name in task_names:
            pipeline.add_task(
                str(
                    _write_python_task(
                        temp_workspace,
                        task_name,
                        resample=125,
                        filter_value=None,
                        annotate_events=False,
                    )
                )
            )

        start = time.time()
        for idx, task_name in enumerate(task_names):
            raw = create_synthetic_raw(
                montage="standard_1020",
                n_channels=32,
                duration=10.0,
                sfreq=250.0,
                seed=200 + idx,
            )
            input_file = temp_workspace / "input" / f"perf_{idx}.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(file_path=input_file, task=task_name.lower())

        avg_time = (time.time() - start) / len(task_names)
        assert avg_time < 10

    def test_concurrent_task_processing(self, temp_workspace: Path):
        """Independent pipeline instances with different task settings should coexist."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        task_variants = {
            "baseline": {"resample": 125, "filter_value": None},
            "filtered": {"resample": None, "filter_value": {"l_freq": 0.5, "h_freq": 30.0}},
        }

        successful_configs: list[str] = []
        for idx, (name, spec) in enumerate(task_variants.items()):
            pipeline = Pipeline(output_dir=temp_workspace / "output" / name, verbose="ERROR")
            pipeline.add_task(
                str(
                    _write_python_task(
                        temp_workspace,
                        f"{name.title()}Task",
                        resample=spec["resample"],
                        filter_value=spec["filter_value"],
                        annotate_events=False,
                    )
                )
            )
            raw = create_synthetic_raw(
                montage="standard_1020",
                n_channels=32,
                duration=12.0,
                sfreq=250.0,
                seed=300 + idx,
            )
            input_file = temp_workspace / "input" / f"{name}_input.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(
                file_path=input_file,
                task=f"{name.title()}Task".lower(),
            )
            if any((temp_workspace / "output" / name).rglob("*_autoclean_metadata.json")):
                successful_configs.append(name)

        assert sorted(successful_configs) == sorted(task_variants.keys())


class TestTaskParameterVariations:
    """Test parameter variations within current Python tasks."""

    @pytest.fixture
    def temp_workspace(self):
        temp_dir = tempfile.mkdtemp(prefix="autoclean_params_")
        workspace = Path(temp_dir)
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "tasks").mkdir()
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_filter_parameter_variations(self, temp_workspace: Path):
        """Different embedded filter settings should all produce outputs."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        filter_configs = [
            {"l_freq": 0.1, "h_freq": 50.0},
            {"l_freq": 0.5, "h_freq": 40.0},
            {"l_freq": 1.0, "h_freq": 30.0},
        ]

        successful_filters: list[str] = []
        for idx, filter_params in enumerate(filter_configs):
            task_name = f"FilterTask{idx}"
            pipeline = Pipeline(
                output_dir=temp_workspace / "output" / f"filter_{idx}",
                verbose="ERROR",
            )
            pipeline.add_task(
                str(
                    _write_python_task(
                        temp_workspace,
                        task_name,
                        resample=None,
                        filter_value=filter_params,
                        annotate_events=False,
                    )
                )
            )
            raw = create_synthetic_raw(
                montage="standard_1020",
                n_channels=32,
                duration=10.0,
                sfreq=250.0,
                seed=400 + idx,
            )
            input_file = temp_workspace / "input" / f"filter_{idx}.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(file_path=input_file, task=task_name.lower())
            if any((temp_workspace / "output" / f"filter_{idx}").rglob("*_autoclean_metadata.json")):
                successful_filters.append(task_name)

        assert len(successful_filters) == len(filter_configs)

    def test_epoch_parameter_variations(self, temp_workspace: Path):
        """Epoch-enabled tasks with annotations should produce exported epoch outputs."""
        configure_logger(verbose="ERROR", output_dir=temp_workspace)

        successful_epochs: list[str] = []
        for idx in range(3):
            task_name = f"EpochTask{idx}"
            pipeline = Pipeline(
                output_dir=temp_workspace / "output" / f"epoch_{idx}",
                verbose="ERROR",
            )
            pipeline.add_task(
                str(
                    _write_python_task(
                        temp_workspace,
                        task_name,
                        resample=125,
                        filter_value=None,
                        annotate_events=True,
                    )
                )
            )
            raw = create_synthetic_raw(
                montage="standard_1020",
                n_channels=32,
                duration=20.0,
                sfreq=250.0,
                seed=500 + idx,
            )
            raw.annotations.append(
                onset=[2.0, 5.0, 8.0, 11.0],
                duration=[0.1] * 4,
                description=["stimulus"] * 4,
            )
            input_file = temp_workspace / "input" / f"epoch_{idx}.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            pipeline.process_file(file_path=input_file, task=task_name.lower())
            if any((temp_workspace / "output" / f"epoch_{idx}").rglob("*_autoclean_metadata.json")):
                successful_epochs.append(task_name)

        assert len(successful_epochs) == 3
