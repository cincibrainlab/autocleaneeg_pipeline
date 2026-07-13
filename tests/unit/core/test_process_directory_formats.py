"""Acceptance test for issue #216.

Given a directory containing .set, .raw, .mff, .bdf, and .edf EEG inputs,
the default --format pattern for process_directory must discover all of
them as top-level inputs (with .mff treated as a single package, not
descended into), and it should never regress back to a narrower default.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autoclean.cli import _is_single_eeg_input_path, cmd_process
from autoclean.core.pipeline import Pipeline


@pytest.fixture
def mixed_format_dir(tmp_path):
    """Create a directory with .set, .raw, .mff, .bdf, .edf stub inputs."""
    (tmp_path / "subject01.set").touch()
    (tmp_path / "subject02.raw").touch()
    mff_dir = tmp_path / "subject03.mff"
    mff_dir.mkdir()
    (mff_dir / "info.xml").touch()  # simulate internal package contents
    (mff_dir / "signal1.bin").touch()
    (tmp_path / "subject04.bdf").touch()
    (tmp_path / "subject05.edf").touch()
    return tmp_path


def test_default_pattern_discovers_all_common_formats(mixed_format_dir, tmp_path):
    """process_directory's default pattern must discover .set/.raw/.mff/.bdf/.edf,
    treating .mff as one input rather than descending into its contents."""
    output_dir = tmp_path / "output"
    pipeline = Pipeline(output_dir=output_dir)

    processed_paths = []
    with patch.object(
        pipeline,
        "_entrypoint",
        side_effect=lambda p, task, *a, **kw: processed_paths.append(p),
    ):
        pipeline.process_directory(directory=mixed_format_dir, task="RestingEyesOpen")

    processed_names = {p.name for p in processed_paths}
    assert processed_names == {
        "subject01.set",
        "subject02.raw",
        "subject03.mff",
        "subject04.bdf",
        "subject05.edf",
    }
    # .mff must be discovered as ONE input, not have its internals processed separately
    assert "info.xml" not in processed_names
    assert "signal1.bin" not in processed_names
    assert len(processed_paths) == 5


def test_no_regression_to_narrower_default_pattern():
    """Guard against the exact regression from #216: default pattern must
    include mff and edf, not just raw/set/bdf."""
    import inspect

    default_pattern = (
        inspect.signature(Pipeline.process_directory).parameters["pattern"].default
    )
    assert "mff" in default_pattern
    assert "edf" in default_pattern


def test_direct_mff_directory_is_single_eeg_input(tmp_path):
    mff_dir = tmp_path / "direct_recording.mff"
    mff_dir.mkdir()

    assert _is_single_eeg_input_path(mff_dir)


def test_process_command_routes_direct_mff_to_process_file(tmp_path, monkeypatch):
    mff_dir = tmp_path / "direct_recording.mff"
    mff_dir.mkdir()
    output_dir = tmp_path / "output"
    calls = []

    class DummyPipeline:
        def __init__(self, **_kwargs):
            pass

        def process_file(self, file_path, task):
            calls.append(("file", file_path, task))

        def process_directory(self, **_kwargs):
            calls.append(("directory",))

    monkeypatch.setattr("autoclean.cli.PIPELINE_AVAILABLE", True)
    monkeypatch.setattr("autoclean.cli.Pipeline", DummyPipeline)
    monkeypatch.setattr("autoclean.cli.get_task_by_name", lambda _name: object())
    monkeypatch.setattr("autoclean.cli.has_logged_errors", lambda: False)

    args = SimpleNamespace(
        output=output_dir,
        automation=None,
        verbose=False,
        task_file=None,
        final_task="RestingEyesOpen",
        final_input=mff_dir,
        dry_run=False,
        format="*.{raw,set,bdf,mff,edf}",
        recursive=False,
        parallel=None,
    )

    assert cmd_process(args) == 0
    assert calls == [("file", mff_dir, "RestingEyesOpen")]
