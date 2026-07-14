"""Focused tests for optional analysis in the built-in ASSR task."""

from pathlib import Path

import pytest

from autoclean.data.builtins.tasks.auditory import ASSR_40Hz as assr_module


def _task(tmp_path: Path, settings: dict) -> assr_module.ASSR_40Hz:
    task = assr_module.ASSR_40Hz.__new__(assr_module.ASSR_40Hz)
    task.settings = settings
    task.config = {"unprocessed_file": tmp_path / "subject_01.set"}
    task.epochs = object()
    task._resolve_report_path = lambda key: tmp_path / "reports" / key
    return task


def test_run_assr_analysis_forwards_profile_overrides_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(
        tmp_path,
        {
            "assr_analysis": {
                "enabled": True,
                "value": {
                    "profile": "assr_epochs",
                    "baseline": [-0.2, 0.0],
                    "save_tfr": True,
                },
            }
        },
    )
    epochs = task.epochs
    captured: dict[str, object] = {}

    def fake_analyze_assr(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(assr_module, "analyze_assr", fake_analyze_assr)

    task.run_assr_analysis()

    assert captured == {
        "output_dir": tmp_path / "reports" / "assr",
        "save_results": True,
        "epochs": epochs,
        "file_basename": "subject_01",
        "analysis_profile": "assr_epochs",
        "analysis_config": {
            "baseline": [-0.2, 0.0],
            "save_tfr": True,
        },
    }


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "assr_analysis": {
                "enabled": False,
                "value": {"profile": "assr_epochs"},
            }
        },
    ],
)
def test_run_assr_analysis_skips_absent_or_disabled_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings: dict,
) -> None:
    task = _task(tmp_path, settings)

    def fail_if_called(**kwargs) -> None:
        raise AssertionError("disabled ASSR analysis must not run")

    monkeypatch.setattr(assr_module, "analyze_assr", fail_if_called)

    task.run_assr_analysis()


def test_run_assr_analysis_requires_epochs_when_enabled(tmp_path: Path) -> None:
    task = _task(
        tmp_path,
        {"assr_analysis": {"enabled": True, "value": {}}},
    )
    task.epochs = None

    with pytest.raises(RuntimeError, match="requires cleaned epochs"):
        task.run_assr_analysis()
