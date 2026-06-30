from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autoclean.core.task import Task
from autoclean.utils.exclusion_list import evaluate_exclusion_list


def _table(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_exclusion_list_tags_matching_recording(tmp_path: Path) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,exclude,reason\nsubject01.set,yes,participant withdrew\n",
    )

    result = evaluate_exclusion_list({"path": str(table)}, tmp_path / "subject01.set")

    assert result.excluded is True
    assert result.reason == "participant withdrew"
    assert result.warning is None
    assert result.metadata["matched"] is True
    assert result.metadata["matched_by"] == "file"


def test_exclusion_list_keeps_non_excluded_recording(tmp_path: Path) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,exclude,reason\nsubject01.set,no,usable\n",
    )

    result = evaluate_exclusion_list({"path": str(table)}, tmp_path / "subject01.set")

    assert result.excluded is False
    assert result.reason == "usable"
    assert result.metadata["excluded"] is False


def test_exclusion_list_warns_when_unmatched_unless_strict(tmp_path: Path) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,exclude,reason\nother.set,yes,bad task\n",
    )

    result = evaluate_exclusion_list({"path": str(table)}, tmp_path / "subject01.set")

    assert result.excluded is False
    assert result.metadata["matched"] is False
    assert "No exclusion-list row matched" in result.warning

    with pytest.raises(ValueError, match="No exclusion-list row matched"):
        evaluate_exclusion_list(
            {"path": str(table), "strict": True}, tmp_path / "subject01.set"
        )


def test_exclusion_list_can_match_subject_and_session_fields(tmp_path: Path) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,subject,session,exclude,reason\n"
        "ambiguous.set,sub-01,ses-1,no,include\n"
        "ambiguous.set,sub-02,ses-1,yes,artifact\n",
    )

    result = evaluate_exclusion_list(
        {
            "path": str(table),
            "subject_column": "subject",
            "subject": "sub-02",
            "session_column": "session",
            "session": "ses-1",
        },
        tmp_path / "ambiguous.set",
    )

    assert result.excluded is True
    assert result.reason == "artifact"
    assert result.metadata["matched_by"] == "field"


def test_exclusion_list_supports_skip_mode_decision(tmp_path: Path) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,exclude,reason\nsubject01.set,yes,skip me\n",
    )

    result = evaluate_exclusion_list(
        {"path": str(table), "mode": "skip"}, tmp_path / "subject01.set"
    )

    assert result.mode == "skip"
    assert result.excluded is True
    assert result.reason == "skip me"


class TestExclusionTask(Task):
    def __init__(self, config, settings):
        self.settings = settings
        super().__init__(config)
        self.metadata_log = []

    def _update_metadata(self, operation, metadata_dict):
        self.metadata_log.append((operation, metadata_dict))

    def create_bids_path(self, *args, **kwargs):
        return None

    def run(self):
        pass


@patch("autoclean.core.task.save_raw_to_set")
@patch("autoclean.core.task.import_eeg")
def test_import_raw_applies_exclusion_tag_without_losing_short_duration_warning(
    mock_import_eeg, mock_save_raw_to_set, tmp_path: Path
) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,exclude,reason\nsubject01.set,yes,known artifact\n",
    )
    settings = {
        "montage": {"enabled": False},
        "exclusion_list": {"enabled": True, "value": {"path": str(table)}},
    }
    config = {
        "run_id": "test_run_200",
        "unprocessed_file": tmp_path / "subject01.set",
        "task": "TestExclusionTask",
    }
    task = TestExclusionTask(config, settings)
    mock_import_eeg.return_value = SimpleNamespace(duration=30.0)

    task.import_raw()

    assert task.flagged is True
    assert "EXCLUSION_LIST: known artifact" in task.flagged_reasons
    assert any("less than 1 minute" in reason for reason in task.flagged_reasons)
    assert task.metadata_log[-1][0] == "step_exclusion_list"
    assert task.metadata_log[-1][1]["excluded"] is True
    mock_save_raw_to_set.assert_called_once()


@patch("autoclean.core.task.save_raw_to_set")
@patch("autoclean.core.task.import_eeg")
def test_import_raw_ignores_skip_mode_because_pipeline_owns_dispatch_skip(
    mock_import_eeg, mock_save_raw_to_set, tmp_path: Path
) -> None:
    table = _table(
        tmp_path / "exclusions.csv",
        "file,exclude,reason\nsubject01.set,yes,skip before task\n",
    )
    settings = {
        "montage": {"enabled": False},
        "exclusion_list": {
            "enabled": True,
            "value": {"path": str(table), "mode": "skip"},
        },
    }
    config = {
        "run_id": "test_run_200_skip",
        "unprocessed_file": tmp_path / "subject01.set",
        "task": "TestExclusionTask",
    }
    task = TestExclusionTask(config, settings)
    mock_import_eeg.return_value = SimpleNamespace(duration=120.0)

    task.import_raw()

    assert task.flagged is False
    assert task.flagged_reasons == []
    assert task.metadata_log == []
    mock_save_raw_to_set.assert_called_once()
