from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoclean.utils.montage_preflight import (
    build_batch_plan,
    clone_tasks_for_mismatches,
    copy_originals_for_plan,
    detect_hydrocel_montage,
    write_batch_plan_json,
    write_scan_csv,
)


class FakeRaw:
    def __init__(self, ch_names: list[str], ch_types: list[str] | None = None) -> None:
        self.ch_names = ch_names
        self._ch_types = ch_types or ["eeg"] * len(ch_names)

    def get_channel_types(self) -> list[str]:
        return self._ch_types


def _task_file(tmp_path: Path, montage: str = "GSN-HydroCel-128") -> Path:
    path = tmp_path / "RestingState_Basic_128.py"
    path.write_text(
        f'''from autoclean.core.task import Task

config = {{
    "schema_version": "2025.09",
    "montage": {{"enabled": True, "value": "{montage}"}},
}}

class RestingStateBasic128(Task):
    def run(self):
        pass
''',
        encoding="utf-8",
    )
    return path


def _touch(path: Path, content: str = "data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_detect_hydrocel_128_and_129_without_dropping_e129() -> None:
    detected_128 = detect_hydrocel_montage(FakeRaw([f"E{i}" for i in range(1, 129)]))
    detected_129 = detect_hydrocel_montage(FakeRaw([f"E{i}" for i in range(1, 130)]))

    assert detected_128 == ("GSN-HydroCel-128", 128, False, "")
    assert detected_129 == ("GSN-HydroCel-129", 129, True, "")


def test_detect_hydrocel_ambiguous_layout_is_unknown() -> None:
    detected = detect_hydrocel_montage(FakeRaw([f"E{i}" for i in range(1, 128)] + ["E129"]))

    assert detected[0] is None
    assert detected[1] == 128
    assert detected[2] is True


def test_build_batch_plan_groups_matching_mixed_and_unknown_files(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    file_128 = _touch(input_dir / "sub-01.raw")
    file_129 = _touch(input_dir / "sub-02.raw")
    file_unknown = _touch(input_dir / "sub-03.raw")
    _touch(input_dir / "notes.txt")
    task = _task_file(tmp_path)

    def loader(path: Path) -> FakeRaw:
        if path == file_128:
            return FakeRaw([f"E{i}" for i in range(1, 129)])
        if path == file_129:
            return FakeRaw([f"E{i}" for i in range(1, 130)])
        if path == file_unknown:
            return FakeRaw([f"E{i}" for i in range(1, 128)])
        raise AssertionError(f"unexpected path {path}")

    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=loader,
    )

    statuses = {(item.relative_path, item.detected_montage, item.status) for item in plan.files}
    assert ("sub-01.raw", "GSN-HydroCel-128", "match") in statuses
    assert ("sub-02.raw", "GSN-HydroCel-129", "mismatch") in statuses
    assert ("sub-03.raw", None, "unknown") in statuses
    assert str(file_unknown) in plan.unknown_files
    assert str(file_128) in plan.actionable_files
    assert str(file_129) in plan.actionable_files
    assert str(input_dir / "notes.txt") not in plan.actionable_files


def test_write_scan_csv_and_batch_plan_json(tmp_path: Path) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )

    csv_path = write_scan_csv(plan, tmp_path / "out")
    json_path = write_batch_plan_json(plan, tmp_path / "out")

    assert "e129_present" in csv_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["expected_montage"] == "GSN-HydroCel-128"
    assert payload["files"][0]["detected_montage"] == "GSN-HydroCel-128"


def test_copy_originals_preserves_source_and_skips_unknown(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    file_128 = _touch(input_dir / "sub-01.raw", "alpha")
    file_unknown = _touch(input_dir / "sub-02.raw", "beta")
    task = _task_file(tmp_path)

    def loader(path: Path) -> FakeRaw:
        if path == file_128:
            return FakeRaw([f"E{i}" for i in range(1, 129)])
        if path == file_unknown:
            return FakeRaw([f"E{i}" for i in range(1, 128)])
        raise AssertionError(f"unexpected path {path}")

    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=loader,
    )
    result = copy_originals_for_plan(plan, split_output_root=tmp_path / "split")

    copied_path = tmp_path / "split" / "GSN-HydroCel-128" / "sub-01.raw"
    assert copied_path.read_text(encoding="utf-8") == "alpha"
    assert file_128.read_text(encoding="utf-8") == "alpha"
    assert str(file_unknown) in result.skipped_files
    assert result.required_bytes >= len("alpha")


def test_copy_originals_blocks_existing_destinations(tmp_path: Path) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )
    _touch(tmp_path / "split" / "GSN-HydroCel-128" / "sub-01.raw", "exists")

    with pytest.raises(FileExistsError):
        copy_originals_for_plan(plan, split_output_root=tmp_path / "split")


def test_direct_mff_input_copy_preserves_package_name(tmp_path: Path) -> None:
    mff_dir = tmp_path / "sub-01.mff"
    _touch(mff_dir / "signal1.bin", "alpha")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=mff_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )

    result = copy_originals_for_plan(plan, split_output_root=tmp_path / "split")

    copied_package = tmp_path / "split" / "GSN-HydroCel-128" / "sub-01.mff"
    assert copied_package.is_dir()
    assert (copied_package / "signal1.bin").read_text(encoding="utf-8") == "alpha"
    assert result.copied_files[0]["destination"] == str(copied_package)


def test_task_clone_changes_montage_class_name_and_adds_provenance(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    _touch(input_dir / "sub-01.raw")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 130)]),
    )

    clones = clone_tasks_for_mismatches(plan=plan, task_output_dir=tmp_path / "tasks")

    assert len(clones) == 1
    clone_text = Path(clones[0].cloned_task).read_text(encoding="utf-8")
    assert "Montage preflight clone note" in clone_text
    assert '"value": "GSN-HydroCel-129"' in clone_text
    assert "class RestingStateBasic128_GSN_HydroCel_129(Task):" in clone_text
    assert '"schema_version": "2025.09"' in clone_text
