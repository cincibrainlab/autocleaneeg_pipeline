from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoclean.utils.montage_preflight import (
    MontageCopyError,
    MontageMoveError,
    build_batch_plan,
    clone_tasks_for_mismatches,
    copy_originals_for_plan,
    detect_hydrocel_montage,
    estimate_copy_originals_for_plan,
    estimate_move_originals_for_plan,
    move_originals_for_plan,
    write_batch_plan_json,
    write_planned_move_manifest,
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
        f"""from autoclean.core.task import Task

config = {{
    "schema_version": "2025.09",
    "montage": {{"enabled": True, "value": "{montage}"}},
}}

class RestingStateBasic128(Task):
    def run(self):
        pass
""",
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
    detected = detect_hydrocel_montage(
        FakeRaw([f"E{i}" for i in range(1, 128)] + ["E129"])
    )

    assert detected[0] is None
    assert detected[1] == 128
    assert detected[2] is True


def test_build_batch_plan_groups_matching_mixed_and_unknown_files(
    tmp_path: Path,
) -> None:
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

    statuses = {
        (item.relative_path, item.detected_montage, item.status) for item in plan.files
    }
    assert ("sub-01.raw", "GSN-HydroCel-128", "match") in statuses
    assert ("sub-02.raw", "GSN-HydroCel-129", "mismatch") in statuses
    assert ("sub-03.raw", None, "unknown") in statuses
    assert str(file_unknown) in plan.unknown_files
    assert str(file_128) in plan.actionable_files
    assert str(file_129) in plan.actionable_files
    assert str(input_dir / "notes.txt") not in plan.actionable_files


def test_build_batch_plan_discovers_plugin_registered_edf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_dir = tmp_path / "input"
    edf_file = _touch(input_dir / "sub-01.edf")
    task = _task_file(tmp_path)
    discovery_calls = []

    def fake_discover_plugins() -> None:
        from autoclean.io.import_ import register_format

        discovery_calls.append(True)
        register_format("edf", "EDF_FORMAT")

    monkeypatch.setattr(
        "autoclean.utils.montage_preflight.discover_plugins",
        fake_discover_plugins,
    )

    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )

    assert discovery_calls
    assert [Path(result.path) for result in plan.files] == [edf_file]
    assert plan.files[0].format_id == "EDF_FORMAT"


def test_mff_package_internals_are_not_scanned(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    mff_dir = input_dir / "sub-01.mff"
    _touch(mff_dir / "signal.raw")
    task = _task_file(tmp_path)

    def loader(path: Path) -> FakeRaw:
        assert path == mff_dir
        return FakeRaw([f"E{i}" for i in range(1, 129)])

    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=loader,
    )

    assert [Path(result.path) for result in plan.files] == [mff_dir]
    assert plan.files[0].relative_path == "sub-01.mff"


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


def test_estimate_copy_originals_reports_size_and_free_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    monkeypatch.setattr(
        "autoclean.utils.montage_preflight.shutil.disk_usage",
        lambda _path: type("DiskUsage", (), {"free": 1_000})(),
    )

    estimate = estimate_copy_originals_for_plan(
        plan,
        split_output_root=tmp_path / "missing-parent" / "split",
    )

    assert estimate.split_output_root == str(tmp_path / "missing-parent" / "split")
    assert estimate.actionable_file_count == 1
    assert estimate.skipped_file_count == 1
    assert estimate.required_bytes == file_128.stat().st_size
    assert estimate.free_bytes_before == 1_000
    assert estimate.free_bytes_after_estimate == 1_000 - file_128.stat().st_size


def test_copy_originals_allows_missing_destination_parent(tmp_path: Path) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw", "alpha")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )

    result = copy_originals_for_plan(
        plan,
        split_output_root=tmp_path / "missing-parent" / "split",
    )

    copied_path = (
        tmp_path / "missing-parent" / "split" / "GSN-HydroCel-128" / "sub-01.raw"
    )
    assert copied_path.read_text(encoding="utf-8") == "alpha"
    assert result.completed is True


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


def test_copy_originals_reports_partial_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_dir = tmp_path / "input"
    file_a = _touch(input_dir / "a.raw", "alpha")
    file_b = _touch(input_dir / "b.raw", "beta")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )

    def fake_copy2(source: Path, destination: Path):
        if Path(source) == file_b:
            raise PermissionError("blocked")
        Path(destination).write_text(Path(source).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        "autoclean.utils.montage_preflight.shutil.copy2",
        fake_copy2,
    )

    with pytest.raises(MontageCopyError) as exc_info:
        copy_originals_for_plan(plan, split_output_root=tmp_path / "split")

    partial = exc_info.value.partial_result
    assert partial.completed is False
    assert partial.copied_files[0]["source"] == str(file_a)
    assert partial.errors == [
        {
            "source": str(file_b),
            "destination": str(tmp_path / "split" / "GSN-HydroCel-128" / "b.raw"),
            "size_bytes": 4,
            "error": "blocked",
        }
    ]


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


def test_move_originals_refuses_unknown_files(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="unknown or unsupported"):
        write_planned_move_manifest(
            plan,
            output_dir=tmp_path / "out",
            split_output_root=tmp_path / "split",
        )

    assert file_128.exists()
    assert file_unknown.exists()
    assert not (tmp_path / "out" / "autoclean_montage_move_manifest.json").exists()


def test_move_originals_blocks_existing_destinations(tmp_path: Path) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw", "alpha")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )
    _touch(tmp_path / "split" / "GSN-HydroCel-128" / "sub-01.raw", "exists")

    with pytest.raises(FileExistsError, match="overwrite"):
        write_planned_move_manifest(
            plan,
            output_dir=tmp_path / "out",
            split_output_root=tmp_path / "split",
        )

    assert input_file.read_text(encoding="utf-8") == "alpha"


def test_move_originals_writes_manifest_before_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw", "alpha")
    task = _task_file(tmp_path)
    output_dir = tmp_path / "out"
    split_root = tmp_path / "split"
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=output_dir,
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )
    manifest = write_planned_move_manifest(
        plan,
        output_dir=output_dir,
        split_output_root=split_root,
    )
    calls = []

    def fake_copy2(source: Path, destination: Path):
        assert manifest.is_file()
        calls.append((source, destination))
        Path(destination).write_text(Path(source).read_text(encoding="utf-8"))

    monkeypatch.setattr("autoclean.utils.montage_preflight.shutil.copy2", fake_copy2)

    result = move_originals_for_plan(
        plan,
        split_output_root=split_root,
        planned_manifest=manifest,
    )

    assert calls
    assert result.planned_manifest == str(manifest)
    assert result.completed is True


def test_move_originals_verifies_before_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw", "alpha")
    task = _task_file(tmp_path)
    split_root = tmp_path / "split"
    destination = split_root / "GSN-HydroCel-128" / "sub-01.raw"
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )
    manifest = write_planned_move_manifest(
        plan,
        output_dir=tmp_path / "out",
        split_output_root=split_root,
    )
    expected_size = input_file.stat().st_size
    original_unlink = Path.unlink
    delete_checks = []

    def checked_unlink(self, *args, **kwargs):
        if self == input_file:
            delete_checks.append(destination.stat().st_size)
            assert destination.exists()
            assert destination.stat().st_size == expected_size
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", checked_unlink)

    move_originals_for_plan(
        plan,
        split_output_root=split_root,
        planned_manifest=manifest,
    )

    assert delete_checks == [expected_size]
    assert not input_file.exists()
    assert destination.read_text(encoding="utf-8") == "alpha"


def test_move_originals_preserves_source_on_verification_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw", "alpha")
    task = _task_file(tmp_path)
    split_root = tmp_path / "split"
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )
    manifest = write_planned_move_manifest(
        plan,
        output_dir=tmp_path / "out",
        split_output_root=split_root,
    )

    def fake_copy2(_source: Path, destination: Path):
        Path(destination).write_text("truncated", encoding="utf-8")

    monkeypatch.setattr("autoclean.utils.montage_preflight.shutil.copy2", fake_copy2)

    with pytest.raises(MontageMoveError) as exc_info:
        move_originals_for_plan(
            plan,
            split_output_root=split_root,
            planned_manifest=manifest,
        )

    assert input_file.read_text(encoding="utf-8") == "alpha"
    assert exc_info.value.partial_result.completed is False
    assert exc_info.value.partial_result.errors[0]["size_bytes"] == 5
    assert "size mismatch" in exc_info.value.partial_result.errors[0]["error"]


def test_move_originals_refuses_insufficient_temporary_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_file = _touch(tmp_path / "input" / "sub-01.raw", "alpha")
    task = _task_file(tmp_path)
    plan = build_batch_plan(
        input_path=input_file,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=lambda _path: FakeRaw([f"E{i}" for i in range(1, 129)]),
    )
    monkeypatch.setattr(
        "autoclean.utils.montage_preflight.shutil.disk_usage",
        lambda _path: type("DiskUsage", (), {"free": 1})(),
    )

    estimate = estimate_move_originals_for_plan(
        plan,
        split_output_root=tmp_path / "split",
    )

    assert estimate.required_bytes == input_file.stat().st_size
    with pytest.raises(RuntimeError, match="Insufficient temporary free space"):
        write_planned_move_manifest(
            plan,
            output_dir=tmp_path / "out",
            split_output_root=tmp_path / "split",
            estimate=estimate,
        )
    assert input_file.exists()


def test_move_originals_synthetic_temp_dir_move(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    file_128 = _touch(input_dir / "sub-01.raw", "alpha")
    file_129 = _touch(input_dir / "sub-02.raw", "bravo")
    task = _task_file(tmp_path)

    def loader(path: Path) -> FakeRaw:
        if path == file_128:
            return FakeRaw([f"E{i}" for i in range(1, 129)])
        if path == file_129:
            return FakeRaw([f"E{i}" for i in range(1, 130)])
        raise AssertionError(f"unexpected path {path}")

    plan = build_batch_plan(
        input_path=input_dir,
        task_path=task,
        output_dir=tmp_path / "out",
        raw_loader=loader,
    )
    manifest = write_planned_move_manifest(
        plan,
        output_dir=tmp_path / "out",
        split_output_root=tmp_path / "split",
    )

    result = move_originals_for_plan(
        plan,
        split_output_root=tmp_path / "split",
        planned_manifest=manifest,
    )

    assert result.completed is True
    assert len(result.moved_files) == 2
    assert not file_128.exists()
    assert not file_129.exists()
    assert (tmp_path / "split" / "GSN-HydroCel-128" / "sub-01.raw").read_text(
        encoding="utf-8"
    ) == "alpha"
    assert (tmp_path / "split" / "GSN-HydroCel-129" / "sub-02.raw").read_text(
        encoding="utf-8"
    ) == "bravo"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [Path(item["source"]).name for item in manifest_payload["moves"]] == [
        "sub-01.raw",
        "sub-02.raw",
    ]


def test_task_clone_changes_montage_class_name_and_adds_provenance(
    tmp_path: Path,
) -> None:
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
