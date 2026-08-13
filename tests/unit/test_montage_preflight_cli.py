from __future__ import annotations

from types import SimpleNamespace

from autoclean.cli import cmd_montage_preflight, create_parser
from autoclean.utils.montage_preflight import (
    MontageBatchPlan,
    MontagePreflightFileResult,
    MontagePreflightGroup,
)


def test_montage_preflight_parser_defaults() -> None:
    parser = create_parser()
    args = parser.parse_args(
        [
            "montage",
            "preflight",
            "--input",
            "/tmp/input",
            "--task",
            "/tmp/task.py",
            "--output",
            "/tmp/output",
        ]
    )

    assert args.command == "montage"
    assert args.montage_action == "preflight"
    assert args.apply is False
    assert args.copy_originals is False
    assert args.clone_tasks is False


def test_cmd_montage_preflight_writes_dry_run_artifacts(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")
    output_dir = tmp_path / "out"
    plan = MontageBatchPlan(
        input_path=str(input_file),
        task_path=str(task_file),
        expected_montage="GSN-HydroCel-128",
        output_dir=str(output_dir),
        groups=[
            MontagePreflightGroup(
                detected_montage="GSN-HydroCel-128",
                status="match",
                file_count=1,
                total_size_bytes=3,
                examples=["sub-01.raw"],
            )
        ],
        files=[
            MontagePreflightFileResult(
                path=str(input_file),
                relative_path="sub-01.raw",
                format_id="EGI_RAW",
                expected_montage="GSN-HydroCel-128",
                detected_montage="GSN-HydroCel-128",
                status="match",
                eeg_channel_count=128,
                e129_present=False,
                size_bytes=3,
            )
        ],
        unknown_files=[],
        actionable_files=[str(input_file)],
    )

    monkeypatch.setattr("autoclean.cli.build_batch_plan", lambda **kwargs: plan)

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=output_dir,
        dry_run=False,
        apply=False,
        copy_originals=False,
        split_output_root=None,
        clone_tasks=False,
        task_output_dir=None,
        yes=False,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 0
    assert (output_dir / "autoclean_montage_scan.csv").is_file()
    assert (output_dir / "autoclean_montage_batch_plan.json").is_file()
    assert not (output_dir / "autoclean_montage_apply_summary.json").exists()


def test_cmd_montage_preflight_apply_clone_uses_keyword_plan(
    monkeypatch, tmp_path
) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")
    output_dir = tmp_path / "out"
    task_output_dir = tmp_path / "tasks"
    plan = MontageBatchPlan(
        input_path=str(input_file),
        task_path=str(task_file),
        expected_montage="GSN-HydroCel-128",
        output_dir=str(output_dir),
        groups=[],
        files=[],
        unknown_files=[],
        actionable_files=[],
    )
    calls = {}

    def fake_clone_tasks_for_mismatches(**kwargs):
        calls.update(kwargs)
        return []

    monkeypatch.setattr("autoclean.cli.build_batch_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        "autoclean.cli.clone_tasks_for_mismatches",
        fake_clone_tasks_for_mismatches,
    )

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=output_dir,
        dry_run=False,
        apply=True,
        copy_originals=False,
        split_output_root=None,
        clone_tasks=True,
        task_output_dir=task_output_dir,
        yes=True,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 0
    assert calls["plan"] is plan
    assert calls["task_output_dir"] == task_output_dir
    assert (output_dir / "autoclean_montage_apply_summary.json").is_file()
