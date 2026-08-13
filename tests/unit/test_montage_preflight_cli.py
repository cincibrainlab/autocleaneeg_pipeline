from __future__ import annotations

from types import SimpleNamespace

from autoclean.cli import cmd_montage_preflight, create_parser
from autoclean.utils.montage_preflight import (
    MontageBatchPlan,
    MontageCopyError,
    MontageCopyEstimate,
    MontageCopyResult,
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


def test_cmd_montage_preflight_rejects_apply_flags_without_apply(
    monkeypatch, tmp_path
) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")
    messages = []

    monkeypatch.setattr(
        "autoclean.cli.build_batch_plan",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("build should not run")),
    )
    monkeypatch.setattr(
        "autoclean.cli.message",
        lambda _level, text: messages.append(text),
    )

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=tmp_path / "out",
        dry_run=False,
        apply=False,
        copy_originals=True,
        split_output_root=tmp_path / "split",
        clone_tasks=True,
        task_output_dir=tmp_path / "tasks",
        yes=True,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 1
    assert any("require --apply" in text for text in messages)


def test_cmd_montage_preflight_rejects_apply_and_dry_run(monkeypatch, tmp_path) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")

    monkeypatch.setattr(
        "autoclean.cli.build_batch_plan",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("build should not run")),
    )

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=tmp_path / "out",
        dry_run=True,
        apply=True,
        copy_originals=False,
        split_output_root=None,
        clone_tasks=False,
        task_output_dir=None,
        yes=True,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 1


def test_cmd_montage_preflight_apply_copy_uses_split_output_root(
    monkeypatch, tmp_path
) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")
    output_dir = tmp_path / "out"
    split_output_root = tmp_path / "split"
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

    def fake_copy_originals_for_plan(plan_arg, **kwargs):
        calls["plan"] = plan_arg
        calls.update(kwargs)
        return MontageCopyResult(
            split_output_root=str(split_output_root),
            copied_files=[],
            skipped_files=[],
            required_bytes=0,
            free_bytes_before=100,
            free_bytes_after_estimate=100,
        )

    monkeypatch.setattr("autoclean.cli.build_batch_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        "autoclean.cli.copy_originals_for_plan",
        fake_copy_originals_for_plan,
    )

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=output_dir,
        dry_run=False,
        apply=True,
        copy_originals=True,
        split_output_root=split_output_root,
        clone_tasks=False,
        task_output_dir=None,
        yes=True,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 0
    assert calls["plan"] is plan
    assert calls["split_output_root"] == split_output_root
    assert (output_dir / "autoclean_montage_apply_summary.json").is_file()


def test_cmd_montage_preflight_prints_copy_estimate_before_copy(
    monkeypatch, tmp_path, capsys
) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")
    output_dir = tmp_path / "out"
    split_output_root = tmp_path / "split"
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

    def fake_estimate_copy_originals_for_plan(plan_arg, **kwargs):
        calls["estimate_seen_copy"] = "copy" in calls
        calls["estimate_plan"] = plan_arg
        calls["estimate_kwargs"] = kwargs
        return MontageCopyEstimate(
            split_output_root=str(split_output_root),
            actionable_file_count=2,
            skipped_file_count=1,
            required_bytes=256,
            free_bytes_before=1024,
            free_bytes_after_estimate=768,
        )

    def fake_copy_originals_for_plan(plan_arg, **kwargs):
        calls["copy"] = True
        return MontageCopyResult(
            split_output_root=str(split_output_root),
            copied_files=[],
            skipped_files=[],
            required_bytes=256,
            free_bytes_before=1024,
            free_bytes_after_estimate=768,
        )

    monkeypatch.setattr("autoclean.cli.build_batch_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        "autoclean.cli.estimate_copy_originals_for_plan",
        fake_estimate_copy_originals_for_plan,
    )
    monkeypatch.setattr(
        "autoclean.cli.copy_originals_for_plan",
        fake_copy_originals_for_plan,
    )

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=output_dir,
        dry_run=False,
        apply=True,
        copy_originals=True,
        split_output_root=split_output_root,
        clone_tasks=False,
        task_output_dir=None,
        yes=True,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 0
    assert calls["estimate_seen_copy"] is False
    assert calls["estimate_plan"] is plan
    assert calls["estimate_kwargs"]["split_output_root"] == split_output_root
    assert calls["copy"] is True
    output = capsys.readouterr().out
    assert "Copy Originals Estimate" in output
    assert str(split_output_root) in output
    assert "Required space" in output
    assert "256 bytes" in output
    assert "Available space" in output
    assert "1,024 bytes" in output


def test_cmd_montage_preflight_copy_failure_writes_partial_summary(
    monkeypatch, tmp_path
) -> None:
    input_file = tmp_path / "sub-01.raw"
    input_file.write_text("raw", encoding="utf-8")
    task_file = tmp_path / "Task.py"
    task_file.write_text("config = {}", encoding="utf-8")
    output_dir = tmp_path / "out"
    split_output_root = tmp_path / "split"
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
    partial_result = MontageCopyResult(
        split_output_root=str(split_output_root),
        copied_files=[
            {
                "source": str(input_file),
                "destination": str(
                    split_output_root / "GSN-HydroCel-128" / "sub-01.raw"
                ),
                "detected_montage": "GSN-HydroCel-128",
                "size_bytes": 3,
            }
        ],
        skipped_files=[],
        required_bytes=6,
        free_bytes_before=100,
        free_bytes_after_estimate=94,
        completed=False,
        errors=[{"source": "sub-02.raw", "destination": "dest", "error": "blocked"}],
    )

    def fake_copy_originals_for_plan(*_args, **_kwargs):
        raise MontageCopyError("blocked", partial_result)

    monkeypatch.setattr("autoclean.cli.build_batch_plan", lambda **kwargs: plan)
    monkeypatch.setattr(
        "autoclean.cli.copy_originals_for_plan",
        fake_copy_originals_for_plan,
    )

    args = SimpleNamespace(
        input=input_file,
        task=task_file,
        output=output_dir,
        dry_run=False,
        apply=True,
        copy_originals=True,
        split_output_root=split_output_root,
        clone_tasks=False,
        task_output_dir=None,
        yes=True,
        overwrite=False,
        no_color=True,
        quiet=True,
    )

    assert cmd_montage_preflight(args) == 1
    summary = output_dir / "autoclean_montage_apply_summary.json"
    assert summary.is_file()
    assert '"completed": false' in summary.read_text(encoding="utf-8")


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
