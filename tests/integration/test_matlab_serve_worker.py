"""Integration tests for MATLAB-aware serve worker execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoclean.api.tasks import process_file


def _write_runtime_cli(runtime_dir: Path, script_body: str) -> Path:
    cli_path = runtime_dir / ".venv" / "bin" / "autocleaneeg-pipeline"
    cli_path.parent.mkdir(parents=True, exist_ok=True)
    cli_path.write_text(script_body, encoding="utf-8")
    cli_path.chmod(0o755)
    return cli_path


def _write_matlab_taskfile(path: Path) -> None:
    path.write_text(
        (
            "from autoclean.functions.matlab import call_matlab\n"
            "def run():\n"
            "    return call_matlab('sqrt', 4.0)\n"
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("stderr_message", "expected_fragment"),
    [
        ("MATLAB Engine API unavailable: missing matlabengine", "MATLAB preflight failed"),
        ("MATLAB R2025b installation not found", "MATLAB preflight failed"),
    ],
)
def test_process_file_fails_fast_when_runtime_cli_reports_missing_engine(
    tmp_path: Path,
    stderr_message: str,
    expected_fragment: str,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    runtime_dir = workspace / "runtimes" / "test"
    taskfile = workspace / "custom_matlab_task.py"
    taskfile.parent.mkdir(parents=True, exist_ok=True)
    _write_matlab_taskfile(taskfile)

    _write_runtime_cli(
        runtime_dir,
        "\n".join(
            [
                "#!/bin/sh",
                'if [ \"$1\" = \"matlab\" ] && [ \"$2\" = \"doctor\" ]; then',
                f"  echo {stderr_message!r} 1>&2",
                "  exit 1",
                "fi",
                "echo unexpected invocation 1>&2",
                "exit 9",
            ]
        )
        + "\n",
    )
    cli_path = runtime_dir / ".venv" / "bin" / "autocleaneeg-pipeline"
    monkeypatch.setattr("autoclean.utils.ingestion.resolve_runtime_cli", lambda _runtime: cli_path)

    result = process_file(
        file_path=str(tmp_path / "subject.set"),
        workspace_dir=str(workspace),
        mode="test",
        route_id="route-matlab",
        taskfile=str(taskfile),
        montage="biosemi64",
        dry_run=False,
    )

    assert result["status"] == "failed"
    assert expected_fragment in result["error"]


def test_process_file_completes_and_writes_route_artifact_for_matlab_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    runtime_dir = workspace / "runtimes" / "test"
    taskfile = workspace / "custom_matlab_task.py"
    taskfile.parent.mkdir(parents=True, exist_ok=True)
    _write_matlab_taskfile(taskfile)
    route_output_dir = workspace / "automations" / "route-matlab"
    route_output_dir.mkdir(parents=True, exist_ok=True)
    process_log = workspace / "runtime-process.log"

    _write_runtime_cli(
        runtime_dir,
        "\n".join(
            [
                "#!/bin/sh",
                'if [ \"$1\" = \"matlab\" ] && [ \"$2\" = \"doctor\" ]; then',
                "  echo 'MATLAB runtime ready'",
                "  exit 0",
                "fi",
                'if [ \"$1\" = \"process\" ]; then',
                "  output_dir=''",
                "  while [ $# -gt 0 ]; do",
                '    if [ \"$1\" = \"--output\" ]; then',
                "      shift",
                "      output_dir=\"$1\"",
                "    fi",
                "    shift",
                "  done",
                f"  echo \"$output_dir\" > {str(process_log)!r}",
                "  mkdir -p \"$output_dir\"",
                "  printf '%s' 'matlab-artifact-ok' > \"$output_dir/matlab_artifact.txt\"",
                "  exit 0",
                "fi",
                "echo unexpected invocation 1>&2",
                "exit 9",
            ]
        )
        + "\n",
    )
    cli_path = runtime_dir / ".venv" / "bin" / "autocleaneeg-pipeline"
    monkeypatch.setattr("autoclean.utils.ingestion.resolve_runtime_cli", lambda _runtime: cli_path)

    result = process_file(
        file_path=str(tmp_path / 'subject.set'),
        workspace_dir=str(workspace),
        mode="test",
        route_id="route-matlab",
        taskfile=str(taskfile),
        montage="biosemi64",
        dry_run=False,
    )

    assert result["status"] == "completed"
    assert result["matlab_preflight"]["required"] is True
    assert (route_output_dir / "matlab_artifact.txt").read_text(encoding="utf-8") == "matlab-artifact-ok"
    assert Path(process_log.read_text(encoding="utf-8").strip()) == route_output_dir
