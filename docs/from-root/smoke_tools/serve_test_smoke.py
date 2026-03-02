#!/usr/bin/env python3
"""One-command serve test-mode end-to-end smoke workflow.

This script is intentionally de-identified and non-destructive by default.
It validates config, deploys test config, ensures runtime is resolvable,
enqueues one sample file, waits for queue status transitions, and verifies
expected output artifacts.

Default mode uses a mock runner so operators can validate orchestration without
real EEG data or long processing runs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow running from source checkout without package install.
def _discover_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "autoclean").exists():
            return candidate
    raise RuntimeError("Could not locate repository root (pyproject.toml + src/autoclean)")


REPO_ROOT = _discover_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import importlib.util

_INGESTION_PATH = SRC_DIR / "autoclean" / "utils" / "ingestion.py"
_spec = importlib.util.spec_from_file_location("ac_ingestion", _INGESTION_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load ingestion module from {_INGESTION_PATH}")
ac_ingestion = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ac_ingestion
_spec.loader.exec_module(ac_ingestion)

IngestionQueue = ac_ingestion.IngestionQueue
ServeConfig = ac_ingestion.ServeConfig
ServeRoute = ac_ingestion.ServeRoute
dispatch_ready_ingestion = ac_ingestion.dispatch_ready_ingestion
load_serve_config = ac_ingestion.load_serve_config
parse_serve_config = ac_ingestion.parse_serve_config
resolve_runtime_cli = ac_ingestion.resolve_runtime_cli


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _info(msg: str) -> None:
    print(f"[serve-smoke] {msg}")


def _resolve_config_path(workspace: Path, mode: str, deployed: bool = False) -> Path:
    if deployed:
        return workspace / "deploy" / f"serve-{mode}.yaml"
    return workspace / f"serve-{mode}.yaml"


def _runtime_python_bin(venv_dir: Path) -> Path:
    if sys.platform.startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@dataclass
class SmokeReport:
    started_at: str
    ended_at: str
    workspace: str
    mode: str
    config_path: str
    deployed_config_path: str
    route_id: str
    queue_path: str
    sample_file: str
    statuses_seen: list[str]
    final_status: str
    artifacts: list[str]
    mock_runner: bool
    notes: list[str]


def _precreate_workspace_paths(raw_config: dict[str, Any], workspace: Path) -> None:
    """Create referenced dirs so strict config validation can run cleanly."""
    runtime_value = raw_config.get("runtime")
    if isinstance(runtime_value, str) and runtime_value:
        runtime_path = Path(runtime_value)
        if not runtime_path.is_absolute():
            runtime_path = (workspace / runtime_path).resolve()
        runtime_path.mkdir(parents=True, exist_ok=True)

    defaults = raw_config.get("defaults") if isinstance(raw_config.get("defaults"), dict) else {}
    automation_root = defaults.get("automation_root", raw_config.get("automation_root"))
    if isinstance(automation_root, str) and automation_root:
        ar = Path(automation_root)
        if not ar.is_absolute():
            ar = (workspace / ar).resolve()
        ar.mkdir(parents=True, exist_ok=True)

    automations = raw_config.get("automations")
    if automations is None:
        automations = [raw_config]
    if isinstance(automations, list):
        for route in automations:
            if not isinstance(route, dict):
                continue
            folders = route.get("ingestion_folders")
            if isinstance(folders, list):
                for folder in folders:
                    if not isinstance(folder, str) or not folder:
                        continue
                    p = Path(folder)
                    if not p.is_absolute():
                        p = (workspace / p).resolve()
                    p.mkdir(parents=True, exist_ok=True)


def _deploy_config(workspace: Path, mode: str) -> Path:
    src = _resolve_config_path(workspace, mode, deployed=False)
    dst = _resolve_config_path(workspace, mode, deployed=True)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    shutil.copy2(src, dst)
    dst.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return dst


def _ensure_runtime(
    *,
    workspace: Path,
    raw_config: dict[str, Any],
    parsed: ServeConfig,
    bootstrap_runtime: bool,
) -> Path:
    """Ensure runtime CLI is resolvable, optionally bootstrapping via uv."""
    try:
        return resolve_runtime_cli(parsed.runtime_path)
    except FileNotFoundError:
        pass

    if not bootstrap_runtime:
        raise FileNotFoundError(
            "Runtime CLI not found. Re-run with --bootstrap-runtime or run "
            "`autocleaneeg-pipeline serve deploy --mode test` first."
        )

    package_spec = raw_config.get("runtime_package", "autocleaneeg-pipeline")
    venv_dir = parsed.runtime_path / ".venv"

    if not shutil.which("uv"):
        raise RuntimeError("uv is required for --bootstrap-runtime but was not found")

    _info(f"Bootstrapping runtime at {parsed.runtime_path} with package={package_spec}")
    if not venv_dir.exists():
        subprocess.run(["uv", "venv", str(venv_dir)], check=True)
    py_bin = _runtime_python_bin(venv_dir)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(py_bin), str(package_spec)],
        check=True,
    )

    return resolve_runtime_cli(parsed.runtime_path)


def _pick_route(parsed: ServeConfig, route_id: Optional[str]) -> ServeRoute:
    enabled = [r for r in parsed.routes if r.enabled]
    if not enabled:
        raise ValueError("No enabled automations found in config")

    if route_id is None:
        return enabled[0]

    for route in enabled:
        if route.id == route_id:
            return route
    raise ValueError(f"Route id not found or disabled: {route_id}")


def _derive_sample_name(route: ServeRoute) -> str:
    suffix = ".set"
    for pattern in route.file_globs:
        if pattern.startswith("*."):
            suffix = pattern[1:]
            break
    return f"deidentified_smoke{suffix}"


def _ensure_sample_file(route: ServeRoute, sample_name: Optional[str]) -> Path:
    root = route.ingestion_folders[0]
    root.mkdir(parents=True, exist_ok=True)

    name = sample_name or _derive_sample_name(route)
    sample = root / name
    sample.write_text(
        "deidentified smoke payload\n"
        "subject: smoke-test\n"
        "note: no PHI\n",
        encoding="utf-8",
    )

    if route.sentinel_ext:
        sentinel = sample.with_name(sample.name + route.sentinel_ext)
        sentinel.write_text("ready\n", encoding="utf-8")

    return sample


def _wait_for_status(
    queue: IngestionQueue,
    sample: Path,
    *,
    timeout_seconds: float,
    terminal: set[str],
) -> tuple[list[str], str]:
    start = time.time()
    statuses_seen: list[str] = []
    last: Optional[str] = None

    while time.time() - start < timeout_seconds:
        status = queue.entries().get(str(sample), {}).get("status", "missing")
        if status != last:
            statuses_seen.append(status)
            last = status
        if status in terminal:
            return statuses_seen, status
        time.sleep(0.25)
    return statuses_seen, (last or "missing")


def _mock_runner_factory(artifact_paths: list[Path]):
    def _runner(cmd: list[str]) -> None:
        out_dir: Optional[Path] = None
        in_file: Optional[Path] = None

        for i, token in enumerate(cmd):
            if token == "--output" and i + 1 < len(cmd):
                out_dir = Path(cmd[i + 1])
            if token == "--file" and i + 1 < len(cmd):
                in_file = Path(cmd[i + 1])

        if out_dir is None:
            raise ValueError(f"--output missing in command: {' '.join(cmd)}")

        out_dir.mkdir(parents=True, exist_ok=True)
        smoke_dir = out_dir / "_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)

        summary = smoke_dir / "dispatch-summary.json"
        summary.write_text(
            json.dumps(
                {
                    "generated_at": _utc_now(),
                    "command": cmd,
                    "sample_file": str(in_file) if in_file else None,
                    "status": "ok",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_paths.append(summary)

        log_file = smoke_dir / "runner.log"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{_utc_now()} CMD {' '.join(cmd)}\n")
        artifact_paths.append(log_file)

    return _runner


def run_smoke(args: argparse.Namespace) -> SmokeReport:
    started = _utc_now()
    workspace = args.workspace.expanduser().resolve()
    mode = args.mode

    config_path = _resolve_config_path(workspace, mode, deployed=False)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw_config = load_serve_config(config_path)
    _precreate_workspace_paths(raw_config, workspace)

    parsed, warnings = parse_serve_config(raw_config, workspace, strict=True)
    for w in warnings:
        _info(f"warning: {w}")

    deployed_config = _deploy_config(workspace, mode)
    _info(f"validated+deployed config: {deployed_config}")

    runtime_cli = _ensure_runtime(
        workspace=workspace,
        raw_config=raw_config,
        parsed=parsed,
        bootstrap_runtime=args.bootstrap_runtime,
    )
    _info(f"runtime CLI: {runtime_cli}")

    route = _pick_route(parsed, args.route_id)
    sample = _ensure_sample_file(route, args.sample_name)

    queue_path = workspace / f"queue-{mode}.json"
    queue = IngestionQueue(queue_path)

    if args.reset_existing:
        queue.entries().pop(str(sample), None)
        queue.save()

    queue.enqueue([sample], route_id=route.id, ingestion_root=route.ingestion_folders[0])

    statuses_seen, status = _wait_for_status(
        queue,
        sample,
        timeout_seconds=args.timeout_seconds,
        terminal={"pending", "processed", "failed"},
    )
    if status == "missing":
        raise RuntimeError("Sample file entry never appeared in queue")

    artifacts: list[Path] = []
    runner = None
    if args.mock_runner:
        runner = _mock_runner_factory(artifacts)

    dispatch_ready_ingestion(
        config_path=deployed_config,
        workspace_dir=workspace,
        use_watchfiles=False,
        max_events=1,
        automation=True,
        yes=True,
        runner=runner,
        queue=queue,
    )

    status_seq_2, final_status = _wait_for_status(
        queue,
        sample,
        timeout_seconds=args.timeout_seconds,
        terminal={"processed", "failed"},
    )

    # Merge status transitions without duplicate contiguous values.
    merged = statuses_seen[:]
    for s in status_seq_2:
        if not merged or merged[-1] != s:
            merged.append(s)

    notes: list[str] = []
    if final_status == "failed":
        err = queue.entries().get(str(sample), {}).get("last_error")
        if err:
            notes.append(f"queue.last_error={err}")

    if args.mock_runner:
        expected = route.automation_root / route.workspace_name / "_smoke" / "dispatch-summary.json"
        if not expected.exists():
            raise FileNotFoundError(f"Expected artifact not found: {expected}")
        if expected not in artifacts:
            artifacts.append(expected)
    else:
        # Conservative check for real processing path.
        out_dir = route.automation_root / route.workspace_name
        if not out_dir.exists():
            raise FileNotFoundError(f"Expected output directory not found: {out_dir}")
        artifacts.append(out_dir)

    ended = _utc_now()
    return SmokeReport(
        started_at=started,
        ended_at=ended,
        workspace=str(workspace),
        mode=mode,
        config_path=str(config_path),
        deployed_config_path=str(deployed_config),
        route_id=route.id,
        queue_path=str(queue_path),
        sample_file=str(sample),
        statuses_seen=merged,
        final_status=final_status,
        artifacts=[str(p) for p in artifacts],
        mock_runner=bool(args.mock_runner),
        notes=notes,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve test-mode smoke workflow")
    p.add_argument("--workspace", type=Path, required=True, help="Serve workspace path")
    p.add_argument("--mode", choices=["test", "live"], default="test")
    p.add_argument("--route-id", default=None, help="Optional route id override")
    p.add_argument("--sample-name", default=None, help="Optional sample filename")
    p.add_argument(
        "--bootstrap-runtime",
        action="store_true",
        help="Install runtime package with uv if runtime CLI is missing",
    )
    p.add_argument(
        "--mock-runner",
        action="store_true",
        default=True,
        help="Use mock command runner to produce smoke artifacts (default)",
    )
    p.add_argument(
        "--real-runner",
        action="store_true",
        help="Disable mock runner and execute real processing command",
    )
    p.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="Queue transition timeout",
    )
    p.add_argument(
        "--reset-existing",
        action="store_true",
        help="Remove existing queue entry for sample before enqueue",
    )
    p.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    args = p.parse_args()
    if args.real_runner:
        args.mock_runner = False
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_smoke(args)
    except Exception as exc:
        _info(f"ERROR: {exc}")
        return 1

    payload = asdict(report)
    print(json.dumps(payload, indent=2))

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _info(f"report written: {args.report_json}")

    if report.final_status != "processed":
        _info("workflow completed but final status is not processed")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
