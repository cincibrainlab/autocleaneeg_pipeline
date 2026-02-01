"""RQ task definitions for automation processing."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

def _timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def process_file(
    file_path: str,
    workspace_dir: str,
    mode: str,
    route_id: str,
    taskfile: str,
    montage: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Process a single EEG file.

    This is the main RQ task that processes files through the pipeline.

    Args:
        file_path: Path to the EEG file to process.
        workspace_dir: Path to the serve workspace.
        mode: Configuration mode ("test" or "live").
        route_id: Route ID for this file.
        taskfile: Task file or name to use.
        montage: Montage configuration.
        dry_run: If True, log command but don't execute.

    Returns:
        Dict with processing result.
    """
    workspace = Path(workspace_dir)
    file_path_obj = Path(file_path)

    result = {
        "file_path": file_path,
        "route_id": route_id,
        "started_at": _timestamp(),
        "status": "processing",
    }

    try:
        # Find the runtime CLI
        from autoclean.utils.ingestion import resolve_runtime_cli

        runtime_dir = workspace / "runtimes" / mode
        try:
            cli_path = resolve_runtime_cli(runtime_dir)
        except FileNotFoundError:
            import sys
            cli_path = Path(sys.executable).parent / "autocleaneeg-pipeline"

        # Build command
        cmd = [
            str(cli_path),
            "process",
            "--task", taskfile,
            "--file", str(file_path_obj),
            "--output", str(workspace / "automations" / route_id),
            "--automation",
            "--yes",
        ]

        if dry_run:
            result["command"] = cmd
            result["status"] = "dry_run"
            result["ended_at"] = _timestamp()
            return result

        # Execute
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout[-5000:] if proc.stdout else ""  # Last 5KB
        result["stderr"] = proc.stderr[-5000:] if proc.stderr else ""

        if proc.returncode == 0:
            result["status"] = "completed"
        else:
            result["status"] = "failed"
            result["error"] = f"Process exited with code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["status"] = "failed"
        result["error"] = "Processing timed out after 1 hour"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    result["ended_at"] = _timestamp()
    return result


def dispatch_ready_files(
    workspace_dir: str,
    mode: str,
    route_id: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Dispatch all ready files for processing.

    This task scans for ready files and enqueues individual process_file jobs.

    Args:
        workspace_dir: Path to the serve workspace.
        mode: Configuration mode ("test" or "live").
        route_id: Optional route ID to filter (None = all routes).
        dry_run: If True, don't actually process.

    Returns:
        Dict with dispatch results.
    """
    from autoclean.utils.ingestion import (
        IngestionQueue,
        dispatch_ready_ingestion,
        load_serve_config,
        parse_serve_config,
    )

    workspace = Path(workspace_dir)
    config_path = workspace / f"serve-{mode}.yaml"
    queue_path = workspace / f"queue-{mode}.json"

    result = {
        "started_at": _timestamp(),
        "workspace_dir": workspace_dir,
        "mode": mode,
        "files_found": 0,
        "files_dispatched": 0,
        "errors": [],
    }

    try:
        # Load config
        raw_config = load_serve_config(config_path)
        config, _ = parse_serve_config(raw_config, workspace, strict=False)

        # Load queue
        queue = IngestionQueue(queue_path)

        # Get ready files
        dispatch_results = dispatch_ready_ingestion(
            config_path=config_path,
            workspace_dir=workspace,
            config=config,
            queue=queue,
            automation=not dry_run,
            yes=True,
            runner=lambda cmd: None if dry_run else subprocess.run(cmd, check=True),
        )

        for dr in dispatch_results:
            if route_id and dr.route_id != route_id:
                continue

            result["files_found"] += len(dr.ready.ready_files)

            if dr.result:
                result["files_dispatched"] += len(dr.result.processed)
                for path, error in dr.result.failed.items():
                    result["errors"].append({"path": str(path), "error": error})

    except Exception as exc:
        result["errors"].append({"error": str(exc)})

    result["ended_at"] = _timestamp()
    result["status"] = "completed" if not result["errors"] else "completed_with_errors"

    return result


def run_ingestion_cycle(
    workspace_dir: str,
    mode: str,
    max_cycles: int = 1,
    idle_limit: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a complete ingestion cycle.

    Args:
        workspace_dir: Path to the serve workspace.
        mode: Configuration mode ("test" or "live").
        max_cycles: Maximum number of cycles to run.
        idle_limit: Idle cycles before stopping.
        dry_run: If True, don't actually process.

    Returns:
        Dict with cycle results.
    """
    from autoclean.utils.ingestion import (
        IngestionQueue,
        run_ingestion_service,
    )

    workspace = Path(workspace_dir)
    config_path = workspace / f"serve-{mode}.yaml"
    queue_path = workspace / f"queue-{mode}.json"

    result = {
        "started_at": _timestamp(),
        "workspace_dir": workspace_dir,
        "mode": mode,
        "cycles": 0,
        "idle_cycles": 0,
        "route_stats": {},
    }

    def runner(cmd: list[str]) -> None:
        if not dry_run:
            subprocess.run(cmd, check=True)

    try:
        service_result = run_ingestion_service(
            config_path=config_path,
            workspace_dir=workspace,
            max_cycles=max_cycles,
            idle_limit=idle_limit,
            runner=runner,
            queue_path=queue_path,
        )

        result["cycles"] = service_result.cycles
        result["idle_cycles"] = service_result.idle_cycles

        # Aggregate stats
        for loop_result in service_result.loop_results:
            for dispatch in loop_result.dispatch_results:
                stats = result["route_stats"].setdefault(
                    dispatch.route_id,
                    {"ready": 0, "processed": 0, "failed": 0},
                )
                stats["ready"] += len(dispatch.ready.ready_files)
                if dispatch.result:
                    stats["processed"] += len(dispatch.result.processed)
                    stats["failed"] += len(dispatch.result.failed)

        result["status"] = "completed"

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    result["ended_at"] = _timestamp()
    return result
