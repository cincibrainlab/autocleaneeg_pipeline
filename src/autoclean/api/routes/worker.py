"""Worker management API routes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from autoclean.api.models import (
    JobInfo,
    WorkerInfo,
    WorkerStartRequest,
    WorkerStartResponse,
    WorkerStatus,
    WorkerStatusResponse,
    WorkerStopRequest,
    WorkerStopResponse,
)
from autoclean.api.state import api_state

router = APIRouter()

# Track started worker processes
_worker_processes: list[subprocess.Popen] = []


def _get_rq_workers() -> list[dict[str, Any]]:
    """Get information about RQ workers from Redis."""
    try:
        from rq import Worker

        workers = Worker.all(connection=api_state.redis)
        result = []
        for w in workers:
            current_job = w.get_current_job()
            result.append({
                "name": w.name,
                "state": w.state,
                "current_job_id": current_job.id if current_job else None,
                "queues": [q.name for q in w.queues],
                "pid": w.pid,
            })
        return result
    except Exception:
        return []


def _get_queue_job_counts() -> tuple[int, int]:
    """Get active and queued job counts."""
    try:
        from rq import Queue

        q = Queue(connection=api_state.redis)
        return len(q.started_job_registry), len(q)
    except Exception:
        return 0, 0


@router.get("/status", response_model=WorkerStatusResponse)
async def get_worker_status() -> WorkerStatusResponse:
    """Get status of all workers."""
    redis_ok = api_state.check_redis()

    if not redis_ok:
        return WorkerStatusResponse(
            workers=[],
            total_workers=0,
            active_jobs=0,
            queued_jobs=0,
            redis_connected=False,
        )

    workers_data = _get_rq_workers()
    active_jobs, queued_jobs = _get_queue_job_counts()

    workers = []
    for w in workers_data:
        state = w.get("state", "idle")
        if state == "busy":
            status = WorkerStatus.BUSY
        elif state == "idle":
            status = WorkerStatus.IDLE
        else:
            status = WorkerStatus.STOPPED

        workers.append(
            WorkerInfo(
                name=w["name"],
                status=status,
                current_job=w.get("current_job_id"),
                queues=w.get("queues", []),
                pid=w.get("pid"),
            )
        )

    return WorkerStatusResponse(
        workers=workers,
        total_workers=len(workers),
        active_jobs=active_jobs,
        queued_jobs=queued_jobs,
        redis_connected=True,
    )


@router.post("/start", response_model=WorkerStartResponse)
async def start_workers(request: WorkerStartRequest) -> WorkerStartResponse:
    """Start RQ worker processes."""
    global _worker_processes

    if not api_state.check_redis():
        raise HTTPException(status_code=503, detail="Redis not available")

    if not api_state.workspace_dir:
        raise HTTPException(status_code=500, detail="Workspace not configured")

    started_pids = []

    for _ in range(request.count):
        # Build worker command
        queues_str = ",".join(request.queues)
        cmd = [
            sys.executable,
            "-m",
            "rq.cli",
            "worker",
            "--url",
            api_state.redis_url,
            queues_str,
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            _worker_processes.append(proc)
            started_pids.append(proc.pid)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Failed to start worker: {exc}"
            )

    return WorkerStartResponse(started=len(started_pids), pids=started_pids)


@router.post("/stop", response_model=WorkerStopResponse)
async def stop_workers(request: WorkerStopRequest) -> WorkerStopResponse:
    """Stop RQ worker processes."""
    global _worker_processes

    stopped = 0

    # Stop workers we started
    for proc in _worker_processes[:]:
        try:
            if proc.poll() is None:  # Still running
                if request.graceful:
                    proc.terminate()
                else:
                    proc.kill()
                proc.wait(timeout=10)
                stopped += 1
            _worker_processes.remove(proc)
        except Exception:
            pass

    # Also try to stop workers registered in Redis
    if api_state.check_redis():
        try:
            from rq import Worker

            workers = Worker.all(connection=api_state.redis)
            for w in workers:
                try:
                    if request.graceful:
                        os.kill(w.pid, signal.SIGTERM)
                    else:
                        os.kill(w.pid, signal.SIGKILL)
                    stopped += 1
                except (ProcessLookupError, PermissionError):
                    pass
        except Exception:
            pass

    return WorkerStopResponse(stopped=stopped)


@router.get("/jobs", response_model=list[JobInfo])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
) -> list[JobInfo]:
    """List jobs in the queue."""
    if not api_state.check_redis():
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        from rq import Queue
        from rq.job import Job
        from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry

        q = Queue(connection=api_state.redis)
        jobs = []

        # Get jobs from different registries based on status filter
        if status in (None, "queued"):
            for job_id in q.job_ids[:limit]:
                try:
                    job = Job.fetch(job_id, connection=api_state.redis)
                    jobs.append(_job_to_info(job))
                except Exception:
                    pass

        if status in (None, "started"):
            registry = StartedJobRegistry(queue=q)
            for job_id in registry.get_job_ids()[:limit]:
                try:
                    job = Job.fetch(job_id, connection=api_state.redis)
                    jobs.append(_job_to_info(job))
                except Exception:
                    pass

        if status in (None, "finished"):
            registry = FinishedJobRegistry(queue=q)
            for job_id in registry.get_job_ids()[:limit]:
                try:
                    job = Job.fetch(job_id, connection=api_state.redis)
                    jobs.append(_job_to_info(job))
                except Exception:
                    pass

        if status in (None, "failed"):
            registry = FailedJobRegistry(queue=q)
            for job_id in registry.get_job_ids()[:limit]:
                try:
                    job = Job.fetch(job_id, connection=api_state.redis)
                    jobs.append(_job_to_info(job))
                except Exception:
                    pass

        return jobs[:limit]

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _job_to_info(job) -> JobInfo:
    """Convert RQ Job to JobInfo model."""
    return JobInfo(
        id=job.id,
        status=job.get_status() or "unknown",
        func_name=job.func_name or "",
        args=list(job.args) if job.args else [],
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        ended_at=job.ended_at.isoformat() if job.ended_at else None,
        result=job.result if job.is_finished else None,
        error=str(job.exc_info) if job.is_failed and job.exc_info else None,
    )


@router.post("/enqueue/{task_name}")
async def enqueue_task(
    task_name: str,
    workspace_dir: Optional[str] = None,
    mode: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enqueue a task for execution.

    Available tasks:
    - dispatch_ready: Scan and dispatch ready files
    - ingestion_cycle: Run a complete ingestion cycle
    """
    if not api_state.check_redis():
        raise HTTPException(status_code=503, detail="Redis not available")

    ws_dir = workspace_dir or str(api_state.workspace_dir)
    task_mode = mode or api_state.mode

    if not ws_dir:
        raise HTTPException(status_code=400, detail="Workspace directory required")

    from autoclean.api import tasks

    task_map = {
        "dispatch_ready": tasks.dispatch_ready_files,
        "ingestion_cycle": tasks.run_ingestion_cycle,
    }

    if task_name not in task_map:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task_name}. Available: {list(task_map.keys())}",
        )

    try:
        job = api_state.rq_queue.enqueue(
            task_map[task_name],
            workspace_dir=ws_dir,
            mode=task_mode,
            dry_run=dry_run,
        )

        return {
            "job_id": job.id,
            "task": task_name,
            "status": "queued",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/job/{job_id}", response_model=JobInfo)
async def get_job(job_id: str) -> JobInfo:
    """Get information about a specific job."""
    if not api_state.check_redis():
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=api_state.redis)
        return _job_to_info(job)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {exc}")


@router.delete("/job/{job_id}")
async def cancel_job(job_id: str) -> dict[str, str]:
    """Cancel a queued job."""
    if not api_state.check_redis():
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        from rq.job import Job

        job = Job.fetch(job_id, connection=api_state.redis)
        job.cancel()

        return {"status": "cancelled", "job_id": job_id}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {exc}")
