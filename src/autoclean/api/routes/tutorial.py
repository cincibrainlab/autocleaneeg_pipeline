"""Tutorial setup and cleanup endpoints.

Provides a guided onboarding experience by generating synthetic EEG data
and setting up a sample processing route.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from autoclean.api.state import api_state

logger = logging.getLogger(__name__)

router = APIRouter()


def _require_workspace():
    """Raise 500 if workspace is not configured."""
    if not api_state.workspace_dir:
        raise HTTPException(status_code=500, detail="Workspace not configured")


def _generate_tutorial_data(tutorial_dir: Path) -> Path:
    """Synchronous helper: generate synthetic EEG and export to disk.

    CPU-bound and I/O-bound work lives here so the caller can run it in a
    thread via asyncio.to_thread(), keeping the async event loop free.

    Returns the path of the exported sample file.
    Raises on failure.
    """
    # Locate the synthetic_data module relative to the package.
    try:
        from autoclean.tests.fixtures.synthetic_data import create_synthetic_raw
    except ImportError:
        import importlib.util

        candidates = [
            Path(__file__).parents[5] / "tests" / "fixtures" / "synthetic_data.py",
            Path(__file__).parents[4] / "tests" / "fixtures" / "synthetic_data.py",
        ]
        spec = None
        for candidate in candidates:
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(
                    "synthetic_data", candidate
                )
                break
        if spec is None:
            raise ImportError("Cannot locate synthetic_data.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        create_synthetic_raw = mod.create_synthetic_raw

    # Use standard_1020 with 32 channels to match BiotrialResting1020,
    # which is the same setup proven to work with real clinical data
    # (201001_D1BL_EC.set processed in 23s). 120s gives plenty of data
    # to survive artifact rejection and produce clean epochs.
    import numpy as np

    raw = create_synthetic_raw(
        montage="standard_1020",
        n_channels=32,
        duration=120.0,
        sfreq=512.0,
        seed=42,
    )

    # Add spatial correlation so RANSAC doesn't flag all channels as
    # noisy. Real EEG has strong common-mode signals across electrodes.
    rng = np.random.RandomState(42)
    n_samples = raw.n_times
    times = np.arange(n_samples) / 512.0
    # Common alpha oscillation shared across all channels
    common = 30e-6 * np.sin(2 * np.pi * 10.0 * times)
    for ch_idx in range(raw.info["nchan"]):
        raw._data[ch_idx] += common * (0.7 + 0.3 * rng.rand())
    # Scale to realistic EEG amplitude range (~160µV peak)
    raw._data *= 2.0

    # Export as .set; fall back to .fif if MNE export fails.
    for attempt in ("set", "fif"):
        if attempt == "set":
            out_path = tutorial_dir / "tutorial_resting.set"
            try:
                raw.export(str(out_path), fmt="eeglab", overwrite=True, verbose=False)
                return out_path
            except Exception as exc:
                logger.warning("EEGLAB export failed (%s), trying .fif", exc)
        else:
            out_path = tutorial_dir / "tutorial_resting.fif"
            try:
                raw.save(str(out_path), overwrite=True, verbose=False)
                return out_path
            except Exception as exc:
                logger.warning(".fif export also failed: %s", exc)

    raise RuntimeError("Could not export synthetic EEG file in any format")


@router.post("/setup")
async def tutorial_setup() -> dict[str, Any]:
    """Generate a synthetic EEG .set file for the tutorial.

    Creates the tutorial incoming directory, generates synthetic EEG data,
    exports as EEGLAB .set (falls back to .fif), and returns a suggested
    route configuration.

    The CPU-bound data-generation work runs in a thread pool so the async
    event loop is not blocked.
    """
    _require_workspace()

    workspace = api_state.workspace_dir
    assert workspace is not None

    tutorial_dir = workspace / "incoming" / "tutorial"
    tutorial_dir.mkdir(parents=True, exist_ok=True)

    try:
        sample_file = await asyncio.to_thread(_generate_tutorial_data, tutorial_dir)
    except Exception as exc:
        logger.exception("Failed to generate synthetic EEG data")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate synthetic EEG data: {exc}"
        )

    # Build suggested route configuration
    suggested_route = {
        "id": "tutorial-resting",
        "taskfile": "BiotrialResting1020",
        "montage": "standard_1020",
        "ingestion_folders": [str(tutorial_dir)],
        "file_globs": [f"*.{sample_file.suffix.lstrip('.')}"],
        "modes": [api_state.mode],
        "enabled": True,
        "recursive": False,
        "priority": 100,
    }

    return {
        "success": True,
        "sample_file": str(sample_file),
        "suggested_route": suggested_route,
    }


@router.post("/cleanup")
async def tutorial_cleanup() -> dict[str, Any]:
    """Remove tutorial artifacts.

    Deletes the tutorial incoming directory, removes the tutorial-resting
    route if it exists, and syncs routes.
    """
    _require_workspace()

    workspace = api_state.workspace_dir
    assert workspace is not None

    messages: list[str] = []

    # Remove tutorial incoming directory
    tutorial_dir = workspace / "incoming" / "tutorial"
    if tutorial_dir.exists():
        shutil.rmtree(tutorial_dir, ignore_errors=True)
        messages.append("Removed tutorial/incoming directory")

    # Track whether any critical step failed so we can surface it to the caller.
    success = True

    # Remove tutorial-resting route if it exists
    try:
        from autoclean.utils.serve_routes import (
            delete_route_spec,
            load_route_specs,
            set_route_archived,
        )

        route_id = "tutorial-resting"
        specs = load_route_specs(workspace)
        exists = any(s.get("id") == route_id for s in specs)

        if exists:
            # Archive first (required by delete_route_spec), then delete
            try:
                set_route_archived(workspace, route_id, archived=True)
            except Exception:
                pass
            ok, err = delete_route_spec(workspace, route_id)
            if ok:
                messages.append(f"Removed route '{route_id}'")
            else:
                success = False
                messages.append(f"Failed to delete route '{route_id}': {err}")
    except Exception as exc:
        success = False
        logger.warning("Could not remove tutorial route: %s", exc)
        messages.append(f"Route removal failed: {exc}")

    # Sync routes to update generated configs
    try:
        from autoclean.utils.serve_routes import sync_route_registry

        sync_route_registry(workspace, modes=("test", "live"))
        messages.append("Routes synced")
    except Exception as exc:
        success = False
        logger.warning("Route sync after cleanup failed: %s", exc)
        messages.append(f"Route sync failed: {exc}")

    return {
        "success": success,
        "message": "; ".join(messages) if messages else "Nothing to clean up",
    }
