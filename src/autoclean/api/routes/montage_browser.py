"""Montage browser endpoints — visual montage discovery for the web UI.

Lists all available EEG electrode montages from configs/montages.yaml,
enriches them with MNE channel data, and serves server-rendered 2D
topomaps as base64-encoded PNGs so operators can browse and compare
montages without needing a local Python environment.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib
if matplotlib.get_backend() != "Agg":
    matplotlib.use("Agg")  # Non-interactive backend — only set if not already Agg
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Constants ───────────────────────────────────────────────────────

# Maps montage name prefix/substring to a display category
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("GSN-", "geodesic"),
    ("EGI_", "geodesic"),
    ("biosemi", "biosemi"),
    ("mgh", "specialty"),
    ("easycap", "specialty"),
    ("artinis", "specialty"),
    ("MEA", "custom"),
    ("Mouse", "custom"),
    ("Grael", "custom"),
    ("standard_", "standard"),
]


def _categorize(name: str) -> str:
    """Return a category label for a montage name."""
    for prefix, category in _CATEGORY_RULES:
        if name.startswith(prefix):
            return category
    return "standard"


def _montage_data_dir() -> Path:
    """Return the path to bundled custom montage files."""
    import autoclean
    import inspect
    pkg_root = Path(inspect.getfile(autoclean)).parent
    return pkg_root / "data" / "montages"


def _load_montage(name: str) -> Any:
    """Load an MNE montage object for the given montage name.

    Tries ``make_standard_montage`` first; falls back to
    ``read_custom_montage`` from the bundled data directory for custom
    montages. Returns ``None`` if neither succeeds.
    """
    import mne

    # Standard montage
    try:
        return mne.channels.make_standard_montage(name)
    except Exception:
        pass

    # Custom montage from bundled sfp files
    data_dir = _montage_data_dir()
    for suffix in (".sfp", ".elc", ".xyz", ".txt"):
        candidate = data_dir / f"{name}{suffix}"
        if candidate.exists():
            try:
                return mne.channels.read_custom_montage(str(candidate))
            except Exception as exc:
                logger.debug("Could not read custom montage '%s' from %s: %s", name, candidate, exc)

    return None


_topomap_cache: dict[str, str] = {}


def _generate_topomap(montage: Any, name: str = "") -> str:
    """Generate a 2D electrode position map as a base64-encoded PNG.

    Uses a top-down head view: X = left/right, Y = front/back (nose = +Y).
    Cached by montage name since electrode positions are static.

    Returns an empty string if the montage has no channel positions.
    """
    if name and name in _topomap_cache:
        return _topomap_cache[name]

    positions = montage.get_positions()
    ch_pos: dict[str, Any] = positions.get("ch_pos", {})
    if not ch_pos:
        return ""

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=120)
    try:
        fig.patch.set_facecolor("#1a1a1a")
        ax.set_facecolor("#1a1a1a")

        xs: list[float] = []
        ys: list[float] = []
        names: list[str] = []
        for ch_name, pos in ch_pos.items():
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
            names.append(ch_name)

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)

        head_radius = 0.095
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(head_radius * np.cos(theta), head_radius * np.sin(theta),
                color="#3ecf8e", linewidth=1.5, alpha=0.4)
        ax.plot([0, 0.01, 0], [head_radius, head_radius + 0.01, head_radius],
                color="#3ecf8e", linewidth=1.5, alpha=0.4)
        for sign in (-1, 1):
            ax.plot([sign * head_radius, sign * (head_radius + 0.01), sign * head_radius],
                    [0.01, 0, -0.01], color="#3ecf8e", linewidth=1.5, alpha=0.3)

        ax.scatter(xs_arr, ys_arr, c="#3ecf8e", s=20, alpha=0.8, zorder=5)

        if len(names) <= 64:
            for ch_name, x, y in zip(names, xs, ys):
                ax.annotate(ch_name, (x, y), fontsize=4, color="#a1a1aa",
                            ha="center", va="bottom", xytext=(0, 2),
                            textcoords="offset points")

        ax.set_xlim(-0.12, 0.12)
        ax.set_ylim(-0.12, 0.12)
        ax.set_aspect("equal")
        ax.axis("off")

        buf = BytesIO()
        fig.savefig(
            buf,
            format="png",
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="#1a1a1a",
            edgecolor="none",
        )
        buf.seek(0)
        result = base64.b64encode(buf.read()).decode("utf-8")
        if name:
            _topomap_cache[name] = result
        return result
    finally:
        plt.close(fig)


def _load_montage_yaml() -> dict[str, str]:
    """Load valid_montages from configs/montages.yaml.

    Returns a dict of {name: description}.
    """
    import inspect
    import autoclean

    pkg_root = Path(inspect.getfile(autoclean)).parent

    # Walk up to find the configs directory (sits alongside src/)
    for ancestor in [pkg_root, *pkg_root.parents]:
        candidate = ancestor / "configs" / "montages.yaml"
        if candidate.exists():
            try:
                import yaml  # type: ignore[import-untyped]
                with candidate.open() as fh:
                    data = yaml.safe_load(fh)
                return data.get("valid_montages", {})
            except Exception as exc:
                logger.warning("Failed to parse montages.yaml at %s: %s", candidate, exc)
                return {}

    logger.warning("configs/montages.yaml not found relative to package root %s", pkg_root)
    return {}


def _discover_task_montages() -> dict[str, list[str]]:
    """Return a mapping of montage_name → [task_names] using task discovery.

    Gracefully returns an empty dict if task discovery fails.
    """
    try:
        from autoclean.utils.task_discovery import safe_discover_tasks
        import importlib.util
        import sys

        valid_tasks, _invalid, _skipped = safe_discover_tasks()
    except Exception as exc:
        logger.warning("Task discovery failed in montage browser: %s", exc)
        return {}

    montage_tasks: dict[str, list[str]] = {}

    for task in valid_tasks:
        try:
            module_name = f"_montage_browser_task_{abs(hash(str(task.source)))}"
            spec = importlib.util.spec_from_file_location(module_name, task.source)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
            finally:
                sys.modules.pop(module_name, None)

            raw_config: dict[str, Any] = getattr(module, "config", {})
            montage_val = raw_config.get("montage", {})
            if isinstance(montage_val, dict):
                montage_name = montage_val.get("value", "")
            elif isinstance(montage_val, str):
                montage_name = montage_val
            else:
                montage_name = ""

            if montage_name:
                montage_tasks.setdefault(montage_name, []).append(task.name)

        except Exception as exc:
            logger.debug("Could not read montage from task '%s': %s", task.name, exc)

    return montage_tasks


# ── Response models ─────────────────────────────────────────────────

class MontageInfo(BaseModel):
    """Summary info for the montage list view."""

    name: str
    n_channels: int
    category: str
    description: str
    channel_names: list[str]
    compatible_tasks: list[str]


class MontageListResponse(BaseModel):
    """Top-level response for GET /api/montages."""

    montages: list[MontageInfo]
    total: int


class ChannelPosition(BaseModel):
    """3-D position of a single electrode in metres."""

    name: str
    x: float
    y: float
    z: float


class MontageDetail(BaseModel):
    """Full montage detail including topomap PNG."""

    name: str
    n_channels: int
    category: str
    description: str
    channels: list[ChannelPosition]
    topomap_png: str
    compatible_tasks: list[str]
    landmarks: dict[str, list[float]]


# ── Endpoints ───────────────────────────────────────────────────────

@router.get("", response_model=MontageListResponse)
async def list_montages() -> MontageListResponse:
    """Return summary info for all montages listed in configs/montages.yaml.

    MNE is called to resolve channel counts and preview names. Montages
    that MNE cannot load are skipped with a warning so the list always
    reflects what is actually usable.
    """
    yaml_montages = _load_montage_yaml()
    task_montage_map = _discover_task_montages()

    results: list[MontageInfo] = []

    for name, description in yaml_montages.items():
        montage = _load_montage(name)
        if montage is None:
            logger.warning("Skipping montage '%s' — could not load via MNE", name)
            continue

        try:
            positions = montage.get_positions()
            ch_pos: dict[str, Any] = positions.get("ch_pos", {})
            channel_names = list(ch_pos.keys())
            n_channels = len(channel_names)

            results.append(
                MontageInfo(
                    name=name,
                    n_channels=n_channels,
                    category=_categorize(name),
                    description=description,
                    channel_names=channel_names[:10],
                    compatible_tasks=task_montage_map.get(name, []),
                )
            )
        except Exception as exc:
            logger.warning("Error reading positions for montage '%s': %s", name, exc)

    return MontageListResponse(montages=results, total=len(results))


@router.get("/{name:path}", response_model=MontageDetail)
async def get_montage_detail(name: str) -> MontageDetail:
    """Return full channel positions, landmarks, and a topomap PNG for one montage.

    The ``name`` path parameter supports slashes so names like
    ``GSN-HydroCel-129`` are handled correctly when URL-encoded.
    """
    yaml_montages = _load_montage_yaml()
    task_montage_map = _discover_task_montages()

    description = yaml_montages.get(name, "")

    montage = _load_montage(name)
    if montage is None:
        raise HTTPException(
            status_code=404,
            detail=f"Montage '{name}' not found or could not be loaded via MNE",
        )

    try:
        positions = montage.get_positions()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read positions for montage '{name}': {exc}",
        ) from exc

    ch_pos: dict[str, Any] = positions.get("ch_pos", {})
    channels = [
        ChannelPosition(
            name=ch_name,
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
        )
        for ch_name, pos in ch_pos.items()
    ]

    # Landmarks: nasion, lpa, rpa — may be None in custom montages
    landmarks: dict[str, list[float]] = {}
    for lm_key in ("nasion", "lpa", "rpa"):
        lm_val = positions.get(lm_key)
        if lm_val is not None:
            landmarks[lm_key] = [float(v) for v in lm_val]

    try:
        topomap_png = _generate_topomap(montage, name=name)
    except Exception as exc:
        logger.warning("Topomap generation failed for '%s': %s", name, exc)
        topomap_png = ""

    return MontageDetail(
        name=name,
        n_channels=len(channels),
        category=_categorize(name),
        description=description,
        channels=channels,
        topomap_png=topomap_png,
        compatible_tasks=task_montage_map.get(name, []),
        landmarks=landmarks,
    )
