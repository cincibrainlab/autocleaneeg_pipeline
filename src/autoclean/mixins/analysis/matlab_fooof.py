"""Proxy mixin exposing the bundled MATLAB FOOOF block to Task classes."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_block_mixin():
    """Load the bundled block mixin from the blocks directory."""
    mixin_path = (
        Path(__file__).resolve().parents[2]
        / "blocks"
        / "analysis"
        / "matlab_fooof"
        / "mixin.py"
    )
    spec = importlib.util.spec_from_file_location(
        "matlab_fooof_block_mixin",
        mixin_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MATLAB FOOOF block mixin from {mixin_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MatlabFooofBlockMixin


_BlockMixin = _load_block_mixin()


class MatlabFooofMixin(_BlockMixin):
    """Auto-discovered task mixin proxy for the bundled MATLAB FOOOF block."""
