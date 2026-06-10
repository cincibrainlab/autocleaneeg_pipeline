"""Bundled MATLAB FOOOF block mixin."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from autoclean.functions.matlab import call_matlab
from autoclean.utils.logging import message


def _load_algorithm_module():
    """Load algorithm.py from the same directory as this mixin file."""
    mixin_path = Path(__file__).parent
    algorithm_path = mixin_path / "algorithm.py"

    spec = importlib.util.spec_from_file_location(
        "matlab_fooof_algorithm",
        algorithm_path,
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    raise ImportError(f"Could not load algorithm module from {algorithm_path}")


_algorithm = _load_algorithm_module()
load_matlab_fooof_manifest = _algorithm.load_matlab_fooof_manifest
resolve_matlab_fooof_context = _algorithm.resolve_matlab_fooof_context


class MatlabFooofBlockMixin:
    """Mixin implementing a MATLAB-backed FOOOF analysis block."""

    def apply_matlab_fooof(
        self,
        stage_name: str = "apply_matlab_fooof",
    ) -> tuple[dict[str, Any], Path] | tuple[None, None]:
        """Run the bundled MATLAB FOOOF analysis for the current input file."""
        is_enabled, config_value = self._check_step_enabled("apply_matlab_fooof")
        if not is_enabled:
            message("info", "MATLAB FOOOF step is disabled")
            return None, None

        params = (config_value or {}).get("value", config_value or {})
        if not isinstance(params, dict):
            raise ValueError("apply_matlab_fooof configuration must be a mapping.")

        required_keys = ("vhtp_path", "eeglab_path")
        for key in required_keys:
            if not params.get(key):
                raise ValueError(f"apply_matlab_fooof requires '{key}' in config.")

        module_dir = Path(__file__).resolve().parent
        context = resolve_matlab_fooof_context(
            self.config,
            params,
            module_dir=module_dir,
        )

        message("header", "Running MATLAB FOOOF analysis")
        message("info", f"Input file: {context['input_file'].name}")
        message("info", f"Output dir: {context['block_root']}")

        manifest_path = call_matlab(
            "autoclean_eeglab_fooof",
            str(context["input_file"]),
            str(context["block_root"]),
            str(Path(str(params["vhtp_path"])).expanduser().resolve()),
            str(Path(str(params["eeglab_path"])).expanduser().resolve()),
            float(context["freq_range"][0]),
            float(context["freq_range"][1]),
            bool(params.get("save_fooof_img", False)),
            bool(params.get("parallel", False)),
            nargout=1,
            startup_options=str(params.get("startup_options", "-nodesktop")),
            license_file=(
                str(Path(str(params["license_file"])).expanduser().resolve())
                if params.get("license_file")
                else None
            ),
            startup_timeout_seconds=float(params.get("startup_timeout_seconds", 60.0)),
            path_entries=[str(context["matlab_assets_dir"])],
        )

        manifest = load_matlab_fooof_manifest(str(manifest_path))
        self.matlab_fooof_result = manifest

        block_info = self._get_block_info("matlab_fooof")
        metadata = {
            "block_name": "matlab_fooof",
            "subject_id": manifest.get("subject_id", context["subject_id"]),
            "input_file": str(context["input_file"]),
            "output_dir": str(context["block_root"]),
            "manifest_path": str(Path(str(manifest_path)).expanduser().resolve()),
            "summary_csv": manifest.get("summary_csv"),
            "aperiodic_csv": manifest.get("aperiodic_csv"),
            "matlab_output_dir": manifest.get("matlab_output_dir"),
            "n_channels": manifest.get("n_channels"),
            "n_epochs": manifest.get("n_epochs"),
            "sampling_rate": manifest.get("sampling_rate"),
            "summary_row_count": manifest.get("summary_row_count"),
            "aperiodic_row_count": manifest.get("aperiodic_row_count"),
            "freq_range": list(context["freq_range"]),
            "save_fooof_img": bool(params.get("save_fooof_img", False)),
            "parallel": bool(params.get("parallel", False)),
            "vhtp_path": str(Path(str(params["vhtp_path"])).expanduser().resolve()),
            "eeglab_path": str(Path(str(params["eeglab_path"])).expanduser().resolve()),
        }
        if block_info:
            metadata.update(block_info)

        self._update_metadata(f"step_{stage_name}", metadata)
        return manifest, Path(str(manifest_path)).expanduser().resolve()
