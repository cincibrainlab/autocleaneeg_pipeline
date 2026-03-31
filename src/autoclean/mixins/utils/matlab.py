"""Mixin for optional MATLAB-backed execution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from autoclean.functions.matlab import execute_matlab_config
from autoclean.utils.logging import message
from autoclean.utils.matlab_runtime import call_matlab_function, run_matlab_script


class MatlabExecutionMixin:
    """Mixin exposing thin MATLAB wrappers to task classes."""

    def execute_matlab_step(
        self,
        config_key: str,
        *,
        stage_name: Optional[str] = None,
        base_path: str | Path | None = None,
    ) -> Any:
        """Execute a MATLAB step config from the task settings."""
        if not hasattr(self, "settings") or not isinstance(self.settings, dict):
            raise ValueError("Task settings are not available for MATLAB execution.")

        step_config = self.settings.get(config_key)
        if not isinstance(step_config, dict):
            raise ValueError(f"Missing MATLAB step config: {config_key}")
        if not step_config.get("enabled", False):
            message("info", f"Skipping disabled MATLAB step: {config_key}")
            return None

        value = step_config.get("value")
        if not isinstance(value, dict):
            raise ValueError(f"Invalid MATLAB step config for {config_key}: missing value mapping")

        metadata = {
            "creationDateTime": datetime.now().isoformat(),
            "config_key": config_key,
            "entrypoint_type": value.get("kind"),
            "entrypoint": value.get("entrypoint"),
            "license_file": value.get("license_file"),
            "startup_timeout_seconds": value.get("startup_timeout_seconds", 60.0),
            "paths": list(value.get("paths") or []),
            "toolbox_requirements": list(value.get("toolbox_requirements") or []),
            "outputs": dict(value.get("outputs") or {}),
        }
        if value.get("kind") == "function":
            metadata["nargout"] = value.get("nargout", 1)

        result = execute_matlab_config(step_config, base_path=base_path)
        self._update_matlab_metadata(stage_name or config_key, metadata)
        return result

    def call_matlab_function(
        self,
        function_name: str,
        *args: Any,
        nargout: int = 1,
        startup_options: str = "-nodesktop",
        license_file: Optional[str] = None,
        startup_timeout_seconds: float = 60.0,
        path_entries: Optional[Iterable[str]] = None,
        stage_name: str = "call_matlab_function",
    ) -> Any:
        """Call a MATLAB function and record basic metadata when possible."""
        message("header", f"MATLAB function: {function_name}")
        result = call_matlab_function(
            function_name,
            *args,
            nargout=nargout,
            startup_options=startup_options,
            license_file=license_file,
            startup_timeout_seconds=startup_timeout_seconds,
            path_entries=path_entries,
        )
        self._update_matlab_metadata(
            stage_name,
            {
                "creationDateTime": datetime.now().isoformat(),
                "entrypoint_type": "function",
                "entrypoint": function_name,
                "nargout": nargout,
                "license_file": str(license_file) if license_file else None,
                "startup_timeout_seconds": startup_timeout_seconds,
                "paths": list(path_entries or []),
            },
        )
        return result

    def run_matlab_file(
        self,
        script_path: str | Path,
        *,
        startup_options: str = "-nodesktop",
        license_file: Optional[str] = None,
        startup_timeout_seconds: float = 60.0,
        path_entries: Optional[Iterable[str]] = None,
        stage_name: str = "run_matlab_file",
    ) -> None:
        """Run a MATLAB script file and record basic metadata when possible."""
        script = Path(script_path).expanduser().resolve()
        message("header", f"MATLAB script: {script.name}")
        run_matlab_script(
            script,
            startup_options=startup_options,
            license_file=license_file,
            startup_timeout_seconds=startup_timeout_seconds,
            path_entries=path_entries,
        )
        self._update_matlab_metadata(
            stage_name,
            {
                "creationDateTime": datetime.now().isoformat(),
                "entrypoint_type": "script",
                "entrypoint": str(script),
                "license_file": str(license_file) if license_file else None,
                "startup_timeout_seconds": startup_timeout_seconds,
                "paths": list(path_entries or []),
            },
        )

    def _update_matlab_metadata(self, operation: str, metadata_dict: dict[str, Any]) -> None:
        """Update task metadata if the host class supports metadata tracking."""
        if hasattr(self, "_update_metadata"):
            self._update_metadata(operation, metadata_dict)
