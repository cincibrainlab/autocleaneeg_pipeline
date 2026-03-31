"""Thin MATLAB wrapper helpers for AutoClean tasks and utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from autoclean.utils.matlab_runtime import call_matlab_function, run_matlab_script


def call_matlab(
    function_name: str,
    *args: Any,
    nargout: int = 1,
    startup_options: str = "-nodesktop",
    license_file: Optional[str] = None,
    startup_timeout_seconds: float = 60.0,
    path_entries: Optional[Iterable[str]] = None,
) -> Any:
    """Call a MATLAB function through the shared AutoClean runtime."""
    return call_matlab_function(
        function_name,
        *args,
        nargout=nargout,
        startup_options=startup_options,
        license_file=license_file,
        startup_timeout_seconds=startup_timeout_seconds,
        path_entries=path_entries,
    )


def run_matlab_file(
    script_path: str,
    *,
    startup_options: str = "-nodesktop",
    license_file: Optional[str] = None,
    startup_timeout_seconds: float = 60.0,
    path_entries: Optional[Iterable[str]] = None,
) -> None:
    """Run a MATLAB script file through the shared AutoClean runtime."""
    run_matlab_script(
        script_path,
        startup_options=startup_options,
        license_file=license_file,
        startup_timeout_seconds=startup_timeout_seconds,
        path_entries=path_entries,
    )


def execute_matlab_config(
    step_config: dict[str, Any],
    *,
    base_path: str | Path | None = None,
) -> Any:
    """Execute a validated MATLAB step config through the shared runtime."""
    if not step_config.get("enabled", False):
        return None

    value = step_config.get("value")
    if not isinstance(value, dict):
        raise ValueError("MATLAB step config must contain a mapping under 'value'.")

    base_dir = Path(base_path).expanduser().resolve() if base_path else None

    def _resolve_optional_path(path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None
        candidate = Path(path_value).expanduser()
        if candidate.is_absolute() or base_dir is None:
            return str(candidate.resolve())
        return str((base_dir / candidate).resolve())

    def _resolve_path_entries(entries: Optional[Iterable[str]]) -> list[str] | None:
        if not entries:
            return None
        return [_resolve_optional_path(str(entry)) or str(entry) for entry in entries]

    kind = value["kind"]
    entrypoint = str(value["entrypoint"])
    args = list(value.get("args") or [])
    path_entries = _resolve_path_entries(value.get("paths"))
    startup_options = str(value.get("startup_options", "-nodesktop"))
    startup_timeout_seconds = float(value.get("startup_timeout_seconds", 60.0))
    license_file = _resolve_optional_path(value.get("license_file"))

    if kind == "function":
        return call_matlab(
            entrypoint,
            *args,
            nargout=int(value.get("nargout", 1)),
            startup_options=startup_options,
            license_file=license_file,
            startup_timeout_seconds=startup_timeout_seconds,
            path_entries=path_entries,
        )

    script_path = _resolve_optional_path(entrypoint)
    if script_path is None:
        raise ValueError("MATLAB script entrypoint must resolve to a file path.")
    run_matlab_file(
        script_path,
        startup_options=startup_options,
        license_file=license_file,
        startup_timeout_seconds=startup_timeout_seconds,
        path_entries=path_entries,
    )
    return None
