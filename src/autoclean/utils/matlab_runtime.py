"""Optional MATLAB runtime helpers.

This module must remain safe to import on machines that do not have MATLAB or
the MATLAB Engine API for Python installed.
"""

from __future__ import annotations

import ast
import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


class MatlabRuntimeError(RuntimeError):
    """Base error for MATLAB runtime issues."""


class MatlabEngineUnavailableError(MatlabRuntimeError):
    """Raised when the MATLAB Engine API for Python is unavailable."""


class MatlabEnvironmentError(MatlabRuntimeError):
    """Raised when MATLAB environment validation fails."""


class MatlabEngineStartupError(MatlabRuntimeError):
    """Raised when the MATLAB engine cannot start."""


class MatlabTimeoutError(MatlabRuntimeError):
    """Raised when MATLAB startup or execution exceeds a timeout."""


class MatlabExecutionError(MatlabRuntimeError):
    """Raised when MATLAB command or function execution fails."""


@dataclass
class MatlabRuntimeReport:
    """Structured runtime status for the active Python environment."""

    python_version: str
    python_executable: str
    is_64_bit: bool
    platform: str
    engine_package_installed: bool = False
    engine_package_version: Optional[str] = None
    matlab_root: Optional[str] = None
    matlab_binary: Optional[str] = None
    matlab_on_path: bool = False
    license_file: Optional[str] = None
    engine_start_ok: bool = False
    route_environment_supported: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when the current environment is ready for MATLAB use."""
        return self.engine_package_installed and self.engine_start_ok


@dataclass
class MatlabTaskfileInspection:
    """Static inspection result for a Python task file."""

    taskfile: str
    requires_matlab: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    matlab_config_keys: list[str] = field(default_factory=list)


def _import_matlab_modules() -> tuple[Any, Any]:
    """Import matlab and matlab.engine lazily."""
    try:
        import matlab  # type: ignore
        import matlab.engine  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local MATLAB install
        raise MatlabEngineUnavailableError(str(exc)) from exc
    return matlab, matlab.engine


def _engine_metadata_version() -> Optional[str]:
    try:
        from importlib import metadata

        return metadata.version("matlabengine")
    except Exception:
        return None


def _read_engine_arch_file(matlab_pkg: Any) -> tuple[Optional[str], Optional[str]]:
    """Read MATLAB root and binary path from the engine arch file when available."""
    try:
        matlab_init = Path(matlab_pkg.__file__).resolve()
        arch_file = matlab_init.parent / "engine" / "_arch.txt"
        if not arch_file.exists():
            return None, None

        lines = [
            line.strip()
            for line in arch_file.read_text(encoding="utf-8").splitlines()
        ]
        if len(lines) < 2:
            return None, None

        matlab_bin = Path(lines[1]).resolve()
        matlab_root = matlab_bin.parent.parent if matlab_bin.parent.name else None
        return str(matlab_root) if matlab_root else None, str(matlab_bin)
    except Exception:
        return None, None


def _node_to_dotted_name(node: ast.AST) -> Optional[str]:
    """Return dotted name for Name/Attribute AST nodes when possible."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _node_to_dotted_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


def _extract_literal_dict_keys(node: ast.AST) -> list[str]:
    """Extract literal string keys from a dict AST node."""
    if not isinstance(node, ast.Dict):
        return []
    keys: list[str] = []
    for key_node in node.keys:
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            keys.append(key_node.value)
    return keys


def inspect_taskfile_for_matlab(taskfile: Path | str) -> MatlabTaskfileInspection:
    """Statically inspect a Python task file for MATLAB-backed execution usage."""
    taskfile_path = Path(taskfile).expanduser().resolve()
    inspection = MatlabTaskfileInspection(taskfile=str(taskfile_path))

    if taskfile_path.suffix != ".py":
        return inspection
    if not taskfile_path.exists():
        inspection.warnings.append(f"Task file not found for MATLAB inspection: {taskfile_path}")
        return inspection

    try:
        source = taskfile_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(taskfile_path))
    except Exception as exc:
        inspection.warnings.append(f"Unable to inspect task file for MATLAB usage: {exc}")
        return inspection

    matlab_aliases: set[str] = set()
    matlab_call_aliases: set[str] = set()
    matlab_mixin_aliases: set[str] = set()
    matlab_config_keys = {"apply_matlab", "run_matlab", "apply_matlab_fooof"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "matlab" or alias.name.startswith("matlab."):
                    inspection.requires_matlab = True
                    inspection.reasons.append(f"imports {alias.name}")
                    matlab_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "matlab" or module.startswith("matlab."):
                inspection.requires_matlab = True
                inspection.reasons.append(f"imports from {module}")
                for alias in node.names:
                    matlab_aliases.add(alias.asname or alias.name)
            elif module in {
                "autoclean.functions",
                "autoclean.functions.matlab",
                "autoclean.utils.matlab_runtime",
            }:
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    if alias.name in {
                        "call_matlab",
                        "run_matlab_file",
                        "call_matlab_function",
                        "run_matlab_script",
                    }:
                        matlab_call_aliases.add(local_name)
                        inspection.requires_matlab = True
                        inspection.reasons.append(f"imports MATLAB helper {alias.name}")
            elif module == "autoclean.mixins.utils.matlab":
                for alias in node.names:
                    if alias.name == "MatlabExecutionMixin":
                        matlab_mixin_aliases.add(alias.asname or alias.name)
                        inspection.requires_matlab = True
                        inspection.reasons.append("imports MatlabExecutionMixin")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "config":
                    config_keys = set(_extract_literal_dict_keys(node.value))
                    matched = sorted(config_keys.intersection(matlab_config_keys))
                    if matched:
                        inspection.requires_matlab = True
                        inspection.matlab_config_keys.extend(matched)
                        inspection.reasons.append(
                            f"declares MATLAB config keys: {', '.join(matched)}"
                        )

    if "matlab" in matlab_aliases or "engine" in matlab_aliases:
        inspection.requires_matlab = True

    matlab_call_names = {
        "call_matlab",
        "run_matlab_file",
        "call_matlab_function",
        "run_matlab_script",
        "apply_matlab_fooof",
        *matlab_call_aliases,
    }
    matlab_mixin_names = {"MatlabExecutionMixin", *matlab_mixin_aliases}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = _node_to_dotted_name(base)
                if not base_name:
                    continue
                if base_name in matlab_mixin_names or base_name.endswith(".MatlabExecutionMixin"):
                    inspection.requires_matlab = True
                    inspection.reasons.append(
                        f"class {node.name} inherits {base_name.split('.')[-1]}"
                    )

        if isinstance(node, ast.Call):
            call_name = _node_to_dotted_name(node.func)
            if not call_name:
                continue
            if call_name in matlab_call_names or call_name.split(".")[-1] in matlab_call_names:
                inspection.requires_matlab = True
                inspection.reasons.append(f"calls {call_name.split('.')[-1]}")

    if inspection.requires_matlab:
        inspection.reasons = list(dict.fromkeys(inspection.reasons))
        inspection.matlab_config_keys = list(dict.fromkeys(inspection.matlab_config_keys))

    return inspection


def detect_matlab_engine(
    *,
    license_file: Optional[str] = None,
    check_engine_start: bool = False,
    startup_options: str = "-nodesktop",
    startup_timeout_seconds: float = 60.0,
) -> MatlabRuntimeReport:
    """Inspect the current Python environment for MATLAB runtime readiness."""
    report = MatlabRuntimeReport(
        python_version=platform.python_version(),
        python_executable=os.path.realpath(os.sys.executable),
        is_64_bit=os.sys.maxsize > 2**32,
        platform=platform.platform(),
        matlab_binary=shutil.which("matlab"),
        matlab_on_path=shutil.which("matlab") is not None,
        license_file=license_file,
    )

    if not report.is_64_bit:
        report.errors.append("Python interpreter is not 64-bit.")

    try:
        matlab_pkg, _ = _import_matlab_modules()
        report.engine_package_installed = True
        report.engine_package_version = _engine_metadata_version()
        matlab_root, matlab_binary = _read_engine_arch_file(matlab_pkg)
        if matlab_root:
            report.matlab_root = matlab_root
        if matlab_binary:
            report.matlab_binary = matlab_binary
            report.matlab_on_path = True
    except MatlabEngineUnavailableError as exc:
        report.errors.append(f"MATLAB Engine API unavailable: {exc}")
        return report

    if report.matlab_root is None:
        report.warnings.append(
            "MATLAB root could not be derived from the installed engine package."
        )

    if check_engine_start:
        try:
            eng = start_matlab_engine(
                startup_options=startup_options,
                license_file=license_file,
                startup_timeout_seconds=startup_timeout_seconds,
            )
        except MatlabRuntimeError as exc:
            report.errors.append(str(exc))
        else:
            report.engine_start_ok = True
            report.route_environment_supported = True
            try:
                eng.quit()
            except Exception:
                report.warnings.append("Engine started but did not shut down cleanly.")
    else:
        report.route_environment_supported = report.engine_package_installed

    return report


def validate_matlab_environment(
    *,
    license_file: Optional[str] = None,
    startup_options: str = "-nodesktop",
    require_engine_start: bool = True,
    startup_timeout_seconds: float = 60.0,
) -> MatlabRuntimeReport:
    """Validate that the current environment can run MATLAB-backed features."""
    report = detect_matlab_engine(
        license_file=license_file,
        check_engine_start=require_engine_start,
        startup_options=startup_options,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    if report.errors:
        raise MatlabEnvironmentError("; ".join(report.errors))
    return report


def start_matlab_engine(
    *,
    startup_options: str = "-nodesktop",
    license_file: Optional[str] = None,
    startup_timeout_seconds: float = 60.0,
) -> Any:
    """Start a MATLAB engine session."""
    _, matlab_engine = _import_matlab_modules()

    previous_license = os.environ.get("MLM_LICENSE_FILE")
    if license_file:
        os.environ["MLM_LICENSE_FILE"] = str(license_file)

    future = None
    try:
        future = matlab_engine.start_matlab(startup_options, background=True)
        return future.result(timeout=startup_timeout_seconds)
    except MatlabRuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - depends on local MATLAB runtime
        if "timeout" in exc.__class__.__name__.lower():
            try:
                future.cancel()
            except Exception:
                pass
            raise MatlabTimeoutError(
                f"MATLAB engine startup timed out after {startup_timeout_seconds} seconds"
            ) from exc
        raise MatlabEngineStartupError(f"Unable to start MATLAB engine: {exc}") from exc
    finally:
        if license_file:
            if previous_license is None:
                os.environ.pop("MLM_LICENSE_FILE", None)
            else:
                os.environ["MLM_LICENSE_FILE"] = previous_license


def shutdown_matlab_engine(engine: Any) -> None:
    """Shut down a MATLAB engine session safely."""
    if engine is None:
        return
    try:
        engine.quit()
    except Exception as exc:  # pragma: no cover - depends on local MATLAB runtime
        raise MatlabRuntimeError(f"Failed to shut down MATLAB engine: {exc}") from exc


def _add_search_paths(engine: Any, path_entries: Optional[Iterable[str]]) -> None:
    """Add MATLAB search paths to an engine session."""
    if not path_entries:
        return
    for entry in path_entries:
        engine.addpath(str(Path(entry).expanduser().resolve()), nargout=0)


def _validate_engine_reuse_arguments(engine: Any, keep_engine: bool) -> None:
    """Reject lifecycle combinations that would leak engine sessions."""
    if keep_engine and engine is None:
        raise MatlabRuntimeError(
            "keep_engine=True requires an explicit engine owned by the caller."
        )


def run_matlab_function(
    function_name: str,
    *args: Any,
    nargout: int = 1,
    engine: Any = None,
    startup_options: str = "-nodesktop",
    license_file: Optional[str] = None,
    startup_timeout_seconds: float = 60.0,
    path_entries: Optional[Iterable[str]] = None,
    keep_engine: bool = False,
) -> Any:
    """Run a MATLAB function using the current engine or a new engine session."""
    _validate_engine_reuse_arguments(engine, keep_engine)
    local_engine = engine
    created_engine = False
    if local_engine is None:
        local_engine = start_matlab_engine(
            startup_options=startup_options,
            license_file=license_file,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        created_engine = True

    try:
        _add_search_paths(local_engine, path_entries)
        matlab_callable = getattr(local_engine, function_name, None)
        if matlab_callable is None:
            raise MatlabExecutionError(f"MATLAB function not found: {function_name}")
        return matlab_callable(*args, nargout=nargout)
    except MatlabRuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - depends on local MATLAB runtime
        raise MatlabExecutionError(
            f"Failed to execute MATLAB function '{function_name}': {exc}"
        ) from exc
    finally:
        if created_engine and not keep_engine:
            shutdown_matlab_engine(local_engine)


def run_matlab_script(
    script_path: str | Path,
    *,
    engine: Any = None,
    startup_options: str = "-nodesktop",
    license_file: Optional[str] = None,
    startup_timeout_seconds: float = 60.0,
    path_entries: Optional[Iterable[str]] = None,
    keep_engine: bool = False,
) -> None:
    """Run a MATLAB script via `run('<script>')`."""
    script = Path(script_path).expanduser().resolve()
    if not script.exists():
        raise MatlabExecutionError(f"MATLAB script not found: {script}")

    _validate_engine_reuse_arguments(engine, keep_engine)
    local_engine = engine
    created_engine = False
    if local_engine is None:
        local_engine = start_matlab_engine(
            startup_options=startup_options,
            license_file=license_file,
            startup_timeout_seconds=startup_timeout_seconds,
        )
        created_engine = True

    try:
        combined_paths = list(path_entries or [])
        combined_paths.append(str(script.parent))
        _add_search_paths(local_engine, combined_paths)
        local_engine.eval(f"run('{script.as_posix()}')", nargout=0)
    except MatlabRuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - depends on local MATLAB runtime
        raise MatlabExecutionError(f"Failed to run MATLAB script '{script}': {exc}") from exc
    finally:
        if created_engine and not keep_engine:
            shutdown_matlab_engine(local_engine)


def call_matlab_function(
    function_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Thin alias for `run_matlab_function` for task/helper ergonomics."""
    return run_matlab_function(function_name, *args, **kwargs)
