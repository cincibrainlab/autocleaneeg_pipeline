"""Helper utilities for managing processing block dependencies.

This module provides tools for detecting, checking, and managing dependencies
for dynamically loaded processing blocks, with special support for uv tool install.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_manifest_dependencies(manifest_path: Path) -> Dict[str, str]:
    """Extract package dependencies from a block manifest.json file.

    Parameters
    ----------
    manifest_path : Path
        Path to the block's manifest.json file

    Returns
    -------
    dict
        Dictionary mapping package names to version specifiers
        Example: {"meegkit": ">=0.1.9", "numpy": ">=2.0.0"}
    """
    if not manifest_path.exists():
        return {}

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        deps = manifest.get("dependencies", {})
        packages = deps.get("packages", {})

        return {name: spec for name, spec in packages.items() if isinstance(spec, str)}
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def check_package_installed(package: str, version_spec: Optional[str] = None) -> bool:
    """Check if a package is installed and optionally meets version requirements.

    Parameters
    ----------
    package : str
        Package name to check (e.g., "meegkit")
    version_spec : str, optional
        Version specifier (e.g., ">=0.1.9"). If None, only checks if package exists.

    Returns
    -------
    bool
        True if package is installed and meets version requirements
    """
    # Check if package can be imported
    spec = importlib.util.find_spec(package)
    if spec is None:
        return False

    # If no version requirement, just return True
    if version_spec is None:
        return True

    # Try to check version (basic implementation, doesn't handle all PEP 440 cases)
    try:
        module = importlib.import_module(package)
        if not hasattr(module, "__version__"):
            # Can't verify version, assume OK
            return True

        installed_version = module.__version__

        # Simple version check (supports >=, >, ==, <=, <)
        return _check_version_constraint(installed_version, version_spec)
    except (ImportError, AttributeError):
        # Can't verify, assume installed but unknown version
        return True


def _check_version_constraint(installed: str, constraint: str) -> bool:
    """Simple version constraint checker.

    Only handles basic constraints like >=0.1.9, not complex PEP 440.
    """
    constraint = constraint.strip()

    # Extract operator and version
    if constraint.startswith(">="):
        op, required = ">=", constraint[2:].strip()
    elif constraint.startswith("<="):
        op, required = "<=", constraint[2:].strip()
    elif constraint.startswith(">"):
        op, required = ">", constraint[1:].strip()
    elif constraint.startswith("<"):
        op, required = "<", constraint[1:].strip()
    elif constraint.startswith("=="):
        op, required = "==", constraint[2:].strip()
    else:
        # Unknown format, assume satisfied
        return True

    try:
        # Parse versions as tuples of integers
        installed_parts = tuple(int(x) for x in installed.split(".")[:3])
        required_parts = tuple(int(x) for x in required.split(".")[:3])

        # Pad to same length
        max_len = max(len(installed_parts), len(required_parts))
        installed_parts += (0,) * (max_len - len(installed_parts))
        required_parts += (0,) * (max_len - len(required_parts))

        # Apply operator
        if op == ">=":
            return installed_parts >= required_parts
        elif op == ">":
            return installed_parts > required_parts
        elif op == "==":
            return installed_parts == required_parts
        elif op == "<=":
            return installed_parts <= required_parts
        elif op == "<":
            return installed_parts < required_parts
    except (ValueError, AttributeError):
        # Can't parse versions, assume satisfied
        return True

    return True


def get_missing_dependencies(
    block_name: str, manifest_path: Path
) -> List[Tuple[str, str]]:
    """Get list of missing dependencies for a block.

    Parameters
    ----------
    block_name : str
        Name of the block (for reference)
    manifest_path : Path
        Path to the block's manifest.json

    Returns
    -------
    list of tuple
        List of (package_name, version_spec) tuples for missing dependencies
    """
    deps = parse_manifest_dependencies(manifest_path)
    missing = []

    for package, version_spec in deps.items():
        # Skip packages that are part of core pipeline
        if package in {"mne", "numpy", "scipy", "matplotlib", "pandas"}:
            continue

        if not check_package_installed(package, version_spec):
            missing.append((package, version_spec))

    return missing


def is_uv_tool_install() -> bool:
    """Detect if the current Python is running from a uv tool install.

    Returns
    -------
    bool
        True if running from ~/.local/share/uv/tools/ or similar uv tool location
    """
    # Check if sys.prefix looks like a uv tool environment
    prefix = Path(sys.prefix)

    # Common uv tool paths
    uv_indicators = [
        ".local/share/uv/tools",
        "Library/Application Support/uv/tools",  # macOS
        "AppData/Local/uv/tools",  # Windows
    ]

    prefix_str = str(prefix)
    return any(indicator in prefix_str for indicator in uv_indicators)


def generate_install_command(
    dependencies: List[Tuple[str, str]], use_blocks_all: bool = False
) -> str:
    """Generate the uv tool install command for dependencies.

    Parameters
    ----------
    dependencies : list of tuple
        List of (package_name, version_spec) tuples
    use_blocks_all : bool, default False
        If True, suggest installing [blocks-all] extra instead of individual packages

    Returns
    -------
    str
        Command to run for installation
    """
    if use_blocks_all:
        return 'uv tool install "autocleaneeg-pipeline[blocks-all]"'

    if not dependencies:
        return ""

    # Build --with arguments
    with_args = " ".join(f"--with {pkg}" for pkg, _ in dependencies)
    return f"uv tool install --reinstall autocleaneeg-pipeline {with_args}"


def format_dependency_error(
    block_name: str,
    dependencies: List[Tuple[str, str]],
    include_install_commands: bool = True,
) -> str:
    """Format a user-friendly dependency error message.

    Parameters
    ----------
    block_name : str
        Name of the block requiring dependencies
    dependencies : list of tuple
        List of (package_name, version_spec) tuples
    include_install_commands : bool, default True
        Whether to include installation command suggestions

    Returns
    -------
    str
        Formatted error message
    """
    if not dependencies:
        return f"Block '{block_name}' has missing dependencies."

    lines = [f"Block '{block_name}' requires additional dependencies:"]
    for pkg, version in dependencies:
        lines.append(f"  • {pkg}{version}")

    if include_install_commands and is_uv_tool_install():
        lines.append("")
        lines.append("You installed via 'uv tool install'. To enable this block:")

        # Option 1: Automated helper command (if we implement it)
        lines.append(f"  autocleaneeg-pipeline blocks enable {block_name}")

        # Option 2: Manual reinstall
        lines.append("")
        lines.append("Or manually:")
        lines.append(f"  {generate_install_command(dependencies)}")

        # Option 3: Install all blocks
        lines.append("")
        lines.append("Or install all block dependencies:")
        lines.append(f"  {generate_install_command(dependencies, use_blocks_all=True)}")
    elif include_install_commands:
        lines.append("")
        lines.append("To install:")
        for pkg, version in dependencies:
            lines.append(f"  pip install '{pkg}{version}'")

    return "\n".join(lines)


def get_block_dependency_status(
    block_name: str, manifest_path: Path
) -> Dict[str, any]:
    """Get comprehensive dependency status for a block.

    Parameters
    ----------
    block_name : str
        Name of the block
    manifest_path : Path
        Path to block's manifest.json

    Returns
    -------
    dict
        Dictionary containing:
        - all_deps: dict of all dependencies
        - missing: list of (package, version) tuples for missing deps
        - satisfied: list of (package, version) tuples for satisfied deps
        - has_issues: bool indicating if there are missing deps
    """
    all_deps = parse_manifest_dependencies(manifest_path)
    missing = get_missing_dependencies(block_name, manifest_path)

    satisfied = [
        (pkg, ver) for pkg, ver in all_deps.items()
        if (pkg, ver) not in missing
    ]

    return {
        "all_deps": all_deps,
        "missing": missing,
        "satisfied": satisfied,
        "has_issues": len(missing) > 0,
    }
