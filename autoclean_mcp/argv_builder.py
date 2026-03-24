"""Registry-backed argv construction for typed MCP wrappers."""

from __future__ import annotations

from typing import Any

from autoclean_mcp.registry import get_registry_entry


def _preferred_option(option_strings: list[str]) -> str:
    long_options = [opt for opt in option_strings if opt.startswith("--")]
    if long_options:
        return long_options[0]
    if option_strings:
        return option_strings[0]
    raise ValueError("No option strings available for non-positional argument")


def build_registry_argv(command_path: list[str], arguments: dict[str, Any] | None = None) -> list[str]:
    """Build CLI argv from a registry entry and typed argument mapping."""
    entry = get_registry_entry(command_path)
    if entry is None:
        raise ValueError(f"Unknown CLI command path: {' '.join(command_path)}")

    provided = dict(arguments or {})
    argv = list(command_path)

    for spec in entry.arguments:
        if spec.name not in provided:
            continue
        value = provided[spec.name]
        if value is None:
            continue

        if spec.positional:
            if isinstance(value, list):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
            continue

        option = _preferred_option(spec.option_strings)
        if spec.action_kind in {"_StoreTrueAction", "_StoreFalseAction"}:
            if bool(value):
                argv.append(option)
            continue

        if isinstance(value, list):
            for item in value:
                argv.extend([option, str(item)])
            continue

        argv.extend([option, str(value)])

    return argv
