from __future__ import annotations

import re
from pathlib import Path

import pytest

from autoclean_mcp.registry import build_registry
from autoclean_mcp.server import (
    _ensure_mutation_confirmed,
    _prepare_compatibility_session,
    _prepare_registered_one_shot,
    _prepare_registered_session,
)


def test_ensure_mutation_confirmed_rejects_missing_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm=True"):
        _ensure_mutation_confirmed(False, ["serve", "deploy"])


def test_ensure_mutation_confirmed_allows_confirmed_mutation() -> None:
    _ensure_mutation_confirmed(True, ["serve", "deploy"])


def test_prepare_registered_one_shot_rejects_mutating_without_confirm() -> None:
    with pytest.raises(ValueError, match="confirm=True"):
        _prepare_registered_one_shot(["serve", "deploy"], confirm=False)


def test_prepare_registered_one_shot_rejects_managed_session_commands() -> None:
    with pytest.raises(ValueError, match="start_registered_cli_session"):
        _prepare_registered_one_shot(["serve", "run"])


def test_prepare_registered_one_shot_rejects_compatibility_commands_by_default() -> None:
    with pytest.raises(ValueError, match="run_compatibility_cli_command"):
        _prepare_registered_one_shot(["view"])


def test_prepare_registered_one_shot_allows_compatibility_mode_when_requested() -> None:
    entry = _prepare_registered_one_shot(["view"], allow_compatibility=True)
    assert entry.wrapper_kind == "compatibility_wrapper"


def test_prepare_registered_session_requires_managed_session_command() -> None:
    with pytest.raises(ValueError, match="not a managed-session command"):
        _prepare_registered_session(["serve", "deploy"])


def test_prepare_registered_session_accepts_managed_session_command() -> None:
    entry = _prepare_registered_session(["serve", "run"])
    assert entry.wrapper_kind == "managed_session"


def test_prepare_compatibility_session_rejects_non_compatibility_command() -> None:
    with pytest.raises(ValueError, match="not a compatibility-wrapper command"):
        _prepare_compatibility_session(["serve", "deploy"])


def test_prepare_compatibility_session_accepts_compatibility_command() -> None:
    entry = _prepare_compatibility_session(["serve", "tui"])
    assert entry.wrapper_kind == "compatibility_wrapper"


def test_every_registry_command_has_a_named_cli_wrapper() -> None:
    server_source = Path("autoclean_mcp/server.py").read_text(encoding="utf-8")
    wrapped = set(re.findall(r"def (cli_[a-z0-9_]+)\(", server_source))

    missing: list[str] = []
    for entry in build_registry():
        function_name = "cli_" + "_".join(
            part.replace("-", "_") for part in entry.path
        )
        if function_name not in wrapped:
            missing.append(" ".join(entry.path))

    assert missing == []
