from __future__ import annotations

from autoclean_mcp.argv_builder import build_registry_argv


def test_build_registry_argv_for_simple_command() -> None:
    argv = build_registry_argv(["version"])
    assert argv == ["version"]


def test_build_registry_argv_for_command_with_options() -> None:
    argv = build_registry_argv(
        ["serve", "validate"],
        {
            "path": "/tmp/workspace",
            "mode": "test",
        },
    )

    assert argv == ["serve", "validate", "--mode", "test", "--path", "/tmp/workspace"]


def test_build_registry_argv_for_boolean_flag() -> None:
    argv = build_registry_argv(
        ["serve", "up"],
        {
            "force": True,
        },
    )

    assert "--force" in argv


def test_build_registry_argv_for_choice_expanded_subcommand() -> None:
    argv = build_registry_argv(
        ["serve", "workspace", "status"],
        {
            "path": "/tmp/workspace",
        },
    )

    assert argv == ["serve", "workspace", "status", "--path", "/tmp/workspace"]


def test_build_registry_argv_for_repeated_option() -> None:
    argv = build_registry_argv(
        ["events", "discover"],
        {
            "input_file": "sample.set",
            "exclude": ["BAD", "EDGE"],
        },
    )

    assert argv == [
        "events",
        "discover",
        "sample.set",
        "--exclude",
        "BAD",
        "--exclude",
        "EDGE",
    ]


def test_build_registry_argv_for_boolean_and_optional_flags() -> None:
    argv = build_registry_argv(
        ["serve", "route", "list"],
        {
            "path": "/tmp/workspace",
            "mode": "live",
            "include_archived": True,
        },
    )

    assert argv == [
        "serve",
        "route",
        "list",
        "--path",
        "/tmp/workspace",
        "--mode",
        "live",
        "--include-archived",
    ]
