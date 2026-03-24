from __future__ import annotations

from autoclean.cli import create_parser
from autoclean_mcp.cli_inventory import extract_cli_inventory


def test_extract_cli_inventory_includes_known_leaf_commands() -> None:
    inventory = extract_cli_inventory(create_parser())
    paths = {tuple(spec.path) for spec in inventory}

    assert ("task", "list") in paths
    assert ("serve", "up") in paths
    assert ("serve", "route", "upsert") in paths
    assert ("serve", "workspace", "status") in paths
    assert ("serve", "workspace", "doctor") in paths


def test_extract_cli_inventory_classifies_long_running_and_interactive() -> None:
    inventory = extract_cli_inventory(create_parser())
    by_path = {tuple(spec.path): spec for spec in inventory}

    assert by_path[("serve", "up")].execution_style == "long_running"
    assert by_path[("review",)].execution_style == "interactive"
    assert by_path[("task", "list")].execution_style == "one_shot"
