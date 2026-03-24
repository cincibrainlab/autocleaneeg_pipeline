"""Inventory helpers for the AutoClean CLI surface."""

from __future__ import annotations

import argparse
from typing import Iterable

from autoclean_mcp.models import CLIArgumentSpec, CLICommandSpec


def _classify_execution_style(path: list[str]) -> str:
    """Classify a CLI command by execution style."""
    if not path:
        return "one_shot"

    full = " ".join(path)
    interactive_commands = {
        "wizard",
        "task set",
        "montage set",
        "workspace explore",
        "review",
        "exclude",
        "serve tui",
    }
    long_running_commands = {
        "serve up",
        "serve api",
        "serve run",
        "serve worker",
    }

    if full in interactive_commands:
        return "interactive"
    if full in long_running_commands:
        return "long_running"
    return "one_shot"


def _iter_subparsers(
    parser: argparse.ArgumentParser,
) -> Iterable[tuple[str, argparse.ArgumentParser]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):  # type: ignore[attr-defined]
            yield from action.choices.items()


def _extract_argument_specs(parser: argparse.ArgumentParser) -> list[CLIArgumentSpec]:
    specs: list[CLIArgumentSpec] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):  # type: ignore[attr-defined]
            continue
        option_strings = list(action.option_strings)
        positional = len(option_strings) == 0
        choices = [str(choice) for choice in action.choices] if action.choices else []
        specs.append(
            CLIArgumentSpec(
                name=action.dest,
                positional=positional,
                required=bool(getattr(action, "required", False)),
                option_strings=option_strings,
                nargs=action.nargs,
                choices=choices,
                help=action.help or "",
                action_kind=type(action).__name__,
            )
        )
    return specs


def _expand_choice_actions(
    current: argparse.ArgumentParser,
    path: list[str],
) -> list[CLICommandSpec]:
    """Expand positional choice-actions that behave like subcommands."""
    expandable: list[tuple[argparse.Action, list[str]]] = []
    for action in current._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):  # type: ignore[attr-defined]
            continue
        option_strings = list(action.option_strings)
        positional = len(option_strings) == 0
        choices = [str(choice) for choice in action.choices] if action.choices else []
        if (
            positional
            and choices
            and (action.dest.endswith("_action") or action.dest == "action")
        ):
            expandable.append((action, choices))

    if len(expandable) != 1:
        return []

    action, choices = expandable[0]
    base_arguments = [
        spec
        for spec in _extract_argument_specs(current)
        if spec.name != action.dest
    ]
    return [
        CLICommandSpec(
            path=[*path, choice],
            help=current.description or current.format_usage().strip(),
            description=current.description or "",
            execution_style=_classify_execution_style([*path, choice]),
            arguments=base_arguments,
        )
        for choice in choices
    ]


def extract_cli_inventory(parser: argparse.ArgumentParser) -> list[CLICommandSpec]:
    """Walk the argparse tree and return every CLI leaf command."""

    inventory: list[CLICommandSpec] = []

    def walk(current: argparse.ArgumentParser, path: list[str]) -> None:
        children = list(_iter_subparsers(current))
        if not children:
            expanded = _expand_choice_actions(current, path)
            if expanded:
                inventory.extend(expanded)
                return
            inventory.append(
                CLICommandSpec(
                    path=path,
                    help=current.description or current.format_usage().strip(),
                    description=current.description or "",
                    execution_style=_classify_execution_style(path),
                    arguments=_extract_argument_specs(current),
                )
            )
            return

        for name, child in children:
            walk(child, [*path, name])

    walk(parser, [])
    return inventory


def load_cli_inventory() -> list[CLICommandSpec]:
    """Build inventory from the real AutoClean CLI parser."""
    from autoclean.cli import create_parser

    return extract_cli_inventory(create_parser())
