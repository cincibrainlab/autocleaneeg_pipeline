"""Maintained MCP registry derived from the AutoClean CLI inventory."""

from __future__ import annotations

from collections import Counter

from autoclean_mcp.cli_inventory import load_cli_inventory
from autoclean_mcp.models import CLICommandSpec, MCPRegistryEntry


READ_ONLY_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("help",),
    ("version",),
    ("whoami",),
    ("auth0-diagnostics",),
    ("list-tasks",),
    ("task", "list"),
    ("task", "show"),
    ("task", "schema", "export"),
    ("task", "search"),
    ("task", "diff"),
    ("task", "diagnose"),
    ("montage", "list"),
    ("montage", "test"),
    ("events", "discover"),
    ("events", "analyze"),
    ("events", "epochs"),
    ("config", "show"),
    ("workspace", "show"),
    ("workspace", "size"),
    ("report", "chat"),
    ("serve", "docs"),
    ("serve", "workspace", "status"),
    ("serve", "workspace", "doctor"),
    ("serve", "list"),
    ("serve", "route", "list"),
    ("serve", "validate"),
    ("serve", "service", "status"),
    ("serve", "mode", "status"),
    ("serve", "queue", "status"),
    ("serve", "queue", "list"),
    ("serve", "status"),
    ("serve", "share", "status"),
)

MUTATING_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("wizard",),
    ("process",),
    ("login",),
    ("logout",),
    ("task", "delete"),
    ("task", "copy"),
    ("task", "edit"),
    ("task", "set"),
    ("task", "unset"),
    ("task", "use"),
    ("task", "create"),
    ("task", "install"),
    ("task", "sync"),
    ("task", "update"),
    ("montage", "set"),
    ("blocks", "update"),
    ("blocks", "install"),
    ("blocks", "lock"),
    ("source", "set"),
    ("source", "unset"),
    ("input", "set"),
    ("input", "unset"),
    ("config", "setup"),
    ("config", "reset"),
    ("config", "export"),
    ("config", "import"),
    ("workspace", "set"),
    ("workspace", "unset"),
    ("workspace", "default"),
    ("export-access-log",),
    ("clean-task",),
    ("report", "create"),
    ("serve", "workspace", "use"),
    ("serve", "route", "upsert"),
    ("serve", "route", "promote"),
    ("serve", "route", "archive"),
    ("serve", "route", "unarchive"),
    ("serve", "route", "delete"),
    ("serve", "route", "sync"),
    ("serve", "deploy"),
    ("serve", "service", "start"),
    ("serve", "service", "stop"),
    ("serve", "mode", "test"),
    ("serve", "mode", "live"),
    ("serve", "queue", "retry"),
    ("serve", "queue", "retry-failed"),
    ("serve", "queue", "clear-processed"),
    ("serve", "queue", "remove"),
    ("serve", "up"),
    ("serve", "down"),
    ("serve", "restart"),
    ("serve", "share", "start"),
    ("serve", "share", "stop"),
    ("serve", "share", "setup"),
    ("serve", "share", "clear"),
    ("tutorial",),
    ("settings", "theme"),
    ("auth",),
)

DESTRUCTIVE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("task", "delete"),
    ("task", "unset"),
    ("source", "unset"),
    ("input", "unset"),
    ("config", "reset"),
    ("workspace", "unset"),
    ("serve", "route", "archive"),
    ("serve", "route", "delete"),
    ("serve", "queue", "clear-processed"),
    ("serve", "queue", "remove"),
    ("serve", "down"),
    ("serve", "restart"),
)

COMPATIBILITY_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("login",),
    ("task", "explore"),
    ("task", "edit"),
    ("workspace", "cd"),
    ("report", "chat"),
    ("view",),
)


def _path_matches_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def _family_for_path(path: tuple[str, ...]) -> str:
    if not path:
        return "root"
    head = path[0]
    if head in {"task", "list-tasks"}:
        return "tasks"
    if head == "montage":
        return "montages"
    if head == "events":
        return "events"
    if head in {"process", "review", "exclude", "report", "view", "clean-task"}:
        return "processing_review_reporting"
    if head in {"workspace", "config", "settings", "source", "input"}:
        return "workspace_configuration"
    if head == "serve":
        if len(path) > 1 and path[1] == "workspace":
            return "serve_workspace"
        if len(path) > 1 and path[1] == "route":
            return "serve_routes"
        if len(path) > 1 and path[1] in {"validate", "deploy"}:
            return "serve_validate_deploy"
        if len(path) > 1 and path[1] == "service":
            return "serve_service"
        if len(path) > 1 and path[1] == "queue":
            return "serve_queue"
        if len(path) > 1 and path[1] in {"mode", "share"}:
            return "serve_mode_share"
        return "serve_launcher_process"
    return head.replace("-", "_")


def _wrapper_kind(spec: CLICommandSpec) -> str:
    path = tuple(spec.path)
    if spec.execution_style == "long_running":
        return "managed_session"
    if any(_path_matches_prefix(path, prefix) for prefix in COMPATIBILITY_PREFIXES):
        return "compatibility_wrapper"
    if spec.execution_style == "interactive":
        return "compatibility_wrapper"
    return "typed_wrapper"


def _output_mode(spec: CLICommandSpec) -> str:
    path = tuple(spec.path)
    if any(_path_matches_prefix(path, prefix) for prefix in COMPATIBILITY_PREFIXES):
        return "raw_compatible"
    if spec.execution_style == "interactive":
        return "raw_compatible"
    if spec.execution_style == "long_running":
        return "partially_structured"
    return "partially_structured"


def _notes_for_entry(path: tuple[str, ...], spec: CLICommandSpec) -> list[str]:
    notes: list[str] = []
    if spec.execution_style == "interactive":
        notes.append("Requires compatibility or managed-session handling for parity.")
    if spec.execution_style == "long_running":
        notes.append("Requires managed session lifecycle support.")
    if any(arg.nargs not in (None, "?", 1) for arg in spec.arguments):
        notes.append("Contains non-trivial argument cardinality that needs explicit schema mapping.")
    if _family_for_path(path).startswith("serve_"):
        notes.append("Serve-family command.")
    return notes


def build_registry() -> list[MCPRegistryEntry]:
    """Build the maintained MCP registry from the CLI inventory."""
    inventory = load_cli_inventory()
    entries: list[MCPRegistryEntry] = []
    for spec in inventory:
        path = tuple(spec.path)
        mutating = any(_path_matches_prefix(path, prefix) for prefix in MUTATING_PREFIXES)
        if any(_path_matches_prefix(path, prefix) for prefix in READ_ONLY_PREFIXES):
            mutating = False
        destructive = any(
            _path_matches_prefix(path, prefix) for prefix in DESTRUCTIVE_PREFIXES
        )
        entries.append(
            MCPRegistryEntry(
                command_id="__".join(path),
                path=list(path),
                family=_family_for_path(path),
                execution_style=spec.execution_style,
                wrapper_kind=_wrapper_kind(spec),
                output_mode=_output_mode(spec),
                mutating=mutating,
                destructive=destructive,
                help=spec.help,
                description=spec.description,
                arguments=spec.arguments,
                notes=_notes_for_entry(path, spec),
            )
        )
    return entries


def registry_summary() -> dict[str, object]:
    """Return high-level summary counts for the maintained registry."""
    entries = build_registry()
    family_counts = Counter(entry.family for entry in entries)
    wrapper_counts = Counter(entry.wrapper_kind for entry in entries)
    output_counts = Counter(entry.output_mode for entry in entries)
    return {
        "count": len(entries),
        "families": dict(sorted(family_counts.items())),
        "wrappers": dict(sorted(wrapper_counts.items())),
        "output_modes": dict(sorted(output_counts.items())),
        "mutating_count": sum(1 for entry in entries if entry.mutating),
        "destructive_count": sum(1 for entry in entries if entry.destructive),
        "interactive_count": sum(
            1 for entry in entries if entry.execution_style == "interactive"
        ),
        "long_running_count": sum(
            1 for entry in entries if entry.execution_style == "long_running"
        ),
    }


def get_registry_entry(command_path: list[str]) -> MCPRegistryEntry | None:
    """Return one registry entry by exact CLI path."""
    target = tuple(command_path)
    for entry in build_registry():
        if tuple(entry.path) == target:
            return entry
    return None
