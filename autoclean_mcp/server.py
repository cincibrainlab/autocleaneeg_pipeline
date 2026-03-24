"""FastMCP server for controlling the AutoClean CLI."""

from __future__ import annotations

from typing import Any

from autoclean_mcp.argv_builder import build_registry_argv
from autoclean_mcp.cli_adapter import execute_cli, get_canonical_cli_command
from autoclean_mcp.cli_inventory import load_cli_inventory
from autoclean_mcp.models import MCPRegistryEntry
from autoclean_mcp.registry import build_registry, get_registry_entry, registry_summary
from autoclean_mcp.session_manager import SESSION_MANAGER

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - exercised only when dependency missing
    FastMCP = None  # type: ignore[assignment]


def _ensure_mutation_confirmed(confirm: bool, command_path: list[str]) -> None:
    """Require explicit opt-in before running a mutating typed wrapper."""
    if confirm:
        return
    joined = " ".join(command_path)
    raise ValueError(
        f"Refusing to run mutating command '{joined}' without confirm=True."
    )


def _get_registry_entry_or_raise(command_path: list[str]) -> MCPRegistryEntry:
    """Load one registry entry or raise a useful error."""
    entry = get_registry_entry(command_path)
    if entry is None:
        joined = " ".join(command_path)
        raise ValueError(f"Unknown CLI command path: {joined}")
    return entry


def _prepare_registered_one_shot(
    command_path: list[str],
    *,
    confirm: bool = False,
    allow_compatibility: bool = False,
) -> MCPRegistryEntry:
    """Validate one-shot execution mode for a registry command."""
    entry = _get_registry_entry_or_raise(command_path)
    if entry.wrapper_kind == "managed_session":
        joined = " ".join(command_path)
        raise ValueError(
            f"Command '{joined}' must be started with start_registered_cli_session."
        )
    if entry.wrapper_kind == "compatibility_wrapper" and not allow_compatibility:
        joined = " ".join(command_path)
        raise ValueError(
            f"Command '{joined}' must be run with run_compatibility_cli_command."
        )
    if entry.mutating:
        _ensure_mutation_confirmed(confirm, command_path)
    return entry


def _prepare_registered_session(command_path: list[str]) -> MCPRegistryEntry:
    """Validate managed-session execution mode for a registry command."""
    entry = _get_registry_entry_or_raise(command_path)
    if entry.wrapper_kind != "managed_session":
        joined = " ".join(command_path)
        raise ValueError(
            f"Command '{joined}' is not a managed-session command."
        )
    return entry


def _prepare_compatibility_session(command_path: list[str]) -> MCPRegistryEntry:
    """Validate session-backed execution mode for a compatibility command."""
    entry = _get_registry_entry_or_raise(command_path)
    if entry.wrapper_kind != "compatibility_wrapper":
        joined = " ".join(command_path)
        raise ValueError(
            f"Command '{joined}' is not a compatibility-wrapper command."
        )
    return entry


def create_mcp_server():
    """Create the FastMCP server instance."""
    if FastMCP is None:
        raise RuntimeError(
            "fastmcp is not installed. Install project dependencies before running the MCP server."
        )

    startup_report = SESSION_MANAGER.initialize_startup()
    mcp = FastMCP("AutoCleanEEG MCP")

    def _run_registered(
        command_path: list[str],
        *,
        arguments: dict[str, Any] | None = None,
        cwd: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        argv = build_registry_argv(command_path, arguments)
        result = execute_cli(argv, cwd=cwd, timeout_seconds=timeout_seconds)
        payload = result.to_dict()
        payload["argv"] = argv
        payload["command_path"] = command_path
        return payload

    def _start_registered_wrapper_session(
        command_path: list[str],
        *,
        arguments: dict[str, Any] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        compatibility: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if compatibility:
            _prepare_compatibility_session(command_path)
            if get_registry_entry(command_path).mutating:  # type: ignore[union-attr]
                _ensure_mutation_confirmed(confirm, command_path)
        else:
            entry = _prepare_registered_session(command_path)
            if entry.mutating:
                _ensure_mutation_confirmed(confirm, command_path)
        argv = build_registry_argv(command_path, arguments)
        status = SESSION_MANAGER.start(
            get_canonical_cli_command(argv),
            cwd=cwd,
            env=env,
        )
        payload = status.to_dict()
        payload["argv"] = argv
        payload["command_path"] = command_path
        if compatibility:
            payload["compatibility_mode"] = True
        return payload

    @mcp.tool
    def list_cli_commands() -> dict[str, Any]:
        """Return the CLI command inventory extracted from argparse."""
        commands = [spec.to_dict() for spec in load_cli_inventory()]
        return {"commands": commands, "count": len(commands)}

    @mcp.tool
    def get_cli_registry_summary() -> dict[str, Any]:
        """Return summary counts for the maintained MCP registry."""
        return registry_summary()

    @mcp.tool
    def list_cli_registry_entries() -> dict[str, Any]:
        """Return the maintained MCP registry entries."""
        entries = [entry.to_dict() for entry in build_registry()]
        return {"entries": entries, "count": len(entries)}

    @mcp.tool
    def get_cli_registry_entry(command_path: list[str]) -> dict[str, Any]:
        """Return one maintained MCP registry entry by exact CLI path."""
        entry = get_registry_entry(command_path)
        if entry is None:
            return {"found": False, "command_path": command_path}
        payload = entry.to_dict()
        payload["found"] = True
        return payload

    @mcp.tool
    def run_cli(
        argv: list[str],
        cwd: str | None = None,
        timeout_seconds: float | None = 60.0,
    ) -> dict[str, Any]:
        """Run one AutoClean CLI command through the canonical subprocess adapter."""
        result = execute_cli(argv, cwd=cwd, timeout_seconds=timeout_seconds)
        return result.to_dict()

    @mcp.tool
    def run_registered_cli_command(
        command_path: list[str],
        arguments: dict[str, Any] | None = None,
        confirm: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = 60.0,
    ) -> dict[str, Any]:
        """Run one CLI command by registry path with structured argument mapping."""
        _prepare_registered_one_shot(command_path, confirm=confirm)
        return _run_registered(
            command_path,
            arguments=arguments,
            cwd=cwd,
            timeout_seconds=timeout_seconds or 60.0,
        )

    @mcp.tool
    def run_compatibility_cli_command(
        command_path: list[str],
        arguments: dict[str, Any] | None = None,
        confirm: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = 60.0,
    ) -> dict[str, Any]:
        """Run one compatibility-wrapped CLI command by registry path."""
        entry = _prepare_registered_one_shot(
            command_path,
            confirm=confirm,
            allow_compatibility=True,
        )
        if entry.wrapper_kind != "compatibility_wrapper":
            joined = " ".join(command_path)
            raise ValueError(
                f"Command '{joined}' is not classified as a compatibility wrapper."
            )
        payload = _run_registered(
            command_path,
            arguments=arguments,
            cwd=cwd,
            timeout_seconds=timeout_seconds or 60.0,
        )
        payload["compatibility_mode"] = True
        return payload

    @mcp.tool
    def start_registered_cli_session(
        command_path: list[str],
        arguments: dict[str, Any] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Start one managed-session CLI command by registry path."""
        _prepare_registered_session(command_path)
        argv = build_registry_argv(command_path, arguments)
        status = SESSION_MANAGER.start(
            get_canonical_cli_command(argv),
            cwd=cwd,
            env=env,
        )
        payload = status.to_dict()
        payload["argv"] = argv
        payload["command_path"] = command_path
        return payload

    @mcp.tool
    def start_compatibility_cli_session(
        command_path: list[str],
        arguments: dict[str, Any] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Start one compatibility-wrapper CLI command as a managed session."""
        _prepare_compatibility_session(command_path)
        argv = build_registry_argv(command_path, arguments)
        status = SESSION_MANAGER.start(
            get_canonical_cli_command(argv),
            cwd=cwd,
            env=env,
        )
        payload = status.to_dict()
        payload["argv"] = argv
        payload["command_path"] = command_path
        payload["compatibility_mode"] = True
        return payload

    @mcp.tool
    def get_cli_entrypoint() -> dict[str, Any]:
        """Return the canonical subprocess command prefix used by MCP."""
        return {"command_prefix": get_canonical_cli_command([])}

    @mcp.tool
    def get_cli_session_startup_report() -> dict[str, Any]:
        """Return the current session startup recovery policy report."""
        return dict(startup_report)

    @mcp.tool
    def cli_version(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline version`."""
        return _run_registered(["version"], cwd=cwd)

    @mcp.tool
    def cli_whoami(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline whoami`."""
        return _run_registered(["whoami"], cwd=cwd)

    @mcp.tool
    def cli_auth0_diagnostics(
        verbose: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth0-diagnostics`."""
        return _run_registered(
            ["auth0-diagnostics"],
            arguments={"verbose": verbose},
            cwd=cwd,
        )

    @mcp.tool
    def cli_help(topic: str | None = None, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline help`."""
        return _run_registered(["help"], arguments={"topic": topic}, cwd=cwd)

    @mcp.tool
    def cli_list_tasks(
        verbose: bool | None = None,
        overrides: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline list-tasks`."""
        return _run_registered(
            ["list-tasks"],
            arguments={"verbose": verbose, "overrides": overrides},
            cwd=cwd,
        )

    @mcp.tool
    def cli_workspace_show(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline workspace show`."""
        return _run_registered(["workspace", "show"], cwd=cwd)

    @mcp.tool
    def cli_workspace_size(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline workspace size`."""
        return _run_registered(["workspace", "size"], cwd=cwd)

    @mcp.tool
    def cli_config_show(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline config show`."""
        return _run_registered(["config", "show"], cwd=cwd)

    @mcp.tool
    def cli_source_show(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline source show`."""
        return _run_registered(["source", "show"], cwd=cwd)

    @mcp.tool
    def cli_input_show(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline input show`."""
        return _run_registered(["input", "show"], cwd=cwd)

    @mcp.tool
    def cli_blocks_list(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline blocks list`."""
        return _run_registered(["blocks", "list"], cwd=cwd)

    @mcp.tool
    def cli_blocks_info(block_name: str, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline blocks info`."""
        return _run_registered(
            ["blocks", "info"],
            arguments={"block_name": block_name},
            cwd=cwd,
        )

    @mcp.tool
    def cli_blocks_deps(block_name: str, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline blocks deps`."""
        return _run_registered(
            ["blocks", "deps"],
            arguments={"block_name": block_name},
            cwd=cwd,
        )

    @mcp.tool
    def cli_events_discover(
        input_file: str,
        montage: str | None = None,
        no_config: bool | None = None,
        exclude: list[str] | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline events discover`."""
        return _run_registered(
            ["events", "discover"],
            arguments={
                "input_file": input_file,
                "montage": montage,
                "no_config": no_config,
                "exclude": exclude,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_events_analyze(
        input_file: str,
        montage: str | None = None,
        gap_threshold: float | None = None,
        top_transitions: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline events analyze`."""
        return _run_registered(
            ["events", "analyze"],
            arguments={
                "input_file": input_file,
                "montage": montage,
                "gap_threshold": gap_threshold,
                "top_transitions": top_transitions,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_events_epochs(input_file: str, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline events epochs`."""
        return _run_registered(
            ["events", "epochs"],
            arguments={"input_file": input_file},
            cwd=cwd,
        )

    @mcp.tool
    def cli_montage_list(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline montage list`."""
        return _run_registered(["montage", "list"], cwd=cwd)

    @mcp.tool
    def cli_montage_test(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline montage test`."""
        return _run_registered(["montage", "test"], cwd=cwd)

    @mcp.tool
    def cli_task_list(
        source: str | None = None,
        status: str | None = None,
        category: str | None = None,
        format: str | None = "json",
        verbose: bool | None = None,
        overrides: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task list`."""
        return _run_registered(
            ["task", "list"],
            arguments={
                "source": source,
                "status": status,
                "category": category,
                "format": format,
                "verbose": verbose,
                "overrides": overrides,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_search(
        query: str,
        source: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task search`."""
        return _run_registered(
            ["task", "search"],
            arguments={"query": query, "source": source},
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_show(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task show`."""
        return _run_registered(["task", "show"], cwd=cwd)

    @mcp.tool
    def cli_task_diagnose(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task diagnose`."""
        return _run_registered(["task", "diagnose"], cwd=cwd)

    @mcp.tool
    def cli_task_diff(
        task_name: str,
        color: bool | None = None,
        context: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task diff`."""
        return _run_registered(
            ["task", "diff"],
            arguments={"task_name": task_name, "color": color, "context": context},
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_schema_export(
        output: str | None = None,
        indent: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task schema export`."""
        return _run_registered(
            ["task", "schema", "export"],
            arguments={"output": output, "indent": indent},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_status(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve status`."""
        return _run_registered(["serve", "status"], cwd=cwd)

    @mcp.tool
    def cli_serve_docs(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve docs`."""
        return _run_registered(["serve", "docs"], cwd=cwd)

    @mcp.tool
    def cli_serve_list(
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve list`."""
        return _run_registered(
            ["serve", "list"],
            arguments={"path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_route_list(
        path: str | None = None,
        mode: str | None = None,
        include_archived: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route list`."""
        return _run_registered(
            ["serve", "route", "list"],
            arguments={
                "path": path,
                "mode": mode,
                "include_archived": include_archived,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_workspace_status(
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve workspace status`."""
        return _run_registered(
            ["serve", "workspace", "status"],
            arguments={"path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_workspace_doctor(
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve workspace doctor`."""
        return _run_registered(
            ["serve", "workspace", "doctor"],
            arguments={"path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_mode_status(
        mode_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve mode status`."""
        return _run_registered(
            ["serve", "mode", "status"],
            arguments={"mode_port": mode_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_service_status(
        service_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve service status`."""
        return _run_registered(
            ["serve", "service", "status"],
            arguments={"service_port": service_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_queue_status(
        queue_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve queue status`."""
        return _run_registered(
            ["serve", "queue", "status"],
            arguments={"queue_port": queue_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_queue_list(
        queue_port: int | None = None,
        status: str | None = None,
        route_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve queue list`."""
        return _run_registered(
            ["serve", "queue", "list"],
            arguments={
                "queue_port": queue_port,
                "status": status,
                "route_id": route_id,
                "limit": limit,
                "offset": offset,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_validate(
        mode: str | None = None,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve validate`."""
        return _run_registered(
            ["serve", "validate"],
            arguments={"mode": mode, "path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_deploy(
        confirm: bool,
        mode: str | None = None,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve deploy`."""
        command_path = ["serve", "deploy"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"mode": mode, "path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_share_status(cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve share status`."""
        return _run_registered(["serve", "share", "status"], cwd=cwd)

    @mcp.tool
    def cli_serve_route_sync(
        confirm: bool,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route sync`."""
        command_path = ["serve", "route", "sync"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_route_promote(
        confirm: bool,
        route_id: str,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route promote`."""
        command_path = ["serve", "route", "promote"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"route_id": route_id, "path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_route_archive(
        confirm: bool,
        route_id: str,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route archive`."""
        command_path = ["serve", "route", "archive"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"route_id": route_id, "path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_route_unarchive(
        confirm: bool,
        route_id: str,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route unarchive`."""
        command_path = ["serve", "route", "unarchive"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"route_id": route_id, "path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_route_delete(
        confirm: bool,
        route_id: str,
        path: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route delete`."""
        command_path = ["serve", "route", "delete"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"route_id": route_id, "path": path, "force": force},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_service_start(
        confirm: bool,
        service_port: int | None = None,
        max_cycles: int | None = None,
        idle_limit: int | None = None,
        sleep_seconds: float | None = None,
        no_watch: bool | None = None,
        no_sentinel: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve service start`."""
        command_path = ["serve", "service", "start"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "service_port": service_port,
                "max_cycles": max_cycles,
                "idle_limit": idle_limit,
                "sleep_seconds": sleep_seconds,
                "no_watch": no_watch,
                "no_sentinel": no_sentinel,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_service_stop(
        confirm: bool,
        service_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve service stop`."""
        command_path = ["serve", "service", "stop"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"service_port": service_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_mode_test(
        confirm: bool,
        mode_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve mode test`."""
        command_path = ["serve", "mode", "test"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"mode_port": mode_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_mode_live(
        confirm: bool,
        mode_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve mode live`."""
        command_path = ["serve", "mode", "live"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"mode_port": mode_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_queue_retry_failed(
        confirm: bool,
        queue_port: int | None = None,
        paths: list[str] | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve queue retry-failed`."""
        command_path = ["serve", "queue", "retry-failed"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"queue_port": queue_port, "paths": paths},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_down(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve down`."""
        command_path = ["serve", "down"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_serve_restart(
        confirm: bool,
        host: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve restart`."""
        command_path = ["serve", "restart"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"host": host, "force": force},
            cwd=cwd,
        )

    @mcp.tool
    def cli_wizard(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline wizard`."""
        return run_compatibility_cli_command(
            ["wizard"],
            confirm=confirm,
            cwd=cwd,
        )

    @mcp.tool
    def cli_process_ica(
        confirm: bool,
        metadata_dir: str | None = None,
        dry_run: bool | None = None,
        verbose: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline process ica`."""
        command_path = ["process", "ica"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "metadata_dir": metadata_dir,
                "dry_run": dry_run,
                "verbose": verbose,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_review(
        output: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline review`."""
        return run_compatibility_cli_command(
            ["review"],
            arguments={"output": output},
            cwd=cwd,
        )

    @mcp.tool
    def cli_exclude(
        path: str | None = None,
        exports: str | None = None,
        task_root: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline exclude`."""
        return run_compatibility_cli_command(
            ["exclude"],
            arguments={"path": path, "exports": exports, "task_root": task_root},
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_delete(
        confirm: bool,
        target: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task delete`."""
        command_path = ["task", "delete"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"target": target, "force": force},
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_explore(cwd: str | None = None) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline task explore`."""
        return run_compatibility_cli_command(["task", "explore"], cwd=cwd)

    @mcp.tool
    def cli_task_edit(
        confirm: bool,
        target: str | None = None,
        name: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline task edit`."""
        return run_compatibility_cli_command(
            ["task", "edit"],
            arguments={"target": target, "name": name, "force": force},
            confirm=confirm,
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_copy(
        confirm: bool,
        source: str | None = None,
        name: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task copy`."""
        command_path = ["task", "copy"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"source": source, "name": name, "force": force},
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_set(
        confirm: bool,
        task_name: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline task set`."""
        return run_compatibility_cli_command(
            ["task", "set"],
            arguments={"task_name": task_name},
            confirm=confirm,
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_unset(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task unset`."""
        command_path = ["task", "unset"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_task_use(
        confirm: bool,
        task_name: str | None = None,
        force: bool | None = None,
        no_activate: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task use`."""
        command_path = ["task", "use"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "task_name": task_name,
                "force": force,
                "no_activate": no_activate,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_create(
        confirm: bool,
        class_name: str,
        file_name: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task create`."""
        command_path = ["task", "create"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "class_name": class_name,
                "file_name": file_name,
                "force": force,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_install(
        confirm: bool,
        task_source: str,
        source: str | None = None,
        name: str | None = None,
        force: bool | None = None,
        activate: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task install`."""
        command_path = ["task", "install"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "task_source": task_source,
                "source": source,
                "name": name,
                "force": force,
                "activate": activate,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_sync(
        confirm: bool,
        update: bool | None = None,
        dry_run: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task sync`."""
        command_path = ["task", "sync"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"update": update, "dry_run": dry_run},
            cwd=cwd,
        )

    @mcp.tool
    def cli_task_update(
        confirm: bool,
        no_network: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline task update`."""
        command_path = ["task", "update"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"no_network": no_network},
            cwd=cwd,
        )

    @mcp.tool
    def cli_montage_set(
        confirm: bool,
        montage_name: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline montage set`."""
        return run_compatibility_cli_command(
            ["montage", "set"],
            arguments={"montage_name": montage_name, "force": force},
            confirm=confirm,
            cwd=cwd,
        )

    @mcp.tool
    def cli_blocks_update(
        confirm: bool,
        no_network: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline blocks update`."""
        command_path = ["blocks", "update"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"no_network": no_network},
            cwd=cwd,
        )

    @mcp.tool
    def cli_blocks_install(
        confirm: bool,
        block_name: str | None = None,
        commit: str | None = None,
        locked: bool | None = None,
        lock_file: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline blocks install`."""
        command_path = ["blocks", "install"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "block_name": block_name,
                "commit": commit,
                "locked": locked,
                "lock_file": lock_file,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_blocks_lock(
        confirm: bool,
        output: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline blocks lock`."""
        command_path = ["blocks", "lock"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"output": output},
            cwd=cwd,
        )

    @mcp.tool
    def cli_source_set(
        confirm: bool,
        source_path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline source set`."""
        command_path = ["source", "set"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"source_path": source_path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_source_unset(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline source unset`."""
        command_path = ["source", "unset"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_input_set(
        confirm: bool,
        source_path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline input set`."""
        command_path = ["input", "set"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"source_path": source_path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_input_unset(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline input unset`."""
        command_path = ["input", "unset"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_config_setup(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline config setup`."""
        command_path = ["config", "setup"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_config_reset(
        confirm: bool,
        cli_confirm: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline config reset`."""
        command_path = ["config", "reset"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"confirm": cli_confirm},
            cwd=cwd,
        )

    @mcp.tool
    def cli_config_export(
        confirm: bool,
        export_path: str,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline config export`."""
        command_path = ["config", "export"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"export_path": export_path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_config_import(
        confirm: bool,
        import_path: str,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline config import`."""
        command_path = ["config", "import"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"import_path": import_path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_workspace_explore(cwd: str | None = None) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline workspace explore`."""
        return run_compatibility_cli_command(["workspace", "explore"], cwd=cwd)

    @mcp.tool
    def cli_workspace_set(
        confirm: bool,
        path: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline workspace set`."""
        command_path = ["workspace", "set"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_workspace_unset(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline workspace unset`."""
        command_path = ["workspace", "unset"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_workspace_default(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline workspace default`."""
        command_path = ["workspace", "default"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_workspace_cd(
        spawn: bool | None = None,
        print: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline workspace cd`."""
        return run_compatibility_cli_command(
            ["workspace", "cd"],
            arguments={"spawn": spawn, "print": print},
            cwd=cwd,
        )

    @mcp.tool
    def cli_export_access_log(
        confirm: bool,
        output: str | None = None,
        format: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        operation: str | None = None,
        verify_only: bool | None = None,
        database: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline export-access-log`."""
        command_path = ["export-access-log"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "output": output,
                "format": format,
                "start_date": start_date,
                "end_date": end_date,
                "operation": operation,
                "verify_only": verify_only,
                "database": database,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_login(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline login`."""
        return run_compatibility_cli_command(
            ["login"],
            confirm=confirm,
            cwd=cwd,
        )

    @mcp.tool
    def cli_logout(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline logout`."""
        command_path = ["logout"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_clean_task(
        confirm: bool,
        task: str,
        output_dir: str | None = None,
        force: bool | None = None,
        dry_run: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline clean-task`."""
        command_path = ["clean-task"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "task": task,
                "output_dir": output_dir,
                "force": force,
                "dry_run": dry_run,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_report_create(
        confirm: bool,
        run_id: str | None = None,
        context_json: str | None = None,
        out_dir: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline report create`."""
        command_path = ["report", "create"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "run_id": run_id,
                "context_json": context_json,
                "out_dir": out_dir,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_report_chat(
        context_json: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline report chat`."""
        return run_compatibility_cli_command(
            ["report", "chat"],
            arguments={"context_json": context_json},
            cwd=cwd,
        )

    @mcp.tool
    def cli_view(
        file: str | None = None,
        no_view: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline view`."""
        return run_compatibility_cli_command(
            ["view"],
            arguments={"file": file, "no_view": no_view},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_workspace_use(
        confirm: bool,
        path: str | None = None,
        mode: str | None = None,
        skip_uv: bool | None = None,
        no_test: bool | None = None,
        package: str | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve workspace use`."""
        command_path = ["serve", "workspace", "use"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "path": path,
                "mode": mode,
                "skip_uv": skip_uv,
                "no_test": no_test,
                "package": package,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_route_upsert(
        confirm: bool,
        route_id: str,
        path: str | None = None,
        mode: str | None = None,
        taskfile: str | None = None,
        montage: str | None = None,
        version: str | None = None,
        ingestion_folders: list[str] | None = None,
        ingestion_excludes: list[str] | None = None,
        file_globs: list[str] | None = None,
        priority: int | None = None,
        automation_root: str | None = None,
        workspace_name: str | None = None,
        sentinel_ext: str | None = None,
        enabled: bool | None = None,
        recursive: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve route upsert`."""
        command_path = ["serve", "route", "upsert"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={
                "route_id": route_id,
                "path": path,
                "mode": mode,
                "taskfile": taskfile,
                "montage": montage,
                "version": version,
                "ingestion_folders": ingestion_folders,
                "ingestion_excludes": ingestion_excludes,
                "file_globs": file_globs,
                "priority": priority,
                "automation_root": automation_root,
                "workspace_name": workspace_name,
                "sentinel_ext": sentinel_ext,
                "enabled": enabled,
                "recursive": recursive,
            },
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_run(
        mode: str | None = None,
        path: str | None = None,
        max_cycles: int | None = None,
        idle_limit: int | None = None,
        file_glob: str | None = None,
        sentinel_ext: str | None = None,
        no_sentinel: bool | None = None,
        no_watch: bool | None = None,
        max_events: int | None = None,
        sleep_seconds: float | None = None,
        queue_path: str | None = None,
        dry_run: bool | None = None,
        use_operator: bool | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Managed-session MCP wrapper for `autocleaneeg-pipeline serve run`."""
        return _start_registered_wrapper_session(
            ["serve", "run"],
            arguments={
                "mode": mode,
                "path": path,
                "max_cycles": max_cycles,
                "idle_limit": idle_limit,
                "file_glob": file_glob,
                "sentinel_ext": sentinel_ext,
                "no_sentinel": no_sentinel,
                "no_watch": no_watch,
                "max_events": max_events,
                "sleep_seconds": sleep_seconds,
                "queue_path": queue_path,
                "dry_run": dry_run,
                "use_operator": use_operator,
            },
            cwd=cwd,
            env=env,
        )

    @mcp.tool
    def cli_serve_tui(
        mode: str | None = None,
        path: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Compatibility MCP wrapper for `autocleaneeg-pipeline serve tui`."""
        return _start_registered_wrapper_session(
            ["serve", "tui"],
            arguments={"mode": mode, "path": path},
            cwd=cwd,
            env=env,
            compatibility=True,
        )

    @mcp.tool
    def cli_serve_api(
        mode: str | None = None,
        path: str | None = None,
        host: str | None = None,
        api_port: int | None = None,
        redis_url: str | None = None,
        reload: bool | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Managed-session MCP wrapper for `autocleaneeg-pipeline serve api`."""
        return _start_registered_wrapper_session(
            ["serve", "api"],
            arguments={
                "mode": mode,
                "path": path,
                "host": host,
                "api_port": api_port,
                "redis_url": redis_url,
                "reload": reload,
            },
            cwd=cwd,
            env=env,
        )

    @mcp.tool
    def cli_serve_worker(
        queues: str | None = None,
        redis_url: str | None = None,
        burst: bool | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Managed-session MCP wrapper for `autocleaneeg-pipeline serve worker`."""
        return _start_registered_wrapper_session(
            ["serve", "worker"],
            arguments={"queues": queues, "redis_url": redis_url, "burst": burst},
            cwd=cwd,
            env=env,
        )

    @mcp.tool
    def cli_serve_queue_clear_processed(
        confirm: bool,
        queue_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve queue clear-processed`."""
        command_path = ["serve", "queue", "clear-processed"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"queue_port": queue_port},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_queue_remove(
        confirm: bool,
        path: str,
        queue_port: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve queue remove`."""
        command_path = ["serve", "queue", "remove"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"queue_port": queue_port, "path": path},
            cwd=cwd,
        )

    @mcp.tool
    def cli_serve_up(
        confirm: bool,
        host: str | None = None,
        force: bool | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Managed-session MCP wrapper for `autocleaneeg-pipeline serve up`."""
        return _start_registered_wrapper_session(
            ["serve", "up"],
            arguments={"host": host, "force": force},
            cwd=cwd,
            env=env,
            confirm=confirm,
        )

    @mcp.tool
    def cli_serve_share_start(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve share start`."""
        command_path = ["serve", "share", "start"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_serve_share_stop(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve share stop`."""
        command_path = ["serve", "share", "stop"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_serve_share_setup(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve share setup`."""
        command_path = ["serve", "share", "setup"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_serve_share_clear(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline serve share clear`."""
        command_path = ["serve", "share", "clear"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_tutorial(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline tutorial`."""
        command_path = ["tutorial"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_login(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth login`."""
        command_path = ["auth", "login"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_logout(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth logout`."""
        command_path = ["auth", "logout"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_whoami(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth whoami`."""
        command_path = ["auth", "whoami"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_diagnostics(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth diagnostics`."""
        command_path = ["auth", "diagnostics"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_setup(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth setup`."""
        command_path = ["auth", "setup"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_enable(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth enable`."""
        command_path = ["auth", "enable"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_auth_disable(confirm: bool, cwd: str | None = None) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline auth disable`."""
        command_path = ["auth", "disable"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(command_path, cwd=cwd)

    @mcp.tool
    def cli_settings_theme(
        confirm: bool,
        theme_name: str | None = None,
        clear: bool | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        """Typed MCP wrapper for `autocleaneeg-pipeline settings theme`."""
        command_path = ["settings", "theme"]
        _ensure_mutation_confirmed(confirm, command_path)
        return _run_registered(
            command_path,
            arguments={"theme_name": theme_name, "clear": clear},
            cwd=cwd,
        )

    @mcp.tool
    def start_cli_session(argv: list[str], cwd: str | None = None) -> dict[str, Any]:
        """Start a managed long-running AutoClean CLI session."""
        status = SESSION_MANAGER.start(get_canonical_cli_command(argv), cwd=cwd)
        return status.to_dict()

    @mcp.tool
    def get_cli_session(session_id: str) -> dict[str, Any]:
        """Inspect one managed CLI session."""
        status = SESSION_MANAGER.get(session_id)
        if status is None:
            return {"session_id": session_id, "found": False}
        payload = status.to_dict()
        payload["found"] = True
        return payload

    @mcp.tool
    def list_cli_sessions() -> dict[str, Any]:
        """List managed CLI sessions."""
        sessions = [status.to_dict() for status in SESSION_MANAGER.list()]
        return {"sessions": sessions, "count": len(sessions)}

    @mcp.tool
    def stop_cli_session(session_id: str) -> dict[str, Any]:
        """Stop one managed CLI session."""
        status = SESSION_MANAGER.stop(session_id)
        if status is None:
            return {"session_id": session_id, "found": False}
        payload = status.to_dict()
        payload["found"] = True
        return payload

    return mcp


def main() -> None:
    """Run the FastMCP server."""
    create_mcp_server().run()


if __name__ == "__main__":
    main()
