"""Shared models for the AutoClean MCP server."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CLIArgumentSpec:
    """Typed description of one CLI argument."""

    name: str
    positional: bool
    required: bool
    option_strings: list[str] = field(default_factory=list)
    nargs: str | int | None = None
    choices: list[str] = field(default_factory=list)
    help: str = ""
    action_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CLICommandSpec:
    """Typed description of one CLI leaf command."""

    path: list[str]
    help: str = ""
    description: str = ""
    execution_style: str = "one_shot"
    arguments: list[CLIArgumentSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "help": self.help,
            "description": self.description,
            "execution_style": self.execution_style,
            "arguments": [arg.to_dict() for arg in self.arguments],
        }


@dataclass(slots=True)
class MCPRegistryEntry:
    """Maintained registry entry for one CLI leaf command."""

    command_id: str
    path: list[str]
    family: str
    execution_style: str
    wrapper_kind: str
    output_mode: str
    mutating: bool
    destructive: bool
    help: str = ""
    description: str = ""
    arguments: list[CLIArgumentSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "path": self.path,
            "family": self.family,
            "execution_style": self.execution_style,
            "wrapper_kind": self.wrapper_kind,
            "output_mode": self.output_mode,
            "mutating": self.mutating,
            "destructive": self.destructive,
            "help": self.help,
            "description": self.description,
            "arguments": [arg.to_dict() for arg in self.arguments],
            "notes": self.notes,
        }


@dataclass(slots=True)
class CLIExecutionResult:
    """Normalized subprocess execution result."""

    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    started_at: str
    finished_at: str
    duration_ms: int
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionStatus:
    """Status for one managed background session."""

    session_id: str
    command: list[str]
    cwd: str
    state: str
    pid: int | None
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
