"""Helpers for applying user-provided recording exclusion lists."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoclean.utils.metadata_table import load_metadata_table, match_recording_row

_TRUE_VALUES = {"1", "true", "yes", "y", "exclude", "excluded", "skip", "x"}
_FALSE_VALUES = {"", "0", "false", "no", "n", "include", "included", "keep"}


@dataclass(frozen=True)
class ExclusionListResult:
    """Resolved exclusion-list decision for one recording."""

    mode: str
    excluded: bool
    reason: str | None
    warning: str | None
    metadata: dict[str, Any]


def evaluate_exclusion_list(
    config_value: dict[str, Any], recording_path: str | Path
) -> ExclusionListResult:
    """Evaluate a configured exclusion table for one recording.

    ``tag`` mode flags and reports matching recordings while still processing them.
    ``skip`` mode returns the same match decision so dispatch code can avoid
    starting the task run at all.
    """

    table_path = config_value.get("path")
    if not table_path:
        raise ValueError("exclusion_list is enabled but no value.path was set")

    mode = str(config_value.get("mode", "tag")).strip().casefold()
    if mode not in {"tag", "skip"}:
        raise ValueError("exclusion_list mode must be 'tag' or 'skip'.")

    file_column = config_value.get("file_column", "file")
    subject_column = config_value.get("subject_column")
    session_column = config_value.get("session_column")
    exclude_column = config_value.get("exclude_column", "exclude")
    reason_column = config_value.get("reason_column", "reason")
    strict = bool(config_value.get("strict", False))

    rows = load_metadata_table(table_path, delimiter=config_value.get("delimiter"))
    field_matches: dict[str, str] = {}
    if subject_column and config_value.get("subject"):
        field_matches[subject_column] = str(config_value["subject"])
    if session_column and config_value.get("session"):
        field_matches[session_column] = str(config_value["session"])

    match = match_recording_row(
        rows,
        recording_path,
        file_column=file_column,
        field_matches=field_matches,
    )

    metadata: dict[str, Any] = {
        "path": str(table_path),
        "mode": mode,
        "matched": False,
        "matched_by": None,
        "excluded": False,
        "reason": None,
    }

    if match is None:
        warning = f"No exclusion-list row matched input file {recording_path!s}"
        if strict:
            raise ValueError(warning)
        metadata["warning"] = warning
        return ExclusionListResult(
            mode=mode,
            excluded=False,
            reason=None,
            warning=warning,
            metadata=metadata,
        )

    if exclude_column not in match.row:
        raise ValueError(f"Exclusion list is missing exclude column {exclude_column!r}")

    excluded = _parse_excluded(match.row.get(exclude_column), exclude_column)
    reason = str(match.row.get(reason_column, "")).strip() or None

    metadata.update(
        {
            "matched": True,
            "matched_by": match.matched_by,
            "excluded": excluded,
            "reason": reason,
        }
    )
    return ExclusionListResult(
        mode=mode,
        excluded=excluded,
        reason=reason,
        warning=None,
        metadata=metadata,
    )


def _parse_excluded(value: object, column: str) -> bool:
    text = "" if value is None else str(value).strip().casefold()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Exclusion list column {column!r} must contain a boolean-like value; "
        f"got {value!r}."
    )
