"""Helpers for matching external metadata tables to input recordings."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MetadataMatch:
    """Result of matching a metadata table row to an input recording."""

    row: dict[str, str]
    matched_by: str


def load_metadata_table(
    path: str | Path,
    *,
    delimiter: str | None = None,
) -> list[dict[str, str]]:
    """Load a small CSV/TSV metadata table as normalised string rows."""

    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Metadata table not found: {table_path}")

    resolved_delimiter = delimiter or _delimiter_for_path(table_path)
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=resolved_delimiter)
        if not reader.fieldnames:
            raise ValueError(f"Metadata table has no header row: {table_path}")

        rows: list[dict[str, str]] = []
        for raw_row in reader:
            rows.append(
                {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in raw_row.items()
                    if key is not None
                }
            )
    return rows


def require_columns(rows: list[dict[str, str]], columns: Iterable[str]) -> None:
    """Raise when any required column is absent from a loaded table."""

    if not rows:
        return

    available = set(rows[0].keys())
    missing = [column for column in columns if column not in available]
    if missing:
        raise ValueError(
            "Metadata table is missing required column(s): "
            f"{', '.join(missing)}. Available columns: {', '.join(sorted(available))}"
        )


def match_recording_row(
    rows: list[dict[str, str]],
    recording_path: str | Path,
    *,
    file_column: str = "file",
    field_matches: Mapping[str, str] | None = None,
) -> MetadataMatch | None:
    """Match a recording to one table row by file and optional exact fields."""

    field_matches = {
        column: value
        for column, value in (field_matches or {}).items()
        if column and value not in (None, "")
    }
    required_columns = [file_column, *field_matches.keys()]
    require_columns(rows, required_columns)

    file_matches = _match_rows_by_file(rows, recording_path, file_column=file_column)
    if field_matches and file_matches:
        exact_matches = _match_rows_by_fields(file_matches, field_matches)
        return _single_match(exact_matches, "field", str(recording_path))
    return _single_match(file_matches, "file", str(recording_path))


def split_channels(value: object) -> list[str]:
    """Parse a user channel-list cell into ordered unique channel names."""

    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    channels: list[str] = []
    for part in text.replace(";", ",").replace("|", ",").split(","):
        channel = part.strip()
        if channel and channel not in channels:
            channels.append(channel)
    return channels


def _match_rows_by_file(
    rows: list[dict[str, str]],
    recording_path: str | Path,
    *,
    file_column: str,
) -> list[dict[str, str]]:
    recording = Path(recording_path)
    candidates = {
        recording.name.casefold(),
        recording.stem.casefold(),
        str(recording).casefold(),
    }
    return [
        row
        for row in rows
        if _normalise_file_cell(row.get(file_column, "")) in candidates
    ]


def _match_rows_by_fields(
    rows: list[dict[str, str]], field_matches: Mapping[str, str]
) -> list[dict[str, str]]:
    expected = {
        column: str(value).strip().casefold() for column, value in field_matches.items()
    }
    return [
        row
        for row in rows
        if all(
            str(row.get(column, "")).strip().casefold() == value
            for column, value in expected.items()
        )
    ]


def _single_match(
    matches: list[dict[str, str]], matched_by: str, recording_label: str
) -> MetadataMatch | None:
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"Metadata table has {len(matches)} rows matching recording "
            f"{recording_label!r}; expected exactly one."
        )
    return MetadataMatch(row=matches[0], matched_by=matched_by)


def _delimiter_for_path(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"
    return ","


def _normalise_file_cell(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""

    path = Path(text)
    if path.name:
        text = path.name
    return text.casefold()
