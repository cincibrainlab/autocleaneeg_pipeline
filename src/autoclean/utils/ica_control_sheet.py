"""Utilities for managing ICA component control sheets.

This module provides helper functions to maintain a CSV control sheet that
tracks automatic and manual Independent Component Analysis (ICA) decisions.
The sheet acts as a single source of truth for which components have been
removed from each recording and allows idempotent re-processing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Default columns for the control sheet
COLUMNS: List[str] = [
    "original_file",
    "ica_fif",
    "auto_initial",
    "final_removed",
    "manual_add",
    "manual_drop",
    "status",
    "last_run_iso",
]


def load_control_sheet(path: str | Path) -> pd.DataFrame:
    """Load an existing control sheet or create an empty one.

    Parameters
    ----------
    path:
        Location of the CSV control sheet.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the control sheet columns. Missing columns are
        added automatically.
    """

    csv_path = Path(path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[COLUMNS]


def save_control_sheet(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a control sheet DataFrame to disk."""

    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def _parse_components(value: str | Iterable[int] | None) -> List[int]:
    """Parse a component list from a string or iterable.

    Returns a sorted list of unique integers. Empty values yield an empty list.
    """

    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return sorted({int(p) for p in parts})
    try:
        return sorted({int(v) for v in value})
    except TypeError:
        return [int(value)]


def _format_components(values: Iterable[int]) -> str:
    """Format a list of component indices as a CSV string."""

    items = sorted({int(v) for v in values})
    return ",".join(str(v) for v in items)


def apply_manual_edits(row: pd.Series) -> pd.Series:
    """Apply pending manual edits to a single control sheet row.

    This combines the existing ``final_removed`` components with any
    ``manual_add`` or ``manual_drop`` entries. The manual columns are cleared
    after applying and the status is updated to ``"applied"`` if changes were
    made, otherwise ``"auto"``.
    """

    base = _parse_components(row.get("final_removed", ""))
    add = _parse_components(row.get("manual_add", ""))
    drop = _parse_components(row.get("manual_drop", ""))

    final = sorted(set(base).union(add).difference(drop))

    row["final_removed"] = _format_components(final) if final else ""
    row["manual_add"] = ""
    row["manual_drop"] = ""
    row["last_run_iso"] = datetime.now(timezone.utc).isoformat()

    if final != base:
        row["status"] = "applied"
    else:
        row["status"] = "auto"

    return row


def update_pending_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply manual edits for all pending rows in a control sheet.

    Rows are considered pending if their ``status`` is ``"pending"`` or if
    ``manual_add``/``manual_drop`` contain values. The DataFrame is modified in
    place and returned for convenience.
    """

    for idx, row in df.iterrows():
        if (
            str(row.get("manual_add", "")).strip()
            or str(row.get("manual_drop", "")).strip()
            or row.get("status") == "pending"
        ):
            df.loc[idx] = apply_manual_edits(row)
    return df
