from pathlib import Path

import pytest

from autoclean.utils.metadata_table import (
    load_metadata_table,
    match_recording_row,
    require_columns,
    split_channels,
)


def test_load_metadata_table_infers_csv_and_trims_cells(tmp_path: Path) -> None:
    table = tmp_path / "bad_channels.csv"
    table.write_text("file,bad_channels\n subject01.set , E1; E2 \n", encoding="utf-8")

    rows = load_metadata_table(table)

    assert rows == [{"file": "subject01.set", "bad_channels": "E1; E2"}]


def test_load_metadata_table_infers_tsv(tmp_path: Path) -> None:
    table = tmp_path / "bad_channels.tsv"
    table.write_text("file\tbad_channels\nsubject01\tE3|E4\n", encoding="utf-8")

    rows = load_metadata_table(table)

    assert rows == [{"file": "subject01", "bad_channels": "E3|E4"}]


def test_match_recording_row_by_filename_or_stem(tmp_path: Path) -> None:
    rows = [
        {"file": "other.set", "bad_channels": "E1"},
        {"file": "subject01", "bad_channels": "E2"},
    ]

    match = match_recording_row(rows, tmp_path / "subject01.set")

    assert match is not None
    assert match.row["bad_channels"] == "E2"
    assert match.matched_by == "file"


def test_match_recording_row_rejects_ambiguous_matches(tmp_path: Path) -> None:
    rows = [
        {"file": "subject01.set", "bad_channels": "E1"},
        {"file": "subject01.set", "bad_channels": "E2"},
    ]

    with pytest.raises(ValueError, match="expected exactly one"):
        match_recording_row(rows, tmp_path / "subject01.set")


def test_require_columns_reports_available_columns() -> None:
    rows = [{"filename": "subject01.set", "channels": "E1"}]

    with pytest.raises(ValueError, match="filename"):
        require_columns(rows, ["file"])


def test_split_channels_accepts_common_delimiters_and_deduplicates() -> None:
    assert split_channels("E1; E2, E3|E1") == ["E1", "E2", "E3"]


def test_match_recording_row_by_exact_subject_session_fields(tmp_path: Path) -> None:
    rows = [
        {
            "file": "same.set",
            "subject": "sub-01",
            "session": "ses-1",
            "bad_channels": "E1",
        },
        {
            "file": "same.set",
            "subject": "sub-02",
            "session": "ses-1",
            "bad_channels": "E2",
        },
    ]

    match = match_recording_row(
        rows,
        tmp_path / "same.set",
        field_matches={"subject": "sub-02", "session": "ses-1"},
    )

    assert match is not None
    assert match.row["bad_channels"] == "E2"
    assert match.matched_by == "field"


def test_match_recording_row_falls_back_to_file_when_field_values_absent(
    tmp_path: Path,
) -> None:
    rows = [
        {"file": "subject01.set", "subject": "", "bad_channels": "E1"},
        {"file": "other.set", "subject": "sub-02", "bad_channels": "E2"},
    ]

    match = match_recording_row(
        rows,
        tmp_path / "subject01.set",
        field_matches={"subject": ""},
    )

    assert match is not None
    assert match.row["bad_channels"] == "E1"
    assert match.matched_by == "file"


def test_field_matching_does_not_select_different_file(tmp_path: Path) -> None:
    rows = [
        {"file": "other.set", "subject": "sub-02", "bad_channels": "E9"},
        {"file": "subject01.set", "subject": "sub-01", "bad_channels": "E1"},
    ]

    match = match_recording_row(
        rows,
        tmp_path / "subject01.set",
        field_matches={"subject": "sub-02"},
    )

    assert match is None
