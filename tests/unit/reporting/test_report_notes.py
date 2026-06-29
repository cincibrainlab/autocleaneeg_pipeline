"""Tests for report-note collection."""

from autoclean.step_functions.reports import _collect_report_notes


def test_collect_report_notes_from_all_step_metadata():
    metadata = {
        "step_apply_source_localization": {
            "report_notes": [
                "source note",
                "source note",
                "second note",
            ]
        },
        "step_future_analysis": {
            "report_notes": [
                "future note",
                123,
                "123",
            ]
        },
    }

    assert _collect_report_notes(metadata) == [
        "source note",
        "second note",
        "future note",
        "123",
    ]


def test_collect_report_notes_ignores_missing_source_metadata():
    assert _collect_report_notes({}) == []


def test_collect_report_notes_ignores_malformed_report_notes():
    metadata = {
        "step_string_notes": {"report_notes": "not a list"},
        "step_none_notes": {"report_notes": None},
        "step_non_dict": ["not", "metadata"],
        "step_valid": {"report_notes": ["kept", "", None, False, "kept"]},
    }

    assert _collect_report_notes(metadata) == ["kept"]
