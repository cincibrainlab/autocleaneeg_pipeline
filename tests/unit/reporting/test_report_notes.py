"""Tests for report-note collection."""

from autoclean.step_functions.reports import _collect_report_notes


def test_collect_report_notes_from_source_localization_metadata():
    metadata = {
        "step_apply_source_localization": {
            "report_notes": [
                "source note",
                "source note",
                "second note",
            ]
        }
    }

    assert _collect_report_notes(metadata) == ["source note", "second note"]


def test_collect_report_notes_ignores_missing_source_metadata():
    assert _collect_report_notes({}) == []
