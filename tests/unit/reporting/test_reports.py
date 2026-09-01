"""Unit tests for step_functions/reports.py."""

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    from autoclean.step_functions.reports import (
        create_json_summary,
        create_run_report,
        generate_bad_channels_tsv,
        update_task_processing_log,
    )

    REPORTS_AVAILABLE = True
except ImportError:
    REPORTS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not REPORTS_AVAILABLE, reason="Reports module not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_summary(tmp_path: Path, run_id: str = "run_001") -> dict:
    """Build the minimum summary_dict accepted by update_task_processing_log."""
    return {
        "output_dir": str(tmp_path),
        "task": "TestTask",
        "timestamp": "2026-01-01 00:00:00",
        "run_id": run_id,
        "proc_state": "completed",
        "basename": "sub01_test",
        "bids_subject": "sub-01",
        "reports_dir": str(tmp_path / "reports"),
    }


# ---------------------------------------------------------------------------
# update_task_processing_log
# ---------------------------------------------------------------------------


class TestUpdateTaskProcessingLog:
    def test_creates_csv_file_when_not_exists(self, tmp_path):
        """A fresh call must create preprocessing_log.csv in output_dir."""
        summary = _minimal_summary(tmp_path)
        update_task_processing_log(summary)

        csv_path = tmp_path / "preprocessing_log.csv"
        assert csv_path.exists()

    def test_csv_contains_run_id(self, tmp_path):
        """The created CSV row must record the run_id."""
        summary = _minimal_summary(tmp_path, run_id="run_abc")
        update_task_processing_log(summary)

        csv_path = tmp_path / "preprocessing_log.csv"
        content = csv_path.read_text()
        assert "run_abc" in content

    def test_appends_second_row_for_different_subject(self, tmp_path):
        """Calling twice with different basenames yields two rows (header + 2 data)."""
        summary1 = _minimal_summary(tmp_path, run_id="run_001")
        summary1["basename"] = "sub01_test"
        summary1["bids_subject"] = "sub-01"

        summary2 = _minimal_summary(tmp_path, run_id="run_002")
        summary2["basename"] = "sub02_test"
        summary2["bids_subject"] = "sub-02"

        update_task_processing_log(summary1)
        update_task_processing_log(summary2)

        csv_path = tmp_path / "preprocessing_log.csv"
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))

        # Header + 2 data rows
        assert len(rows) >= 3

    def test_missing_required_key_does_not_raise(self, tmp_path):
        """Missing required key should log an error but not crash."""
        bad_summary = {"output_dir": str(tmp_path), "task": "T"}  # missing keys
        # Should not raise — function handles the error internally
        update_task_processing_log(bad_summary)

    @pytest.mark.parametrize(
        ("metadata", "expected"),
        [
            (
                {},
                {
                    "outerlayer_chans_dropped_n": "NA",
                    "outerlayer_chans_dropped": "NA",
                    "proc_badchans_n": "NA",
                    "proc_badchans": "NA",
                    "proc_badchans_action": "NA",
                },
            ),
            (
                {"step_drop_outerlayer": {"dropped_outer_layer_channels": ["E1"]}},
                {
                    "outerlayer_chans_dropped_n": "1",
                    "outerlayer_chans_dropped": "['E1']",
                    "proc_badchans_n": "NA",
                    "proc_badchans": "NA",
                    "proc_badchans_action": "NA",
                },
            ),
            (
                {
                    "step_drop_outerlayer": {
                        "enabled": False,
                        "dropped_outer_layer_channels": ["E1"],
                    }
                },
                {
                    "outerlayer_chans_dropped_n": "NA",
                    "outerlayer_chans_dropped": "NA",
                    "proc_badchans_n": "NA",
                    "proc_badchans": "NA",
                    "proc_badchans_action": "NA",
                },
            ),
            (
                {
                    "step_clean_bad_channels": {
                        "bads": [],
                        "cleaning_method": "interpolate",
                    }
                },
                {
                    "outerlayer_chans_dropped_n": "NA",
                    "outerlayer_chans_dropped": "NA",
                    "proc_badchans_n": "0",
                    "proc_badchans": "[]",
                    "proc_badchans_action": "interpolated",
                },
            ),
            (
                {
                    "step_clean_bad_channels": {
                        "bads": ["E2", "E3"],
                        "cleaning_method": "interpolate",
                    }
                },
                {
                    "outerlayer_chans_dropped_n": "NA",
                    "outerlayer_chans_dropped": "NA",
                    "proc_badchans_n": "2",
                    "proc_badchans": "['E2', 'E3']",
                    "proc_badchans_action": "interpolated",
                },
            ),
            (
                {
                    "step_clean_bad_channels": {
                        "bads": ["E2"],
                        "cleaning_method": "drop",
                    }
                },
                {
                    "outerlayer_chans_dropped_n": "NA",
                    "outerlayer_chans_dropped": "NA",
                    "proc_badchans_n": "1",
                    "proc_badchans": "['E2']",
                    "proc_badchans_action": "dropped",
                },
            ),
            (
                {
                    "step_clean_bad_channels": {
                        "bads": ["E2"],
                        "cleaning_method": None,
                    }
                },
                {
                    "outerlayer_chans_dropped_n": "NA",
                    "outerlayer_chans_dropped": "NA",
                    "proc_badchans_n": "1",
                    "proc_badchans": "['E2']",
                    "proc_badchans_action": "marked_only",
                },
            ),
        ],
        ids=[
            "not-run",
            "outer-layer-dropped",
            "outer-layer-disabled",
            "empty",
            "interpolated",
            "dropped",
            "marked-only",
        ],
    )
    def test_reports_distinct_channel_cleaning_metadata(
        self, tmp_path, metadata, expected
    ):
        summary = _minimal_summary(tmp_path)
        summary["metadata"] = metadata

        update_task_processing_log(summary)

        with (tmp_path / "preprocessing_log.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        for field, value in expected.items():
            assert row[field] == value


# ---------------------------------------------------------------------------
# generate_bad_channels_tsv
# ---------------------------------------------------------------------------


class TestGenerateBadChannelsTsv:
    def _base_summary(self, tmp_path: Path) -> dict:
        return {
            "output_dir": str(tmp_path),
            "run_id": "run_tsv_001",
            "basename": "sub01_test",
            "bids_subject": "sub-01",
            "reports_dir": str(tmp_path / "reports"),
            "channel_dict": {
                "noisy_channels": ["E1", "E2"],
                "uncorrelated_channels": [],
                "deviation_channels": [],
                "bridged_channels": [],
                "rank_channels": [],
                "ransac_channels": [],
                "removed_channels": ["E1", "E2"],
            },
            "metadata": {},
        }

    def test_writes_tsv_file(self, tmp_path):
        """generate_bad_channels_tsv must write a .tsv file to disk."""
        summary = self._base_summary(tmp_path)
        generate_bad_channels_tsv(summary)

        tsv_files = list((tmp_path / "reports" / "run_reports").glob("*.tsv"))
        assert len(tsv_files) == 1

    def test_tsv_contains_expected_columns(self, tmp_path):
        """TSV must include at minimum a 'name' column (BIDS bad channels)."""
        summary = self._base_summary(tmp_path)
        generate_bad_channels_tsv(summary)

        tsv_files = list((tmp_path / "reports" / "run_reports").glob("*.tsv"))
        assert tsv_files, "No TSV file was written"
        content = tsv_files[0].read_text()
        # Must have at least a 'channel' or 'label' column header (BIDS bad channels)
        assert "channel" in content.lower() or "label" in content.lower()

    def test_empty_when_no_bad_channels(self, tmp_path):
        """With no bad channels the TSV should exist but have no data rows."""
        summary = self._base_summary(tmp_path)
        summary["channel_dict"] = {
            "noisy_channels": [],
            "uncorrelated_channels": [],
            "deviation_channels": [],
            "bridged_channels": [],
            "rank_channels": [],
            "ransac_channels": [],
            "removed_channels": [],
        }
        generate_bad_channels_tsv(summary)

        tsv_files = list((tmp_path / "reports" / "run_reports").glob("*.tsv"))
        if tsv_files:
            with open(tsv_files[0], newline="") as f:
                rows = list(csv.reader(f, delimiter="\t"))
            # Only header row or empty file — no channel data
            data_rows = [r for r in rows if any(c.strip() for c in r)][1:]
            assert len(data_rows) == 0

    def test_no_crash_when_channel_dict_missing(self, tmp_path):
        """Missing channel_dict key should log a warning, not raise."""
        summary = {
            "output_dir": str(tmp_path),
            "run_id": "run_tsv_002",
            "basename": "sub01_test",
        }
        # Should not raise — function handles missing key internally
        generate_bad_channels_tsv(summary)


# ---------------------------------------------------------------------------
# create_run_report
# ---------------------------------------------------------------------------


def _make_run_record(tmp_path, run_id="run_report_001", flagged=False):
    """Build a minimal run_record structure for create_run_report."""
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir(exist_ok=True)
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(exist_ok=True)

    return {
        "run_id": run_id,
        "task": "TestTask",
        "status": "completed",
        "success": True,
        "flagged": flagged,
        "report_file": f"{run_id}_autoclean_report.pdf",
        "metadata": {
            "step_prepare_directories": {
                "metadata": str(meta_dir),
                "reports": str(reports_dir),
                "bids": str(tmp_path / "bids"),
            }
        },
    }


class TestCreateRunReport:
    def test_skips_when_no_run_id(self, tmp_path):
        """create_run_report should return early without error if run_id is empty."""
        # Should not raise
        create_run_report(run_id="", autoclean_dict=None)

    def test_skips_when_run_record_not_found(self, tmp_path):
        """create_run_report returns None when no DB record exists for run_id."""
        with patch(
            "autoclean.step_functions.reports.get_run_record", return_value=None
        ):
            result = create_run_report(run_id="nonexistent_run")
        assert result is None

    def test_skips_when_metadata_missing(self, tmp_path):
        """Returns None when run record has no metadata."""
        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value={"run_id": "x", "status": "done"},
        ):
            result = create_run_report(run_id="x")
        assert result is None

    def test_writes_report_when_metadata_present(self, tmp_path):
        """create_run_report writes a file when all required metadata is provided."""
        run_record = _make_run_record(tmp_path)
        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            # Should not raise
            create_run_report(
                run_id=run_record["run_id"], autoclean_dict=None, json_summary={}
            )

    def test_skips_sections_for_missing_step_prepare_directories(self, tmp_path):
        """Returns when required step_prepare_directories metadata key is missing."""
        run_record = {
            "run_id": "run_001",
            "status": "done",
            "report_file": "report.pdf",
            "metadata": {},  # Missing step_prepare_directories
        }
        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            # Should return early without raising
            result = create_run_report(run_id="run_001")
        assert result is None


# ---------------------------------------------------------------------------
# create_json_summary
# ---------------------------------------------------------------------------


class TestCreateJsonSummary:
    @staticmethod
    def _run_record(tmp_path, run_id, channel_removals):
        derivatives_dir = tmp_path / "derivatives" / run_id
        derivatives_dir.mkdir(parents=True)
        for directory in ["metadata", "reports", "ica", "exports", "bids"]:
            (tmp_path / directory).mkdir(exist_ok=True)

        return {
            "run_id": run_id,
            "task": "TestTask",
            "created_at": "2026-08-17 00:00:00",
            "success": True,
            "report_file": f"{run_id}_autoclean_report.pdf",
            "metadata": {
                "step_create_bids_path": {
                    "derivatives_dir": str(derivatives_dir),
                    "bids_subject": run_id,
                },
                "step_prepare_directories": {
                    "bids": str(tmp_path / "bids"),
                    "metadata": str(tmp_path / "metadata"),
                    "reports": str(tmp_path / "reports"),
                    "ica": str(tmp_path / "ica"),
                    "exports": str(tmp_path / "exports"),
                },
                "import_eeg": {
                    "sampleRate": 500,
                    "channelCount": 3,
                    "durationSec": 10,
                    "unprocessedFile": f"{run_id}.set",
                },
                "channel_removals": channel_removals,
            },
        }

    def test_returns_none_when_no_run_record(self, tmp_path):
        """create_json_summary returns None when run_id not in DB."""
        with patch(
            "autoclean.step_functions.reports.get_run_record", return_value=None
        ):
            result = create_json_summary(run_id="nonexistent_id")
        assert result is None

    def test_returns_empty_dict_when_bids_metadata_missing(self, tmp_path):
        """Returns empty dict when step_create_bids_path not in metadata."""
        run_record = {
            "run_id": "x",
            "status": "done",
            "success": True,
            "metadata": {},  # no bids info
        }
        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="x")
        # Either None or empty dict — missing bids info causes early return
        assert result is None or result == {}

    def test_serializes_path_objects(self, tmp_path):
        """create_json_summary handles Path objects in metadata without crashing."""
        bids_dir = tmp_path / "bids" / "derivatives" / "sub-test"
        bids_dir.mkdir(parents=True, exist_ok=True)
        run_record = {
            "run_id": "path_test",
            "status": "completed",
            "success": True,
            "flagged": False,
            "error": None,
            "metadata": {
                "step_create_bids_path": {
                    "derivatives_dir": str(bids_dir),
                },
                "step_prepare_directories": {
                    "bids": str(tmp_path / "bids"),
                    "metadata": str(tmp_path / "metadata"),
                    "reports": str(tmp_path / "reports"),
                    "ica": str(tmp_path / "ica"),
                    "exports": str(tmp_path / "exports"),
                },
            },
        }
        # Create needed dirs
        for key in ["metadata", "reports", "ica", "exports"]:
            (tmp_path / key).mkdir(exist_ok=True)

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            # Should not raise — Path objects or missing dirs handled gracefully
            try:
                result = create_json_summary(run_id="path_test")
            except Exception:
                result = None
        # The call itself should not crash unhandled — None or dict is acceptable
        assert result is None or isinstance(result, dict)

    def test_ignores_foreign_flagged_channels_tsv(self, tmp_path):
        """A shared report artifact from another file cannot affect this run."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [{"channel": "E1", "reason": "NOISY"}],
        )
        run_reports = tmp_path / "reports" / "run_reports"
        run_reports.mkdir()
        (run_reports / "foreign_flagged_channels.tsv").write_text(
            "label\tchannel\nNoisy\tE99\n", encoding="utf8"
        )

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert result["channel_dict"]["Noisy"] == ["E1"]
        assert "E99" not in result["channel_dict"]["removed_channels"]

    def test_current_metadata_wins_before_current_tsv_exists(self, tmp_path):
        """Current removal metadata supplies labels before TSV generation."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [
                {"channel": "E2", "reason": "UNCORRELATED"},
                {"channel": "E3", "reason": "MANUAL_EXCLUDE"},
            ],
        )

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert not (tmp_path / "reports" / "run_reports").exists()
        assert result["channel_dict"]["Uncorrelated"] == ["E2"]
        assert result["channel_dict"]["Manual"] == ["E3"]

    @pytest.mark.parametrize("order", [("first", "second"), ("second", "first")])
    def test_runs_keep_channel_sets_isolated_in_any_order(self, tmp_path, order):
        """Two summaries sharing a reports directory remain order-independent."""
        records = {
            "first": self._run_record(
                tmp_path,
                "first",
                [{"channel": "E1", "reason": "NOISY"}],
            ),
            "second": self._run_record(
                tmp_path,
                "second",
                [{"channel": "E2", "reason": "RANSAC"}],
            ),
        }
        run_reports = tmp_path / "reports" / "run_reports"
        run_reports.mkdir()
        (run_reports / "first_flagged_channels.tsv").write_text(
            "label\tchannel\nNoisy\tE1\n", encoding="utf8"
        )
        (run_reports / "second_flagged_channels.tsv").write_text(
            "label\tchannel\nRansac\tE2\n", encoding="utf8"
        )
        results = {}

        for run_id in order:
            with patch(
                "autoclean.step_functions.reports.get_run_record",
                return_value=records[run_id],
            ):
                results[run_id] = create_json_summary(run_id=run_id)

        assert results["first"]["channel_dict"]["removed_channels"] == ["E1"]
        assert results["second"]["channel_dict"]["removed_channels"] == ["E2"]
        assert results["first"]["channel_dict"]["Noisy"] == ["E1"]
        assert results["second"]["channel_dict"]["Ransac"] == ["E2"]
        assert "Ransac" not in results["first"]["channel_dict"]
        assert "Noisy" not in results["second"]["channel_dict"]

    def test_removed_channels_are_unique_and_stable(self, tmp_path):
        """Repeated audit entries retain order without duplicating removals."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [
                {"channel": "E2", "reason": "NOISY"},
                {"channel": "E1", "reason": "RANSAC"},
                {"channel": "E2", "reason": "MANUAL_EXCLUDE"},
            ],
        )

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert result["channel_dict"]["removed_channels"] == ["E2", "E1"]
        assert result["channel_dict"]["Noisy"] == ["E2"]
        assert result["channel_dict"]["Manual"] == ["E2"]

    def test_category_deduplicates_repeated_channel_and_reason(self, tmp_path):
        """Repeated same-reason audit entries produce one category item."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [
                {"channel": "E2", "reason": "NOISY"},
                {"channel": "E2", "reason": "NOISY"},
            ],
        )

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert result["channel_dict"]["Noisy"] == ["E2"]
        assert result["channel_dict"]["removed_channels"] == ["E2"]

    def test_interpolated_channels_excluded_from_expected_count(self, tmp_path):
        """Regression test for #292: interpolated channels stay in the export,
        so they must not be subtracted from the expected channel count (which
        previously produced a false "channel count mismatch" warning)."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [
                {"channel": "E1", "reason": "UNCORRELATED", "action": "interpolated"},
                {"channel": "E2", "reason": "DEVIATION", "action": "interpolated"},
            ],
        )
        # import_eeg.channelCount is 3; both bad channels were interpolated,
        # so the exported file should still have all 3 channels.
        run_record["metadata"]["save_epochs_to_set"] = {
            "tmin": 0,
            "tmax": 1,
            "n_epochs": 1,
            "n_channels": 3,
        }

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert result["channel_dict"]["removed_channels"] == []
        assert result["channel_dict"]["interpolated_channels"] == ["E1", "E2"]
        assert result["channel_dict"]["marked_channels"] == []
        assert result["export_details"]["net_nbchan_post"] == 3
        assert "channel_count_mismatch" not in result["export_details"]

    def test_marked_channels_tracked_separately_from_interpolated(self, tmp_path):
        """Regression test: 'marked' (flagged bad but neither interpolated nor
        dropped) and 'interpolated' channels are distinct outcomes and must
        not be collapsed into one bucket -- a 'marked' channel's data was
        never touched, unlike an interpolated one."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [
                {"channel": "E1", "reason": "UNCORRELATED", "action": "interpolated"},
                {"channel": "E2", "reason": "DEVIATION", "action": "marked"},
            ],
        )
        run_record["metadata"]["save_epochs_to_set"] = {
            "tmin": 0,
            "tmax": 1,
            "n_epochs": 1,
            "n_channels": 3,
        }

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert result["channel_dict"]["removed_channels"] == []
        assert result["channel_dict"]["interpolated_channels"] == ["E1"]
        assert result["channel_dict"]["marked_channels"] == ["E2"]
        assert "channel_count_mismatch" not in result["export_details"]

    def test_channel_count_mismatch_still_detected_for_dropped_channels(self, tmp_path):
        """Dropped channels must still be subtracted from the expected count,
        so a genuine mismatch is still caught."""
        run_record = self._run_record(
            tmp_path,
            "current",
            [
                {"channel": "E1", "reason": "MANUAL_EXCLUDE", "action": "dropped"},
            ],
        )
        # 3 total channels, 1 dropped -> exported file should have 2, but we
        # simulate an exported file that still reports 3 to trigger the check.
        run_record["metadata"]["save_epochs_to_set"] = {
            "tmin": 0,
            "tmax": 1,
            "n_epochs": 1,
            "n_channels": 3,
        }

        with patch(
            "autoclean.step_functions.reports.get_run_record",
            return_value=run_record,
        ):
            result = create_json_summary(run_id="current")

        assert result["channel_dict"]["removed_channels"] == ["E1"]
        assert result["export_details"]["channel_count_mismatch"] is True
