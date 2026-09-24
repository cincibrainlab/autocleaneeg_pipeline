"""Regression tests for importing the standalone Exclude GUI module."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_autoclean_exclude_import_does_not_abort_python():
    """Importing Exclude should force qtpy onto PyQt6 before Qt imports."""

    env = os.environ.copy()
    env["MNE_DONTWRITE_HOME"] = "true"
    env.pop("QT_API", None)
    src_dir = Path(__file__).resolve().parents[3] / "src"
    env["PYTHONPATH"] = str(src_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; "
            "import autoclean.tools.autoclean_exclude as exclude; "
            "print(os.environ.get('QT_API')); "
            "print(hasattr(exclude, '_unique_channels')); "
            "print('import ok')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "pyqt6" in output
    assert "True" in output
    assert "import ok" in output


def test_pymupdf_import_is_available_for_pdf_previews():
    """PDF previews use PyMuPDF instead of QtPdf."""

    result = subprocess.run(
        [sys.executable, "-c", "import fitz"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_reprocess_selected_ica_remove_does_not_abort_python(tmp_path):
    """Removing a selected component must survive Qt signal callbacks."""

    env = os.environ.copy()
    env["MNE_DONTWRITE_HOME"] = "true"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["QT_QPA_PLATFORM"] = "offscreen"
    src_dir = Path(__file__).resolve().parents[3] / "src"
    env["PYTHONPATH"] = str(src_dir)
    script = textwrap.dedent(
        """
        from qtpy.QtWidgets import QApplication
        from autoclean.tools.autoclean_exclude import ReprocessWidget

        app = QApplication([])
        widget = ReprocessWidget()
        widget.load_from_metadata(
            {
                "bad_channels": [],
                "rejected_ica": [],
                "valid_channels": ["Cz"],
                "max_components": 2,
            }
        )
        changes = []
        widget.values_changed.connect(lambda: changes.append(widget.get_current_values()))

        widget.remove_ica_btn.click()
        assert changes == []

        widget.add_ica_btn.click()
        widget.ica_list.setCurrentRow(0)
        widget.remove_ica_btn.click()
        app.processEvents()

        assert widget.get_current_values()["rejected_ica"] == []
        assert changes == [
            {"bad_channels": [], "rejected_ica": [0]},
            {"bad_channels": [], "rejected_ica": []},
        ]

        original_widget = ReprocessWidget()
        original_widget.load_from_metadata(
            {
                "bad_channels": [],
                "rejected_ica": [0],
                "valid_channels": ["Cz"],
                "max_components": 2,
            }
        )
        original_widget.ica_list.setCurrentRow(0)
        original_widget.remove_ica_btn.click()
        original_widget.add_ica_btn.click()
        app.processEvents()

        assert original_widget.has_changes() is False
        assert original_widget._modification_mode is None
        assert original_widget.add_channel_btn.isEnabled()
        assert original_widget.add_ica_btn.isEnabled()

        no_ica_widget = ReprocessWidget()
        no_ica_widget.load_from_metadata(
            {
                "bad_channels": ["Cz"],
                "rejected_ica": [],
                "valid_channels": ["Cz"],
                "max_components": 0,
            }
        )
        no_ica_widget.channels_list.setCurrentRow(0)
        no_ica_widget.remove_channel_btn.click()
        app.processEvents()

        assert no_ica_widget.add_ica_btn.isEnabled() is False
        print("remove ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "remove ok" in output


def test_epoch_save_preserves_reprocess_edits_and_pre_ica_positions(tmp_path):
    env = os.environ.copy()
    env["MNE_DONTWRITE_HOME"] = "true"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["QT_QPA_PLATFORM"] = "offscreen"
    src_dir = Path(__file__).resolve().parents[3] / "src"
    env["PYTHONPATH"] = str(src_dir)
    script = textwrap.dedent(
        f"""
        from pathlib import Path

        from autoclean.tools.autoclean_exclude import ExclusionFileSelector

        class FakeEpochs:
            ch_names = ["Fp1", "Fp2"]
            info = {{"sfreq": 250.0}}
            selection = [10, 20, 30, 40]
            events = [
                [0, 0, 101],
                [250, 0, 102],
                [500, 0, 103],
                [750, 0, 104],
            ]

            def __len__(self):
                return 4

        class FakeReprocessWidget:
            def __init__(self):
                self.manual_bad_epoch_indices = []
                self.manual_bad_epoch_pre_ica_indices = []
                self.manual_bad_epoch_times = []
                self.manual_bad_epoch_events = []
                self.summary_updates = 0

            def _update_changes_summary(self):
                self.summary_updates += 1

            def load_from_metadata(self, _metadata):
                raise AssertionError("epoch save must not reload manual edits")

        selector = ExclusionFileSelector.__new__(ExclusionFileSelector)
        selector.current_key = "subject01"
        selector.selected_file_path = Path({str(tmp_path / "subject01.set")!r})
        selector.current_epochs = FakeEpochs()
        selector.decisions = {{}}
        selector.row_lookup = {{}}
        selector.plot_widget = None
        selector.reprocess_widget = FakeReprocessWidget()
        selector._relative_path = lambda path: path.name
        selector._schedule_save = lambda: None

        selector._save_epoch_changes_immediately({{1, 3}})

        record = selector.decisions["subject01"]
        assert record["bad_epoch_indices"] == "1,3"
        assert record["bad_epoch_pre_ica_indices"] == "20,40"
        assert record["bad_epoch_times"] == "1.000,3.000"
        assert record["bad_epoch_events"] == "102,104"
        assert selector.reprocess_widget.manual_bad_epoch_indices == [1, 3]
        assert selector.reprocess_widget.manual_bad_epoch_pre_ica_indices == [20, 40]
        assert selector.reprocess_widget.summary_updates == 1
        print("epoch save ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )

    output = result.stdout + result.stderr
    if "Incompatible processor" in output and "neon" in output:
        pytest.skip("Local Qt runtime aborts before GUI-state regression can run")
    assert result.returncode == 0, output
    assert "epoch save ok" in output
