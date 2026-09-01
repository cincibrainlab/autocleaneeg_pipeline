"""Regression tests for importing the standalone Exclude GUI module."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


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
