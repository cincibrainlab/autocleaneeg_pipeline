"""Regression tests for importing the standalone Exclude GUI module."""

from __future__ import annotations

import os
import subprocess
import sys
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
