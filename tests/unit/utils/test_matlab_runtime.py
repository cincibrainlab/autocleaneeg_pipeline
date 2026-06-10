import types
from pathlib import Path

import pytest

from autoclean.utils import matlab_runtime


def test_detect_matlab_engine_reports_missing_package(monkeypatch):
    def _boom():
        raise matlab_runtime.MatlabEngineUnavailableError("missing engine")

    monkeypatch.setattr(matlab_runtime, "_import_matlab_modules", _boom)

    report = matlab_runtime.detect_matlab_engine()

    assert report.engine_package_installed is False
    assert report.errors
    assert "missing engine" in report.errors[0]


def test_detect_matlab_engine_reads_arch_file(tmp_path, monkeypatch):
    matlab_dir = tmp_path / "matlab"
    engine_dir = matlab_dir / "engine"
    engine_dir.mkdir(parents=True)
    (matlab_dir / "__init__.py").write_text("# test\n", encoding="utf-8")
    (engine_dir / "_arch.txt").write_text(
        "maca64\n/Applications/MATLAB_R2025b.app/bin/maca64\n"
        "/Applications/MATLAB_R2025b.app/extern/engines/python/dist/matlab/engine/maca64\n"
        "/Applications/MATLAB_R2025b.app/extern/bin/maca64\n",
        encoding="utf-8",
    )
    fake_matlab = types.SimpleNamespace(__file__=str(matlab_dir / "__init__.py"))
    fake_engine = types.SimpleNamespace()

    monkeypatch.setattr(
        matlab_runtime,
        "_import_matlab_modules",
        lambda: (fake_matlab, fake_engine),
    )
    monkeypatch.setattr(
        matlab_runtime,
        "_engine_metadata_version",
        lambda: "25.2.2",
    )

    report = matlab_runtime.detect_matlab_engine()

    assert report.engine_package_installed is True
    assert report.engine_package_version == "25.2.2"
    assert report.matlab_root == "/Applications/MATLAB_R2025b.app"
    assert report.matlab_binary == "/Applications/MATLAB_R2025b.app/bin/maca64"


def test_run_matlab_script_rejects_missing_file():
    with pytest.raises(matlab_runtime.MatlabExecutionError):
        matlab_runtime.run_matlab_script(Path("/tmp/does-not-exist.m"))


def test_start_matlab_engine_timeout(monkeypatch):
    class FakeFuture:
        def __init__(self):
            self.cancel_called = False

        def result(self, timeout=None):
            raise TimeoutError("timed out")

        def cancel(self):
            self.cancel_called = True
            return True

    fake_future = FakeFuture()
    fake_engine = types.SimpleNamespace(
        start_matlab=lambda option, background=True: fake_future
    )

    monkeypatch.setattr(
        matlab_runtime,
        "_import_matlab_modules",
        lambda: (types.SimpleNamespace(), fake_engine),
    )

    with pytest.raises(matlab_runtime.MatlabTimeoutError):
        matlab_runtime.start_matlab_engine(startup_timeout_seconds=0.01)

    assert fake_future.cancel_called is True


def test_run_matlab_function_rejects_keep_engine_without_explicit_engine():
    with pytest.raises(matlab_runtime.MatlabRuntimeError):
        matlab_runtime.run_matlab_function("sqrt", 4.0, keep_engine=True)


def test_inspect_taskfile_for_matlab_detects_wrapper_usage(tmp_path: Path) -> None:
    taskfile = tmp_path / "matlab_task.py"
    taskfile.write_text(
        """
from autoclean.functions.matlab import call_matlab


def run():
    return call_matlab("sqrt", 4.0)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    inspection = matlab_runtime.inspect_taskfile_for_matlab(taskfile)

    assert inspection.requires_matlab is True
    assert any("calls call_matlab" == reason for reason in inspection.reasons)


def test_inspect_taskfile_for_matlab_detects_mixin_usage(tmp_path: Path) -> None:
    taskfile = tmp_path / "matlab_task.py"
    taskfile.write_text(
        """
from autoclean.mixins.utils.matlab import MatlabExecutionMixin


class ExampleTask(MatlabExecutionMixin):
    pass
""".strip()
        + "\n",
        encoding="utf-8",
    )

    inspection = matlab_runtime.inspect_taskfile_for_matlab(taskfile)

    assert inspection.requires_matlab is True
    assert any("inherits MatlabExecutionMixin" in reason for reason in inspection.reasons)


def test_inspect_taskfile_for_matlab_detects_config_key(tmp_path: Path) -> None:
    taskfile = tmp_path / "matlab_task.py"
    taskfile.write_text(
        """
config = {
    "schema_version": "2025.09",
    "apply_matlab_fooof": {
        "enabled": True,
        "value": {
            "vhtp_path": "/opt/vhtp",
            "eeglab_path": "/opt/eeglab"
        }
    }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    inspection = matlab_runtime.inspect_taskfile_for_matlab(taskfile)

    assert inspection.requires_matlab is True
    assert "apply_matlab_fooof" in inspection.matlab_config_keys
