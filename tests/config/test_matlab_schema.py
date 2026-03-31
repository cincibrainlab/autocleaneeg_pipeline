"""MATLAB task config schema tests."""

from pathlib import Path

import pytest

from autoclean.configkit.schema import SCHEMA_VERSION, validate_task_module_config
from autoclean.functions import execute_matlab_config


def _minimal_config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "montage": {"enabled": True, "value": None},
        "resample_step": {"enabled": False, "value": None},
        "filtering": {
            "enabled": False,
            "value": {
                "l_freq": None,
                "h_freq": None,
                "notch_freqs": None,
                "notch_widths": None,
            },
        },
        "drop_outerlayer": {"enabled": False, "value": None},
        "eog_step": {"enabled": False, "value": None},
        "trim_step": {"enabled": False, "value": 0},
        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
        "reference_step": {"enabled": False, "value": None},
        "ICA": {"enabled": False, "value": {"method": "fastica"}},
        "component_rejection": {
            "enabled": False,
            "method": "iclabel",
            "value": {
                "ic_flags_to_reject": [],
                "ic_rejection_threshold": 0,
            },
        },
        "epoch_settings": {
            "enabled": False,
            "value": {"tmin": None, "tmax": None},
            "event_id": None,
            "remove_baseline": {"enabled": False, "window": None},
            "threshold_rejection": {"enabled": False, "volt_threshold": {}},
        },
    }


def test_schema_accepts_apply_matlab_function_config() -> None:
    config = _minimal_config()
    config["apply_matlab"] = {
        "enabled": True,
        "value": {
            "kind": "function",
            "entrypoint": "fooof_wrapper",
            "args": ["input.set", 250.0],
            "paths": ["temp"],
            "startup_timeout_seconds": 30.0,
            "nargout": 1,
            "toolbox_requirements": ["Signal Processing Toolbox"],
            "outputs": {"artifacts_subdir": "matlab/fooof", "capture_stdout": True},
        },
    }

    validated = validate_task_module_config(config)

    assert validated["apply_matlab"]["value"]["kind"] == "function"


def test_schema_rejects_negative_matlab_timeout() -> None:
    config = _minimal_config()
    config["run_matlab"] = {
        "enabled": True,
        "value": {
            "kind": "script",
            "entrypoint": "temp/run_fooof_batch.m",
            "startup_timeout_seconds": -1,
        },
    }

    with pytest.raises(Exception):
        validate_task_module_config(config)


def test_schema_accepts_matlab_fooof_block_config() -> None:
    config = _minimal_config()
    config["apply_matlab_fooof"] = {
        "enabled": True,
        "value": {
            "vhtp_path": "/opt/vhtp",
            "eeglab_path": "/opt/eeglab",
            "spect_freqs": [1, 55],
            "save_fooof_img": False,
            "parallel": False,
            "startup_timeout_seconds": 45.0,
            "artifacts_subdir": "matlab/fooof",
        },
    }

    validated = validate_task_module_config(config)

    assert validated["apply_matlab_fooof"]["value"]["vhtp_path"] == "/opt/vhtp"


def test_execute_matlab_config_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = tmp_path / "scripts" / "demo.m"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("% demo\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_matlab_file(
        script_path: str,
        *,
        startup_options: str,
        license_file: str | None,
        startup_timeout_seconds: float,
        path_entries,
    ) -> None:
        captured["script_path"] = script_path
        captured["startup_options"] = startup_options
        captured["license_file"] = license_file
        captured["startup_timeout_seconds"] = startup_timeout_seconds
        captured["path_entries"] = list(path_entries or [])

    monkeypatch.setattr("autoclean.functions.matlab.run_matlab_file", _fake_run_matlab_file)

    execute_matlab_config(
        {
            "enabled": True,
            "value": {
                "kind": "script",
                "entrypoint": "scripts/demo.m",
                "paths": ["scripts"],
                "license_file": "licenses/network.lic",
                "startup_timeout_seconds": 15,
            },
        },
        base_path=tmp_path,
    )

    assert captured["script_path"] == str(script_path.resolve())
    assert captured["license_file"] == str((tmp_path / "licenses" / "network.lic").resolve())
    assert captured["path_entries"] == [str(script_path.parent.resolve())]
