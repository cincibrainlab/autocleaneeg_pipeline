from types import SimpleNamespace

from autoclean.cli import (
    _matlab_install_mode,
    _matlab_remediation_guidance,
    _matlab_route_support_label,
    create_parser,
)


def test_matlab_doctor_parser_defaults() -> None:
    parser = create_parser()
    args = parser.parse_args(["matlab", "doctor"])

    assert args.command == "matlab"
    assert args.matlab_action == "doctor"
    assert args.startup_options == "-nodesktop"
    assert args.startup_timeout == 60.0
    assert args.skip_start is False


def test_matlab_test_engine_parser_defaults() -> None:
    parser = create_parser()
    args = parser.parse_args(["matlab", "test-engine"])

    assert args.command == "matlab"
    assert args.matlab_action == "test-engine"
    assert args.startup_options == "-nodesktop"
    assert args.startup_timeout == 60.0


def test_matlab_install_mode_reports_base_venv_without_engine() -> None:
    mode = _matlab_install_mode(
        "/tmp/workspace/.venv/bin/python",
        engine_installed=False,
    )

    assert mode == "base .venv (MATLAB not enabled)"


def test_matlab_route_support_label_handles_skip_start() -> None:
    label = _matlab_route_support_label(
        route_environment_supported=False,
        skip_start=True,
        engine_installed=True,
    )

    assert label == "not verified"


def test_matlab_remediation_guidance_surfaces_arch_and_runtime_advice() -> None:
    report = SimpleNamespace(
        is_64_bit=True,
        python_executable="/tmp/workspace/.venv/bin/python",
        engine_package_installed=True,
        errors=[
            "MATLAB Engine API unavailable: Could not find directory: /Applications/MATLAB_R2025b.app/extern/engines/python/dist/matlab/engine/maca64",
            "Unable to start MATLAB engine: Remote MVMs are disabled for this session",
        ],
    )

    guidance = _matlab_remediation_guidance(report, skip_start=False)

    assert any("matches the installed MATLAB build" in item for item in guidance)
    assert any("not via mwpython" in item for item in guidance)
