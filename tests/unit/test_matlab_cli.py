from autoclean.cli import create_parser


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
