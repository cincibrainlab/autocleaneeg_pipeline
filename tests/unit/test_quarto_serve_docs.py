from types import SimpleNamespace
from unittest.mock import patch

from autoclean.cli import _quarto_missing_message, cmd_serve_docs


def test_quarto_missing_message_explains_scope_and_macos_install(monkeypatch) -> None:
    monkeypatch.setattr("autoclean.cli.sys.platform", "darwin")

    guidance = _quarto_missing_message()

    assert "`autocleaneeg-pipeline serve docs`" in guidance
    assert "`serve`, `serve up`" in guidance
    assert "uv tool install autocleaneeg-pipeline" in guidance
    assert "brew install --cask quarto" in guidance


def test_serve_docs_missing_quarto_uses_actionable_message(tmp_path, monkeypatch) -> None:
    (tmp_path / "plans").mkdir()
    monkeypatch.chdir(tmp_path)

    with (
        patch("autoclean.cli.subprocess.run", side_effect=FileNotFoundError),
        patch("autoclean.cli.message") as mock_message,
    ):
        result = cmd_serve_docs(SimpleNamespace(port=4321, host="127.0.0.1"))

    assert result == 1
    error_message = mock_message.call_args.args[1]
    assert "Quarto CLI is required" in error_message
    assert "serve docs" in error_message
    assert "serve up" in error_message


def test_serve_docs_nonzero_quarto_version_uses_actionable_message(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "plans").mkdir()
    monkeypatch.chdir(tmp_path)

    with (
        patch(
            "autoclean.cli.subprocess.run",
            return_value=SimpleNamespace(returncode=1),
        ),
        patch("autoclean.cli.message") as mock_message,
    ):
        result = cmd_serve_docs(SimpleNamespace(port=4321, host="127.0.0.1"))

    assert result == 1
    error_message = mock_message.call_args.args[1]
    assert "Quarto CLI is required" in error_message
    assert "serve docs" in error_message
    assert "serve up" in error_message
