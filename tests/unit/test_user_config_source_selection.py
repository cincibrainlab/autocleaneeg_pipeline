import builtins

from autoclean.utils.user_config import user_config


def test_select_source_basic_retries_on_invalid_path(monkeypatch, tmp_path):
    missing = tmp_path / "missing.eeg"
    inputs = iter(["1", str(missing), "3"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    result = user_config._select_source_basic()
    assert result == "NONE"
