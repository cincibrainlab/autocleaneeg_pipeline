from pathlib import Path

from autoclean.utils import user_config as user_config_module
from autoclean.utils.user_config import UserConfigManager, safe_path_exists


def test_init_does_not_crash_when_saved_workspace_is_inaccessible(monkeypatch, capsys):
    inaccessible = Path("/Volumes/srv2/autoclean")

    monkeypatch.setattr(
        UserConfigManager,
        "_get_workspace_path",
        lambda self: inaccessible,
    )

    def raise_for_inaccessible(self, *args, **kwargs):
        if self == inaccessible or self == inaccessible / "tasks":
            raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "mkdir", raise_for_inaccessible)

    manager = UserConfigManager()

    assert manager.config_dir == inaccessible
    assert manager.tasks_dir == inaccessible / "tasks"
    assert manager.workspace_accessible is False
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "workspace set <path>" in captured.err


def test_safe_path_exists_returns_false_for_permission_errors(monkeypatch):
    inaccessible = Path("/Volumes/srv2")

    def raise_for_inaccessible(self):
        if self == inaccessible:
            raise PermissionError("permission denied")
        return True

    monkeypatch.setattr(Path, "exists", raise_for_inaccessible)

    assert safe_path_exists(inaccessible) is False
    assert safe_path_exists(Path("/tmp")) is True


def test_is_workspace_valid_returns_false_for_inaccessible_workspace(
    monkeypatch, tmp_path
):
    config_root = tmp_path / "config"
    config_root.mkdir()
    setup_json = config_root / "setup.json"
    setup_json.write_text("{}", encoding="utf-8")
    inaccessible = Path("/Volumes/srv2/autoclean")

    manager = UserConfigManager.__new__(UserConfigManager)
    manager.config_dir = inaccessible
    manager.tasks_dir = inaccessible / "tasks"

    monkeypatch.setattr(
        user_config_module.platformdirs,
        "user_config_dir",
        lambda *args: str(config_root),
    )

    original_exists = Path.exists

    def raise_for_inaccessible(self):
        if self == inaccessible or self == inaccessible / "tasks":
            raise PermissionError("permission denied")
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", raise_for_inaccessible)

    assert manager._is_workspace_valid() is False


def test_list_custom_tasks_returns_empty_for_inaccessible_tasks_dir(monkeypatch):
    inaccessible = Path("/Volumes/srv2/autoclean/tasks")

    manager = UserConfigManager.__new__(UserConfigManager)
    manager.config_dir = inaccessible.parent
    manager.tasks_dir = inaccessible

    def raise_for_inaccessible(self):
        if self == inaccessible:
            raise PermissionError("permission denied")
        return True

    monkeypatch.setattr(Path, "exists", raise_for_inaccessible)

    assert manager.list_custom_tasks() == {}


def test_get_active_task_preserves_task_when_tasks_dir_is_inaccessible(
    monkeypatch, tmp_path
):
    config_root = tmp_path / "config"
    config_root.mkdir()
    setup_json = config_root / "setup.json"
    setup_json.write_text('{"active_task": "MyTask"}', encoding="utf-8")
    inaccessible = Path("/Volumes/srv2/autoclean/tasks")

    manager = UserConfigManager.__new__(UserConfigManager)
    manager.config_dir = inaccessible.parent
    manager.tasks_dir = inaccessible
    manager.workspace_accessible = True

    monkeypatch.setattr(
        user_config_module.platformdirs,
        "user_config_dir",
        lambda *args: str(config_root),
    )

    def raise_for_inaccessible(self):
        if self == inaccessible:
            raise PermissionError("permission denied")
        return True

    monkeypatch.setattr(Path, "exists", raise_for_inaccessible)
    monkeypatch.setattr(
        manager,
        "set_active_task",
        lambda task_name=None: (_ for _ in ()).throw(
            AssertionError("active task should not be cleared")
        ),
    )

    assert manager.get_active_task() == "MyTask"
    assert setup_json.read_text(encoding="utf-8") == '{"active_task": "MyTask"}'
