import json
from types import SimpleNamespace

from autoclean.cli import cmd_task_copy
from autoclean.utils.user_config import user_config


def test_task_copy_can_set_active(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda app, appauthor=None: str(tmp_path)
    )
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    user_config.tasks_dir = tasks_dir
    user_config.config_dir = tmp_path

    source = tmp_path / "src_task.py"
    source.write_text(
        """from autoclean.core.task import Task\nclass SrcTask(Task):\n    pass\n"""
    )

    args = SimpleNamespace(source=str(source), name="My Task", force=True)
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **kw: True)

    cmd_task_copy(args)

    with open(tmp_path / "setup.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config.get("active_task") == "MyTask"
