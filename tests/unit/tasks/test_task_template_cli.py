import argparse
import importlib.util
from pathlib import Path

from autoclean.cli import cmd_task_template
from autoclean.task_config_schema import CONFIG_VERSION, validate_task_config
from autoclean.utils.user_config import user_config

def import_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_cmd_task_template_creates_valid_file(tmp_path, monkeypatch):
    monkeypatch.setattr(user_config, "tasks_dir", tmp_path)
    args = argparse.Namespace()
    assert cmd_task_template(args) == 0
    created = list(tmp_path.glob("*.py"))
    assert len(created) == 1
    mod = import_module_from_file(created[0])
    assert mod.config["version"] == CONFIG_VERSION
    validate_task_config(mod.config)


def test_cmd_task_template_auto_naming(tmp_path, monkeypatch):
    monkeypatch.setattr(user_config, "tasks_dir", tmp_path)
    args = argparse.Namespace()
    cmd_task_template(args)
    cmd_task_template(args)
    files = sorted(p.name for p in tmp_path.glob("*.py"))
    assert files == ["custom_task.py", "custom_task_1.py"]
