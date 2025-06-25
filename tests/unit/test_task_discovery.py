"""Tests for the task discovery utility."""

from pathlib import Path
from unittest.mock import patch

import pytest

from autoclean.utils.task_discovery import (
    DiscoveredTask,
    InvalidTaskFile,
    get_task_by_name,
    safe_discover_tasks,
)


def test_safe_discover_tasks(monkeypatch):
    """Test that safe_discover_tasks correctly identifies good and bad tasks."""
    # Mock the user_config.tasks_dir to point to our test fixtures
    monkeypatch.setattr(
        "autoclean.utils.user_config.user_config.tasks_dir",
        Path("tests/fixtures/tasks"),
    )
    
    valid_tasks, invalid_files = safe_discover_tasks()
    
    # Check that we have discovered tasks (built-in + custom)
    assert len(valid_tasks) > 0
    
    # Check that the good task was discovered
    assert any(task.name == "GoodTask" for task in valid_tasks)
    
    # Check that invalid files were identified
    assert len(invalid_files) >= 2  # At least our two bad test files
    
    # Check that the bad syntax task was identified as invalid
    bad_syntax_files = [f for f in invalid_files if "bad_syntax_task.py" in f.source]
    assert len(bad_syntax_files) > 0
    assert any("Syntax error" in f.error for f in bad_syntax_files)
    
    # Check that the bad import task was identified as invalid
    bad_import_files = [f for f in invalid_files if "bad_import_task.py" in f.source]
    assert len(bad_import_files) > 0
    assert any("Missing dependency" in f.error or "ModuleNotFoundError" in f.error 
               for f in bad_import_files)


def test_discovered_task_structure():
    """Test that DiscoveredTask has the expected structure."""
    task = DiscoveredTask(
        name="TestTask",
        description="Test description",
        source="/path/to/task.py",
        class_obj=None
    )
    
    assert task.name == "TestTask"
    assert task.description == "Test description"
    assert task.source == "/path/to/task.py"
    assert task.class_obj is None


def test_invalid_task_file_structure():
    """Test that InvalidTaskFile has the expected structure."""
    invalid = InvalidTaskFile(
        source="/path/to/bad.py",
        error="SyntaxError: invalid syntax"
    )
    
    assert invalid.source == "/path/to/bad.py"
    assert invalid.error == "SyntaxError: invalid syntax"


def test_get_task_by_name(monkeypatch):
    """Test retrieving a task by name."""
    # Mock the user_config.tasks_dir
    monkeypatch.setattr(
        "autoclean.utils.user_config.user_config.tasks_dir",
        Path("tests/fixtures/tasks"),
    )
    
    # Try to get the GoodTask
    task_class = get_task_by_name("GoodTask")
    assert task_class is not None
    assert task_class.__name__ == "GoodTask"
    
    # Try to get a non-existent task
    non_existent = get_task_by_name("NonExistentTask")
    assert non_existent is None


def test_duplicate_task_handling(monkeypatch, tmp_path):
    """Test that duplicate task names are handled correctly."""
    # Create a temporary tasks directory with duplicate task names
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    
    # Create two files with the same task name
    task1 = tasks_dir / "task1.py"
    task1.write_text("""
from autoclean.core.task import Task

class DuplicateTask(Task):
    '''First duplicate task.'''
    def run(self):
        pass
""")
    
    task2 = tasks_dir / "task2.py"
    task2.write_text("""
from autoclean.core.task import Task

class DuplicateTask(Task):
    '''Second duplicate task.'''
    def run(self):
        pass
""")
    
    monkeypatch.setattr(
        "autoclean.utils.user_config.user_config.tasks_dir",
        tasks_dir,
    )
    
    valid_tasks, _ = safe_discover_tasks()
    
    # Should only have one DuplicateTask (the first one found)
    duplicate_tasks = [t for t in valid_tasks if t.name == "DuplicateTask"]
    assert len(duplicate_tasks) == 1
    assert duplicate_tasks[0].description == "First duplicate task."


def test_template_and_private_files_skipped(monkeypatch, tmp_path):
    """Test that template and private files are skipped."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    
    # Create files that should be skipped
    template_file = tasks_dir / "template_task.py"
    template_file.write_text("""
from autoclean.core.task import Task

class TemplateTask(Task):
    def run(self):
        pass
""")
    
    private_file = tasks_dir / "_private_task.py"
    private_file.write_text("""
from autoclean.core.task import Task

class PrivateTask(Task):
    def run(self):
        pass
""")
    
    # Create a file that should be found
    normal_file = tasks_dir / "normal_task.py"
    normal_file.write_text("""
from autoclean.core.task import Task

class NormalTask(Task):
    def run(self):
        pass
""")
    
    monkeypatch.setattr(
        "autoclean.utils.user_config.user_config.tasks_dir",
        tasks_dir,
    )
    
    valid_tasks, _ = safe_discover_tasks()
    
    # Filter out built-in tasks
    custom_tasks = [t for t in valid_tasks if str(tasks_dir) in t.source]
    
    # Should only find NormalTask
    assert len(custom_tasks) == 1
    assert custom_tasks[0].name == "NormalTask"
    
    # Should not find TemplateTask or PrivateTask
    task_names = [t.name for t in custom_tasks]
    assert "TemplateTask" not in task_names
    assert "PrivateTask" not in task_names


def test_error_messages_are_helpful(monkeypatch, tmp_path):
    """Test that error messages provide helpful information."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    
    # Create a file with a specific syntax error
    syntax_error_file = tasks_dir / "syntax_error.py"
    syntax_error_file.write_text("""
from autoclean.core.task import Task

class SyntaxErrorTask(Task):
    def run(self)  # Missing colon
        pass
""")
    
    monkeypatch.setattr(
        "autoclean.utils.user_config.user_config.tasks_dir",
        tasks_dir,
    )
    
    _, invalid_files = safe_discover_tasks()
    
    # Find the syntax error file
    syntax_errors = [f for f in invalid_files if "syntax_error.py" in f.source]
    assert len(syntax_errors) > 0
    
    error = syntax_errors[0].error
    assert "Syntax error" in error
    assert "line" in error.lower()  # Should mention line number