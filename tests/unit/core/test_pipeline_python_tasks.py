"""Unit tests for Pipeline Python task file functionality."""

import tempfile
from pathlib import Path

import pytest

try:
    from autoclean.core.pipeline import Pipeline
    from autoclean.core.task import Task

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelinePythonTasks:
    """Test Pipeline functionality for Python task files."""

    def test_pipeline_init_without_yaml(self):
        """Test Pipeline initialization with simplified API (no YAML config needed)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # In the new simplified API, we don't use YAML configs
            assert hasattr(pipeline, "output_dir")
            assert hasattr(pipeline, "session_task_registry")
            assert isinstance(pipeline.session_task_registry, dict)
            assert hasattr(pipeline, "add_task")

    def test_pipeline_init_with_default_config(self):
        """Test Pipeline initialization uses default configuration approach."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # New implementation doesn't use external YAML configs
            # All configuration comes from Python task files
            assert pipeline.output_dir == Path(temp_dir).absolute()
            assert hasattr(pipeline, "session_task_registry")
            assert isinstance(pipeline.session_task_registry, dict)

    def test_add_task_method(self):
        """Test add_task method functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Create a mock task file
            task_content = """
from typing import Any, Dict
from autoclean.core.task import Task

config = {
    'resample_step': {'enabled': True, 'value': 250}
}

class MockTestTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        pass
"""

            task_file = Path(temp_dir) / "mock_task.py"
            task_file.write_text(task_content)

            # Test adding the task
            pipeline.add_task(str(task_file))

            # Test adding the task returns the task name
            task_name = pipeline.add_task(str(task_file))
            assert task_name == "MockTestTask"

            # Verify task was registered (tasks are stored with lowercase keys)
            assert "mocktesttask" in pipeline.session_task_registry
            task_class = pipeline.session_task_registry["mocktesttask"]
            assert issubclass(task_class, Task)

    def test_add_task_file_not_found(self):
        """Test add_task with non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            with pytest.raises(FileNotFoundError):
                pipeline.add_task("/nonexistent/task.py")

    def test_add_task_malformed_file(self):
        """Test add_task with malformed Python file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Create malformed file
            malformed_content = "invalid python syntax {"
            task_file = Path(temp_dir) / "malformed.py"
            task_file.write_text(malformed_content)

            with pytest.raises(ImportError):
                pipeline.add_task(str(task_file))

    def test_add_task_no_task_class(self):
        """Test add_task with file containing no Task classes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Create file without Task class
            no_task_content = """
def some_function():
    pass

class NotATask:
    pass
"""
            task_file = Path(temp_dir) / "no_task.py"
            task_file.write_text(no_task_content)

            with pytest.raises(ImportError, match="No Task"):
                pipeline.add_task(str(task_file))

    def test_list_tasks_python_only(self):
        """Test list_tasks includes built-in tasks plus added Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Should have built-in tasks
            initial_tasks = pipeline.list_tasks()
            assert len(initial_tasks) > 0

            # Add a Python task
            task_content = """
from autoclean.core.task import Task

class TestListTask(Task):
    def run(self):
        pass
"""
            task_file = Path(temp_dir) / "test_list.py"
            task_file.write_text(task_content)

            pipeline.add_task(str(task_file))

            tasks = pipeline.list_tasks()
            # Should have more tasks now (built-in + added)
            assert len(tasks) > len(initial_tasks)
            # Check that our added task is there (stored as lowercase)
            assert "testlisttask" in tasks

    def test_list_tasks_mixed(self):
        """Test list_tasks with built-in and Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Get initial built-in tasks
            initial_tasks = pipeline.list_tasks()
            initial_count = len(initial_tasks)

            # Add Python task
            task_content = """
from autoclean.core.task import Task

class PythonTask(Task):
    def run(self):
        pass
"""
            task_file = Path(temp_dir) / "python_task.py"
            task_file.write_text(task_content)

            pipeline.add_task(str(task_file))

            tasks = pipeline.list_tasks()

            # Should have more tasks now (built-in + Python task)
            assert len(tasks) == initial_count + 1

            # Check that our Python task was added (stored as lowercase)
            assert "pythontask" in tasks

            # Check that some built-in tasks are still there
            assert any(task in tasks for task in initial_tasks)

    def test_validate_task_python(self):
        """Test _validate_task for Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Add a Python task
            class MockPythonTask(Task):
                def run(self):
                    pass

            pipeline.session_task_registry["mockpythontask"] = MockPythonTask

            # Should validate successfully
            result = pipeline._validate_task("MockPythonTask")
            assert result == "MockPythonTask"

    def test_validate_task_case_insensitive(self):
        """Test that task validation is case-insensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            class CamelCaseTask(Task):
                def run(self):
                    pass

            pipeline.session_task_registry["camelcasetask"] = CamelCaseTask

            # Should work with different cases
            assert pipeline._validate_task("CamelCaseTask") == "CamelCaseTask"
            assert pipeline._validate_task("camelcasetask") == "camelcasetask"
            assert pipeline._validate_task("CAMELCASETASK") == "CAMELCASETASK"

    def test_session_task_registry_initialization(self):
        """Test that session task registry is properly initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Check that session_task_registry exists and contains built-in tasks
            assert hasattr(pipeline, "session_task_registry")
            assert isinstance(pipeline.session_task_registry, dict)

            # Should have some built-in tasks
            task_list = pipeline.list_tasks()
            assert len(task_list) > 0

            # Check for some common built-in tasks
            task_list_lower = [t.lower() for t in task_list]
            assert any(
                task in task_list_lower
                for task in ["assrdefault", "chirpdefault", "rawtoset"]
            )

    def test_load_python_task_multiple_classes(self):
        """Test _load_python_task with multiple Task classes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Create file with multiple Task classes
            multi_task_content = """
from autoclean.core.task import Task

class FirstTask(Task):
    def run(self):
        pass

class SecondTask(Task):
    def run(self):
        pass
"""
            task_file = Path(temp_dir) / "multi_task.py"
            task_file.write_text(multi_task_content)

            # Should pick the first one found
            task_class = pipeline._load_python_task(task_file)
            assert issubclass(task_class, Task)
            assert task_class.__name__ in ["FirstTask", "SecondTask"]


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelineBackwardCompatibility:
    """Test Pipeline backward compatibility with YAML tasks."""

    def test_built_in_and_python_task_coexistence(self):
        """Test that built-in and Python tasks can coexist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Get initial built-in tasks
            initial_tasks = pipeline.list_tasks()
            assert len(initial_tasks) > 0

            # Add Python task
            python_content = """
from autoclean.core.task import Task

class PythonTask(Task):
    def run(self):
        pass
"""
            task_file = Path(temp_dir) / "python_task.py"
            task_file.write_text(python_content)

            pipeline.add_task(str(task_file))

            # Both built-in and Python tasks should be available
            final_tasks = pipeline.list_tasks()

            # Should have all initial tasks plus the new Python task
            assert len(final_tasks) == len(initial_tasks) + 1
            assert "pythontask" in final_tasks

            # All initial tasks should still be there
            for task in initial_tasks:
                assert task in final_tasks

    def test_simplified_configuration_approach(self):
        """Test that pipeline works without external YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Pipeline should work with embedded task configurations
            # No external YAML files are needed
            assert hasattr(pipeline, "output_dir")
            assert hasattr(pipeline, "session_task_registry")

            # Should be able to process with built-in tasks
            tasks = pipeline.list_tasks()
            assert len(tasks) > 0


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelineUtilities:
    """Test Pipeline utility functions for Python tasks."""

    def test_list_stage_files_functionality(self):
        """Test list_stage_files method provides expected stages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            stage_files = pipeline.list_stage_files()

            # Should have standard stages
            expected_stages = [
                "post_import",
                "post_basic_steps",
                "post_clean_raw",
                "post_epochs",
                "post_comp",
            ]

            assert isinstance(stage_files, list)
            for stage in expected_stages:
                assert stage in stage_files

    def test_task_extraction_from_python_file(self):
        """Test that Python task settings are properly extracted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)

            # Create task with settings
            task_with_settings = """
from typing import Any, Dict
from autoclean.core.task import Task

config = {
    'montage': {'enabled': True, 'value': 'GSN-HydroCel-129'},
    'filtering': {'enabled': True, 'value': {'l_freq': 1, 'h_freq': 40}}
}

class SettingsTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        pass
"""

            task_file = Path(temp_dir) / "settings_task.py"
            task_file.write_text(task_with_settings)

            pipeline.add_task(str(task_file))

            # Create minimal config for testing
            minimal_config = {
                "run_id": "test",
                "unprocessed_file": Path("/fake/path"),
                "task": "SettingsTask",
                "tasks": {},
                "stage_files": {},
            }

            # Instantiate task to test settings
            task_class = pipeline.session_task_registry["settingstask"]
            task_instance = task_class(minimal_config)

            assert hasattr(task_instance, "settings")
            assert "montage" in task_instance.settings
            assert task_instance.settings["montage"]["value"] == "GSN-HydroCel-129"


if __name__ == "__main__":
    pytest.main([__file__])
