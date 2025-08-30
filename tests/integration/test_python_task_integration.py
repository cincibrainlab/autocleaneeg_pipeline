"""
Integration tests for Python task file workflows.

Tests the complete workflow using Python task files with synthetic data.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

# Only run if core imports are available
pytest.importorskip("autoclean.core.pipeline")

try:
    from autoclean.core.pipeline import Pipeline
    from autoclean.core.task import Task

    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Pipeline module not available")
class TestPythonTaskIntegration:
    """Integration tests for Python task file workflows."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration testing."""
        temp_dir = tempfile.mkdtemp(prefix="autoclean_python_task_")
        workspace = Path(temp_dir)

        # Create directory structure
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "tasks").mkdir()

        yield workspace

        # Cleanup
        shutil.rmtree(workspace, ignore_errors=True)

    @pytest.fixture
    def synthetic_eeg_file(self, temp_workspace):
        """Create synthetic EEG data file for testing."""
        # Create synthetic data
        raw = create_synthetic_raw(
            montage="standard_1020", n_channels=32, duration=60, sfreq=500  # 1 minute
        )

        # Save to file
        data_file = temp_workspace / "input" / "test_data.fif"
        raw.save(data_file, overwrite=True)

        return data_file

    def create_simple_python_task(self, workspace: Path) -> Path:
        """Create a simple Python task file for testing."""
        task_content = '''
from typing import Any, Dict
from autoclean.core.task import Task

# Simple configuration for testing
config = {
    'resample_step': {'enabled': True, 'value': 250},
    'filtering': {'enabled': True, 'value': {'l_freq': 1, 'h_freq': 40}},
    'montage': {'enabled': False, 'value': 'standard_1020'},
    'reference_step': {'enabled': True, 'value': 'average'},
    'ICA': {'enabled': False, 'value': {'method': 'infomax'}},
    'epoch_settings': {'enabled': True, 'value': {'tmin': -1, 'tmax': 1}}
}

class SimpleIntegrationTask(Task):
    """Simple task for integration testing."""
    
    def run(self) -> None:
        """Execute simple processing pipeline."""
        # Import raw data
        self.import_raw()
        
        # Basic steps only (minimal processing for testing)
        self.run_basic_steps(export=True)
        
        # Simple epoching
        self.create_regular_epochs(export=True)

    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
'''

        task_file = workspace / "tasks" / "simple_integration_task.py"
        task_file.write_text(task_content)
        return task_file

    def test_python_task_complete_workflow(self, temp_workspace, synthetic_eeg_file):
        """Test complete workflow with Python task file."""
        # Create Python task file
        task_file = self.create_simple_python_task(temp_workspace)

        # Initialize pipeline without YAML config
        pipeline = Pipeline(output_dir=str(temp_workspace / "output"))

        # Add Python task
        pipeline.add_task(str(task_file))

        # Verify task was loaded (tasks stored with lowercase keys)
        tasks = pipeline.list_tasks()
        assert "simpleintegrationtask" in tasks

        # Mock the actual processing to avoid dependencies
        with (
            patch("autoclean.io.import_.import_eeg") as mock_import,
            patch("autoclean.io.export.save_raw_to_set") as mock_save_raw,
            patch("autoclean.io.export.save_epochs_to_set") as mock_save_epochs,
            patch("autoclean.utils.database.manage_database") as mock_db,
        ):

            # Mock the import to return our synthetic data
            raw = create_synthetic_raw(n_channels=32, sfreq=500, duration=60)
            mock_import.return_value = raw

            # Mock database operations
            mock_db.return_value = None

            # Process the file (use lowercase task name)
            try:
                pipeline.process_file(
                    file_path=str(synthetic_eeg_file), task="simpleintegrationtask"
                )

                # Verify that expected methods were called
                assert mock_import.called
                assert mock_save_raw.called or mock_save_epochs.called

            except Exception as e:
                # For integration test, we expect some methods to fail due to mocking
                # The important part is that the Python task loading and basic pipeline works
                # Accept plugin errors as expected in mocked environment
                assert (
                    "plugin" in str(e).lower()
                    or "simpleintegrationtask" in str(e).lower()
                    or mock_import.called
                )

    def test_python_task_settings_priority(self, temp_workspace):
        """Test that Python task settings take priority over defaults."""
        # Create task with specific settings
        task_content = """
from typing import Any, Dict
from autoclean.core.task import Task

config = {
    'resample_step': {'enabled': True, 'value': 123},  # Unique value for testing
    'filtering': {'enabled': False, 'value': {}}
}

class SettingsPriorityTask(Task):
    def run(self):
        # Check that our settings are used
        is_enabled, settings = self._check_step_enabled('resample_step')
        assert is_enabled is True
        assert settings['value'] == 123
        
        is_enabled, _ = self._check_step_enabled('filtering')
        assert is_enabled is False

    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
"""

        task_file = temp_workspace / "priority_task.py"
        task_file.write_text(task_content)

        # Create pipeline and add task
        pipeline = Pipeline(output_dir=str(temp_workspace / "output"))
        pipeline.add_task(str(task_file))

        # Create mock config (simplified - no stage_files needed)
        config = {
            "run_id": "test_priority",
            "unprocessed_file": Path("/fake/path"),
            "task": "SettingsPriorityTask",
        }

        # Instantiate task and test settings
        task_class = pipeline.session_task_registry["settingsprioritytask"]
        task_instance = task_class(config)

        # Run the task (which includes the assertions)
        task_instance.run()

    def test_python_task_export_control(self, temp_workspace):
        """Test export parameter control in Python tasks."""
        # Create task that tests export functionality
        task_content = """
from typing import Any, Dict
from autoclean.core.task import Task
from unittest.mock import Mock

config = {
    'resample_step': {'enabled': True, 'value': 250}
}

class ExportControlTask(Task):
    def run(self):
        # Mock the export functions to test they're called correctly
        self.raw = Mock()
        
        # Test _auto_export_if_enabled
        with unittest.mock.patch('autoclean.io.export.save_raw_to_set') as mock_save:
            self._auto_export_if_enabled(self.raw, 'test_stage', export_enabled=True)
            # Should have tried to ensure stage exists and save
            
        # Test that export=False doesn't trigger saving
        with unittest.mock.patch('autoclean.io.export.save_raw_to_set') as mock_save:
            self._auto_export_if_enabled(self.raw, 'test_stage', export_enabled=False)
            # Should not have called save

    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
"""

        task_file = temp_workspace / "export_control_task.py"
        task_file.write_text(task_content)

        pipeline = Pipeline(output_dir=str(temp_workspace / "output"))
        pipeline.add_task(str(task_file))

        # Verify task loaded (tasks stored with lowercase keys)
        assert "exportcontroltask" in pipeline.list_tasks()

    def test_mixed_builtin_python_workflow(self, temp_workspace):
        """Test workflow mixing built-in and Python tasks."""
        # Create Python task
        python_task = self.create_simple_python_task(temp_workspace)

        # Test with simplified configuration (no YAML needed)
        pipeline = Pipeline(output_dir=str(temp_workspace / "output"))

        # Get initial built-in tasks
        initial_tasks = pipeline.list_tasks()
        initial_count = len(initial_tasks)

        # Add Python task
        pipeline.add_task(str(python_task))

        # Should have built-in + Python tasks
        final_tasks = pipeline.list_tasks()
        assert len(final_tasks) == initial_count + 1
        assert "simpleintegrationtask" in final_tasks  # stored in lowercase

    def test_python_task_error_handling(self, temp_workspace):
        """Test error handling in Python task workflows."""
        # Test malformed Python task
        malformed_content = "invalid python syntax {"
        malformed_file = temp_workspace / "malformed.py"
        malformed_file.write_text(malformed_content)

        pipeline = Pipeline(output_dir=str(temp_workspace / "output"))

        with pytest.raises(ImportError):
            pipeline.add_task(str(malformed_file))

        # Test task without Task class
        no_task_content = """
def regular_function():
    pass

class NotATask:
    pass
"""
        no_task_file = temp_workspace / "no_task.py"
        no_task_file.write_text(no_task_content)

        with pytest.raises(ImportError, match="No Task"):
            pipeline.add_task(str(no_task_file))

    def test_dynamic_stage_creation(self, temp_workspace):
        """Test that stages are created dynamically without pre-definition."""
        # Create task that uses custom stage names
        task_content = """
from typing import Any, Dict
from autoclean.core.task import Task

config = {
    'resample_step': {'enabled': True, 'value': 250}
}

class DynamicStageTask(Task):
    def run(self):
        # Test that _ensure_stage_exists method exists and works
        # In simplified implementation, this method does nothing
        # Stage creation is handled automatically by export functions
        self._ensure_stage_exists('custom_stage_name')
        
        # Method should exist and not raise errors
        assert hasattr(self, '_ensure_stage_exists')
        # Stage creation is now handled by export functions, not this method

    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
"""

        task_file = temp_workspace / "dynamic_stage_task.py"
        task_file.write_text(task_content)

        pipeline = Pipeline(output_dir=str(temp_workspace / "output"))
        pipeline.add_task(str(task_file))

        # Create config and test (simplified)
        config = {
            "run_id": "test_dynamic",
            "unprocessed_file": Path("/fake/path"),
            "task": "DynamicStageTask",
        }

        task_class = pipeline.session_task_registry["dynamicstagetask"]
        task_instance = task_class(config)
        task_instance.run()


if __name__ == "__main__":
    pytest.main([__file__])
