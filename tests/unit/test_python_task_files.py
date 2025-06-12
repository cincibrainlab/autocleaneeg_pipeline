"""Unit tests for Python task file functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

from autoclean.core.pipeline import Pipeline
from autoclean.core.task import Task


class TestPythonTaskFiles:
    """Test suite for Python task file loading and execution."""

    def test_pipeline_without_yaml_config(self):
        """Test that Pipeline can be initialized without YAML config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)
            # In simplified implementation, no external YAML config needed
            assert hasattr(pipeline, 'output_dir')
            assert hasattr(pipeline, 'session_task_registry')
            assert isinstance(pipeline.session_task_registry, dict)

    def test_add_task_method_exists(self):
        """Test that Pipeline has add_task method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)
            assert hasattr(pipeline, 'add_task')
            assert callable(pipeline.add_task)

    def test_python_task_creation(self):
        """Test creating a simple Python task file."""
        # Create a mock Python task
        task_content = '''
from typing import Any, Dict
from autoclean.core.task import Task

config = {
    'resample_step': {'enabled': True, 'value': 250},
    'filtering': {'enabled': True, 'value': {'l_freq': 1, 'h_freq': 40}}
}

class TestTask(Task):
    def __init__(self, config: Dict[str, Any]):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(task_content)
            task_file = f.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pipeline = Pipeline(output_dir=temp_dir)
                
                # Test adding the task
                pipeline.add_task(task_file)
                
                # Check that task was registered
                assert 'testtask' in pipeline.session_task_registry
                
                # Check that the task class is correct
                task_class = pipeline.session_task_registry['testtask']
                assert issubclass(task_class, Task)
                assert task_class.__name__ == 'TestTask'
                
        finally:
            Path(task_file).unlink()

    def test_task_settings_priority(self):
        """Test that self.settings takes priority over YAML config."""
        # Create a mock task with settings
        class MockTask(Task):
            def __init__(self, config):
                self.settings = {
                    'resample_step': {'enabled': True, 'value': 500}
                }
                super().__init__(config)
            
            def run(self):
                pass
        
        # Mock config with different settings
        config = {
            'run_id': 'test',
            'unprocessed_file': Path('/fake/path'),
            'task': 'MockTask',
            'tasks': {
                'MockTask': {
                    'settings': {
                        'resample_step': {'enabled': True, 'value': 250}
                    }
                }
            },
            'stage_files': {}
        }
        
        task = MockTask(config)
        
        # Check that self.settings takes priority
        is_enabled, settings = task._check_step_enabled('resample_step')
        assert is_enabled is True
        assert settings['value'] == 500  # From self.settings, not YAML

    def test_dynamic_stage_creation(self):
        """Test that stages are created dynamically without required_stages."""
        class MockTask(Task):
            def __init__(self, config):
                # No self.required_stages defined
                super().__init__(config)
            
            def run(self):
                pass
        
        config = {
            'run_id': 'test',
            'unprocessed_file': Path('/fake/path'),
            'task': 'MockTask',
            'tasks': {},
            'stage_files': {}
        }
        
        # Should not raise an error even without required_stages
        task = MockTask(config)
        assert task.config == config

    def test_export_parameter_functionality(self):
        """Test that export parameters work in mixin methods."""
        from autoclean.mixins.base import BaseMixin
        
        class MockTaskWithMixin(Task, BaseMixin):  # Fixed MRO by putting Task first
            def __init__(self, config):
                self.settings = {
                    'resample_step': {'enabled': True, 'value': 250}
                }
                super().__init__(config)
                self.raw = Mock()
                self.config = config
            
            def run(self):
                pass
        
        config = {
            'run_id': 'test',
            'unprocessed_file': Path('/fake/path'),
            'task': 'MockTask',
            'tasks': {},
            'stage_files': {}
        }
        
        task = MockTaskWithMixin(config)
        
        # Test _auto_export_if_enabled method exists
        assert hasattr(task, '_auto_export_if_enabled')
        assert callable(task._auto_export_if_enabled)
        
        # Test _ensure_stage_exists method exists
        assert hasattr(task, '_ensure_stage_exists')
        assert callable(task._ensure_stage_exists)

    def test_built_in_tasks_with_python_tasks(self):
        """Test that built-in tasks work alongside Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)
            
            # Get initial built-in tasks
            initial_tasks = pipeline.list_tasks()
            initial_count = len(initial_tasks)
            assert initial_count > 0
            
            # Add a Python task
            python_task_content = '''
from autoclean.core.task import Task

config = {
    'resample_step': {'enabled': True, 'value': 250}
}

class PythonTask(Task):
    def __init__(self, config):
        self.settings = globals()['config']
        super().__init__(config)
    
    def run(self):
        pass
'''
            task_file = Path(temp_dir) / "python_task.py"
            task_file.write_text(python_task_content)
            
            # Add the Python task
            pipeline.add_task(str(task_file))
            
            # Should have both built-in and Python tasks
            final_tasks = pipeline.list_tasks()
            assert len(final_tasks) == initial_count + 1
            assert 'pythontask' in final_tasks
            
            # All original tasks should still be there
            for task in initial_tasks:
                assert task in final_tasks

    def test_missing_task_file_error(self):
        """Test error handling for missing task files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)
            
            with pytest.raises(FileNotFoundError):
                pipeline.add_task("/nonexistent/task.py")

    def test_malformed_python_task_error(self):
        """Test error handling for malformed Python task files."""
        # Create a malformed task file
        malformed_content = '''
# This file has syntax errors
class TestTask(Task:  # Missing closing parenthesis
    def __init__(self, config):
        pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(malformed_content)
            task_file = f.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pipeline = Pipeline(output_dir=temp_dir)
                
                with pytest.raises(ImportError):
                    pipeline.add_task(task_file)
                    
        finally:
            Path(task_file).unlink()

    def test_no_task_class_error(self):
        """Test error handling for Python files without Task classes."""
        # Create a file without Task classes
        no_task_content = '''
def some_function():
    pass

class NotATask:
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(no_task_content)
            task_file = f.name
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pipeline = Pipeline(output_dir=temp_dir)
                
                with pytest.raises(ImportError, match="No Task"):
                    pipeline.add_task(task_file)
                    
        finally:
            Path(task_file).unlink()

    def test_validate_task_for_python_tasks(self):
        """Test task validation for Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(output_dir=temp_dir)
            
            # Add a mock Python task
            class MockPythonTask(Task):
                def run(self):
                    pass
            
            pipeline.session_task_registry['mockpythontask'] = MockPythonTask
            
            # Should validate successfully
            result = pipeline._validate_task('MockPythonTask')
            assert result == 'MockPythonTask'

    def test_hash_encoding_without_yaml_config(self):
        """Test configuration hashing when no YAML config is provided."""
        from autoclean.utils.config import hash_and_encode_yaml
        
        # Test with minimal config (no file)
        minimal_config = {"version": "1.0", "type": "python_tasks_only"}
        b64_encoded, config_hash = hash_and_encode_yaml(minimal_config, is_file=False)
        
        assert isinstance(b64_encoded, str)
        assert isinstance(config_hash, str)
        assert len(config_hash) == 64  # SHA256 hash length


class TestExportFunctionality:
    """Test suite for export parameter functionality."""

    def test_auto_export_if_enabled(self):
        """Test _auto_export_if_enabled method."""
        from autoclean.mixins.base import BaseMixin
        
        class MockTask(BaseMixin):
            def __init__(self):
                self.config = {
                    'stage_files': {},
                    'run_id': 'test'
                }
        
        task = MockTask()
        mock_data = Mock()
        
        # Test with export disabled
        with patch.object(task, '_ensure_stage_exists') as mock_ensure:
            task._auto_export_if_enabled(mock_data, 'test_stage', export_enabled=False)
            mock_ensure.assert_not_called()
        
        # Test with export enabled
        with patch.object(task, '_ensure_stage_exists') as mock_ensure:
            task._auto_export_if_enabled(mock_data, 'test_stage', export_enabled=True)
            mock_ensure.assert_called_once_with('test_stage')

    def test_ensure_stage_exists(self):
        """Test _ensure_stage_exists method (simplified implementation)."""
        from autoclean.mixins.base import BaseMixin
        
        class MockTask(BaseMixin):
            def __init__(self):
                self.config = {}
        
        task = MockTask()
        
        # In the simplified implementation, _ensure_stage_exists does nothing
        # Export functions handle stage creation automatically
        task._ensure_stage_exists('post_custom_stage')
        
        # Should not raise any errors and method should exist
        assert hasattr(task, '_ensure_stage_exists')
        # Stage creation is now handled by export functions, not this method
        assert True  # Test passes if no exception was raised

    def test_generate_stage_name(self):
        """Test _generate_stage_name method."""
        from autoclean.mixins.base import BaseMixin
        
        task = BaseMixin()
        
        # Test method name mapping
        assert task._generate_stage_name('run_basic_steps') == 'post_basic_steps'
        assert task._generate_stage_name('run_ica') == 'post_ica'
        assert task._generate_stage_name('create_regular_epochs') == 'post_epochs'
        assert task._generate_stage_name('custom_method') == 'post_custom_method'


class TestUtilityFunctions:
    """Test suite for updated utility functions."""

    def test_validate_eeg_system_with_python_tasks(self):
        """Test EEG system validation for Python tasks."""
        from autoclean.utils.config import validate_eeg_system
        
        # Test with Python task configuration
        config = {
            'tasks': {},  # No YAML tasks
            'task_config': {
                'montage': {'value': 'GSN-HydroCel-129'}
            }
        }
        
        result = validate_eeg_system(config, 'PythonTask')
        assert result == 'GSN-HydroCel-129'

    def test_validate_eeg_system_without_montage(self):
        """Test EEG system validation when no montage is specified."""
        from autoclean.utils.config import validate_eeg_system
        
        # Test with no montage configuration
        config = {
            'tasks': {},
            'task_config': {}
        }
        
        result = validate_eeg_system(config, 'PythonTask')
        assert result is None  # Should skip validation gracefully


if __name__ == '__main__':
    pytest.main([__file__])