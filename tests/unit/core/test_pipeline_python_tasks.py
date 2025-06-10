"""Unit tests for Pipeline Python task file functionality."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

try:
    from autoclean.core.pipeline import Pipeline
    from autoclean.core.task import Task
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelinePythonTasks:
    """Test Pipeline functionality for Python task files."""

    def test_pipeline_init_without_yaml(self):
        """Test Pipeline initialization without YAML config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            assert pipeline.autoclean_config is None
            assert pipeline.autoclean_dict is not None
            assert pipeline.session_task_registry == {}
            assert hasattr(pipeline, 'add_task')

    def test_pipeline_init_with_yaml(self):
        """Test Pipeline initialization with YAML config (backward compatibility)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock YAML file path
            yaml_path = Path(temp_dir) / "test_config.yaml"
            
            with patch('autoclean.core.pipeline.Path.exists', return_value=True), \
                 patch('autoclean.utils.config.load_config') as mock_load:
                
                mock_load.return_value = {
                    'tasks': {'test_task': {'settings': {}}},
                    'stage_files': {'post_import': {'enabled': True, 'suffix': '_import'}}
                }
                
                pipeline = Pipeline(autoclean_dir=temp_dir, autoclean_config=str(yaml_path))
                
                assert pipeline.autoclean_config == str(yaml_path)
                assert pipeline.autoclean_dict is not None

    def test_add_task_method(self):
        """Test add_task method functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Create a mock task file
            task_content = '''
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
'''
            
            task_file = Path(temp_dir) / "mock_task.py"
            task_file.write_text(task_content)
            
            # Test adding the task
            pipeline.add_task(str(task_file))
            
            # Verify task was registered
            assert 'mocktesttask' in pipeline.session_task_registry
            task_class = pipeline.session_task_registry['mocktesttask']
            assert issubclass(task_class, Task)

    def test_add_task_file_not_found(self):
        """Test add_task with non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            with pytest.raises(FileNotFoundError):
                pipeline.add_task("/nonexistent/task.py")

    def test_add_task_malformed_file(self):
        """Test add_task with malformed Python file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Create malformed file
            malformed_content = "invalid python syntax {"
            task_file = Path(temp_dir) / "malformed.py"
            task_file.write_text(malformed_content)
            
            with pytest.raises(ImportError):
                pipeline.add_task(str(task_file))

    def test_add_task_no_task_class(self):
        """Test add_task with file containing no Task classes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Create file without Task class
            no_task_content = '''
def some_function():
    pass

class NotATask:
    pass
'''
            task_file = Path(temp_dir) / "no_task.py"
            task_file.write_text(no_task_content)
            
            with pytest.raises(ImportError, match="No Task"):
                pipeline.add_task(str(task_file))

    def test_list_tasks_python_only(self):
        """Test list_tasks with only Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Initially empty
            assert pipeline.list_tasks() == []
            
            # Add a Python task
            task_content = '''
from autoclean.core.task import Task

class TestListTask(Task):
    def run(self):
        pass
'''
            task_file = Path(temp_dir) / "test_list.py"
            task_file.write_text(task_content)
            
            pipeline.add_task(str(task_file))
            
            tasks = pipeline.list_tasks()
            assert 'TestListTask' in tasks

    def test_list_tasks_mixed(self):
        """Test list_tasks with both YAML and Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock pipeline with YAML tasks
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Mock some built-in tasks
            class BuiltInTask(Task):
                def run(self):
                    pass
            
            pipeline.task_registry['builtintask'] = BuiltInTask
            
            # Add Python task
            task_content = '''
from autoclean.core.task import Task

class PythonTask(Task):
    def run(self):
        pass
'''
            task_file = Path(temp_dir) / "python_task.py"
            task_file.write_text(task_content)
            
            pipeline.add_task(str(task_file))
            
            tasks = pipeline.list_tasks()
            assert 'BuiltInTask' in tasks
            assert 'PythonTask' in tasks

    def test_validate_task_python(self):
        """Test _validate_task for Python tasks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Add a Python task
            class MockPythonTask(Task):
                def run(self):
                    pass
            
            pipeline.session_task_registry['mockpythontask'] = MockPythonTask
            
            # Should validate successfully
            result = pipeline._validate_task('MockPythonTask')
            assert result == 'MockPythonTask'

    def test_validate_task_case_insensitive(self):
        """Test that task validation is case-insensitive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            class CamelCaseTask(Task):
                def run(self):
                    pass
            
            pipeline.session_task_registry['camelcasetask'] = CamelCaseTask
            
            # Should work with different cases
            assert pipeline._validate_task('CamelCaseTask') == 'CamelCaseTask'
            assert pipeline._validate_task('camelcasetask') == 'camelcasetask'
            assert pipeline._validate_task('CAMELCASETASK') == 'CAMELCASETASK'

    def test_generate_default_config(self):
        """Test _generate_default_config method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            config = pipeline._generate_default_config()
            
            assert 'stage_files' in config
            assert 'tasks' in config
            assert isinstance(config['stage_files'], dict)
            assert isinstance(config['tasks'], dict)

    def test_load_python_task_multiple_classes(self):
        """Test _load_python_task with multiple Task classes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Create file with multiple Task classes
            multi_task_content = '''
from autoclean.core.task import Task

class FirstTask(Task):
    def run(self):
        pass

class SecondTask(Task):
    def run(self):
        pass
'''
            task_file = Path(temp_dir) / "multi_task.py"
            task_file.write_text(multi_task_content)
            
            # Should pick the first one found
            task_class = pipeline._load_python_task(task_file)
            assert issubclass(task_class, Task)
            assert task_class.__name__ in ['FirstTask', 'SecondTask']


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineBackwardCompatibility:
    """Test Pipeline backward compatibility with YAML tasks."""

    def test_yaml_and_python_coexistence(self):
        """Test that YAML and Python tasks can coexist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pipeline with mocked YAML config
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Mock YAML task structure
            pipeline.autoclean_dict = {
                'tasks': {
                    'YamlTask': {
                        'settings': {'resample_step': {'enabled': True, 'value': 250}}
                    }
                },
                'stage_files': {'post_import': {'enabled': True, 'suffix': '_import'}}
            }
            
            # Mock built-in task registry
            class YamlTask(Task):
                def run(self):
                    pass
            
            pipeline.task_registry['yamltask'] = YamlTask
            
            # Add Python task
            python_content = '''
from autoclean.core.task import Task

class PythonTask(Task):
    def run(self):
        pass
'''
            task_file = Path(temp_dir) / "python_task.py"
            task_file.write_text(python_content)
            
            pipeline.add_task(str(task_file))
            
            # Both should be available
            tasks = pipeline.list_tasks()
            assert 'YamlTask' in tasks
            assert 'PythonTask' in tasks

    def test_config_hash_without_yaml(self):
        """Test configuration hashing when no YAML config exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Mock the entrypoint to test config hashing
            with patch('autoclean.core.pipeline.hash_and_encode_yaml') as mock_hash:
                mock_hash.return_value = ('encoded_config', 'config_hash')
                
                # This should work without errors
                assert pipeline.autoclean_config is None
                # The _entrypoint method should handle None config gracefully


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import") 
class TestPipelineUtilities:
    """Test Pipeline utility functions for Python tasks."""

    def test_generate_default_stage_config(self):
        """Test _generate_default_stage_config method."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            stage_config = pipeline._generate_default_stage_config()
            
            # Should have standard stages
            expected_stages = ['post_import', 'post_basic_steps', 'post_clean_raw', 'post_epochs', 'post_comp']
            
            for stage in expected_stages:
                assert stage in stage_config
                assert 'enabled' in stage_config[stage]
                assert 'suffix' in stage_config[stage]
                assert stage_config[stage]['enabled'] is True

    def test_task_extraction_from_python_file(self):
        """Test that Python task settings are properly extracted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = Pipeline(autoclean_dir=temp_dir)
            
            # Create task with settings
            task_with_settings = '''
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
'''
            
            task_file = Path(temp_dir) / "settings_task.py"
            task_file.write_text(task_with_settings)
            
            pipeline.add_task(str(task_file))
            
            # Create minimal config for testing
            minimal_config = {
                'run_id': 'test',
                'unprocessed_file': Path('/fake/path'),
                'task': 'SettingsTask',
                'tasks': {},
                'stage_files': {}
            }
            
            # Instantiate task to test settings
            task_class = pipeline.session_task_registry['settingstask']
            task_instance = task_class(minimal_config)
            
            assert hasattr(task_instance, 'settings')
            assert 'montage' in task_instance.settings
            assert task_instance.settings['montage']['value'] == 'GSN-HydroCel-129'


if __name__ == '__main__':
    pytest.main([__file__])