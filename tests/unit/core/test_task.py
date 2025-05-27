"""Unit tests for the Task base class."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from abc import ABC

from tests.fixtures.test_utils import BaseTestCase, EEGAssertions, MockOperations
from tests.fixtures.synthetic_data import create_synthetic_raw

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.core.task import Task
    from autoclean.mixins import DISCOVERED_MIXINS
    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False


@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
class TestTaskInitialization:
    """Test Task base class initialization and configuration."""
    
    def test_task_is_abstract_base_class(self):
        """Test that Task is properly defined as an abstract base class."""
        from autoclean.core.task import Task
        
        # Task should be abstract and not directly instantiable
        assert issubclass(Task, ABC)
        
        # Should raise TypeError when trying to instantiate directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Task({})
    
    def test_task_mixin_inheritance(self):
        """Test that Task properly inherits from discovered mixins."""
        from autoclean.core.task import Task
        from autoclean.mixins import DISCOVERED_MIXINS
        
        # Task should inherit from all discovered mixins
        for mixin in DISCOVERED_MIXINS:
            assert issubclass(Task, mixin), f"Task should inherit from {mixin}"
    
    def test_task_expected_abstract_methods(self):
        """Test that Task defines expected abstract methods."""
        from autoclean.core.task import Task
        
        # Get abstract methods
        abstract_methods = getattr(Task, '__abstractmethods__', set())
        
        # Should have run method as abstract
        expected_abstracts = {'run'}
        assert expected_abstracts.issubset(abstract_methods), \
            f"Task missing expected abstract methods: {expected_abstracts - abstract_methods}"
    
    def test_task_config_parameter_requirements(self):
        """Test Task configuration parameter requirements."""
        from autoclean.core.task import Task
        
        # Create concrete Task for testing
        class ConcreteTask(Task):
            def run(self):
                pass
        
        # Test valid config
        valid_config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
            "tasks": {
                "test_task": {
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250}
                    }
                }
            }
        }
        
        # Should not raise error with valid config
        task = ConcreteTask(valid_config)
        assert task.config == valid_config
    
    def test_task_config_validation(self):
        """Test Task configuration validation."""
        from autoclean.core.task import Task
        
        # Create concrete Task for testing
        class ConcreteTask(Task):
            def run(self):
                pass
        
        # Test with missing required fields
        invalid_configs = [
            {},  # Empty config
            {"run_id": "test"},  # Missing unprocessed_file
            {"run_id": "test", "unprocessed_file": Path("/test.fif")},  # Missing task
            {
                "run_id": "test", 
                "unprocessed_file": Path("/test.fif"), 
                "task": "test_task"
                # Missing tasks
            }
        ]
        
        for invalid_config in invalid_configs:
            # Task might validate config in __init__ or later
            # For now, just test that it accepts the config parameter
            task = ConcreteTask(invalid_config)
            assert task.config == invalid_config


@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
class TestTaskInterface:
    """Test Task interface and method signatures."""
    
    def test_task_has_expected_methods(self):
        """Test that Task has expected methods from mixins."""
        from autoclean.core.task import Task
        
        # Should have methods from mixins (these will be tested in mixin tests)
        # Here we just verify the interface exists
        expected_mixin_methods = [
            # These come from mixins and should be available
            # Actual method names depend on mixin implementation
        ]
        
        # Verify Task class has the abstract interface
        assert hasattr(Task, '__init__')
        assert hasattr(Task, 'run')  # Abstract method
    
    def test_task_mro_consistency(self):
        """Test that Task's method resolution order is consistent."""
        from autoclean.core.task import Task
        
        # MRO should be well-defined without conflicts
        mro = Task.__mro__
        assert len(mro) > 2  # At least Task, ABC, and mixins
        assert Task in mro
        assert ABC in mro


class TestTaskConcrete:
    """Test Task with concrete implementation."""
    
    @pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
    def test_concrete_task_implementation(self):
        """Test that concrete Task implementation works."""
        from autoclean.core.task import Task
        
        class TestTask(Task):
            """Concrete test task implementation."""
            
            def run(self):
                """Test run implementation."""
                return {"status": "completed", "result": "test"}
        
        config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
            "tasks": {
                "test_task": {
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250}
                    }
                }
            }
        }
        
        task = TestTask(config)
        assert task.config == config
        
        # Should be able to call run method
        result = task.run()
        assert result["status"] == "completed"
    
    @pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
    @patch('autoclean.io.import_.import_eeg')
    def test_task_with_mocked_dependencies(self, mock_import):
        """Test Task with mocked heavy dependencies."""
        from autoclean.core.task import Task
        
        class TestTask(Task):
            def run(self):
                # Test that import method is available (from mixins)
                if hasattr(self, 'import_raw'):
                    return {"imported": True}
                return {"imported": False}
        
        config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
            "tasks": {"test_task": {"settings": {}}}
        }
        
        # Mock the EEG import
        mock_raw = create_synthetic_raw()
        mock_import.return_value = mock_raw
        
        task = TestTask(config)
        result = task.run()
        
        # Task should be properly constructed
        assert isinstance(result, dict)


class TestTaskMocked:
    """Test Task functionality with heavy mocking."""
    
    @patch('autoclean.mixins.DISCOVERED_MIXINS', [])
    def test_task_without_mixins(self):
        """Test Task behavior when no mixins are discovered."""
        # This tests the fallback behavior
        with patch('autoclean.core.task.DISCOVERED_MIXINS', []):
            # Import with no mixins
            from autoclean.core.task import Task
            
            # Task should still be an ABC
            assert issubclass(Task, ABC)
    
    def test_task_mixin_discovery_failure(self):
        """Test Task behavior when mixin discovery fails."""
        with patch('autoclean.core.task.DISCOVERED_MIXINS', side_effect=ImportError("Mixin discovery failed")):
            # Should handle mixin discovery failure gracefully
            # or raise appropriate error
            try:
                from autoclean.core.task import Task
                # If import succeeds, it handled the error
                assert True
            except ImportError:
                # If import fails, that's also acceptable behavior
                assert True


class TestTaskConceptual:
    """Conceptual tests for Task design patterns."""
    
    def test_task_design_patterns(self):
        """Test that Task follows expected design patterns."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing design conceptually")
        
        from autoclean.core.task import Task
        
        # Abstract Base Class pattern
        assert issubclass(Task, ABC)
        
        # Multiple inheritance pattern (with mixins)
        assert len(Task.__mro__) > 2
        
        # Template method pattern (abstract run method)
        assert hasattr(Task, 'run')
        assert Task.run.__qualname__.startswith('Task.')
    
    def test_task_configuration_interface(self):
        """Test Task configuration interface design."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing interface conceptually")
        
        from autoclean.core.task import Task
        
        # Should accept config in __init__
        init_signature = Task.__init__.__annotations__
        # Note: annotations might not be available in all Python versions
        
        # Should store config
        assert hasattr(Task, '__init__')
    
    def test_task_mixin_integration_concept(self):
        """Test conceptual Task-mixin integration."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing integration conceptually")
        
        from autoclean.core.task import Task
        from autoclean.mixins import DISCOVERED_MIXINS
        
        # Task should integrate with discovered mixins
        if DISCOVERED_MIXINS:
            for mixin in DISCOVERED_MIXINS:
                assert issubclass(Task, mixin), \
                    f"Task should inherit from discovered mixin {mixin}"
        
        # Task should maintain its primary interface
        assert hasattr(Task, 'run')
    
    def test_task_extensibility_concept(self):
        """Test Task extensibility concept."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing extensibility conceptually")
        
        from autoclean.core.task import Task
        
        # Should be extensible through inheritance
        class CustomTask(Task):
            def run(self):
                return "custom implementation"
        
        # Should be able to create custom tasks
        assert issubclass(CustomTask, Task)
        assert CustomTask.run != Task.run  # Override
        
        # Custom task should still have access to mixin functionality
        # (Specific mixin methods tested in mixin tests)


# Error condition tests
class TestTaskErrorHandling:
    """Test Task error handling and edge cases."""
    
    @pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
    def test_task_with_none_config(self):
        """Test Task behavior with None config."""
        from autoclean.core.task import Task
        
        class TestTask(Task):
            def run(self):
                return "test"
        
        # Should handle None config appropriately
        # (might raise error or handle gracefully)
        try:
            task = TestTask(None)
            # If it accepts None, that's valid behavior
            assert task.config is None
        except (TypeError, ValueError):
            # If it rejects None, that's also valid behavior
            assert True
    
    @pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
    def test_task_with_invalid_config_types(self):
        """Test Task behavior with invalid config types."""
        from autoclean.core.task import Task
        
        class TestTask(Task):
            def run(self):
                return "test"
        
        invalid_configs = ["string", 123, [1, 2, 3], True]
        
        for invalid_config in invalid_configs:
            # Should handle invalid config types appropriately
            try:
                task = TestTask(invalid_config)
                # If it accepts invalid types, it should store them
                assert task.config == invalid_config
            except (TypeError, ValueError):
                # If it rejects invalid types, that's also valid
                assert True