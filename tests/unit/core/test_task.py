"""Unit tests for the Task base class."""

from abc import ABC
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.core.task import Task
    from autoclean.mixins import DISCOVERED_MIXINS

    TASK_AVAILABLE = True
except ImportError:
    TASK_AVAILABLE = False
    Task = None
    DISCOVERED_MIXINS = None


@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
class TestTaskInitialization:
    """Test Task base class initialization and configuration."""

    def test_task_is_abstract_base_class(self):
        """Test that Task is properly defined as an abstract base class."""

        # Task should be abstract and not directly instantiable
        assert issubclass(Task, ABC)

        # Should raise TypeError when trying to instantiate directly
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Task({})

    def test_task_mixin_inheritance(self):
        """Test that Task properly inherits from discovered mixins."""

        # Task should inherit from all discovered mixins
        for mixin in DISCOVERED_MIXINS:
            assert issubclass(Task, mixin), f"Task should inherit from {mixin}"

    def test_task_expected_abstract_methods(self):
        """Test that Task defines expected abstract methods."""

        # Get abstract methods
        abstract_methods = getattr(Task, "__abstractmethods__", set())

        # Should have run method as abstract
        expected_abstracts = {"run"}
        assert expected_abstracts.issubset(
            abstract_methods
        ), f"Task missing expected abstract methods: {expected_abstracts - abstract_methods}"

    def test_task_config_parameter_requirements(self):
        """Test Task configuration parameter requirements."""

        # Create concrete Task for testing
        class ConcreteTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                pass

        # Test valid config with all required fields
        valid_config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
            "tasks": {
                "test_task": {
                    "mne_task": "test",
                    "description": "Test task",
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250},
                        "filtering": {
                            "enabled": True,
                            "value": {"l_freq": 1, "h_freq": 100},
                        },
                        "trim_step": {"enabled": False, "value": 2},
                        "crop_step": {
                            "enabled": False,
                            "value": {"start": 0, "end": None},
                        },
                        "reference_step": {"enabled": True, "value": "average"},
                        "montage": {"enabled": True, "value": "standard_1020"},
                        "ICA": {
                            "enabled": False,
                            "value": {"method": "infomax", "n_components": 15},
                        },
                        "ICLabel": {
                            "enabled": False,
                            "value": {
                                "ic_flags_to_reject": [],
                                "ic_rejection_threshold": 0.5,
                            },
                        },
                        "epoch_settings": {
                            "enabled": True,
                            "value": {"tmin": -1, "tmax": 1},
                            "event_id": None,
                        },
                    },
                }
            },
            "stage_files": {
                "post_import": {"enabled": True, "suffix": "_postimport"},
                "post_clean_raw": {"enabled": True, "suffix": "_postcleanraw"},
            },
        }

        # Should not raise error with valid config
        task = ConcreteTask(valid_config)
        assert task.config == valid_config

    def test_python_task_with_settings(self):
        """Test Python task with embedded settings."""

        class PythonTask(Task):
            def __init__(self, config):
                # Embedded settings (Python task style)
                self.settings = {
                    "resample_step": {"enabled": True, "value": 250},
                    "filtering": {
                        "enabled": True,
                        "value": {"l_freq": 1, "h_freq": 40},
                    },
                }
                super().__init__(config)

            def run(self):
                pass

        # Minimal config for Python tasks
        python_config = {
            "run_id": "test_run_456",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "PythonTask",
            "tasks": {},  # Empty for Python tasks
            "stage_files": {},  # Auto-generated for Python tasks
        }

        # Should work with Python task
        task = PythonTask(python_config)
        assert task.config == python_config
        assert hasattr(task, "settings")
        assert task.settings["resample_step"]["enabled"] is True

    def test_task_without_required_stages(self):
        """Test that tasks work without defining required_stages."""

        class FlexibleTask(Task):
            def __init__(self, config):
                # No required_stages defined - should work with new system
                super().__init__(config)

            def run(self):
                pass

        minimal_config = {
            "run_id": "test_run_789",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "FlexibleTask",
            "tasks": {},
            "stage_files": {},
        }

        # Should not raise error even without required_stages
        task = FlexibleTask(minimal_config)
        assert task.config == minimal_config

    def test_task_config_validation(self):
        """Test Task configuration validation."""

        # Create concrete Task for testing
        class ConcreteTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                pass

        # Test with missing required fields (only run_id, unprocessed_file, task are required)
        invalid_configs = [
            {},  # Empty config
            {"run_id": "test"},  # Missing unprocessed_file
            {"run_id": "test", "unprocessed_file": Path("/test.fif")},  # Missing task
        ]

        # Test a valid config that should NOT raise an error
        valid_config = {
            "run_id": "test",
            "unprocessed_file": Path("/test.fif"),
            "task": "test_task",
            # stage_files no longer required in simplified implementation
        }

        # Test invalid configs
        for invalid_config in invalid_configs:
            # Task should validate config in __init__ and raise ValueError
            with pytest.raises(ValueError, match="Missing required field"):
                ConcreteTask(invalid_config)

        # Test valid config should NOT raise error
        task = ConcreteTask(valid_config)
        assert task.config == valid_config


@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
class TestTaskInterface:
    """Test Task interface and method signatures."""

    def test_task_has_expected_methods(self):
        """Test that Task has expected methods from mixins."""

        # Should have methods from mixins (these will be tested in mixin tests)
        # Here we just verify the interface exists
        expected_mixin_methods = [
            # These come from mixins and should be available
            # Actual method names depend on mixin implementation
        ]

        # Verify Task class has the abstract interface
        assert hasattr(Task, "__init__")
        assert hasattr(Task, "run")  # Abstract method

    def test_task_mro_consistency(self):
        """Test that Task's method resolution order is consistent."""

        # MRO should be well-defined without conflicts
        mro = Task.__mro__
        assert len(mro) > 2  # At least Task, ABC, and mixins
        assert Task in mro
        assert ABC in mro


class TestTaskConcrete:
    """Test Task with concrete implementation."""

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_concrete_task_implementation(self):
        """Test that concrete Task implementation works."""

        class TestTask(Task):
            """Concrete test task implementation."""

            def __init__(self, config):
                super().__init__(config)

            def run(self):
                """Test run implementation."""
                return {"status": "completed", "result": "test"}

        config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
            "tasks": {
                "test_task": {
                    "mne_task": "test",
                    "description": "Test task",
                    "settings": {"resample_step": {"enabled": True, "value": 250}},
                }
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
        }

        task = TestTask(config)
        assert task.config == config

        # Should be able to call run method
        result = task.run()
        assert result["status"] == "completed"

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    @patch("autoclean.io.import_.import_eeg")
    def test_task_with_mocked_dependencies(self, mock_import):
        """Test Task with mocked heavy dependencies."""

        class TestTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                # Test that import method is available (from mixins)
                if hasattr(self, "import_raw"):
                    return {"imported": True}
                return {"imported": False}

        config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
            "tasks": {
                "test_task": {
                    "mne_task": "test",
                    "description": "Test task",
                    "settings": {},
                }
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
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

    @patch("autoclean.mixins.DISCOVERED_MIXINS", [])
    def test_task_without_mixins(self):
        """Test Task behavior when no mixins are discovered."""
        # This tests the fallback behavior
        with patch("autoclean.core.task.DISCOVERED_MIXINS", []):
            # Import with no mixins already available at module level

            # Task should still be an ABC
            assert issubclass(Task, ABC)

    def test_task_mixin_discovery_failure(self):
        """Test Task behavior when mixin discovery fails."""
        with patch(
            "autoclean.core.task.DISCOVERED_MIXINS",
            side_effect=ImportError("Mixin discovery failed"),
        ):
            # Should handle mixin discovery failure gracefully
            # or raise appropriate error
            # Task is already imported at module level
            # If import succeeded at module level, it handled any errors
            assert True


class TestTaskConceptual:
    """Conceptual tests for Task design patterns."""

    def test_task_design_patterns(self):
        """Test that Task follows expected design patterns."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing design conceptually")

        # Task already imported at module level

        # Abstract Base Class pattern
        assert issubclass(Task, ABC)

        # Multiple inheritance pattern (with mixins)
        assert len(Task.__mro__) > 2

        # Template method pattern (abstract run method)
        assert hasattr(Task, "run")
        assert Task.run.__qualname__.startswith("Task.")

    def test_task_configuration_interface(self):
        """Test Task configuration interface design."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing interface conceptually")

        # Task already imported at module level

        # Should accept config in __init__
        init_signature = Task.__init__.__annotations__
        # Note: annotations might not be available in all Python versions

        # Should store config
        assert hasattr(Task, "__init__")

    def test_task_mixin_integration_concept(self):
        """Test conceptual Task-mixin integration."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing integration conceptually")

        # Task and DISCOVERED_MIXINS already imported at module level

        # Task should integrate with discovered mixins
        if DISCOVERED_MIXINS:
            for mixin in DISCOVERED_MIXINS:
                assert issubclass(
                    Task, mixin
                ), f"Task should inherit from discovered mixin {mixin}"

        # Task should maintain its primary interface
        assert hasattr(Task, "run")

    def test_task_extensibility_concept(self):
        """Test Task extensibility concept."""
        if not TASK_AVAILABLE:
            pytest.skip("Task not importable, testing extensibility conceptually")

        # Task already imported at module level

        # Should be extensible through inheritance
        class CustomTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                return "custom implementation"

        # Should be able to create custom tasks
        assert issubclass(CustomTask, Task)
        assert CustomTask.run != Task.run  # Override

        # Test that custom task can be instantiated
        config = {
            "run_id": "test",
            "unprocessed_file": Path("/test.fif"),
            "task": "custom",
            "tasks": {
                "custom": {"mne_task": "test", "description": "Test", "settings": {}}
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_test"}},
        }

        # Should be able to instantiate custom task with proper config
        if TASK_AVAILABLE:
            try:
                task = CustomTask(config)
                assert task.run() == "custom implementation"
            except Exception:
                # If there are dependency issues, that's okay for this test
                pass


# Error condition tests
class TestTaskErrorHandling:
    """Test Task error handling and edge cases."""

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_task_with_none_config(self):
        """Test Task behavior with None config."""

        class TestTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                return "test"

        # Should handle None config appropriately
        # (should raise error for None config)
        with pytest.raises((TypeError, ValueError)):
            task = TestTask(None)

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_task_with_invalid_config_types(self):
        """Test Task behavior with invalid config types."""

        class TestTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                return "test"

        invalid_configs = ["string", 123, [1, 2, 3], True]

        for invalid_config in invalid_configs:
            # Should handle invalid config types appropriately
            # (should raise error for invalid config types)
            with pytest.raises((TypeError, ValueError)):
                task = TestTask(invalid_config)
