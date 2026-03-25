"""Unit tests for the Task base class."""

from abc import ABC
from pathlib import Path
from types import SimpleNamespace
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

    def test_task_config_validation_raises_on_empty_config(self):
        """Task should reject an empty config dict with a clear error."""

        class ConcreteTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                pass

        with pytest.raises(ValueError, match="Missing required field"):
            ConcreteTask({})

    def test_task_config_validation_raises_on_missing_unprocessed_file(self):
        """Task should reject config that has run_id but no unprocessed_file."""

        class ConcreteTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                pass

        with pytest.raises(ValueError, match="Missing required field"):
            ConcreteTask({"run_id": "test"})

    def test_task_config_validation_raises_on_missing_task_field(self):
        """Task should reject config that is missing the task name field."""

        class ConcreteTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                pass

        with pytest.raises(ValueError, match="Missing required field"):
            ConcreteTask({"run_id": "test", "unprocessed_file": Path("/test.fif")})

    def test_task_config_validation_accepts_minimal_valid_config(self):
        """Task should accept a config with only the three required fields."""

        class ConcreteTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                pass

        valid_config = {
            "run_id": "test",
            "unprocessed_file": Path("/test.fif"),
            "task": "test_task",
        }
        task = ConcreteTask(valid_config)
        assert task.config == valid_config


@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
class TestTaskInterface:
    """Test Task interface and method signatures."""

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

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_task_propagates_montage_and_flagged_file_defaults(self):
        """Task settings should populate runtime config with EEG system and move policy."""

        class TestTask(Task):
            def __init__(self, config):
                self.settings = {
                    "montage": {"enabled": True, "value": "standard_1020"},
                    "move_flagged_files": False,
                }
                super().__init__(config)

            def run(self):
                pass

        config = {
            "run_id": "test_run_123",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
        }

        task = TestTask(config)

        assert task.config["eeg_system"] == "standard_1020"
        assert task.config["move_flagged_files"] is False

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_task_defaults_move_flagged_files_to_true_without_setting(self):
        """Tasks without explicit move_flagged_files should default to True."""

        class TestTask(Task):
            def __init__(self, config):
                self.settings = {"montage": {"enabled": False}}
                super().__init__(config)

            def run(self):
                pass

        config = {
            "run_id": "test_run_124",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
        }

        task = TestTask(config)

        assert task.config["eeg_system"] == "auto"
        assert task.config["move_flagged_files"] is True

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    @patch("autoclean.core.task.save_raw_to_set")
    @patch("autoclean.core.task.import_eeg")
    def test_import_raw_flags_short_recordings_and_saves_stage(
        self, mock_import_eeg, mock_save_raw_to_set
    ):
        """Short imports should be flagged and exported as the post-import stage."""

        class TestTask(Task):
            def run(self):
                pass

        config = {
            "run_id": "test_run_125",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
        }
        task = TestTask(config)
        task.create_bids_path = lambda *args, **kwargs: None

        mock_raw = SimpleNamespace(duration=30.0)
        mock_import_eeg.return_value = mock_raw

        task.import_raw()

        assert task.raw is mock_raw
        assert task.flagged is True
        assert "less than 1 minute" in task.flagged_reasons[0]
        mock_save_raw_to_set.assert_called_once_with(
            raw=mock_raw,
            autoclean_dict=task.config,
            stage="post_import",
            flagged=True,
        )

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_get_raw_and_get_epochs_raise_when_not_initialized(self):
        """Accessors should fail clearly before data has been imported."""

        class TestTask(Task):
            def run(self):
                pass

        config = {
            "run_id": "test_run_126",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
        }
        task = TestTask(config)

        with pytest.raises(ValueError, match="Raw data is not available"):
            task.get_raw()

        with pytest.raises(ValueError, match="Epochs are not available"):
            task.get_epochs()





@pytest.mark.skipif(not TASK_AVAILABLE, reason="Task module not available for import")
class TestTaskMixinIntegration:
    """Test that Task correctly integrates with the mixin system."""

    def test_task_inherits_from_all_discovered_mixins(self):
        """Task should be a subclass of every mixin returned by the discovery system."""
        for mixin in DISCOVERED_MIXINS:
            assert issubclass(Task, mixin), f"Task should inherit from discovered mixin {mixin}"

    def test_subclass_run_override_is_called(self):
        """A concrete subclass's run() should be what gets invoked, not Task.run."""
        class CustomTask(Task):
            def run(self):
                return "custom implementation"

        config = {
            "run_id": "test",
            "unprocessed_file": Path("/test.fif"),
            "task": "custom",
        }
        task = CustomTask(config)
        assert task.run() == "custom implementation"


# Error condition tests
class TestTaskErrorHandling:
    """Test Task error handling and edge cases."""

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_task_with_none_config(self):
        """Task should raise TypeError when initialized with None instead of a dict."""

        class TestTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                return "test"

        with pytest.raises(TypeError):
            TestTask(None)

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    @patch("autoclean.core.task.save_raw_to_set")
    @patch("autoclean.core.task.import_eeg")
    def test_import_raw_stores_raw_on_task(self, mock_import_eeg, mock_save_raw_to_set):
        """import_raw() sets self.raw to the object returned by import_eeg."""

        class TestTask(Task):
            def run(self):
                pass

        config = {
            "run_id": "test_store",
            "unprocessed_file": Path("/path/to/test.fif"),
            "task": "test_task",
        }
        task = TestTask(config)
        task.create_bids_path = lambda *args, **kwargs: None

        mock_raw = SimpleNamespace(duration=120.0)
        mock_import_eeg.return_value = mock_raw

        task.import_raw()

        assert task.raw is mock_raw

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_get_raw_returns_raw_after_set(self):
        """get_raw() returns self.raw once it has been populated."""

        class TestTask(Task):
            def run(self):
                pass

        config = {
            "run_id": "test_get",
            "unprocessed_file": Path("/test.fif"),
            "task": "test_task",
        }
        task = TestTask(config)
        sentinel = object()
        task.raw = sentinel

        assert task.get_raw() is sentinel

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_get_epochs_returns_epochs_after_set(self):
        """get_epochs() returns self.epochs once it has been populated."""

        class TestTask(Task):
            def run(self):
                pass

        config = {
            "run_id": "test_epochs",
            "unprocessed_file": Path("/test.fif"),
            "task": "test_task",
        }
        task = TestTask(config)
        sentinel = object()
        task.epochs = sentinel

        assert task.get_epochs() is sentinel

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    def test_flagged_reasons_accumulate(self):
        """Multiple assignments to flagged_reasons append, not overwrite."""

        class TestTask(Task):
            def run(self):
                pass

        config = {
            "run_id": "test_flags",
            "unprocessed_file": Path("/test.fif"),
            "task": "test_task",
        }
        task = TestTask(config)
        task.flagged = True
        task.flagged_reasons.append("reason one")
        task.flagged_reasons.append("reason two")

        assert len(task.flagged_reasons) == 2
        assert "reason one" in task.flagged_reasons
        assert "reason two" in task.flagged_reasons

    @pytest.mark.skipif(
        not TASK_AVAILABLE, reason="Task module not available for import"
    )
    @pytest.mark.parametrize("invalid_config", ["string", 123, [1, 2, 3], True])
    def test_task_with_invalid_config_types_raises(self, invalid_config):
        """Task should raise TypeError for any non-dict config value."""

        class TestTask(Task):
            def __init__(self, config):
                super().__init__(config)

            def run(self):
                return "test"

        with pytest.raises(TypeError):
            TestTask(invalid_config)
