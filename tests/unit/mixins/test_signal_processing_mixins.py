"""Unit tests for signal processing mixins."""

from unittest.mock import Mock, patch

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw
from tests.fixtures.test_utils import EEGAssertions, MockOperations

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.mixins.signal_processing.basic_steps import BasicStepsMixin
    from autoclean.mixins.signal_processing.ica import IcaMixin

    SIGNAL_PROCESSING_AVAILABLE = True
except ImportError:
    SIGNAL_PROCESSING_AVAILABLE = False
    BasicStepsMixin = None
    IcaMixin = None


@pytest.mark.skipif(
    not SIGNAL_PROCESSING_AVAILABLE, reason="Signal processing mixins not available"
)
class TestBasicStepsMixin:
    """Test the BasicStepsMixin functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.raw = create_synthetic_raw(duration=10.0, sfreq=1000.0)
        self.config = {
            "tasks": {
                "test_task": {
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250},
                        "filtering": {
                            "enabled": True,
                            "value": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60]},
                        },
                        "trim_step": {"enabled": True, "value": 2},
                        "crop_step": {
                            "enabled": False,
                            "value": {"start": 0, "end": None},
                        },
                        "drop_outerlayer": {"enabled": False, "value": []},
                        "eog_step": {"enabled": False, "value": []},
                    }
                }
            }
        }


    def test_basic_steps_mixin_inheritance(self):
        """Test BasicStepsMixin can be inherited."""
        from autoclean.core.task import Task

        class TestClass(Task):
            def __init__(self):
                self.raw = create_synthetic_raw()
                self.config = {
                    "task": "test_task",
                    "tasks": {
                        "test_task": {
                            "settings": {
                                "resample_step": {"enabled": True, "value": 250},
                                "filtering": {"enabled": True, "value": None},
                                "drop_outerlayer": {"enabled": True, "value": []},
                                "eog_step": {"enabled": True, "value": []},
                                "trim_step": {"enabled": True, "value": 0},
                                "crop_step": {
                                    "enabled": True,
                                    "value": {"start": 0, "end": None},
                                },
                            }
                        }
                    },
                }
                self.epochs = None

            def run(self):
                return None

        test_instance = TestClass()

        # Should be able to call run_basic_steps
        assert hasattr(test_instance, "run_basic_steps")

        # Mock the individual step methods to avoid full processing
        with (
            patch.object(test_instance, "resample_data", return_value=self.raw),
            patch.object(test_instance, "filter_data", return_value=self.raw),
            patch.object(test_instance, "drop_outer_layer", return_value=self.raw),
            patch.object(test_instance, "assign_eog_channels", return_value=self.raw),
            patch.object(test_instance, "trim_edges", return_value=self.raw),
            patch.object(test_instance, "crop_duration", return_value=self.raw),
        ):

            result = test_instance.run_basic_steps()
            assert result is not None

    @patch("autoclean.utils.logging.message")
    def test_basic_steps_sequential_execution(self, mock_message):
        """Test that basic steps execute in correct sequence."""
        from autoclean.core.task import Task

        class TestClass(Task):
            def __init__(self):
                self.raw = create_synthetic_raw()
                self.execution_order = []
                self.config = {
                    "task": "test_task",
                    "tasks": {
                        "test_task": {
                            "settings": {
                                "resample_step": {"enabled": True, "value": 250},
                                "filtering": {
                                    "enabled": True,
                                    "value": {"l_freq": 1, "h_freq": 40},
                                },
                                "drop_outerlayer": {"enabled": True, "value": []},
                                "eog_step": {"enabled": True, "value": []},
                                "trim_step": {"enabled": True, "value": 0},
                                "crop_step": {
                                    "enabled": True,
                                    "value": {"start": 0, "end": None},
                                },
                            }
                        }
                    },
                }
                self.epochs = None

            def run(self):
                return None

            def resample_data(self, data=None, use_epochs=False):
                self.execution_order.append("resample")
                return data if data is not None else self.raw

            def filter_data(self, data=None, use_epochs=False):
                self.execution_order.append("filter")
                return data if data is not None else self.raw

            def drop_outer_layer(self, data=None, use_epochs=False):
                self.execution_order.append("drop_outerlayer")
                return data if data is not None else self.raw

            def assign_eog_channels(self, data=None, use_epochs=False):
                self.execution_order.append("assign_eog")
                return data if data is not None else self.raw

            def trim_edges(self, data=None, use_epochs=False):
                self.execution_order.append("trim")
                return data if data is not None else self.raw

            def crop_duration(self, data=None, use_epochs=False):
                self.execution_order.append("crop")
                return data if data is not None else self.raw

        test_instance = TestClass()
        test_instance.run_basic_steps()

        # Verify execution order
        expected_order = [
            "resample",
            "filter",
            "drop_outerlayer",
            "assign_eog",
            "trim",
            "crop",
        ]
        assert test_instance.execution_order == expected_order

    def test_basic_steps_data_parameter_handling(self):
        """Test that BasicStepsMixin handles data parameter correctly."""
        from autoclean.core.task import Task

        class TestClass(Task):
            def __init__(self):
                self.raw = create_synthetic_raw()
                self.received_data = None
                self.config = {
                    "task": "test_task",
                    "tasks": {
                        "test_task": {
                            "settings": {
                                "resample_step": {"enabled": True, "value": 250},
                                "filtering": {"enabled": True, "value": None},
                                "drop_outerlayer": {"enabled": True, "value": []},
                                "eog_step": {"enabled": True, "value": []},
                                "trim_step": {"enabled": True, "value": 0},
                                "crop_step": {
                                    "enabled": True,
                                    "value": {"start": 0, "end": None},
                                },
                            }
                        }
                    },
                }
                self.epochs = None

            def run(self):
                return None

            def _get_data_object(self, data=None, use_epochs=False):
                self.received_data = data
                return data if data is not None else self.raw

            def resample_data(self, data=None, use_epochs=False):
                return data

            def filter_data(self, data=None, use_epochs=False):
                return data

            def drop_outer_layer(self, data=None, use_epochs=False):
                return data

            def assign_eog_channels(self, data=None, use_epochs=False):
                return data

            def trim_edges(self, data=None, use_epochs=False):
                return data

            def crop_duration(self, data=None, use_epochs=False):
                return data

        test_instance = TestClass()

        # Test with explicit data parameter
        custom_raw = create_synthetic_raw(duration=5.0)
        result = test_instance.run_basic_steps(data=custom_raw)

        assert test_instance.received_data == custom_raw
        assert result == custom_raw

    @pytest.mark.parametrize("eog_value,expected_eog_indices", [
        ({"eog_indices": [1, 2], "eog_drop": False}, [0, 1]),  # dict: 1-based indices 1,2 → positions 0,1
        ([1, 2, 3], [0, 1, 2]),                                 # list: 1-based indices 1,2,3 → positions 0,1,2
        (None, []),                                              # None → no eog channels set
        ([], []),                                                # empty list → no eog channels set
    ])
    def test_assign_eog_channels_marks_correct_channels_as_eog(self, eog_value, expected_eog_indices):
        """assign_eog_channels should set the correct channels to EOG type for each config format."""
        from autoclean.core.task import Task

        class TestTask(Task):
            def __init__(self):
                self.raw = create_synthetic_raw(n_channels=10)
                self.config = {
                    "task": "test_task",
                    "tasks": {"test_task": {"settings": {"eog_step": {"enabled": True, "value": eog_value}}}},
                }

            def run(self):
                return None

        task = TestTask()
        original_types = task.raw.get_channel_types()
        result = task.assign_eog_channels()

        result_types = result.get_channel_types()
        for idx in expected_eog_indices:
            assert result_types[idx] == "eog", f"Channel at position {idx} should be 'eog' type"
        for idx in range(10):
            if idx not in expected_eog_indices:
                assert result_types[idx] == original_types[idx], (
                    f"Channel at position {idx} should be unchanged"
                )

    def test_resample_data_simple_number(self):
        """Test resample_data with simple number format (schema-compliant)."""
        from autoclean.core.task import Task

        class TestTask(Task):
            def __init__(self):
                self.raw = create_synthetic_raw(sfreq=1000.0)
                self.config = {"task": "test_task", "tasks": {"test_task": {"settings": {"resample_step": {"enabled": True, "value": 250}}}}}
                self.flagged = False

            def _save_raw_result(self, result_data, stage_name):
                return None

            def _update_metadata(self, operation, metadata_dict):
                return None

            def run(self):
                return None

        task = TestTask()
        result = task.resample_data()
        assert result is not None
        # Verify resampling occurred
        assert result.info["sfreq"] == 250

    def test_resample_data_dict_format(self):
        """Test resample_data with dict format (advanced options)."""
        from autoclean.core.task import Task

        class TestTask(Task):
            def __init__(self):
                self.raw = create_synthetic_raw(sfreq=1000.0)
                self.config = {
                    "task": "test_task",
                    "tasks": {
                        "test_task": {
                            "settings": {
                                "resample_step": {
                                    "enabled": True,
                                    "value": {"sfreq": 500, "npad": "auto"},
                                }
                            }
                        }
                    },
                }
                self.flagged = False

            def _save_raw_result(self, result_data, stage_name):
                return None

            def _update_metadata(self, operation, metadata_dict):
                return None

            def run(self):
                return None

        task = TestTask()
        result = task.resample_data()
        assert result is not None
        assert result.info["sfreq"] == 500

    def test_filter_data_simple_format(self):
        """Test filter_data with simple schema-compliant format."""
        from autoclean.core.task import Task

        class TestTask(Task):
            def __init__(self):
                self.raw = create_synthetic_raw(sfreq=500.0, duration=10.0)
                self.config = {
                    "task": "test_task",
                    "tasks": {
                        "test_task": {
                            "settings": {
                                "filtering": {
                                    "enabled": True,
                                    "value": {
                                        "l_freq": 1.0,
                                        "h_freq": 100.0,
                                        "notch_freqs": [60],
                                    },
                                }
                            }
                        }
                    }
                }
                self.flagged = False

            def _save_raw_result(self, result_data, stage_name):
                return None

            def _update_metadata(self, operation, metadata_dict):
                return None

            def run(self):
                return None

        task = TestTask()
        result = task.filter_data()
        assert result is not None
        # Verify filtering was applied (data object returned)
        assert hasattr(result, "info")

    def test_filter_data_defaults_applied(self):
        """Test filter_data applies sensible defaults for advanced parameters."""
        from autoclean.core.task import Task
        from unittest.mock import patch

        class TestTask(Task):
            def __init__(self):
                self.raw = create_synthetic_raw(sfreq=500.0, duration=10.0)
                self.config = {
                    "task": "test_task",
                    "tasks": {
                        "test_task": {
                            "settings": {
                                "filtering": {
                                    "enabled": True,
                                    "value": {
                                        "l_freq": 1.0,
                                        "h_freq": 100.0,
                                    },
                                }
                            }
                        }
                    }
                }
                self.flagged = False

            def _save_raw_result(self, result_data, stage_name):
                return None

            def _update_metadata(self, operation, metadata_dict):
                return None

            def run(self):
                return None

        task = TestTask()

        # Patch the standalone function to verify defaults are passed
        with patch('autoclean.mixins.signal_processing.basic_steps.standalone_filter_data') as mock_filter:
            mock_filter.return_value = task.raw
            task.filter_data()

            # Verify default parameters were used
            call_kwargs = mock_filter.call_args.kwargs
            assert call_kwargs.get('method') == 'fir', "Default method should be 'fir'"
            assert call_kwargs.get('phase') == 'zero', "Default phase should be 'zero'"
            assert call_kwargs.get('fir_window') == 'hamming', "Default fir_window should be 'hamming'"


@pytest.mark.skipif(
    not SIGNAL_PROCESSING_AVAILABLE, reason="Signal processing mixins not available"
)
class TestICAMixin:
    """Test the ICAMixin functionality."""

    def test_ica_mixin_exposes_run_ica_and_component_rejection(self):
        """IcaMixin should expose run_ica and apply_ica_component_rejection as callable methods."""
        assert callable(getattr(IcaMixin, "run_ica", None)), "IcaMixin missing run_ica"
        assert callable(getattr(IcaMixin, "apply_ica_component_rejection", None)), (
            "IcaMixin missing apply_ica_component_rejection"
        )






# Error handling tests
class TestSignalProcessingMixinsErrorHandling:
    """Test signal processing mixins error handling."""

    @pytest.mark.skipif(
        not SIGNAL_PROCESSING_AVAILABLE, reason="Signal processing mixins not available"
    )
    def test_basic_steps_error_handling(self):
        """Test BasicStepsMixin error handling."""

        class FailingClass(BasicStepsMixin):
            def _get_data_object(self, data=None, use_epochs=False):
                raise ValueError("No data available")

            def resample_data(self, data=None, use_epochs=False):
                return data

        test_instance = FailingClass()

        # Should propagate errors appropriately
        with pytest.raises(ValueError, match="No data available"):
            test_instance.run_basic_steps()

    @pytest.mark.skipif(
        not SIGNAL_PROCESSING_AVAILABLE, reason="Signal processing mixins not available"
    )
    def test_missing_method_error_handling(self):
        """Test error handling when required methods are missing."""

        class IncompleteClass(BasicStepsMixin):
            # Missing required methods
            pass

        test_instance = IncompleteClass()

        # Should raise AttributeError for missing methods
        with pytest.raises(AttributeError):
            test_instance.run_basic_steps()

    def test_invalid_data_handling(self):
        """Test handling of invalid data types."""
        if not SIGNAL_PROCESSING_AVAILABLE:
            pytest.skip("Signal processing mixins not available")

        # BasicStepsMixin already imported at module level

        class TestClass(BasicStepsMixin):
            def _get_data_object(self, data=None, use_epochs=False):
                return "invalid_data_type"  # Not a Raw object

            def resample_data(self, data=None, use_epochs=False):
                if not hasattr(data, "info"):
                    raise TypeError("Expected Raw object")
                return data

        test_instance = TestClass()

        # Should handle invalid data types appropriately
        with pytest.raises(TypeError, match="Expected Raw object"):
            test_instance.run_basic_steps()


# Performance and optimization tests
class TestSignalProcessingMixinsPerformance:
    """Test signal processing mixins performance considerations."""

    def test_basic_steps_performance_mocked(self):
        """Test BasicStepsMixin performance with mocked operations."""
        # This tests that the mixin doesn't add significant overhead
        mock_raw = create_synthetic_raw()

        call_count = 0

        def mock_step(self, data=None, use_epochs=False):
            nonlocal call_count
            call_count += 1
            return data if data is not None else mock_raw

        if SIGNAL_PROCESSING_AVAILABLE:
            # BasicStepsMixin already imported at module level

            class FastTestClass(BasicStepsMixin):
                def __init__(self):
                    self.raw = mock_raw
                    self.original_raw = None
                    self.config = {
                        "task": "test_task",
                        "tasks": {
                            "test_task": {
                                "settings": {
                                    "resample_step": {"enabled": True, "value": 250},
                                    "filtering": {"enabled": True, "value": None},
                                    "drop_outerlayer": {"enabled": True, "value": []},
                                    "eog_step": {"enabled": True, "value": []},
                                    "trim_step": {"enabled": True, "value": 0},
                                    "crop_step": {
                                        "enabled": True,
                                        "value": {"start": 0, "end": None},
                                    },
                                }
                            }
                        },
                    }

                def _get_data_object(self, data=None, use_epochs=False):
                    return data if data is not None else self.raw

                def _update_instance_data(self, original_data, processed_data, use_epochs):
                    self.raw = processed_data

                def _auto_export_if_enabled(self, processed_data, stage_name, export):
                    return None

                resample_data = mock_step
                filter_data = mock_step
                drop_outer_layer = mock_step
                assign_eog_channels = mock_step
                trim_edges = mock_step
                crop_duration = mock_step

            test_instance = FastTestClass()
            result = test_instance.run_basic_steps()

            # Should call all steps exactly once
            assert call_count == 6  # 6 basic steps
            assert result == mock_raw

