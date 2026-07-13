"""Tests for signal processing mixin classes with BaseEpochs support."""

import mne
import numpy as np
import pytest


class TestGFPCleanEpochsMixin:
    """Test GFP cleaning accepts BaseEpochs subclasses."""

    def test_gfp_clean_epochs_rejects_non_epochs(self):
        """GFP cleaning should reject non-Epochs data."""
        from autoclean.mixins.signal_processing.gfp_clean_epochs import (
            GFPCleanEpochsMixin,
        )

        class MockPipeline(GFPCleanEpochsMixin):
            def __init__(self):
                self.epochs = None

            def _get_data_object(self, data, use_epochs=False):
                return data if data is not None else self.epochs

            def _check_step_enabled(self, step_name):
                return (True, None)

            def _update_metadata(self, *args, **kwargs):
                pass

            def _update_instance_data(self, *args, **kwargs):
                pass

            def _save_epochs_result(self, *args, **kwargs):
                pass

            def _auto_export_if_enabled(self, *args, **kwargs):
                pass

        pipeline = MockPipeline()

        # Should raise TypeError for raw data
        with pytest.raises(TypeError):
            pipeline.gfp_clean_epochs(epochs="not_epochs")

    def test_gfp_clean_epochs_accepts_baseepochs_type(self):
        """GFP cleaning should accept BaseEpochs subclasses."""
        from autoclean.mixins.signal_processing.gfp_clean_epochs import (
            GFPCleanEpochsMixin,
        )

        class MockPipeline(GFPCleanEpochsMixin):
            def __init__(self):
                self.epochs = None

            def _get_data_object(self, data, use_epochs=False):
                return data if data is not None else self.epochs

            def _check_step_enabled(self, step_name):
                return (True, None)

            def _update_metadata(self, *args, **kwargs):
                pass

            def _update_instance_data(self, *args, **kwargs):
                pass

            def _save_epochs_result(self, *args, **kwargs):
                pass

            def _auto_export_if_enabled(self, *args, **kwargs):
                pass

        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        info = mne.create_info(n_channels, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        pipeline = MockPipeline()
        result = pipeline.gfp_clean_epochs(epochs=epochs)
        assert isinstance(result, mne.BaseEpochs)


class TestOutlierDetectionMixin:
    """Test outlier detection accepts BaseEpochs subclasses."""

    def test_detect_outlier_epochs_rejects_non_epochs(self):
        """Outlier detection should reject non-Epochs data."""
        from autoclean.mixins.signal_processing.outlier_detection import (
            OutlierDetectionMixin,
        )

        class MockPipeline(OutlierDetectionMixin):
            def __init__(self):
                self.epochs = None

            def _get_data_object(self, data, use_epochs=False):
                return data if data is not None else self.epochs

            def _check_step_enabled(self, step_name):
                return (True, None)

            def _update_metadata(self, *args, **kwargs):
                pass

            def _update_instance_data(self, *args, **kwargs):
                pass

            def _save_epochs_result(self, *args, **kwargs):
                pass

            def _auto_export_if_enabled(self, *args, **kwargs):
                pass

        pipeline = MockPipeline()

        # Should raise TypeError for raw data
        with pytest.raises(TypeError):
            pipeline.detect_outlier_epochs(epochs="not_epochs")

    def test_detect_outlier_epochs_accepts_baseepochs_type(self):
        """Outlier detection should accept BaseEpochs subclasses."""
        from autoclean.mixins.signal_processing.outlier_detection import (
            OutlierDetectionMixin,
        )

        class MockPipeline(OutlierDetectionMixin):
            def __init__(self):
                self.epochs = None

            def _get_data_object(self, data, use_epochs=False):
                return data if data is not None else self.epochs

            def _check_step_enabled(self, step_name):
                """Mock implementation - always enabled."""
                return (True, None)

            def _update_metadata(self, *args, **kwargs):
                pass

            def _update_instance_data(self, *args, **kwargs):
                pass

            def _save_epochs_result(self, *args, **kwargs):
                pass

            def _auto_export_if_enabled(self, *args, **kwargs):
                pass

        n_channels, n_samples, n_epochs = 4, 256, 10
        data = np.random.randn(n_epochs, n_channels, n_samples)
        info = mne.create_info(n_channels, 256, "eeg")
        epochs = mne.EpochsArray(data, info)

        pipeline = MockPipeline()
        result = pipeline.detect_outlier_epochs(epochs=epochs)
        assert isinstance(result, mne.BaseEpochs)
