"""Test the testing utilities themselves."""


import mne
import numpy as np
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw
from tests.fixtures.test_utils import (
    BaseTestCase,
    EEGAssertions,
    MockOperations,
    TestDataManager,
    validate_bids_structure,
)


class TestBaseTestCase(BaseTestCase):
    """Test the BaseTestCase functionality."""

    def test_temp_directory_creation(self):
        """Test that temporary directories are created properly."""
        assert self.temp_dir.exists()
        assert self.autoclean_dir.exists()
        assert self.autoclean_dir.is_dir()

    def test_create_test_pipeline(self):
        """Test pipeline creation with test configuration."""
        # Skip this test for now since it requires the full autoclean module to be importable
        pytest.skip(
            "Pipeline creation test requires full autoclean import - to be implemented in integration tests"
        )

    def test_minimal_config_structure(self):
        """Test that minimal config has required structure."""
        config = self.get_minimal_config()

        required_keys = ["tasks", "stage_files", "database"]
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"

        assert "TestTask" in config["tasks"]
        assert "settings" in config["tasks"]["TestTask"]


class TestEEGAssertions:
    """Test EEG-specific assertion functions."""

    def test_assert_raw_properties_valid(self):
        """Test raw property assertions with valid data."""
        raw = create_synthetic_raw(duration=10.0, sfreq=500.0, n_channels=64)

        # Should not raise any assertions
        EEGAssertions.assert_raw_properties(
            raw, expected_sfreq=500.0, expected_n_channels=64, expected_duration=10.0
        )

    def test_assert_raw_properties_invalid_sfreq(self):
        """Test raw property assertions with invalid sampling frequency."""
        raw = create_synthetic_raw(sfreq=500.0)

        with pytest.raises(AssertionError, match="Expected sfreq"):
            EEGAssertions.assert_raw_properties(raw, expected_sfreq=1000.0)

    def test_assert_raw_properties_invalid_channels(self):
        """Test raw property assertions with invalid channel count."""
        raw = create_synthetic_raw(n_channels=64)

        with pytest.raises(AssertionError, match="Expected .* channels"):
            EEGAssertions.assert_raw_properties(raw, expected_n_channels=128)

    def test_assert_raw_properties_invalid_duration(self):
        """Test raw property assertions with invalid duration."""
        raw = create_synthetic_raw(duration=5.0)

        with pytest.raises(AssertionError, match="Expected duration"):
            EEGAssertions.assert_raw_properties(raw, expected_duration=20.0)

    def test_assert_data_quality_valid(self):
        """Test data quality assertions with valid data."""
        raw = create_synthetic_raw()

        # Should not raise
        EEGAssertions.assert_data_quality(
            raw, max_amplitude=500e-6, min_good_channels=100
        )

    def test_assert_data_quality_excessive_amplitude(self):
        """Test data quality assertions with excessive amplitude."""
        raw = create_synthetic_raw()

        with pytest.raises(AssertionError, match="amplitude .* exceeds maximum"):
            EEGAssertions.assert_data_quality(
                raw, max_amplitude=1e-6
            )  # Very strict limit

    def test_assert_data_quality_too_many_bad_channels(self):
        """Test data quality assertions with too many bad channels."""
        raw = create_synthetic_raw()
        raw.info["bads"] = raw.ch_names[:100]  # Mark many channels as bad

        with pytest.raises(AssertionError, match="good channels"):
            EEGAssertions.assert_data_quality(raw, min_good_channels=100)

    def test_assert_processing_outputs_valid(self, tmp_path):
        """Test processing output assertions with valid files."""
        # Create some test files
        (tmp_path / "test.txt").touch()
        (tmp_path / "data.csv").touch()
        (tmp_path / "result.json").touch()

        # Should not raise
        EEGAssertions.assert_processing_outputs(tmp_path, ["*.txt", "*.csv", "*.json"])

    def test_assert_processing_outputs_missing_files(self, tmp_path):
        """Test processing output assertions with missing files."""
        with pytest.raises(AssertionError, match="No files matching pattern"):
            EEGAssertions.assert_processing_outputs(tmp_path, ["*.missing"])


class TestMockOperations:
    """Test mock operation utilities."""

    def test_mock_ica_fit(self):
        """Test mock ICA fitting."""
        raw = create_synthetic_raw(n_channels=64)
        mock_ica = MockOperations.mock_ica_fit(raw, n_components=15)

        assert hasattr(mock_ica, "n_components")
        assert hasattr(mock_ica, "mixing_")
        assert hasattr(mock_ica, "unmixing_")
        assert mock_ica.n_components == 15
        assert mock_ica.mixing_.shape == (64, 15)

    def test_mock_ica_apply(self):
        """Test mock ICA apply method."""
        raw = create_synthetic_raw()
        mock_ica = MockOperations.mock_ica_fit(raw)

        # Apply should return a copy
        cleaned_raw = mock_ica.apply(raw, exclude=[0, 1])
        assert cleaned_raw is not raw  # Should be a copy
        assert mock_ica.exclude == [0, 1]

    def test_mock_autoreject_fit(self):
        """Test mock autoreject fitting."""
        raw = create_synthetic_raw(duration=10.0)
        # Create some mock epochs
        events = np.array(
            [[1000, 0, 1], [3000, 0, 1], [5000, 0, 1], [7000, 0, 1], [9000, 0, 1]]
        )
        epochs = mne.Epochs(
            raw, events, tmin=-0.2, tmax=0.5, preload=True, verbose=False
        )
        original_length = len(epochs)

        cleaned_epochs = MockOperations.mock_autoreject_fit(epochs)

        # Should return fewer epochs
        assert len(cleaned_epochs) <= original_length
        assert len(cleaned_epochs) > 0  # But not zero

    def test_mock_ransac_channels(self):
        """Test mock RANSAC channel detection."""
        raw = create_synthetic_raw(n_channels=64)
        bad_channels = MockOperations.mock_ransac_channels(raw, n_bad=5)

        assert len(bad_channels) == 5
        assert all(ch in raw.ch_names for ch in bad_channels)
        assert len(set(bad_channels)) == len(bad_channels)  # No duplicates


class TestTestDataManager:
    """Test the TestDataManager utility."""

    def test_data_manager_context(self):
        """Test data manager as context manager."""
        with TestDataManager() as manager:
            assert manager.base_dir.exists()
            # Manager should clean up automatically

    def test_data_manager_file_creation(self):
        """Test that data manager creates test files."""
        with TestDataManager() as manager:
            # This should trigger file creation
            file_path = manager.get_test_file("resting_129ch_raw")
            assert file_path.exists()
            assert file_path.suffix == ".fif"  # MNE saves as .fif format


class TestBIDSValidation:
    """Test BIDS structure validation."""

    def test_validate_bids_structure_valid(self, tmp_path):
        """Test BIDS validation with valid structure."""
        # Create minimal BIDS structure
        (tmp_path / "sourcedata").mkdir()
        (tmp_path / "derivatives").mkdir()
        (tmp_path / "dataset_description.json").touch()

        assert validate_bids_structure(tmp_path) is True

    def test_validate_bids_structure_invalid(self, tmp_path):
        """Test BIDS validation with invalid structure."""
        # Empty directory
        assert validate_bids_structure(tmp_path) is False

        # Missing required directories
        (tmp_path / "dataset_description.json").touch()
        assert validate_bids_structure(tmp_path) is False
