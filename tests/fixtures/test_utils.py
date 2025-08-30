"""Core testing utilities and base classes for AutoClean EEG tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from mne.io import Raw

from autoclean.core.pipeline import Pipeline


class BaseTestCase:
    """Base test case class with common testing utilities."""

    def setup_method(self):
        """Set up each test method."""
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        self.autoclean_dir = self.temp_dir / "autoclean_output"
        self.autoclean_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up after each test method."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_pipeline(self, config: Optional[Dict[str, Any]] = None) -> Pipeline:
        """Create a test pipeline with minimal configuration.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If None, uses minimal config.

        Returns
        -------
        Pipeline
            Configured pipeline instance
        """
        if config is None:
            config = self.get_minimal_config()

        # Save config to temporary file
        config_file = self.temp_dir / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return Pipeline(output_dir=str(self.autoclean_dir))

    def get_minimal_config(self) -> Dict[str, Any]:
        """Get minimal configuration for testing."""
        return {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    "description": "Test task",
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250},
                        "filtering": {
                            "enabled": True,
                            "value": {"l_freq": 1, "h_freq": 50},
                        },
                        "reference_step": {"enabled": True, "value": "average"},
                        "montage": {"enabled": False},
                        "ICA": {"enabled": False},
                        "epoch_settings": {"enabled": False},
                    },
                }
            },
            "stage_files": {
                "post_import": True,
                "post_clean_raw": False,
                "post_ica": False,
                "post_epochs": False,
            },
            "database": {"enabled": False},
        }


class EEGAssertions:
    """Helper class for EEG-specific assertions."""

    @staticmethod
    def assert_raw_properties(
        raw: Raw,
        expected_sfreq: Optional[float] = None,
        expected_n_channels: Optional[int] = None,
        expected_duration: Optional[float] = None,
    ):
        """Assert Raw object has expected properties.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object to check
        expected_sfreq : float, optional
            Expected sampling frequency
        expected_n_channels : int, optional
            Expected number of channels
        expected_duration : float, optional
            Expected duration in seconds (±1s tolerance)
        """
        # Check if it's a Raw object (including RawArray, RawFIF, etc.)
        assert hasattr(raw, "info") and hasattr(
            raw, "get_data"
        ), f"Expected Raw-like object, got {type(raw)}"

        if expected_sfreq is not None:
            assert (
                raw.info["sfreq"] == expected_sfreq
            ), f"Expected sfreq {expected_sfreq}, got {raw.info['sfreq']}"

        if expected_n_channels is not None:
            assert (
                len(raw.ch_names) == expected_n_channels
            ), f"Expected {expected_n_channels} channels, got {len(raw.ch_names)}"

        if expected_duration is not None:
            actual_duration = raw.times[-1]
            assert (
                abs(actual_duration - expected_duration) <= 1.0
            ), f"Expected duration ~{expected_duration}s, got {actual_duration}s"

    @staticmethod
    def assert_epochs_properties(
        epochs,
        expected_n_epochs: Optional[int] = None,
        expected_tmin: Optional[float] = None,
        expected_tmax: Optional[float] = None,
    ):
        """Assert Epochs object has expected properties.

        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object to check
        expected_n_epochs : int, optional
            Expected number of epochs (minimum)
        expected_tmin : float, optional
            Expected epoch start time
        expected_tmax : float, optional
            Expected epoch end time
        """
        assert hasattr(
            epochs, "get_data"
        ), f"Expected Epochs object, got {type(epochs)}"

        if expected_n_epochs is not None:
            assert (
                len(epochs) >= expected_n_epochs
            ), f"Expected at least {expected_n_epochs} epochs, got {len(epochs)}"

        if expected_tmin is not None:
            assert (
                abs(epochs.tmin - expected_tmin) < 0.01
            ), f"Expected tmin {expected_tmin}, got {epochs.tmin}"

        if expected_tmax is not None:
            assert (
                abs(epochs.tmax - expected_tmax) < 0.01
            ), f"Expected tmax {expected_tmax}, got {epochs.tmax}"

    @staticmethod
    def assert_data_quality(
        raw: Raw, max_amplitude: float = 500e-6, min_good_channels: Optional[int] = None
    ):
        """Assert data quality metrics.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object to check
        max_amplitude : float
            Maximum allowed amplitude in Volts
        min_good_channels : int, optional
            Minimum number of good (non-bad) channels required
        """
        # Check amplitude ranges
        data = raw.get_data()
        max_amp = np.max(np.abs(data))
        assert (
            max_amp <= max_amplitude
        ), f"Data amplitude {max_amp:.2e} exceeds maximum {max_amplitude:.2e}"

        # Check for good channels
        if min_good_channels is not None:
            n_good_channels = len(raw.ch_names) - len(raw.info["bads"])
            assert (
                n_good_channels >= min_good_channels
            ), f"Only {n_good_channels} good channels, need at least {min_good_channels}"

    @staticmethod
    def assert_processing_outputs(output_dir: Path, expected_files: List[str]):
        """Assert expected processing output files exist.

        Parameters
        ----------
        output_dir : Path
            Output directory to check
        expected_files : List[str]
            List of expected file patterns (can use wildcards)
        """
        assert output_dir.exists(), f"Output directory {output_dir} does not exist"

        for file_pattern in expected_files:
            matching_files = list(output_dir.glob(file_pattern))
            assert (
                len(matching_files) > 0
            ), f"No files matching pattern '{file_pattern}' found in {output_dir}"


class MockOperations:
    """Mock implementations of heavy computational operations."""

    @staticmethod
    def mock_ica_fit(raw: Raw, n_components: int = 15) -> object:
        """Create a mock ICA object without fitting.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data (for getting dimensions)
        n_components : int
            Number of ICA components

        Returns
        -------
        object
            Mock ICA object with required attributes
        """

        class MockICA:
            def __init__(self, n_components, n_channels):
                self.n_components = n_components
                self.n_channels = n_channels
                self.mixing_ = np.random.randn(n_channels, n_components)
                self.unmixing_ = np.linalg.pinv(self.mixing_)
                self.labels_ = {}
                self.exclude = []

            def apply(self, raw, exclude=None):
                """Mock apply method."""
                if exclude is not None:
                    self.exclude = exclude
                return raw.copy()

            def get_components(self):
                """Mock get_components method."""
                return self.mixing_

        n_channels = len(raw.ch_names)
        n_components = min(n_components, n_channels)
        return MockICA(n_components, n_channels)

    @staticmethod
    def mock_autoreject_fit(epochs, verbose=False):
        """Mock autoreject fitting - just return a subset of epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        verbose : bool
            Verbose flag (ignored)

        Returns
        -------
        mne.Epochs
            Subset of input epochs
        """
        # Return 80% of epochs to simulate rejection
        n_keep = int(0.8 * len(epochs))
        if n_keep == 0:
            n_keep = len(epochs)  # Keep at least something
        return epochs[:n_keep]

    @staticmethod
    def mock_ransac_channels(raw: Raw, n_bad: int = 2) -> List[str]:
        """Mock RANSAC channel detection - mark some channels as bad.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw data
        n_bad : int
            Number of channels to mark as bad

        Returns
        -------
        List[str]
            List of bad channel names
        """
        if n_bad >= len(raw.ch_names):
            n_bad = max(1, len(raw.ch_names) // 4)  # At most 25% bad

        # Pick channels to mark as bad (avoid first and last for consistency)
        bad_indices = np.linspace(1, len(raw.ch_names) - 2, n_bad, dtype=int)
        return [raw.ch_names[i] for i in bad_indices]

    @staticmethod
    def mock_ica(raw: Raw, n_components: int = 15):
        """Mock ICA fitting for integration tests."""
        return MockOperations.mock_ica_fit(raw, n_components)

    @staticmethod
    def mock_apply_ica(raw: Raw, ica_object=None, exclude=None):
        """Mock ICA application."""
        return raw.copy()

    @staticmethod
    def mock_autoreject(epochs, verbose=False):
        """Mock autoreject for integration tests."""
        return MockOperations.mock_autoreject_fit(epochs, verbose)

    @staticmethod
    def mock_apply_autoreject(epochs, autoreject_object=None):
        """Mock autoreject application."""
        return MockOperations.mock_autoreject_fit(epochs)

    @staticmethod
    def mock_ransac(raw: Raw, n_bad: int = 2):
        """Mock RANSAC for integration tests."""
        return MockOperations.mock_ransac_channels(raw, n_bad)


def setup_test_data_files(test_data_dir: Path) -> Dict[str, Path]:
    """Set up synthetic test data files if they don't exist.

    Parameters
    ----------
    test_data_dir : Path
        Directory for test data files

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping data types to file paths
    """
    from tests.fixtures.synthetic_data import save_synthetic_data_files

    # Create test data directory
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data files
    return save_synthetic_data_files(test_data_dir, overwrite=False)


def validate_bids_structure(bids_root: Path) -> bool:
    """Validate that output follows BIDS structure.

    Parameters
    ----------
    bids_root : Path
        Root of BIDS directory

    Returns
    -------
    bool
        True if valid BIDS structure
    """
    required_dirs = ["sourcedata", "derivatives"]
    required_files = ["dataset_description.json"]

    # Check required directories
    for dir_name in required_dirs:
        if not (bids_root / dir_name).exists():
            return False

    # Check for some expected files (lenient check)
    has_some_files = any((bids_root / f).exists() for f in required_files)

    return has_some_files


class TestDataManager:
    """Manager for test data files and cleanup."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize test data manager.

        Parameters
        ----------
        base_dir : Path, optional
            Base directory for test data. If None, uses temporary directory.
        """
        if base_dir is None:
            self.base_dir = Path(tempfile.mkdtemp())
            self._cleanup_needed = True
        else:
            self.base_dir = base_dir
            self._cleanup_needed = False

        self.data_files = {}

    def get_test_file(self, file_type: str) -> Path:
        """Get path to test file, creating if necessary.

        Parameters
        ----------
        file_type : str
            Type of test file ('resting_129ch', 'chirp_129ch', etc.)

        Returns
        -------
        Path
            Path to test file
        """
        if file_type not in self.data_files:
            self._create_test_files()

        if file_type not in self.data_files:
            raise ValueError(f"Unknown test file type: {file_type}")

        return self.data_files[file_type]

    def _create_test_files(self):
        """Create all test data files."""
        data_dir = self.base_dir / "test_data"
        self.data_files = setup_test_data_files(data_dir)

    def cleanup(self):
        """Clean up temporary files if needed."""
        if self._cleanup_needed and self.base_dir.exists():
            shutil.rmtree(self.base_dir, ignore_errors=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
