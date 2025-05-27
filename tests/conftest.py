"""Pytest configuration and shared fixtures for AutoClean EEG tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import yaml

import pytest
import numpy as np
import mne
from mne.io import RawArray

from tests.fixtures.synthetic_data import (
    create_synthetic_raw,
    create_synthetic_events,
    create_test_montage_info
)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory path."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session") 
def test_config_dir() -> Path:
    """Get the test config directory path."""
    return Path(__file__).parent / "fixtures" / "configs"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def temp_autoclean_dir(temp_dir: Path) -> Path:
    """Create a temporary autoclean output directory structure."""
    autoclean_dir = temp_dir / "autoclean_output"
    autoclean_dir.mkdir()
    
    # Create expected subdirectories
    (autoclean_dir / "logs").mkdir()
    (autoclean_dir / "derivatives").mkdir()
    (autoclean_dir / "sourcedata").mkdir()
    
    return autoclean_dir


@pytest.fixture(scope="session")
def synthetic_raw_129() -> mne.io.Raw:
    """Create synthetic Raw data with GSN-HydroCel-129 montage."""
    return create_synthetic_raw(
        montage="GSN-HydroCel-129",
        n_channels=129,
        duration=60.0,
        sfreq=1000.0
    )


@pytest.fixture(scope="session")
def synthetic_raw_128() -> mne.io.Raw:
    """Create synthetic Raw data with GSN-HydroCel-128 montage."""
    return create_synthetic_raw(
        montage="GSN-HydroCel-128", 
        n_channels=128,
        duration=60.0,
        sfreq=1000.0
    )


@pytest.fixture(scope="session")
def synthetic_raw_1020() -> mne.io.Raw:
    """Create synthetic Raw data with standard 10-20 montage."""
    return create_synthetic_raw(
        montage="standard_1020",
        n_channels=64,
        duration=60.0,
        sfreq=1000.0
    )


@pytest.fixture
def resting_events() -> np.ndarray:
    """Create empty events array for resting state data."""
    return create_synthetic_events(
        paradigm="resting",
        duration=60.0,
        sfreq=1000.0
    )


@pytest.fixture
def chirp_events() -> np.ndarray:
    """Create chirp stimulus events."""
    return create_synthetic_events(
        paradigm="chirp",
        duration=60.0,
        sfreq=1000.0
    )


@pytest.fixture
def mmn_events() -> np.ndarray:
    """Create MMN standard/deviant events."""
    return create_synthetic_events(
        paradigm="mmn",
        duration=60.0,
        sfreq=1000.0
    )


@pytest.fixture
def basic_config() -> Dict[str, Any]:
    """Basic configuration for testing."""
    return {
        "tasks": {
            "TestResting": {
                "mne_task": "rest",
                "description": "Test resting state task",
                "settings": {
                    "resample_step": {"enabled": True, "value": 250},
                    "filtering": {
                        "enabled": True,
                        "value": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60]}
                    },
                    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
                    "reference_step": {"enabled": True, "value": "average"},
                    "epoch_settings": {
                        "enabled": True,
                        "value": {"tmin": -1, "tmax": 1},
                        "event_id": None
                    }
                }
            }
        },
        "stage_files": {
            "post_import": True,
            "post_clean_raw": True,
            "post_ica": True,
            "post_epochs": True
        }
    }


@pytest.fixture
def test_config_file(test_config_dir: Path, basic_config: Dict[str, Any]) -> Path:
    """Create a temporary config file for testing."""
    config_file = test_config_dir / "test_config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(basic_config, f)
    
    return config_file


@pytest.fixture
def mock_heavy_operations(monkeypatch):
    """Mock heavy computational operations for faster testing."""
    
    def mock_run_ica(self, *args, **kwargs):
        """Mock ICA fitting - just add fake ICA object."""
        n_components = min(15, len(self.raw.ch_names))
        self.ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        # Create fake mixing matrix
        self.ica.mixing_ = np.random.randn(len(self.raw.ch_names), n_components)
        self.ica.unmixing_ = np.linalg.pinv(self.ica.mixing_)
        return self
    
    def mock_run_ransac(self, *args, **kwargs):
        """Mock RANSAC channel cleaning - mark a few channels as bad."""
        if len(self.raw.ch_names) > 10:
            self.raw.info['bads'] = [self.raw.ch_names[0], self.raw.ch_names[-1]]
        return self
    
    def mock_autoreject(self, epochs, *args, **kwargs):
        """Mock autoreject - just return epochs with a few dropped."""
        if len(epochs) > 5:
            return epochs[:-2]  # Drop last 2 epochs
        return epochs
    
    # Apply mocks
    monkeypatch.setattr("autoclean.mixins.signal_processing.ica.ICAMixin.run_ica", mock_run_ica)
    monkeypatch.setattr("autoclean.mixins.signal_processing.basic_steps.BasicStepsMixin.run_ransac", mock_run_ransac)
    

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables and settings."""
    # Suppress MNE verbose output during tests
    monkeypatch.setenv("MNE_LOGGING_LEVEL", "ERROR")
    
    # Set test-specific configurations
    monkeypatch.setenv("AUTOCLEAN_TEST_MODE", "1")