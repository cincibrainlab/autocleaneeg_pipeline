"""Test suite for the autoclean pipeline functionality."""

import pytest
from pathlib import Path
import mne
import numpy as np
from autoclean import Pipeline
import yaml
import os
import requests
from urllib.parse import urlparse
import shutil


def download_test_file(url: str, output_path: Path) -> Path:
    """Download test file if it doesn't exist locally.

    Args:
        url: URL to download from (can be file://, http://, or s3://)
        output_path: Where to save the file
    """
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If file exists and not in CI, use it
    if output_path.exists() and not os.getenv("CI"):
        # For .set files, also check for .fdt
        if output_path.suffix == ".set":
            fdt_path = output_path.with_suffix(".fdt")
            if fdt_path.exists():
                return output_path
        else:
            return output_path

    # Parse URL scheme
    scheme = urlparse(url).scheme

    if scheme in ("http", "https"):
        # Download from web URL
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # For .set files, also download .fdt
        if output_path.suffix == ".set":
            fdt_url = url.replace(".set", ".fdt")
            fdt_path = output_path.with_suffix(".fdt")
            response = requests.get(fdt_url, stream=True)
            response.raise_for_status()
            with open(fdt_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    elif scheme == "s3":
        # Download from S3
        import boto3

        s3 = boto3.client("s3")
        bucket = urlparse(url).netloc
        key = urlparse(url).path.lstrip("/")
        s3.download_file(bucket, key, str(output_path))

        # For .set files, also download .fdt
        if output_path.suffix == ".set":
            fdt_key = key.replace(".set", ".fdt")
            fdt_path = output_path.with_suffix(".fdt")
            s3.download_file(bucket, fdt_key, str(fdt_path))

    elif scheme == "file":
        # Local file (useful for development)
        src_path = Path(urlparse(url).path)
        shutil.copy2(src_path, output_path)

        # For .set files, also copy .fdt
        if output_path.suffix == ".set":
            src_fdt = src_path.with_suffix(".fdt")
            dst_fdt = output_path.with_suffix(".fdt")
            if src_fdt.exists():
                shutil.copy2(src_fdt, dst_fdt)
            else:
                raise FileNotFoundError(f"Required .fdt file not found: {src_fdt}")

    else:
        raise ValueError(f"Unsupported URL scheme: {scheme}")

    return output_path


@pytest.fixture(scope="session")
def test_raw_data():
    """Get test EEG data, using real file if available, otherwise synthetic."""
    # Test file locations (in order of preference)
    test_file_urls = [
        # Local development path
        f"file://{os.getenv('TEST_EEG_PATH')}" if os.getenv("TEST_EEG_PATH") else None,
        # S3 path (if configured)
        os.getenv("TEST_EEG_S3_URL"),
        # Public HTTP URL (if available)
        os.getenv("TEST_EEG_HTTP_URL"),
    ]

    # Try each URL in order
    for url in test_file_urls:
        if not url:
            continue

        try:
            # Note: Now expecting .set file
            output_path = Path("tests/test_data/sample_eeg/test_raw.set")
            downloaded_file = download_test_file(url, output_path)

            # Convert to fif format for consistent handling
            raw = mne.io.read_raw_eeglab(downloaded_file, preload=True)
            fif_path = output_path.with_suffix(".fif")
            raw.save(fif_path, overwrite=True)
            return fif_path

        except Exception as e:
            print(f"Failed to get test file from {url}: {e}")
            continue

    # If no real file available, create synthetic data
    print("Using synthetic data for testing")
    sfreq = 250  # 250 Hz sampling rate
    t = np.arange(0, 10, 1 / sfreq)  # 10 seconds of data
    n_channels = 128

    # Generate random data with realistic properties
    data = np.random.randn(n_channels, len(t)) * 20  # ~20µV amplitude

    # Add some artifacts
    # Eye blink artifact
    blink_idx = np.random.randint(0, len(t) - 250, 5)  # 5 blinks
    for idx in blink_idx:
        data[0:10, idx : idx + 250] += 100 * np.sin(np.pi * np.arange(250) / 250)

    # Create MNE Raw object
    ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Save to a temporary file
    temp_dir = Path("tests/test_data")
    temp_dir.mkdir(exist_ok=True)
    test_file = temp_dir / "synthetic_raw.fif"
    raw.save(test_file, overwrite=True)

    return test_file


@pytest.fixture(scope="session")
def test_config(test_raw_data):
    """Create a test configuration based on the test data."""
    # Load test file to get properties
    raw = mne.io.read_raw_fif(test_raw_data, preload=False)
    n_channels = len(raw.ch_names)

    return {
        "tasks": {
            "test_task": {
                "mne_task": "test",
                "description": "Test task",
                "lossless_config": "configs/lossless_config.yaml",
                "settings": {
                    "resample_step": {"enabled": True, "value": 250},
                    "drop_outerlayer": {"enabled": False, "value": None},
                    "eog_step": {
                        "enabled": True,
                        "value": raw.ch_names[:3],
                    },  # Use first 3 channels for EOG
                    "trim_step": {"enabled": True, "value": 1},
                    "reference_step": {"enabled": True, "value": "average"},
                    "montage": {"enabled": True, "value": "GSN-HydroCel-128"},
                },
            }
        },
        "stage_files": {
            "post_import": {"enabled": True, "suffix": "_imported"},
            "post_prepipeline": {"enabled": True, "suffix": "_preprocessed"},
            "post_clean": {"enabled": True, "suffix": "_clean"},
        },
    }


def test_pipeline_initialization(tmp_path, test_config):
    """Test pipeline initialization."""
    # Save test config
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Initialize pipeline
    pipeline = Pipeline(autoclean_dir=str(tmp_path), autoclean_config=str(config_path))

    assert pipeline is not None
    assert pipeline.autoclean_dir == tmp_path
    assert "test_task" in pipeline.list_tasks()


def test_pipeline_processing(test_raw_data, tmp_path, test_config):
    """Test complete pipeline processing."""
    # Save test config
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Initialize pipeline
    pipeline = Pipeline(autoclean_dir=str(tmp_path), autoclean_config=str(config_path))

    # Process file
    try:
        pipeline.process_file(file_path=str(test_raw_data), task="test_task")
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"Pipeline processing failed: {str(e)}")

    assert success

    # Check output files exist
    output_dir = tmp_path / "clean"
    assert output_dir.exists()
    assert len(list(output_dir.glob("*_clean.fif"))) > 0


def test_pipeline_error_handling(tmp_path, test_config):
    """Test pipeline error handling with invalid data."""
    # Save test config
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Initialize pipeline
    pipeline = Pipeline(autoclean_dir=str(tmp_path), autoclean_config=str(config_path))

    # Try to process non-existent file
    with pytest.raises(FileNotFoundError):
        pipeline.process_file(file_path="nonexistent.fif", task="test_task")


def test_pipeline_data_quality(test_raw_data, tmp_path, test_config):
    """Test pipeline output data quality."""
    # Save test config
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # Initialize pipeline
    pipeline = Pipeline(autoclean_dir=str(tmp_path), autoclean_config=str(config_path))

    # Process file
    pipeline.process_file(file_path=str(test_raw_data), task="test_task")

    # Load output file
    output_files = list((tmp_path / "clean").glob("*_clean.fif"))
    assert len(output_files) > 0

    cleaned_raw = mne.io.read_raw_fif(output_files[0], preload=True)

    # Quality checks
    assert cleaned_raw.info["sfreq"] == 250  # Check resampling
    assert len(cleaned_raw.annotations) > 0  # Check annotations exist
    assert cleaned_raw.info["custom_ref_applied"]  # Check referencing

    # Check data is not all zeros or NaNs
    data = cleaned_raw.get_data()
    assert not np.all(data == 0)
    assert not np.any(np.isnan(data))

    # Check reasonable amplitude range (±100µV)
    assert np.all(np.abs(data) < 100)
