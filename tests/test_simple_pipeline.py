"""Test simple pipeline for resting state data processing."""

from pathlib import Path

import requests

from autoclean import Pipeline


def download_eeg_from_googledrive(url: str) -> Path:
    """Download EEG file from Google Drive link and return its path.

    Args:
        url: OneDrive sharing URL

    Returns:
        Path to downloaded file
    """
    # Download to current directory
    output_path = Path("test_raw.raw")

    response = requests.get(url, allow_redirects=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def test_simple_resting_pipeline():
    """Test basic pipeline functionality with resting state data."""

    # Download sample data and get its path
    onedrive_url = "https://drive.google.com/file/d/1Fvztzx0PfYD0IeW3R8UPUYHL4-eKSbyD/view?usp=sharing"
    raw_file = download_eeg_from_googledrive(onedrive_url)

    # Initialize and run pipeline
    pipeline = Pipeline(
        autoclean_dir="outputs", autoclean_config="configs/autoclean_config.yaml"
    )

    pipeline.process_file(file_path=raw_file, task="RestingEyesOpen")
