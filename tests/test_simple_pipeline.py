"""Test simple pipeline for resting state data processing."""

from pathlib import Path

import requests

from autoclean import Pipeline


def download_eeg_from_onedrive(url: str) -> Path:
    """Download EEG file from OneDrive link and return its path.

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
    onedrive_url = "https://cchmc-my.sharepoint.com/:i:/g/personal/gavin_gammoh_cchmc_org/EZWyFUvk1hRHuHWa8WvAQGUBlmVK90F2XRxnrCqr1JusWg?e=2DkfBf"
    raw_file = download_eeg_from_onedrive(onedrive_url)

    # Initialize and run pipeline
    pipeline = Pipeline(
        autoclean_dir="outputs", autoclean_config="configs/autoclean_config.yaml"
    )

    pipeline.process_file(file_path=raw_file, task="RestingEyesOpen")
