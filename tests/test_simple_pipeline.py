"""Test simple pipeline for resting state data processing."""

import json
import os
from pathlib import Path

import requests

from autoclean import Pipeline


def download_eeg_from_googledrive(url: str) -> Path:
    """Download EEG file from Google Drive link and return its path.

    Args:
        url: Google Drive sharing URL

    Returns:
        Path to downloaded file
    """
    # Extract file ID from Google Drive URL
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Download to current directory
    output_path = Path("test_raw.raw")

    print(f"Downloading from URL: {download_url}")

    # Use a session to handle redirects
    session = requests.Session()
    response = session.get(download_url, stream=True)
    response.raise_for_status()

    # Check content type
    content_type = response.headers.get("content-type", "")
    print(f"Content-Type: {content_type}")
    print(f"Response Headers: {json.dumps(dict(response.headers), indent=2)}")

    # Save the file
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Debug file information
    print(f"\nFile details:")
    print(f"File exists: {output_path.exists()}")
    print(f"File size: {output_path.stat().st_size} bytes")

    # Try to read first few bytes to check format
    with open(output_path, "rb") as f:
        header_bytes = f.read(16)
        print(f"First 16 bytes (hex): {header_bytes.hex()}")
        try:
            print(
                f"First 16 bytes (ascii): {header_bytes.decode('ascii', errors='replace')}"
            )
        except:
            print("Could not decode as ASCII")

    return output_path


def test_simple_resting_pipeline():
    """Test basic pipeline functionality with resting state data."""

    # Download sample data and get its path
    googledrive_url = "https://drive.google.com/file/d/1Fvztzx0PfYD0IeW3R8UPUYHL4-eKSbyD/view?usp=sharing"
    raw_file = download_eeg_from_googledrive(googledrive_url)

    # Initialize and run pipeline
    pipeline = Pipeline(
        autoclean_dir="outputs", autoclean_config="configs/autoclean_config.yaml"
    )

    pipeline.process_file(file_path=raw_file, task="RestingEyesOpen")
