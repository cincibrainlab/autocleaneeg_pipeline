"""Test simple pipeline for resting state data processing."""

import numpy as np
import pytest
from pathlib import Path
import time

from autoclean import Pipeline
from .utils import get_test_file


@pytest.mark.timeout(300)  # 5 minute timeout
def test_simple_resting_pipeline():
    """Test basic pipeline functionality with resting state data."""
    
    print("\nStarting test_simple_resting_pipeline...")
    start_time = time.time()

    print("Downloading test file if needed...")
    test_file = get_test_file("0199_rest.raw")
    
    print(f"Test file path: {test_file}")
    print(f"Test file exists, size: {test_file.stat().st_size} bytes")
    
    # Debug: Check file format
    with open(test_file, 'rb') as fid:
        # Read first 4 bytes as little-endian int32
        version = np.fromfile(fid, "<i4", 1)[0]
        print(f"File version: {version} (hex: {hex(version)})")
        
        # Try big-endian if needed
        fid.seek(0)
        version_be = np.fromfile(fid, ">i4", 1)[0]
        print(f"File version (big-endian): {version_be} (hex: {hex(version_be)})")
        
        # Read a bit more for debugging
        fid.seek(0)
        header_bytes = fid.read(32)
        print(f"First 32 bytes: {[hex(b)[2:].zfill(2) for b in header_bytes]}")

    print("Initializing pipeline...")
    # Initialize and run pipeline
    pipeline = Pipeline(
        autoclean_dir="outputs",
        autoclean_config="configs/autoclean_config.yaml"
    )
    
    print("Starting file processing...")
    pipeline.process_file(file_path=test_file, task="RestingEyesOpen")
    
    end_time = time.time()
    print(f"\nTest completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    test_simple_resting_pipeline()
