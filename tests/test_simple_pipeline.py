"""Test simple pipeline for resting state data processing."""

import numpy as np
from pathlib import Path

from autoclean import Pipeline


def test_simple_resting_pipeline():
    """Test basic pipeline functionality with resting state data."""

    # Test file is tracked by Git LFS
    test_file = Path("tests/data/0199_rest.raw")

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}\n"
            "Make sure you have Git LFS installed and have pulled the test data:\n"
            "  git lfs install\n"
            "  git lfs pull"
        )

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

    # Initialize and run pipeline
    pipeline = Pipeline(
        autoclean_dir="outputs", autoclean_config="configs/autoclean_config.yaml"
    )

    pipeline.process_file(file_path=test_file, task="RestingEyesOpen")


if __name__ == "__main__":
    test_simple_resting_pipeline()
