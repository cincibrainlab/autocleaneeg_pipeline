"""Test simple pipeline for resting state data processing."""

from pathlib import Path
import shutil
import os

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
    
    # Initialize and run pipeline
    pipeline = Pipeline(
        autoclean_dir="outputs",
        autoclean_config="configs/autoclean_config.yaml"
    )
    
    pipeline.process_file(file_path=test_file, task="RestingEyesOpen")

if __name__ == "__main__":
    test_simple_resting_pipeline()

