"""Utilities for test data management."""

import os
from pathlib import Path
import urllib.request
import hashlib
from typing import Optional

# Configuration
REPO_OWNER = "cincibrainlab"
REPO_NAME = "autoclean_pipeline"
TEST_DATA_VERSION = "v1.0.0"  # Update this when you create a new release
TEST_DATA_DIR = Path(__file__).parent / "data"

# File registry with their expected SHA256 hashes
TEST_FILES = {
    "0199_rest.raw": "28f9a714baf44cbb98aa055f3b11320f32d35d755c2020481265804df7c7df1c",
}

def get_test_data_dir() -> Path:
    """Get the test data directory, creating it if it doesn't exist."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DATA_DIR

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_test_file(filename: str, force: bool = False) -> Optional[Path]:
    """Download a test file from GitHub Releases if it doesn't exist locally.
    
    Args:
        filename: Name of the file to download
        force: If True, download even if file exists locally
        
    Returns:
        Path to the downloaded file or None if download failed
        
    Raises:
        ValueError: If file is not in the registry
        RuntimeError: If download fails or hash verification fails
    """
    if filename not in TEST_FILES:
        raise ValueError(f"Unknown test file: {filename}")
    
    file_path = get_test_data_dir() / filename
    
    # Check if file exists and hash matches
    if not force and file_path.exists():
        if calculate_file_hash(file_path) == TEST_FILES[filename]:
            return file_path
    
    # Construct GitHub release asset URL
    url = (
        f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/"
        f"{TEST_DATA_VERSION}/0199_rest.raw"  # Use exact filename to avoid any case sensitivity issues
    )
    
    print(f"\nAttempting to download from URL:\n{url}")
    
    try:
        # Add headers to request to handle pre-releases
        headers = {
            "Accept": "application/octet-stream",
            "User-Agent": "Python/urllib"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(file_path, 'wb') as out_file:
                out_file.write(response.read())
        
        # Verify hash
        if calculate_file_hash(file_path) != TEST_FILES[filename]:
            file_path.unlink()  # Delete file if hash doesn't match
            raise RuntimeError(f"Hash verification failed for {filename}")
        
        return file_path
    
    except Exception as e:
        if file_path.exists():
            file_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Failed to download {filename}: {str(e)}")

def get_test_file(filename: str, required: bool = True) -> Optional[Path]:
    """Get a test file, downloading it if necessary.
    
    Args:
        filename: Name of the file to get
        required: If True, raise error when file can't be obtained
                 If False, return None when file can't be obtained
    
    Returns:
        Path to the test file or None if not required and unavailable
    
    Raises:
        RuntimeError: If file is required but can't be obtained
    """
    try:
        return download_test_file(filename)
    except Exception as e:
        if required:
            raise RuntimeError(f"Required test file {filename} not available: {str(e)}")
        return None 