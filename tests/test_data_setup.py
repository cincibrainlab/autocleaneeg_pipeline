"""Setup script to generate synthetic test data files."""

import sys
from pathlib import Path

from tests.fixtures.synthetic_data import (
    create_corrupted_data_samples,
    save_synthetic_data_files,
)

# Add the project root and src to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main():
    """Generate all synthetic test data files."""
    # Get the test data directory
    test_dir = Path(__file__).parent
    data_dir = test_dir / "fixtures" / "data"

    print("Generating synthetic test data files...")

    # Generate standard test files
    file_paths = save_synthetic_data_files(data_dir, overwrite=True)

    print(f"Created {len(file_paths)} test data files:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")

    # Generate corrupted data samples
    print("\nGenerating corrupted data samples...")
    corrupted_paths = create_corrupted_data_samples(data_dir)

    print(f"Created {len(corrupted_paths)} corrupted data files:")
    for name, path in corrupted_paths.items():
        print(f"  {name}: {path}")

    print("\nTest data generation complete!")


if __name__ == "__main__":
    main()
