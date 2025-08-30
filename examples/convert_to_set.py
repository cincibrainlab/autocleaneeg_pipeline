from pathlib import Path

from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path(
    "/path/to/output/directory"
)  # Where processed data will be stored
CONFIG_FILE = Path(
    "configs/autoclean_config.yaml"
)  # Path to config relative to this example

EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = Pipeline(output_dir=EXAMPLE_OUTPUT_DIR, verbose="debug")

directory = Path("/path/to/input/directory")

pipeline.process_directory(directory=directory, task="RawToSet", pattern="*.raw")
