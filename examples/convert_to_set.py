import asyncio
from pathlib import Path

from autoclean import Pipeline

# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("/srv/Analysis/Gavin_Projects/Autoclean")  # Where processed data will be stored
CONFIG_FILE = Path("configs/autoclean_config.yaml")  # Path to config relative to this example

EXAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='Error'
)

directory = Path("/srv2/RAWDATA/1_NBRT_LAB_STUDIES/Raw_P1_EEGs_n141/Rest_RAW")

pipeline.process_directory(
    directory = directory,
    task = "RawToSet",
    pattern ="*.raw"
)
   