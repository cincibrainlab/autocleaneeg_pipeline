from pathlib import Path
from autoclean import Pipeline

EXAMPLE_OUTPUT_DIR = Path("C:/Users/Gam9LG/Documents/Autoclean")  # Where processed data will be stored
CONFIG_FILE = Path("configs/autoclean_config.yaml")  # Path to config relative to this example

pipeline = Pipeline(
    autoclean_dir=EXAMPLE_OUTPUT_DIR,
    autoclean_config=CONFIG_FILE,
    verbose='INFO'
)

pipeline.start_autoclean_review()



