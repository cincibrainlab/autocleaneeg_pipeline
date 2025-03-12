import asyncio
from pathlib import Path

from autoclean import Pipeline

async def main():
    # Define paths - modify these to match your system
    EXAMPLE_OUTPUT_DIR = Path("/srv2/RAWDATA/3_From_Collaborators/SPG302-ALS-001/BioTrials/SET_QEEG/Autoclean_output")  # Where processed data will be stored
    CONFIG_FILE = Path("configs/autoclean_config_rest_4k.yaml")  # Path to config relative to working directory OR absolute path

    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        autoclean_dir=EXAMPLE_OUTPUT_DIR,
        autoclean_config=CONFIG_FILE,
        verbose='INFO'
    )

    # Example INPUT directory path - modify this to point to your EEG files
    directory = Path("/srv2/RAWDATA/3_From_Collaborators/SPG302-ALS-001/BioTrials/SET_QEEG/RAW")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory=directory,
        task="resting_eyesopen_grael4k",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subdirectories
        pattern="*.set", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=1 # Optional: specify the maximum number of concurrent processes
    )

if __name__ == "__main__":
    asyncio.run(main())

