import asyncio
from pathlib import Path

from autoclean import Pipeline

async def main():
    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        verbose='DEBUG'
    )
    # Example INPUT directory path - modify path/to/input/dthis to point to your EEG files
    directory = Path("C:/Users/Gam9LG/Documents/DATA/n141_resting/raw/")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory_path=directory,
        task="RestingEyesOpen",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subfolders
        pattern="*.raw", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=3 # Optional: specify the maximum number of concurrent files to process
    )
    

if __name__ == "__main__":
    asyncio.run(main())

