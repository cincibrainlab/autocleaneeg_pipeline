import asyncio
from pathlib import Path

from autoclean import Pipeline

async def main():
    """Example of batch processing multiple EEG files asynchronously."""
    # Create pipeline instance
    pipeline = Pipeline(
        verbose='WARNING'
    )
    # Example INPUT directory path - modify path/to/input/dthis to point to your EEG files
    directory = Path("C:/Users/Gam9LG/Documents/DATA/stat_learning/raw_set_files/raw_set_files/")

    # Process all files in directory
    await pipeline.process_directory_async(
        directory_path=directory,
        task="Statistical_Learning",  # Choose appropriate task
        sub_directories=False, # Optional: process files in subfolders
        pattern="*.set", # Optional: specify a pattern to filter files (use "*.extention" for all files of that extension)
        max_concurrent=3 # Optional: specify the maximum number of concurrent files to process
    )
    

if __name__ == "__main__":
    asyncio.run(main())

