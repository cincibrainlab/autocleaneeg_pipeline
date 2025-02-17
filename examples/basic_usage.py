
import asyncio
from pathlib import Path

from autoclean import Pipeline


output_dir = Path("C:/Users/Gam9LG/Documents/Autoclean")
config_file = Path("configs/autoclean_config.yaml")

# Create pipeline instance
pipeline = Pipeline(
    autoclean_dir=output_dir,
    autoclean_config=config_file,
    verbose='INFO'
)

def process_single_file():
    file_path = Path("C:/Users/Gam9LG/Documents/DATA/n141_resting/raw/0079_rest.raw")

    pipeline.process_file(
        file_path=file_path,
        task="RestingEyesOpen",
    )

async def batch_process():
    directory = Path("C:/Users/Gam9LG/Documents/DATA/hbcd_mmn")

    await pipeline.process_directory_async(
        directory=directory,
        task="hbcd_mmn",
        sub_directories=False,
        max_concurrent=5
    )

if __name__ == "__main__":
    print("Processing single file...")
    process_single_file()

    print("Batch processing...")
    asyncio.run(batch_process())



