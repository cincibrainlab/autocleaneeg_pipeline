#!/usr/bin/env python3

from pathlib import Path
import mne
from autoclean.calc.source import estimate_source_function_raw  # Import the source function
from autoclean.utils.logging import message  # Assuming this is available for logging
from rich.console import Console
import shutil

# Initialize Rich console for pretty printing
console = Console()

# Define directories
INPUT_DIR = Path("/Users/ernie/Data/autoclean_v3/resting_eyesopen_grael4k/stage/04_postrejection")
OUTPUT_DIR = Path("/Users/ernie/Data/autoclean_v3/resting_eyesopen_grael4k/stage/09_stc")
CONFIG = {
    "stage_dir": OUTPUT_DIR,
    "stage_files": {
        "post_source_localization": {"enabled": True, "suffix": "_stc"}
    },
    "unprocessed_file": "",  # Will be set per file
    "run_id": "resting_eyesopen_grael4k_run",  # Example run ID
    # Add other necessary config fields as per your autoclean setup
}

def load_raw_file(file_path: Path) -> mne.io.Raw:
    """Load a .set file as raw data using MNE."""
    try:
        console.print(f"[bold cyan]Loading raw file: {file_path}[/bold cyan]")
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        raw.pick_channels([ch for ch in raw.ch_names if ch not in ['HEOG', 'VEOG']])
        # Apply the standard 10-20 montage
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False)
        
        message("success", "Successfully configured standard 10-20 montage")
        
        console.print(f"[bold green]Successfully loaded raw data with {len(raw.ch_names)} channels[/bold green]")
        return raw
    except Exception as e:
        console.print(f"[bold red]Error loading raw file {file_path}: {str(e)}[/bold red]")
        return None

def process_and_save_stc(raw: mne.io.Raw, output_dir: Path, config: dict, filename: str):
    """Process raw data to STC and save it."""
    try:
        # Update config with the current file
        config["unprocessed_file"] = str(raw.filenames[0])

        # Apply source localization
        console.print(f"[bold cyan]Computing source estimates for {filename}[/bold cyan]")
        stc = estimate_source_function_raw(raw)  # Assuming this returns an STC object
        console.print(f"[bold green]Successfully computed STC with shape: {stc.data.shape} (vertices x times)[/bold green]")

        # Save the STC
        output_path = output_dir / f"{Path(filename).stem}_stc.h5"
        stc.save(output_path, ftype='h5', overwrite=True)
        console.print(f"[bold green]Saved STC to: {output_path}[/bold green]")
        message("success", f"✓ Saved STC for {filename} to {output_path}")

    except Exception as e:
        console.print(f"[bold red]Error processing {filename} to STC: {str(e)}[/bold red]")

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .set files in the input directory
    set_files = list(INPUT_DIR.glob("*.set"))
    if not set_files:
        console.print(f"[bold yellow]No .set files found in {INPUT_DIR}[/bold yellow]")
        return

    console.print(f"[bold blue]Found {len(set_files)} .set files to process[/bold blue]")
    for file_path in set_files:
        raw = load_raw_file(file_path)
        if raw is not None:
            process_and_save_stc(raw, OUTPUT_DIR, CONFIG, file_path.name)

if __name__ == "__main__":
    console.print("[bold blue]Starting Batch Conversion of .set to STC[/bold blue]")
    try:
        main()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {str(e)}[/bold red]")
    finally:
        console.print("[bold blue]Done![/bold blue]")


# # 3D alignment with fsaverage head model (for source localization context)
# import matplotlib
# from rich.console import Console
# from mne.viz import plot_sensors, plot_alignment
# from mne.datasets import fetch_fsaverage
# fs_dir = fetch_fsaverage()
# subjects_dir = fs_dir.parent
# subject = 'fsaverage'
# trans = 'fsaverage'
# src = mne.read_source_spaces(f'{fs_dir}/bem/fsaverage-ico-5-src.fif')
# bem = mne.read_bem_solution(f'{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif')

# fwd = mne.make_forward_solution(
#     raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=10
# )
# subjects_dir = fs_dir.parent

# mne.viz.plot_alignment(
#     raw.info,
#     src=src,
#     eeg=["original", "projected"],
#     trans='fsaverage',
#     show_axes=True,
#     mri_fiducials=True,
#     dig="fiducials",
# )

# console.print(f"[bold green]3D electrode alignment plot generated for {filename}[/bold green]")