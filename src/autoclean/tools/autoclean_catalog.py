#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
import os
import time

# Initialize Rich console
console = Console()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate metadata CSV for EEG files (.set, .stc, .fif) in autoclean stage directory.")
    parser.add_argument(
        "--autoclean-base",
        type=Path,
        required=True,
        help="Base directory of the autoclean output (e.g., /Users/ernie/Data/autoclean_v3)"
    )
    return parser.parse_args()

def collect_eeg_metadata(stage_dir: Path) -> list:
    """Collect metadata for all .set, .stc, and .fif files in the stage directory."""
    metadata = []
    file_extensions = ["*.set", "*.stc", "*.fif", "*.h5"]  # File extensions to search for
    
    console.print(f"[bold cyan]Scanning directory: {stage_dir} for {', '.join(file_extensions)} files[/bold cyan]")
    for folder in track(list(stage_dir.iterdir()), description="Processing folders..."):
        if folder.is_dir() and folder.name.startswith("0"):  # Assuming stage folders start with numbers (e.g., 01_postimport)
            for ext in file_extensions:
                for file_path in folder.glob(ext):
                    stat = file_path.stat()
                    metadata.append({
                        "full_path": str(file_path.absolute()),
                        "basename": file_path.name,
                        "stage_folder": folder.name,
                        "file_type": ext.lstrip("*"),  # Remove the "*" from "*.set" to get "set"
                        "size_bytes": stat.st_size,
                        "size_mb": stat.st_size / (1024 * 1024),  # Convert to MB
                        "modification_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                    })
    
    return metadata

def create_table(metadata: list) -> Table:
    """Create a rich table for displaying metadata."""
    table = Table(title="EEG File Metadata (.set, .stc, .fif)", show_lines=True)
    table.add_column("Full Path", style="magenta")
    table.add_column("Base Name", style="magenta")
    table.add_column("Stage Folder", style="green")
    table.add_column("File Type", style="blue")
    table.add_column("Size (MB)", justify="right", style="yellow")
    table.add_column("Modification Date", style="cyan")

    for item in metadata:
        table.add_row(
            item["full_path"],
            item["basename"],
            item["stage_folder"],
            item["file_type"],
            f"{item['size_mb']:.2f}",
            item["modification_date"]
        )
    
    return table

def save_to_csv(metadata: list, output_path: Path):
    """Save metadata to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    df = pd.DataFrame(metadata)
    df.to_csv(output_path, index=False)
    console.print(f"[bold green]Metadata saved to: {output_path}[/bold green]")

def main():
    # Parse command-line arguments
    args = parse_args()
    autoclean_base = args.autoclean_base

    # Define directories
    stage_dir = autoclean_base / "stage"
    output_dir = autoclean_base / "metadata"
    output_csv = output_dir / "eeg_metadata.csv"  # Renamed to reflect broader scope

    # Ensure the stage directory exists
    if not stage_dir.exists():
        console.print(f"[bold red]Error: Stage directory {stage_dir} does not exist![/bold red]")
        return

    # Collect metadata
    metadata = collect_eeg_metadata(stage_dir)

    if not metadata:
        console.print("[bold yellow]No .set, .stc, or .fif files found in the stage directory![/bold yellow]")
        return

    # Display table
    console.print(create_table(metadata))

    # Save to CSV
    save_to_csv(metadata, output_csv)

if __name__ == "__main__":
    console.print("[bold blue]Starting EEG Metadata Generator (.set, .stc, .fif)[/bold blue]")
    try:
        main()
    except Exception as e:
        console.print(f"[bold red]An error occurred: {str(e)}[/bold red]")
    finally:
        console.print("[bold blue]Done![/bold blue]")