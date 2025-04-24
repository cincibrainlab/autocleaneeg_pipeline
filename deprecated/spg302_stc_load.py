from pathlib import Path
from autoclean import Pipeline
import mne
from rich.console import Console

console = Console()
def load_stc_file(file_path: Path) -> mne.SourceEstimate:
    """Load an STC file and return the SourceEstimate object."""
    try:
        # Check if the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist!")

        # Load the STC file
        console.print(f"[bold cyan]Loading STC file: {file_path}[/bold cyan]")
        stc = mne.read_source_estimate(file_path)
        console.print(f"[bold green]Successfully loaded STC with shape: {stc.data.shape} (vertices x times)[/bold green]")

        # Calculate tmax using tmin, tstep, and the number of time points
        tmax = stc.tmin + (stc.data.shape[1] - 1) * stc.tstep  # tmax = tmin + (n_times - 1) * tstep
        # Alternatively, you can use stc.times[-1]
        # tmax = stc.times[-1]

        # Print details for verification
        console.print(f"Time range: {stc.tmin:.3f} to {tmax:.3f} s")
        console.print(f"Sampling step: {stc.tstep:.3f} s")
        console.print(f"Number of vertices (left, right): {len(stc.vertices[0])}, {len(stc.vertices[1])}")

        return stc

    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return None
    except Exception as e:
        console.print(f"[bold red]Error loading STC file: {str(e)}[/bold red]")
        return None
        
# Define paths - modify these to match your system
EXAMPLE_OUTPUT_DIR = Path("/Users/ernie/Data/autoclean_v3")  # Where processed data will be stored
CONFIG_FILE = Path("/Users/ernie/Documents/GitHub/autoclean_pipeline/configs/autoclean_config_rest_4k.yaml")  # Path to config relative to working directory OR absolute path

# load stc (h5 format)
file_path = Path("/Users/ernie/Data/autoclean_v3/resting_eyesopen_grael4k/stage/09_stc/140101_C1D1BL_EO_stc-stc.h5")

stc = load_stc_file(file_path)

breakpoint()
