from pathlib import Path
from autoclean import Pipeline
import mne
from rich.console import Console

from autoclean.calc.source import calculate_source_psd, visualize_psd_results

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
file_path = Path("/Users/ernie/Data/autoclean_v3/resting_eyesopen_grael4k/stage/09_stc/140101_C2D1BL_EO_postrejection_raw_stc.h5")

stc = load_stc_file(file_path)

from mne.datasets import fetch_fsaverage
import matplotlib
matplotlib.use("Qt5Agg")
fs_dir = fetch_fsaverage()
subjects_dir = fs_dir.parent
subject = 'fsaverage'
subject_id = 'C2D1BL_EO'
output_dir = '/Users/ernie/Data/spg302_testresults'

# Calculate PSD and save to file
psd_df, file_path = calculate_source_psd(
    stc, 
    subjects_dir=subjects_dir,
    subject='fsaverage',
    n_jobs=10,  # Using 10 jobs as in your source estimation
    output_dir=output_dir,
    subject_id=subject_id
)

# Visualize PSD results
visualize_psd_results(psd_df, output_dir)

# def calculate_source_psd(stc, subjects_dir=None, subject='fsaverage', n_jobs=4, output_dir=None, subject_id=None):

breakpoint()
