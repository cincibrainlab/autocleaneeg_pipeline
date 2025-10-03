"""Source connectivity mixin for autoclean tasks.

This module provides functionality for calculating functional connectivity between
brain regions from source-localized EEG data and computing graph theory metrics.

The SourceConnectivityMixin class implements methods for computing connectivity
using multiple methods (WPLI, PLV, coherence, PLI, amplitude envelope correlation)
across frequency bands and includes graph theory analysis (clustering coefficient,
global efficiency, modularity, characteristic path length, small-worldness).

Functional connectivity analysis enables:
- Brain network characterization
- Connectivity biomarkers for clinical populations
- Network dynamics across frequency bands
- Graph theory metrics for network organization
"""

from pathlib import Path
from typing import Optional, Union
import importlib.util

import mne
import pandas as pd

# Import algorithm function dynamically (for bundled block compatibility)
_algorithm_path = Path(__file__).parent / "algorithm.py"
_spec = importlib.util.spec_from_file_location("_source_connectivity_algorithm", _algorithm_path)
_algorithm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_algorithm_module)
calculate_source_connectivity = _algorithm_module.calculate_source_connectivity


class SourceConnectivityMixin:
    """Mixin class providing source-level functional connectivity analysis.

    This mixin provides methods for calculating functional connectivity between
    brain regions using source estimates (STCs). The implementation computes
    connectivity using multiple methods (WPLI, PLV, coherence, PLI, AEC) across
    frequency bands and includes comprehensive graph theory metrics.

    The mixin automatically retrieves source estimates from the task object (self.stc)
    and computes connectivity matrices for selected ROIs. Results include connectivity
    matrices, pairwise connectivity values, and graph theory metrics.

    References
    ----------
    Vinck M, et al. (2011). The pairwise phase consistency: a bias-free measure of
    rhythmic neuronal synchronization. NeuroImage, 51(1), 112-122.

    Rubinov M & Sporns O (2010). Complex network measures of brain connectivity: Uses
    and interpretations. NeuroImage, 52(3), 1059-1069.
    """

    def apply_source_connectivity(
        self,
        stc: Union[mne.SourceEstimate, None] = None,
        epoch_length: float = 4.0,
        n_epochs: int = 40,
        n_jobs: int = 4,
        stage_name: str = "apply_source_connectivity",
    ) -> tuple:
        """Calculate functional connectivity from source estimates.

        This method computes functional connectivity between brain regions using
        source-localized EEG data. Multiple connectivity methods are calculated:
        - WPLI: Weighted phase lag index (robust to volume conduction)
        - PLV: Phase locking value
        - Coherence: Magnitude-squared coherence
        - PLI: Phase lag index
        - AEC: Amplitude envelope correlation

        The method also computes graph theory metrics including clustering coefficient,
        global efficiency, characteristic path length, modularity, and small-worldness.

        Args:
            stc: Optional SourceEstimate. If None, uses self.stc
            epoch_length: Length of epochs in seconds for connectivity (default: 4s)
            n_epochs: Number of epochs to use for averaging (default: 40)
            n_jobs: Number of parallel jobs for computation (default: 4)
            stage_name: Name for saving and metadata tracking

        Returns:
            tuple: (conn_df, summary_path) where:
                - conn_df: DataFrame with columns [subject, method, band, roi1, roi2, connectivity]
                - summary_path: Path to saved connectivity summary CSV

        Raises:
            AttributeError: If no source estimates found (no self.stc)
            TypeError: If input is not SourceEstimate
            RuntimeError: If connectivity calculation fails

        Example:
            ```python
            # Apply connectivity with default parameters
            conn_df, summary_path = task.apply_source_connectivity()

            # Apply with custom parameters
            conn_df, summary_path = task.apply_source_connectivity(
                epoch_length=2.0,
                n_epochs=60,
                n_jobs=8
            )

            # Access results
            print(f"Calculated {len(conn_df)} connectivity pairs")
            alpha_wpli = conn_df[
                (conn_df['method'] == 'wpli') &
                (conn_df['band'] == 'alpha')
            ]
            ```

        Notes
        -----
        - Requires prior source localization (self.stc must exist)
        - Uses Desikan-Killiany atlas for ROI definition
        - Default ROIs: sensorimotor regions (8 ROIs)
        - Saves three files:
            * {subject}_connectivity_summary.csv: All connectivity pairs
            * {subject}_{method}_{band}_matrix.csv: Full connectivity matrices
            * {subject}_graph_metrics.csv: Graph theory metrics
            * {subject}_connectivity_log.txt: Detailed processing log
        - Frequency bands: delta (1-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-45)
        - Requires networkx and bctpy for graph metrics
        """
        # Check if this step is enabled in the configuration
        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled(
                "apply_source_connectivity"
            )

            if not is_enabled:
                if hasattr(self, "message"):
                    self.message("info", "Source connectivity step is disabled")
                else:
                    print("INFO: Source connectivity step is disabled")
                return None, None

            # Get parameters from config if available
            if config_value and isinstance(config_value, dict):
                epoch_length = config_value.get("epoch_length", epoch_length)
                n_epochs = config_value.get("n_epochs", n_epochs)
                n_jobs = config_value.get("n_jobs", n_jobs)

        # Determine which data to use
        if stc is None:
            # Try to get source estimates from task object
            if hasattr(self, "stc") and self.stc is not None:
                stc = self.stc
            else:
                raise AttributeError(
                    "No source estimates found. Apply source localization first "
                    "(self.stc must exist)."
                )

        # Type checking
        if not isinstance(stc, mne.SourceEstimate):
            raise TypeError(f"Data must be mne.SourceEstimate, got {type(stc)}")

        try:
            # Log start
            if hasattr(self, "message"):
                self.message("header", "Calculating source-level connectivity")
                self.message(
                    "info", f"Epoch length: {epoch_length}s, n_epochs: {n_epochs}"
                )
                self.message("info", f"n_jobs: {n_jobs}")
            else:
                print("=== Calculating Source-Level Connectivity ===")
                print(f"Epoch length: {epoch_length}s, n_epochs: {n_epochs}")
                print(f"n_jobs: {n_jobs}")

            # Prepare parameters
            output_dir = None
            subject_id = None  # Leave as None to let algorithm function handle default

            # Get output directory and subject ID from task config
            if hasattr(self, "config"):
                config = self.config
                # Use derivatives_dir if available (BIDS structure)
                if "derivatives_dir" in config:
                    output_dir = Path(config["derivatives_dir"]) / "connectivity"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_dir = str(output_dir)
                elif "output_dir" in config:
                    output_dir = config["output_dir"]
                # Get subject ID from config - use same method as export functions
                if "unprocessed_file" in config:
                    subject_id = Path(config["unprocessed_file"]).stem
                elif "subject_id" in config:
                    subject_id = config["subject_id"]
                elif "base_fname" in config:
                    subject_id = config["base_fname"]
                elif "original_fname" in config:
                    # Extract just the stem (no extension)
                    subject_id = Path(config["original_fname"]).stem

            # If output_dir not in config, try to construct from file paths
            if output_dir is None and hasattr(self, "file_path"):
                output_dir = (
                    Path(self.file_path).parent / "derivatives" / "source_connectivity"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_dir = str(output_dir)

            # If subject_id not in config, try to extract from filename
            if subject_id is None and hasattr(self, "file_path"):
                subject_id = Path(self.file_path).stem

            # Call algorithm function
            conn_df, summary_path = calculate_source_connectivity(
                stc=stc,
                labels=None,  # Uses Desikan-Killiany atlas
                subjects_dir=None,  # Uses MNE environment variable
                subject="fsaverage",
                n_jobs=n_jobs,
                output_dir=output_dir,
                subject_id=subject_id,
                sfreq=None,  # Uses stc.sfreq
                epoch_length=epoch_length,
                n_epochs=n_epochs,
            )

            # Log completion
            if conn_df is not None and not conn_df.empty:
                n_pairs = len(conn_df)
                n_methods = len(conn_df["method"].unique())
                n_bands = len(conn_df["band"].unique())

                if hasattr(self, "message"):
                    self.message(
                        "success",
                        f"Connectivity complete: {n_pairs} pairs, {n_methods} methods, {n_bands} bands",
                    )
                    if summary_path:
                        self.message("info", f"Saved to: {summary_path}")
                else:
                    print(
                        f"SUCCESS: Connectivity complete: {n_pairs} pairs, {n_methods} methods, {n_bands} bands"
                    )
                    if summary_path:
                        print(f"Saved to: {summary_path}")

                # Update metadata
                if hasattr(self, "_update_metadata"):
                    # Get block info for reproducibility
                    block_info = self._get_block_info("source_connectivity")

                    metadata = {
                        "block_name": "source_connectivity",
                        "epoch_length": epoch_length,
                        "n_epochs": n_epochs,
                        "n_jobs": n_jobs,
                        "n_connectivity_pairs": n_pairs,
                        "n_methods": n_methods,
                        "n_bands": n_bands,
                        "methods": list(conn_df["method"].unique()),
                        "bands": list(conn_df["band"].unique()),
                        "output_file": summary_path,
                    }

                    # Add block version/commit info for reproducibility
                    if block_info:
                        metadata.update(block_info)

                    self._update_metadata("step_apply_source_connectivity", metadata)

                # Store in task object for downstream use
                self.source_connectivity_df = conn_df
                self.source_connectivity_file = summary_path

            else:
                if hasattr(self, "message"):
                    self.message("warning", "No connectivity data was generated")
                else:
                    print("WARNING: No connectivity data was generated")

            return conn_df, summary_path

        except Exception as e:
            error_msg = f"Error during source connectivity calculation: {str(e)}"
            if hasattr(self, "message"):
                self.message("error", error_msg)
            else:
                print(f"ERROR: {error_msg}")
            raise RuntimeError(
                f"Failed to calculate source connectivity: {str(e)}"
            ) from e