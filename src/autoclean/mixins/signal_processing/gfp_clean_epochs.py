"""GFP-based epoch cleaning mixin for autoclean tasks.

This module provides functionality for cleaning epochs based on Global Field Power (GFP),
a measure of the spatial standard deviation across all electrodes at each time point.
GFP is a useful metric for identifying epochs with abnormal activity patterns that
may represent artifacts.

The GFPCleanEpochsMixin class implements methods for calculating GFP values for each
epoch, identifying outliers based on statistical thresholds, and optionally creating
visualization plots to help understand the distribution of GFP values across epochs.

GFP-based cleaning is particularly useful for removing epochs with widespread artifacts
that affect multiple channels simultaneously, such as movement artifacts, muscle activity,
or other global disturbances in the EEG signal.
"""

from typing import Union, Dict, Optional, List, Any, Tuple
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from autoclean.utils.logging import message
from autoclean.utils.database import manage_database

class GFPCleanEpochsMixin:
    """Mixin class providing functionality to clean epochs based on Global Field Power (GFP).
    
    This mixin provides methods for cleaning epochs using Global Field Power (GFP), which
    is a measure of the spatial standard deviation across all electrodes at each time point.
    GFP quantifies the overall variability in the electric field across the scalp and is
    sensitive to both signal strength and spatial complexity.
    
    The cleaning process involves calculating GFP for each epoch, identifying epochs with
    abnormally high or low GFP values (outliers), and optionally creating visualization
    plots to help understand the distribution of GFP values. The mixin focuses on scalp
    electrodes for GFP calculation, excluding non-EEG channels like EOG or reference
    electrodes.
    
    The mixin respects configuration settings from the autoclean_config.yaml file,
    allowing users to customize the GFP threshold, number of epochs to retain, and
    other parameters.
    """
    
    def gfp_clean_epochs(self, epochs: Union[mne.Epochs, None] = None,
                        gfp_threshold: float = 3.0,
                        number_of_epochs: Optional[int] = None,
                        random_seed: Optional[int] = None,
                        stage_name: str = "post_gfp_clean") -> mne.Epochs:
        """Clean an MNE Epochs object by removing outlier epochs based on Global Field Power.
        
        This method calculates the Global Field Power (GFP) for each epoch, identifies
        epochs with abnormal GFP values (outliers), and removes them from the dataset.
        GFP is calculated as the standard deviation across all scalp electrodes at each
        time point, providing a measure of the spatial variability of the electric field.
        
        The method focuses on scalp electrodes for GFP calculation, excluding non-EEG
        channels like EOG or reference electrodes. It can also optionally select a random
        subset of the cleaned epochs, which is useful for ensuring a consistent number
        of epochs across subjects or conditions.
        
        The method generates visualization plots showing the GFP distribution and outliers,
        which are saved to the output directory if a run_id is specified in the configuration.
        
        Args:
            epochs: Optional MNE Epochs object. If None, uses self.epochs
            gfp_threshold: Z-score threshold for GFP-based outlier detection (default: 3.0)
            number_of_epochs: If specified, randomly selects this number of epochs from the cleaned data
            random_seed: Seed for random number generator when selecting epochs
            stage_name: Name for saving and metadata tracking
            
        Returns:
            mne.Epochs: The cleaned epochs object with outlier epochs removed
            
        Raises:
            AttributeError: If self.epochs doesn't exist when needed
            TypeError: If epochs is not an Epochs object
            RuntimeError: If cleaning fails
            
        Example:
            ```python
            # Clean epochs with default parameters
            clean_epochs = task.gfp_clean_epochs()
            
            # Clean epochs with custom parameters and select a specific number of epochs
            clean_epochs = task.gfp_clean_epochs(
                gfp_threshold=2.5,
                number_of_epochs=40,
                random_seed=42
            )
            ```
        """
        # Check if this step is enabled in the configuration
        # is_enabled, config_value = self._check_step_enabled("gfp_clean_epochs")
            
        # if not is_enabled:
        #     message("info", "GFP clean epochs step is disabled in configuration")
        #     return None
            
        # # Get parameters from config if available
        # if config_value and isinstance(config_value, dict):
        #     gfp_threshold = config_value.get("gfp_threshold", gfp_threshold)
        #     number_of_epochs = config_value.get("number_of_epochs", number_of_epochs)
        #     random_seed = config_value.get("random_seed", random_seed)
        
        # Determine which data to use
        epochs = self._get_data_object(epochs, use_epochs=True)
        
        # Type checking
        if not isinstance(epochs, mne.Epochs):
            raise TypeError("Data must be an MNE Epochs object for GFP cleaning")
            
        try:
            message("header", "Cleaning epochs based on Global Field Power (GFP)")
            
            import random
            
            # Force preload to avoid RuntimeError
            if not epochs.preload:
                epochs.load_data()
                
            # Create a copy to work with
            epochs_clean = epochs.copy()
            
            # Define non-scalp electrodes to exclude
            channel_region_map = {
                "E17": "OTHER",
                "E38": "OTHER",
                "E43": "OTHER",
                "E44": "OTHER",
                "E48": "OTHER",
                "E49": "OTHER",
                "E56": "OTHER",
                "E73": "OTHER",
                "E81": "OTHER",
                "E88": "OTHER",
                "E94": "OTHER",
                "E107": "OTHER",
                "E113": "OTHER",
                "E114": "OTHER",
                "E119": "OTHER",
                "E120": "OTHER",
                "E121": "OTHER",
                "E125": "OTHER",
                "E126": "OTHER",
                "E127": "OTHER",
                "E128": "OTHER",
            }
            
            # Get scalp electrode indices (all channels except those in channel_region_map)
            non_scalp_channels = list(channel_region_map.keys())
            all_channels = epochs_clean.ch_names
            scalp_channels = [ch for ch in all_channels if ch not in non_scalp_channels]
            scalp_indices = [epochs_clean.ch_names.index(ch) for ch in scalp_channels if ch in epochs_clean.ch_names]
            
            # Calculate Global Field Power (GFP) only for scalp electrodes
            message("info", "Calculating Global Field Power (GFP) for each epoch using only scalp electrodes")
            gfp = np.sqrt(
                np.mean(epochs_clean.get_data()[:, scalp_indices, :] ** 2, axis=(1, 2))
            )  # Shape: (n_epochs,)
            
            # Epoch Statistics
            epoch_stats = pd.DataFrame(
                {
                    "epoch": np.arange(len(gfp)),
                    "gfp": gfp,
                    "mean_amplitude": epochs_clean.get_data()[:, scalp_indices, :].mean(axis=(1, 2)),
                    "max_amplitude": epochs_clean.get_data()[:, scalp_indices, :].max(axis=(1, 2)),
                    "min_amplitude": epochs_clean.get_data()[:, scalp_indices, :].min(axis=(1, 2)),
                    "std_amplitude": epochs_clean.get_data()[:, scalp_indices, :].std(axis=(1, 2)),
                }
            )
            
            # Remove Outlier Epochs based on GFP
            message("info", "Removing outlier epochs based on GFP z-scores")
            gfp_mean = epoch_stats["gfp"].mean()
            gfp_std = epoch_stats["gfp"].std()
            z_scores = np.abs((epoch_stats["gfp"] - gfp_mean) / gfp_std)
            good_epochs_mask = z_scores < gfp_threshold
            removed_by_gfp = np.sum(~good_epochs_mask)
            epochs_final = epochs_clean[good_epochs_mask]
            epoch_stats_final = epoch_stats[good_epochs_mask]
            message("info", f"Outlier epochs removed based on GFP: {removed_by_gfp}")
            
            # Handle epoch selection with warning if needed
            requested_epochs_exceeded = False
            if number_of_epochs is not None:
                if len(epochs_final) < number_of_epochs:
                    warning_msg = f"Requested number_of_epochs={number_of_epochs} exceeds the available cleaned epochs={len(epochs_final)}. Using all available epochs."
                    message("warning", warning_msg)
                    requested_epochs_exceeded = True
                    number_of_epochs = len(epochs_final)
                
                if random_seed is not None:
                    random.seed(random_seed)
                selected_indices = random.sample(range(len(epochs_final)), number_of_epochs)
                epochs_final = epochs_final[selected_indices]
                epoch_stats_final = epoch_stats_final.iloc[selected_indices]
                message("info", f"Selected {number_of_epochs} epochs from the cleaned data")
            
            # Analyze drop log to tally different annotation types
            drop_log = epochs.drop_log
            total_epochs = len(drop_log)
            good_epochs = sum(1 for log in drop_log if len(log) == 0)
            
            # Dynamically collect all unique annotation types
            annotation_types = {}
            for log in drop_log:
                if len(log) > 0:  # If epoch was dropped
                    for annotation in log:
                        # Convert numpy string to regular string if needed
                        annotation = str(annotation)
                        annotation_types[annotation] = annotation_types.get(annotation, 0) + 1
            
            # Add good and total to the annotation_types dictionary
            annotation_types["KEEP"] = good_epochs
            annotation_types["TOTAL"] = total_epochs
            
            # Create GFP barplot if we have a pipeline with a derivative path
            if hasattr(self, 'pipeline') and hasattr(self, 'config'):
                self._create_gfp_plots(epochs, epoch_stats, epoch_stats_final)
            
            # Update metadata
            metadata = {
                "initial_epochs": len(epochs),
                "final_epochs": len(epochs_final),
                "removed_by_gfp": int(removed_by_gfp),
                "mean_amplitude": float(epoch_stats_final["mean_amplitude"].mean()),
                "max_amplitude": float(epoch_stats_final["max_amplitude"].max()),
                "min_amplitude": float(epoch_stats_final["min_amplitude"].min()),
                "std_amplitude": float(epoch_stats_final["std_amplitude"].mean()),
                "mean_gfp": float(epoch_stats_final["gfp"].mean()),
                "gfp_threshold": float(gfp_threshold),
                "removed_total": int(removed_by_gfp),
                "annotation_types": annotation_types,
                "epoch_duration": epochs.times[-1] - epochs.times[0],
                "samples_per_epoch": epochs.times.shape[0],
                "total_duration_sec": (epochs.times[-1] - epochs.times[0]) * len(epochs_final),
                "total_samples": epochs.times.shape[0] * len(epochs_final),
                "channel_count": len(epochs.ch_names),
                "scalp_channels_used": scalp_channels,
                "requested_epochs_exceeded": requested_epochs_exceeded,
            }
            
            self._update_metadata("gfp_clean_epochs", metadata)
            
            # Store epochs
            if hasattr(self, 'config') and self.config.get("run_id"):
                self.epochs = epochs_final
                
            # Save epochs
            from autoclean.step_functions.io import save_epochs_to_set
            if hasattr(self, 'config'):
                save_epochs_to_set(epochs_final, self.config, stage_name)
                
            message("info", "Epoch GFP cleaning process completed")
            return epochs_final
            
        except Exception as e:
            message("error", f"Error during GFP epoch cleaning: {str(e)}")
            raise RuntimeError(f"Failed to clean epochs using GFP: {str(e)}") from e
            
    def _create_gfp_plots(self, epochs, epoch_stats, epoch_stats_final):
        """Create GFP plots and save them to the derivatives path.
        
        Args:
            epochs: Original epochs object
            epoch_stats: DataFrame with statistics for all epochs
            epoch_stats_final: DataFrame with statistics for kept epochs
        """
        try:
            # Get derivative path
            derivatives_path = self.pipeline.get_derivative_path(self.config["bids_path"])
            
            # Create GFP barplot
            plt.figure(figsize=(12, 4))
            
            # Plot all epochs in red first (marking removed epochs)
            plt.bar(epoch_stats.index, epoch_stats["gfp"], width=0.8, color="red", alpha=0.3)
            
            # Then overlay kept epochs in blue
            plt.bar(epoch_stats_final.index, epoch_stats_final["gfp"], width=0.8, color="blue")
            
            plt.xlabel("Epoch Number")
            plt.ylabel("Global Field Power (GFP)")
            plt.title("GFP Values by Epoch (Red = Removed, Blue = Kept)")
            
            # Save plot
            plot_fname = derivatives_path.copy().update(
                suffix="gfp", extension="png", check=False
            )
            plt.savefig(Path(plot_fname), dpi=150, bbox_inches="tight")
            plt.close()
            
            # Create GFP heatmap with larger figure size and improved readability
            plt.figure(figsize=(30, 18))
            
            # Calculate number of rows and columns for grid layout
            n_epochs = len(epoch_stats)
            n_cols = 8  # Reduced number of columns for larger cells
            n_rows = int(np.ceil(n_epochs / n_cols))
            
            # Create a grid of values
            grid = np.full((n_rows, n_cols), np.nan)
            for i, (idx, gfp) in enumerate(epoch_stats["gfp"].items()):
                row = i // n_cols
                col = i % n_cols
                grid[row, col] = gfp
            
            # Create heatmap with larger spacing between cells
            im = plt.imshow(grid, cmap="RdYlBu_r", aspect="auto")
            plt.colorbar(im, label="GFP Value (×10⁻⁶)", fraction=0.02, pad=0.04)
            
            # Add text annotations with increased font size and spacing
            for i, (idx, gfp) in enumerate(epoch_stats["gfp"].items()):
                row = i // n_cols
                col = i % n_cols
                kept = idx in epoch_stats_final.index
                color = "black" if kept else "red"
                plt.text(
                    col,
                    row,
                    f"ID: {idx}\nGFP: {gfp:.1e}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.8, pad=0.8),
                )
            
            # Improve title and labels with larger font sizes
            plt.title("GFP Heatmap by Epoch (Red = Removed, Black = Kept)", fontsize=14, pad=20)
            plt.xlabel("Column", fontsize=12, labelpad=10)
            plt.ylabel("Row", fontsize=12, labelpad=10)
            
            # Adjust layout to prevent text overlap
            plt.tight_layout()
            
            # Save heatmap plot with higher DPI for better quality
            plot_fname = derivatives_path.copy().update(
                suffix="gfp-heatmap", extension="png", check=False
            )
            plt.savefig(Path(plot_fname), dpi=300, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            message("warning", f"Could not create GFP plots: {str(e)}")
            # Continue without plots
