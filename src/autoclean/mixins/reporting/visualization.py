"""Visualization mixin for EEG data in autoclean tasks.

This module provides specialized visualization functionality for EEG data in the AutoClean
pipeline. It defines methods for generating plots that visualize different aspects of
EEG processing results, such as:

- Raw data overlays comparing original and cleaned data
- Bad channel visualizations with topographies
- PSD and topographical maps
- MMN ERP analyses

These visualizations help users understand the effects of preprocessing steps and
validate the quality of processed data.

Example:
    ```python
    from autoclean.core.task import Task
    
    class MyEEGTask(Task):
        def process(self, raw, pipeline, autoclean_dict):
            # Process the data
            raw_cleaned = self.apply_preprocessing(raw)
            
            # Generate visualizations
            self.plot_raw_vs_cleaned_overlay(raw, raw_cleaned, pipeline, autoclean_dict)
            self.psd_topo_figure(raw, raw_cleaned, pipeline, autoclean_dict)
    ```
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from autoclean.utils.logging import message
from autoclean.utils.montage import get_standard_set_in_montage, validate_channel_set
from autoclean.mixins.reporting.base import ReportingMixin

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")


class VisualizationMixin:
    """Mixin providing visualization methods for EEG data.
    
    This mixin extends the base ReportingMixin with specialized methods for
    generating plots and visualizations of EEG data. It provides a comprehensive
    set of visualization tools for assessing data quality and understanding the
    effects of preprocessing steps.
    
    All visualization methods respect configuration toggles from `autoclean_config.yaml`,
    checking if their corresponding step is enabled before execution. Each method
    can be individually enabled or disabled via configuration.
    
    Available visualization methods include:
    
    - `plot_raw_vs_cleaned_overlay`: Overlay of original and cleaned EEG data
    - `plot_bad_channels_with_topography`: Bad channel visualization with topographies
    - `psd_topo_figure`: Combined PSD and topographical maps for frequency bands
    - `generate_mmn_erp`: MMN ERP analysis for standard/deviant paradigms
    - `step_psd_topo_figure`: Wrapper for backwards compatibility
    """
    
    def plot_raw_vs_cleaned_overlay(
        self,
        raw_original: mne.io.Raw,
        raw_cleaned: mne.io.Raw,
        pipeline: Any,
        autoclean_dict: Dict[str, Any],
        suffix: str = "",
    ) -> None:
        """Plot raw data channels over the full duration, overlaying the original and cleaned data.
        
        Parameters
        ----------
            raw_original : mne.io.Raw
                Original raw EEG data before cleaning.
            raw_cleaned : mne.io.Raw
                Cleaned raw EEG data after preprocessing.
            pipeline : Any
                Pipeline object containing pipeline metadata and utility functions.
            autoclean_dict : Dict[str, Any]
                Dictionary containing metadata about the processing run.
            suffix : str, optional
                Optional suffix to append to the output filename.
            
        Returns
        -------
            None
            
        Notes
        -----
            - The method downsamples the data for plotting to reduce file size
            - The resulting plot is saved to the derivatives directory with appropriate naming
            - Metadata about the plot is stored in the processing database
            - Original data is plotted in red, cleaned data in black.
        """
        # Ensure that the original and cleaned data have the same channels and times
        if raw_original.ch_names != raw_cleaned.ch_names:
            raise ValueError("Channel names in raw_original and raw_cleaned do not match.")
        if raw_original.times.shape != raw_cleaned.times.shape:
            raise ValueError("Time vectors in raw_original and raw_cleaned do not match.")

        # Get raw data
        channel_labels = raw_original.ch_names
        n_channels = len(channel_labels)
        sfreq = raw_original.info["sfreq"]
        times = raw_original.times
        data_original = raw_original.get_data()
        data_cleaned = raw_cleaned.get_data()

        # Increase downsample factor to reduce file size
        desired_sfreq = 100  # Reduced sampling rate to 100 Hz
        downsample_factor = int(sfreq // desired_sfreq)
        if downsample_factor > 1:
            data_original = data_original[:, ::downsample_factor]
            data_cleaned = data_cleaned[:, ::downsample_factor]
            times = times[::downsample_factor]

        # Normalize each channel individually for better visibility
        data_original_normalized = np.zeros_like(data_original)
        data_cleaned_normalized = np.zeros_like(data_cleaned)
        spacing = 10  # Fixed spacing between channels
        for idx in range(n_channels):
            # Original data
            channel_data_original = data_original[idx]
            channel_data_original = channel_data_original - np.mean(
                channel_data_original
            )  # Remove DC offset
            std = np.std(channel_data_original)
            if std == 0:
                std = 1  # Avoid division by zero
            data_original_normalized[idx] = (
                channel_data_original / std
            )  # Normalize to unit variance

            # Cleaned data
            channel_data_cleaned = data_cleaned[idx]
            channel_data_cleaned = channel_data_cleaned - np.mean(
                channel_data_cleaned
            )  # Remove DC offset
            # Use same std for normalization to ensure both signals are on the same scale
            data_cleaned_normalized[idx] = channel_data_cleaned / std

        # Multiply by a scaling factor to control amplitude
        scaling_factor = 2  # Adjust this factor as needed for visibility
        data_original_scaled = data_original_normalized * scaling_factor
        data_cleaned_scaled = data_cleaned_normalized * scaling_factor

        # Calculate offsets for plotting
        offsets = np.arange(n_channels) * spacing

        # Create plot
        total_duration = times[-1] - times[0]
        width_per_second = 0.1  # Adjust this factor as needed
        fig_width = min(total_duration * width_per_second, 50)
        fig_height = max(6, n_channels * 0.25)  # Adjusted for better spacing

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot channels
        for idx in range(n_channels):
            offset = offsets[idx]

            # Plot original data in red
            ax.plot(
                times,
                data_original_scaled[idx] + offset,
                color="red",
                linewidth=0.5,
                linestyle="-",
            )

            # Plot cleaned data in black
            ax.plot(
                times,
                data_cleaned_scaled[idx] + offset,
                color="black",
                linewidth=0.5,
                linestyle="-",
            )

        # Set y-ticks and labels
        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_labels, fontsize=8)

        # Customize axes
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_title("Raw Data Channels: Original vs Cleaned (Full Duration)", fontsize=14)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-spacing, offsets[-1] + spacing)
        ax.set_ylabel("")
        ax.invert_yaxis()

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="red", lw=0.5, linestyle="-", label="Original Data"),
            Line2D([0], [0], color="black", lw=0.5, linestyle="-", label="Cleaned Data"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.tight_layout()

        # Create Artifact Report
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])

        # Target file path
        target_figure = str(
            derivatives_path.copy().update(
                suffix="step_plot_raw_vs_cleaned_overlay", extension=".png", datatype="eeg"
            )
        )

        # Save as PNG with high DPI for quality
        fig.savefig(target_figure, dpi=150, bbox_inches="tight")
        plt.close(fig)

        message("info", f"Raw channels overlay full duration plot saved to {target_figure}")

        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "plot_raw_vs_cleaned_overlay": Path(target_figure).name,
            }
        }

        self._update_metadata("plot_raw_vs_cleaned_overlay", metadata)
        
    def plot_bad_channels_with_topography(
        self,
        raw_original: mne.io.Raw,
        raw_cleaned: mne.io.Raw,
        pipeline: Any,
        autoclean_dict: Dict[str, Any],
        zoom_duration: float = 30,
        zoom_start: float = 0,
    ) -> None:
        """Plot bad channels with a topographical map and time series overlays.
        
        This method creates a comprehensive visualization of bad channels including:
        
        1. A topographical map showing the locations of bad channels
        2. Time series plots comparing original and cleaned data for bad channels
        3. Both full duration and zoomed-in views of the data
        
        The visualization helps to validate the detection and interpolation of bad channels
        during preprocessing.
        
        Parameters
        ----------
            raw_original: mne.io.Raw
                Original raw EEG data before cleaning.
            raw_cleaned: mne.io.Raw
                Cleaned raw EEG data after interpolation of bad channels.
            pipeline: Any
                Pipeline object containing pipeline metadata and utility functions.
            autoclean_dict: Dict[str, Any]
                Dictionary containing metadata about the processing run.
            zoom_duration: float, Optional
                Duration in seconds for the zoomed-in time series plot.
            zoom_start: float, Optional
                Start time in seconds for the zoomed-in window.
            
        Returns
        -------
            None
            
        Notes
        -----
            - If no bad channels are found, the method returns without creating a plot
            - The resulting plot is saved to the derivatives directory
        """
        # Check if this step is enabled in the configuration
        if not self._check_step_enabled("bad_channel_report_step"):
            message("info", "✗ Bad channel visualization step disabled in config")
            return
            
        message("info", "✓ Generating bad channel visualization")
        
        # Collect Bad Channels
        bad_channels_info = {}

        # Mapping from channel to reason(s)
        for reason, channels in pipeline.flags.get("ch", {}).items():
            for ch in channels:
                if ch in bad_channels_info:
                    if reason not in bad_channels_info[ch]:
                        bad_channels_info[ch].append(reason)
                else:
                    bad_channels_info[ch] = [reason]

        bad_channels = list(bad_channels_info.keys())

        if not bad_channels:
            message("info", "No bad channels were identified.")
            return

        # Debugging: Print bad channels
        message("info", f"Identified Bad Channels: {bad_channels}")

        # Identify Good Channels
        all_channels = raw_original.ch_names
        good_channels = [ch for ch in all_channels if ch not in bad_channels]

        # Debugging: Print good channels count
        message("info", f"Number of Good Channels: {len(good_channels)}")

        # Extract Data for Bad Channels
        picks_bad_original = mne.pick_channels(raw_original.ch_names, bad_channels)
        picks_bad_cleaned = mne.pick_channels(raw_cleaned.ch_names, bad_channels)

        if len(picks_bad_original) == 0:
            message("info", "No bad channels found in original data.")
            return

        if len(picks_bad_cleaned) == 0:
            message("info", "No bad channels found in cleaned data.")
            return

        data_original, times = raw_original.get_data(
            picks=picks_bad_original, return_times=True
        )
        data_cleaned = raw_cleaned.get_data(picks=picks_bad_cleaned)

        channel_labels = [raw_original.ch_names[i] for i in picks_bad_original]
        n_channels = len(channel_labels)

        # Debugging: Print number of bad channels being plotted
        message("info", f"Number of Bad Channels to Plot: {n_channels}")

        # Downsample Data if Necessary
        sfreq = raw_original.info["sfreq"]
        desired_sfreq = 100  # Target sampling rate
        downsample_factor = int(sfreq // desired_sfreq)
        if downsample_factor > 1:
            data_original = data_original[:, ::downsample_factor]
            data_cleaned = data_cleaned[:, ::downsample_factor]
            times = times[::downsample_factor]
            message("info", f"Data downsampled by a factor of {downsample_factor} to {desired_sfreq} Hz.")

        # Normalize and Scale Data
        data_original_normalized = np.zeros_like(data_original)
        data_cleaned_normalized = np.zeros_like(data_cleaned)
        spacing = 10  # Fixed spacing between channels

        for idx in range(n_channels):
            # Original data
            channel_data_original = data_original[idx]
            channel_data_original = channel_data_original - np.mean(channel_data_original)  # Remove DC offset
            std = np.std(channel_data_original)
            if std == 0:
                std = 1  # Avoid division by zero
            data_original_normalized[idx] = channel_data_original / std  # Normalize to unit variance

            # Cleaned data
            channel_data_cleaned = data_cleaned[idx]
            channel_data_cleaned = channel_data_cleaned - np.mean(channel_data_cleaned)  # Remove DC offset
            # Use same std for normalization to ensure both signals are on the same scale
            data_cleaned_normalized[idx] = channel_data_cleaned / std

        # Multiply by a scaling factor to control amplitude
        scaling_factor = 2  # Adjust this factor as needed for visibility
        data_original_scaled = data_original_normalized * scaling_factor
        data_cleaned_scaled = data_cleaned_normalized * scaling_factor

        # Calculate offsets for plotting
        offsets = np.arange(n_channels) * spacing

        # Create the figure
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 2])

        # Topographical Subplot
        ax_topo = fig.add_subplot(gs[0, 0])
        self._plot_bad_channels_topo(raw_original, bad_channels, ax_topo)

        # Full Duration Time Series Subplot
        ax_full = fig.add_subplot(gs[0, 1])

        for idx in range(n_channels):
            # Plot original data
            ax_full.plot(
                times,
                data_original_scaled[idx] + offsets[idx],
                color="red",
                linewidth=1,
                linestyle="-",
            )
            # Plot cleaned data
            ax_full.plot(
                times,
                data_cleaned_scaled[idx] + offsets[idx],
                color="black",
                linewidth=1,
                linestyle="-",
            )

        ax_full.set_xlabel("Time (seconds)", fontsize=14)
        ax_full.set_ylabel("Bad Channels", fontsize=14)
        ax_full.set_title(
            "Bad Channels: Original vs Interpolated (Full Duration)", fontsize=16
        )
        ax_full.set_xlim(times[0], times[-1])
        ax_full.set_ylim(-spacing, offsets[-1] + spacing)
        ax_full.set_yticks([])  # Hide y-ticks
        ax_full.invert_yaxis()

        # Add legend
        legend_elements = [
            Line2D([0], [0], color="red", lw=2, linestyle="-", label="Original Data"),
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Interpolated Data"),
        ]
        ax_full.legend(handles=legend_elements, loc="upper right", fontsize=12)

        # Zoomed-In Time Series Subplot
        ax_zoom = fig.add_subplot(gs[1, 1])
        for idx in range(n_channels):
            # Plot original data
            ax_zoom.plot(
                times,
                data_original_scaled[idx] + offsets[idx],
                color="red",
                linewidth=1,
                linestyle="-",
            )
            # Plot cleaned data
            ax_zoom.plot(
                times,
                data_cleaned_scaled[idx] + offsets[idx],
                color="black",
                linewidth=1,
                linestyle="-",
            )

        # Calculate zoom window
        zoom_end = min(zoom_start + zoom_duration, times[-1])
        ax_zoom.set_xlim(zoom_start, zoom_end)
        ax_zoom.set_ylim(-spacing, offsets[-1] + spacing)
        ax_zoom.set_xlabel("Time (seconds)", fontsize=14)
        ax_zoom.set_ylabel("Bad Channels", fontsize=14)
        ax_zoom.set_title(
            f"Bad Channels: Original vs Interpolated (Zoom: {zoom_start}-{zoom_end} s)",
            fontsize=16,
        )
        ax_zoom.set_yticks([])  # Hide y-ticks
        ax_zoom.invert_yaxis()

        # Add legend
        ax_zoom.legend(handles=legend_elements, loc="upper right", fontsize=12)

        # Detailed Bad Channels Info Subplot
        ax_info = fig.add_subplot(gs[1, 0])
        ax_info.axis("off")  # Turn off axes

        info_text = "Bad Channels Information:\n\n"
        for ch, reasons in bad_channels_info.items():
            info_text += f"{ch}: {', '.join(reasons)}\n"

        ax_info.text(
            0.05,
            0.95,
            info_text,
            transform=ax_info.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()

        # Save the figure
        output_dir = Path(pipeline.run_path) / "derivatives" / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = output_dir / f"bad_channels_{timestamp}.png"

        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)

        message("success", f"Bad channels plot saved to {output_file}")
        
        # Save metadata to pipeline
        pipeline.add_metadata({
            "bad_channels_plot": {
                "file": str(output_file),
                "n_bad_channels": len(bad_channels),
                "bad_channels": bad_channels,
                "reasons": {ch: reasons for ch, reasons in bad_channels_info.items()}
            }
        })
        # Check if configuration checking is enabled and the step is enabled
        if hasattr(self, '_check_step_enabled'):
            is_enabled, _ = self._check_step_enabled("bad_channel_report_step")
            if not is_enabled:
                message("info", "✗ Bad channel report generation is disabled in configuration")
                return
            
        # Get bad channels from the cleaned data
        bad_channels = raw_cleaned.info['bads']
        
        if not bad_channels:
            message("info", "No bad channels found. Skipping bad channel visualization.")
            return
            
        message("info", f"Generating bad channel visualization for {len(bad_channels)} channels: {', '.join(bad_channels)}")
        
        # Create Artifact Report
        derivatives_path = pipeline.get_derivative_path(autoclean_dict['bids_path'])
        
        # Target file path
        target_figure = str(
            derivatives_path.copy().update(
                suffix="step_bad_channels_topo", extension=".png", datatype="eeg"
            )
        )
        
        # Create a figure to visualize bad channels
        n_bad_channels = len(bad_channels)
        fig = plt.figure(figsize=(12, 8 + n_bad_channels * 1.5))
        gs = GridSpec(2 + n_bad_channels, 2, height_ratios=[2] + [1.5] * n_bad_channels + [0.5])
        
        # 1. Plot topographical map of all channels with bad channels highlighted
        ax_topo = fig.add_subplot(gs[0, :])
        self._plot_bad_channels_topo(raw_original, bad_channels, ax_topo)
        
        # 2. Plot time series for each bad channel - full duration
        message("info", "Generating full duration time series plots for bad channels")
        sfreq = raw_original.info['sfreq']
        for i, bad_ch in enumerate(bad_channels):
            ax_full = fig.add_subplot(gs[i+1, 0])
            self._plot_bad_channel_timeseries(
                raw_original, raw_cleaned, bad_ch,
                ax_full, full_duration=True, title_suffix="Full Duration"
            )
            
        # 3. Plot time series for each bad channel - zoomed in
        message("info", "Generating zoomed time series plots for bad channels")
        for i, bad_ch in enumerate(bad_channels):
            ax_zoom = fig.add_subplot(gs[i+1, 1])
            self._plot_bad_channel_timeseries(
                raw_original, raw_cleaned, bad_ch,
                ax_zoom, full_duration=False, zoom_start=zoom_start,
                zoom_duration=zoom_duration, title_suffix=f"Zoom {zoom_start}-{zoom_start+zoom_duration}s"
            )
            
        # Add legend at the bottom
        ax_legend = fig.add_subplot(gs[-1, :])
        ax_legend.axis('off')
        ax_legend.legend(
            [plt.Line2D([0], [0], color='red', lw=1), plt.Line2D([0], [0], color='blue', lw=1)],
            ['Original Data', 'Cleaned/Interpolated Data'],
            loc='center', ncol=2, frameon=False
        )
        
        plt.tight_layout()
        
        # Save figure
        fig.savefig(target_figure, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        message("info", f"Bad channels visualization saved to {target_figure}")
        
        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "bad_channels_visualization": Path(target_figure).name,
                "bad_channels": bad_channels,
            }
        }
        
        self._update_metadata("bad_channels_visualization", metadata)
        
    def _plot_bad_channels_topo(self, raw: mne.io.Raw, bad_channels: List[str], ax) -> None:
        """Plot topographical map highlighting bad channels.
        
        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data with channel information.
        bad_channels : list of str
            List of bad channel names.
        ax : matplotlib.axes.Axes
            Axes to plot on.
        """
        # Create data array for topography where bad channels have value 1, good channels 0
        ch_names = raw.ch_names
        data = np.zeros(len(ch_names))
        for bad_ch in bad_channels:
            if bad_ch in ch_names:
                data[ch_names.index(bad_ch)] = 1
                
        # Plot topography
        mne.viz.plot_topomap(
            data, raw.info, axes=ax, show=False,
            cmap='RdBu_r', vmin=0, vmax=1,
            outlines='head', contours=0, sensors=True
        )
        
        # Add title
        ax.set_title(f"Bad Channels Topography (n={len(bad_channels)})")
        
    def _plot_bad_channel_timeseries(
        self,
        raw_original: mne.io.Raw,
        raw_cleaned: mne.io.Raw,
        channel: str,
        ax,
        full_duration: bool = True,
        zoom_start: float = 0,
        zoom_duration: float = 30,
        title_suffix: str = ""
    ) -> None:
        """Plot time series for a specific channel.
        
        Parameters:
        -----------
        raw_original : mne.io.Raw
            Original raw EEG data before cleaning.
        raw_cleaned : mne.io.Raw
            Cleaned raw EEG data after preprocessing.
        channel : str
            Channel name to plot.
        ax : matplotlib.axes.Axes
            Axes to plot on.
        full_duration : bool, optional
            Whether to plot the full duration or a zoomed-in section. Default is True.
        zoom_start : float, optional
            Start time in seconds for the zoomed-in window. Default is 0 seconds.
        zoom_duration : float, optional
            Duration in seconds for the zoomed-in time series plot. Default is 30 seconds.
        title_suffix : str, optional
            Suffix to add to the plot title.
        """
        # Get channel data
        sfreq = raw_original.info['sfreq']
        ch_idx = raw_original.ch_names.index(channel)
        
        if full_duration:
            data_orig = raw_original.get_data(picks=[ch_idx])[0]
            data_clean = raw_cleaned.get_data(picks=[ch_idx])[0]
            times = raw_original.times
            
            # Downsample for faster plotting if too many data points
            if len(times) > 10000:  # Arbitrary threshold
                downsample_factor = max(1, int(len(times) / 10000))
                data_orig = data_orig[::downsample_factor]
                data_clean = data_clean[::downsample_factor]
                times = times[::downsample_factor]
        else:
            # Extract zoomed-in data
            start_idx = int(zoom_start * sfreq)
            end_idx = int((zoom_start + zoom_duration) * sfreq)
            data_orig = raw_original.get_data(picks=[ch_idx], start=start_idx, stop=end_idx)[0]
            data_clean = raw_cleaned.get_data(picks=[ch_idx], start=start_idx, stop=end_idx)[0]
            times = np.arange(len(data_orig)) / sfreq + zoom_start
        
        # Plot data
        ax.plot(times, data_orig, color='red', lw=0.8)
        ax.plot(times, data_clean, color='blue', lw=0.8)
        
        # Add labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f"Channel: {channel} - {title_suffix}")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
    def step_psd_topo_figure(
        self,
        raw_original: mne.io.Raw,
        raw_cleaned: mne.io.Raw,
        pipeline: Any,
        autoclean_dict: Dict[str, Any],
        bands: Optional[List[Tuple[str, float, float]]] = None,
    ) -> None:
        """ Generate and save a single high-resolution image that includes:
       
        - Two PSD plots side by side: Absolute PSD (mV²) and Relative PSD (%).
        - Topographical maps for multiple EEG frequency bands arranged horizontally,
        showing both pre and post cleaning.
        - Annotations for average power and outlier channels.

        Parameters
        ----------
        raw_original : mne.io.Raw
            Original raw EEG data before cleaning.
        raw_cleaned : mne.io.Raw
            Cleaned EEG data after preprocessing.
        pipeline : pylossless.Pipeline
            Pipeline object.
        autoclean_dict : dict
            Dictionary containing autoclean parameters and paths.
        bands : list of tuple, optional
            List of frequency bands to plot. Each tuple should contain
            (band_name, lower_freq, upper_freq).

        Returns
        -------
        image_path : str
            Path to the saved combined figure.
        """

        # Define default frequency bands if none provided
        if bands is None:
            bands = [
                ("Delta", 1, 4),
                ("Theta", 4, 8),
                ("Alpha", 8, 12),
                ("Beta", 12, 30),
                ("Gamma1", 30, 60),
                ("Gamma2", 60, 80),
            ]

        # Create Artifact Report
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])

        # Output figure path
        target_figure = str(
            derivatives_path.copy().update(
                suffix="psd_topo_figure", extension=".png", datatype="eeg"
            )
        )

        # Select all EEG channels
        raw_original.interpolate_bads()

        # Count number of EEG channels
        channel_types = raw_original.get_channel_types()
        n_eeg_channels = channel_types.count("eeg")

        if n_eeg_channels == 0:
            message("warning", "No EEG channels found in raw data.")
        else:
            message("info", f"Number of EEG channels: {n_eeg_channels}")
            raw_original.pick("eeg")

        # Parameters for PSD
        fmin = 0.5
        fmax = 80
        n_fft = int(raw_original.info["sfreq"] * 2)  # Window length of 2 seconds

        # Compute PSD for original and cleaned data
        psd_original = raw_original.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            average="mean",
            verbose=False,
        )
        psd_cleaned = raw_cleaned.compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            average="mean",
            verbose=False,
        )

        freqs = psd_original.freqs
        df = freqs[1] - freqs[0]  # Frequency resolution

        # Convert PSDs to mV^2/Hz
        psd_original_mV2 = psd_original.get_data() * 1e6
        psd_cleaned_mV2 = psd_cleaned.get_data() * 1e6

        # Compute mean PSDs
        psd_original_mean_mV2 = np.mean(psd_original_mV2, axis=0)
        psd_cleaned_mean_mV2 = np.mean(psd_cleaned_mV2, axis=0)

        # Compute relative PSDs
        total_power_orig = np.sum(psd_original_mean_mV2 * df)
        total_power_clean = np.sum(psd_cleaned_mean_mV2 * df)
        psd_original_rel = (psd_original_mean_mV2 * df) / total_power_orig * 100
        psd_cleaned_rel = (psd_cleaned_mean_mV2 * df) / total_power_clean * 100

        # Compute band powers and identify outliers
        band_powers_orig = []
        band_powers_clean = []
        outlier_channels_orig = {}
        outlier_channels_clean = {}
        band_powers_metadata = {}

        for band_name, l_freq, h_freq in bands:
            # Get band powers
            band_power_orig = (
                psd_original.get_data(fmin=l_freq, fmax=h_freq).mean(axis=-1) * df * 1e6
            )
            band_power_clean = (
                psd_cleaned.get_data(fmin=l_freq, fmax=h_freq).mean(axis=-1) * df * 1e6
            )

            band_powers_orig.append(band_power_orig)
            band_powers_clean.append(band_power_clean)

            # Identify outliers
            for power, raw_data, outlier_dict in [
                (band_power_orig, raw_original, outlier_channels_orig),
                (band_power_clean, raw_cleaned, outlier_channels_clean),
            ]:
                mean_power = np.mean(power)
                std_power = np.std(power)
                if std_power > 0:
                    z_scores = (power - mean_power) / std_power
                    outliers = [
                        ch for ch, z in zip(raw_data.ch_names, z_scores) if abs(z) > 3
                    ]
                else:
                    outliers = []
                outlier_dict[band_name] = outliers

            # Store metadata
            band_powers_metadata[band_name] = {
                "frequency_band": f"{l_freq}-{h_freq} Hz",
                "band_power_mean_original_mV2": float(np.mean(band_power_orig)),
                "band_power_std_original_mV2": float(np.std(band_power_orig)),
                "band_power_mean_cleaned_mV2": float(np.mean(band_power_clean)),
                "band_power_std_cleaned_mV2": float(np.std(band_power_clean)),
                "outlier_channels_original": outlier_channels_orig[band_name],
                "outlier_channels_cleaned": outlier_channels_clean[band_name],
            }

        # Create figure and GridSpec
        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(4, len(bands), height_ratios=[2, 1, 1, 1.5], hspace=0.4, wspace=0.3)

        # Create PSD plots
        self._plot_psd(
            fig,
            gs,
            freqs,
            psd_original_mean_mV2,
            psd_cleaned_mean_mV2,
            psd_original_rel,
            psd_cleaned_rel,
            len(bands),
        )

        # Create topographical maps
        self._plot_topomaps(
            fig,
            gs,
            bands,
            band_powers_orig,
            band_powers_clean,
            raw_original,
            raw_cleaned,
            outlier_channels_orig,
            outlier_channels_clean,
        )

        # Add suptitle and adjust layout
        fig.suptitle(os.path.basename(raw_cleaned.filenames[0]), fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save and close figure
        fig.savefig(target_figure, dpi=300)
        plt.close(fig)

        message("info", f"Combined PSD and topographical map figure saved to {target_figure}")

        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "plot_psd_topo_figure": Path(target_figure).name,
            }
        }

        self._update_metadata("step_psd_topo_figure", metadata)

        return target_figure
        
    def _plot_psd(
        self,
        fig,
        gs,
        freqs,
        psd_original_mean_mV2,
        psd_cleaned_mean_mV2,
        psd_original_rel,
        psd_cleaned_rel,
        num_bands,
    ) -> None:
        """Helper function to create PSD plots.
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure object to plot on
        gs : matplotlib.gridspec.GridSpec
            GridSpec for organizing subplots
        freqs : numpy.ndarray
            Frequency vector for PSD
        psd_original_mean_mV2 : numpy.ndarray
            Original mean PSD in mV²/Hz
        psd_cleaned_mean_mV2 : numpy.ndarray
            Cleaned mean PSD in mV²/Hz
        psd_original_rel : numpy.ndarray
            Original relative PSD in %
        psd_cleaned_rel : numpy.ndarray
            Cleaned relative PSD in %
        num_bands : int
            Number of frequency bands for subplot width
        """
        # Plot Absolute PSD
        ax1 = fig.add_subplot(gs[0, :num_bands//2])
        ax1.semilogy(freqs, psd_original_mean_mV2, color='red', label='Original')
        ax1.semilogy(freqs, psd_cleaned_mean_mV2, color='black', label='Cleaned')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('PSD (mV²/Hz)')
        ax1.set_title('Absolute PSD')
        ax1.legend(loc='upper right')
        ax1.grid(True, which="both", ls="--")

        # Plot Relative PSD
        ax2 = fig.add_subplot(gs[0, num_bands//2:])
        ax2.plot(freqs, psd_original_rel, color='red', label='Original')
        ax2.plot(freqs, psd_cleaned_rel, color='black', label='Cleaned')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Relative Power (%)')
        ax2.set_title('Relative PSD')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
    def _plot_topomaps(
        self,
        fig,
        gs,
        bands,
        band_powers_orig,
        band_powers_clean,
        raw_original,
        raw_cleaned,
        outlier_channels_orig,
        outlier_channels_clean,
    ):
        """Helper function to create topographical maps"""
        # Second row: Topomaps for original data
        for i, (band, power) in enumerate(zip(bands, band_powers_orig)):
            band_name, l_freq, h_freq = band
            ax = fig.add_subplot(gs[1, i])
            mne.viz.plot_topomap(
                power, raw_original.info, axes=ax, show=False, contours=0, cmap="jet"
            )
            mean_power = np.mean(power)
            ax.set_title(
                f"Original: {band_name}\n({l_freq}-{h_freq} Hz)\nMean Power: {mean_power:.2e} mV²",
                fontsize=10,
            )
            # Annotate outlier channels
            outliers = outlier_channels_orig[band_name]
            if outliers:
                ax.annotate(
                    f"Outliers:\n{', '.join(outliers)}",
                    xy=(0.5, -0.15),
                    xycoords="axes fraction",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="red",
                )

        # Third row: Topomaps for cleaned data
        for i, (band, power) in enumerate(zip(bands, band_powers_clean)):
            band_name, l_freq, h_freq = band
            ax = fig.add_subplot(gs[2, i])
            mne.viz.plot_topomap(
                power, raw_cleaned.info, axes=ax, show=False, contours=0, cmap="jet"
            )
            mean_power = np.mean(power)
            ax.set_title(
                f"Cleaned: {band_name}\n({l_freq}-{h_freq} Hz)\nMean Power: {mean_power:.2e} mV²",
                fontsize=10,
            )
            # Annotate outlier channels
            outliers = outlier_channels_clean[band_name]
            if outliers:
                ax.annotate(
                    f"Outliers:\n{', '.join(outliers)}",
                    xy=(0.5, -0.15),
                    xycoords="axes fraction",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="red",
                )
            

    # def generate_mmn_erp(
    #     self,
    #     epochs: mne.Epochs,
    #     pipeline: Any,
    #     autoclean_dict: Dict[str, Any]
    # ) -> None:
    #     """Generate ERP plots for MMN paradigm."""
    #     message("header", "generate_mmn_erp")
    #     task = autoclean_dict["task"]
    #     settings = autoclean_dict["tasks"][task]["settings"]
    #     roi_channels = get_standard_set_in_montage(
    #         "mmn_standard", settings["montage"]["value"]
    #     )
    #     roi_channels = validate_channel_set(roi_channels, epochs.ch_names)
        
    #     if not roi_channels:
    #         message("error", "No valid ROI channels found in data")
    #         return None
        
    #     # Get conditions
    #     epoch_settings = settings["epoch_settings"]
    #     event_id = epoch_settings.get("event_id")
        
    #     if event_id is None:
    #         message("warning", "Event ID is not specified in epoch_settings (set to null)")
    #         return None
            
    #     conditions = {
    #         "standard": event_id.get("standard", "DIN2/1"),
    #         "predeviant": event_id.get("predeviant", "DIN2/2"),
    #         "deviant": event_id.get("deviant", "DIN2/3"),
    #     }
        
    #     # Create evoked objects
    #     try:
    #         evoked_dict = {
    #             "standard": epochs[conditions["standard"]].average(),
    #             "deviant": epochs[conditions["deviant"]].average(),
    #         }
            
    #         # Create difference wave (MMN = deviant - standard)
    #         evoked_dict["mmn"] = mne.combine_evoked(
    #             [evoked_dict["deviant"], evoked_dict["standard"]], weights=[1, -1]
    #         )
    #     except Exception as e:
    #         message("error", f"Error creating evoked objects: {str(e)}")
    #         return None
        
    #     # Set up figure output
    #     output_dir = Path(pipeline.run_path) / "derivatives" / "reports"
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     output_file = output_dir / f"mmn_erp_report_{timestamp}.pdf"
        
    #     # Create PDF report
    #     with PdfPages(output_file) as pdf:
    #         # Title page
    #         fig = plt.figure(figsize=(12, 9))
    #         plt.axis('off')
    #         title = f"MMN ERP Analysis Report\n{pipeline.subject_id}"
    #         plt.text(0.5, 0.8, title, ha='center', fontsize=20)
    #         plt.text(0.5, 0.6, f"Task: {task}", ha='center', fontsize=16)
    #         plt.text(0.5, 0.5, f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=14)
    #         plt.text(0.5, 0.4, f"ROI Channels: {', '.join(roi_channels)}", ha='center', fontsize=14)
    #         pdf.savefig(fig)
    #         plt.close()
            
    #         # Create ERP plots for each ROI channel
    #         time_window = (-0.1, 0.5)  # -100 ms to 500 ms
    #         for channel in roi_channels:
    #             fig = plt.figure(figsize=(12, 9))
    #             gs = GridSpec(3, 2, height_ratios=[1, 1, 1])
                
    #             # Plot standard response
    #             ax1 = plt.subplot(gs[0, 0])
    #             evoked_dict["standard"].plot(picks=channel, axes=ax1, time_unit='s', show=False)
    #             ax1.set_title(f"Standard Response - {channel}")
                
    #             # Plot deviant response
    #             ax2 = plt.subplot(gs[1, 0])
    #             evoked_dict["deviant"].plot(picks=channel, axes=ax2, time_unit='s', show=False)
    #             ax2.set_title(f"Deviant Response - {channel}")
                
    #             # Plot MMN (difference wave)
    #             ax3 = plt.subplot(gs[2, 0])
    #             evoked_dict["mmn"].plot(picks=channel, axes=ax3, time_unit='s', show=False)
    #             ax3.set_title(f"MMN (Deviant - Standard) - {channel}")
                
    #             # Find MMN peak in 100-250 ms window
    #             mmn_data = evoked_dict["mmn"].copy().pick(channel)
    #             peak_window = (0.100, 0.250)  # 100-250 ms for MMN
    #             peak_idx, peak_latency = mmn_data.get_peak(
    #                 ch_type='eeg', tmin=peak_window[0], tmax=peak_window[1], mode='neg'
    #             )
    #             peak_amp = mmn_data.data[0, peak_idx]
                
    #             # Highlight MMN window on plot
    #             ax3.axvspan(peak_window[0], peak_window[1], color='yellow', alpha=0.3)
    #             ax3.axvline(peak_latency, color='red', linestyle='--')
    #             ax3.text(
    #                 peak_latency, peak_amp, 
    #                 f"\n{peak_amp:.2f} μV\n{peak_latency*1000:.0f} ms",
    #                 ha='center', va='bottom', color='red'
    #             )
                
    #             # Plot topomaps
    #             # Standard
    #             ax4 = plt.subplot(gs[0, 1])
    #             evoked_dict["standard"].plot_topomap(
    #                 times=peak_latency, axes=ax4, show=False, time_unit='s'
    #             )
    #             ax4.set_title(f"Standard at {peak_latency*1000:.0f} ms")
                
    #             # Deviant
    #             ax5 = plt.subplot(gs[1, 1])
    #             evoked_dict["deviant"].plot_topomap(
    #                 times=peak_latency, axes=ax5, show=False, time_unit='s'
    #             )
    #             ax5.set_title(f"Deviant at {peak_latency*1000:.0f} ms")
                
    #             # MMN
    #             ax6 = plt.subplot(gs[2, 1])
    #             evoked_dict["mmn"].plot_topomap(
    #                 times=peak_latency, axes=ax6, show=False, time_unit='s'
    #             )
    #             ax6.set_title(f"MMN at {peak_latency*1000:.0f} ms")
                
    #             plt.tight_layout()
    #             pdf.savefig(fig)
    #             plt.close(fig)
            
    #         # Global field power analysis
    #         fig = plt.figure(figsize=(12, 9))
    #         gs = GridSpec(2, 1)
            
    #         # GFP for all conditions
    #         ax1 = plt.subplot(gs[0, 0])
    #         for cond, evoked in evoked_dict.items():
    #             gfp = np.sqrt(np.mean(evoked.data ** 2, axis=0))
    #             times = evoked.times
    #             ax1.plot(times, gfp, label=cond.capitalize())
            
    #         ax1.set_title("Global Field Power")
    #         ax1.set_xlabel("Time (s)")
    #         ax1.set_ylabel("GFP (μV)")
    #         ax1.axvline(0, color='k', linestyle='--')
    #         ax1.legend()
    #         ax1.grid(True)
            
    #         # Summary statistics
    #         ax2 = plt.subplot(gs[1, 0])
    #         ax2.axis('off')
            
    #         # Create a summary table
    #         mmn_peak_table = {
    #             "Channel": [],
    #             "Peak Amplitude (μV)": [],
    #             "Peak Latency (ms)": []
    #         }
            
    #         for channel in roi_channels:
    #             mmn_data = evoked_dict["mmn"].copy().pick(channel)
    #             peak_idx, peak_latency = mmn_data.get_peak(
    #                 ch_type='eeg', tmin=peak_window[0], tmax=peak_window[1], mode='neg'
    #             )
    #             peak_amp = mmn_data.data[0, peak_idx]
    #             mmn_peak_table["Channel"].append(channel)
    #             mmn_peak_table["Peak Amplitude (μV)"].append(f"{peak_amp:.2f}")
    #             mmn_peak_table["Peak Latency (ms)"].append(f"{peak_latency*1000:.0f}")
            
    #         # Convert to string format
    #         table_str = "MMN Peak Analysis (100-250 ms window):\n\n"
    #         for i in range(len(mmn_peak_table["Channel"])):
    #             ch = mmn_peak_table["Channel"][i]
    #             amp = mmn_peak_table["Peak Amplitude (μV)"][i]
    #             lat = mmn_peak_table["Peak Latency (ms)"][i]
    #             table_str += f"{ch}: {amp} μV at {lat} ms\n"
            
    #         ax2.text(0.1, 0.9, table_str, fontsize=12, va='top')
            
    #         plt.tight_layout()
    #         pdf.savefig(fig)
    #         plt.close(fig)
        
    #     message("success", f"MMN ERP report saved to {output_file}")
        
    #     # Update pipeline metadata
    #     pipeline.add_metadata({
    #         "mmn_erp_analysis": {
    #             "report_file": str(output_file),
    #             "roi_channels": roi_channels,
    #             "peak_window": peak_window,
    #             "conditions": conditions
    #         }
    #     })

    #     derivatives_path = pipeline.get_derivative_path(autoclean_dict['bids_path'])
        
    #     # Target file path
    #     target_pdf = str(
    #         derivatives_path.copy().update(
    #             suffix="mmn_erp_report", extension=".pdf", datatype="eeg"
    #         )
    #     )
        
    #     # Get channel info and validate standard 10-20 channel set
    #     channel_names = epochs.ch_names
        
    #     # Validate if we have the standard channels needed for MMN analysis
    #     standard_frontal_channels = ['Fz', 'FCz']
    #     available_channels = [ch for ch in standard_frontal_channels if ch in channel_names]
        
    #     if not available_channels:
    #         message("warning", f"No standard frontal channels found for MMN analysis. Looking for: {standard_frontal_channels}")
    #         # Try to find alternative frontal channels
    #         for ch in channel_names:
    #             if ch.startswith('F') and len(ch) <= 3:
    #                 available_channels.append(ch)
            
    #         if not available_channels:
    #             message("error", "Cannot perform MMN analysis: no suitable frontal channels found.")
    #             return
                
    #     message("info", f"Using channels for MMN analysis: {available_channels}")
        
    #     # Extract evoked responses
    #     standard_evoked = epochs[standard_key].average()
    #     deviant_evoked = epochs[deviant_key].average()
        
    #     # Create MMN (deviant - standard)
    #     mmn_evoked = mne.combine_evoked([deviant_evoked, -standard_evoked], weights='equal')
    #     mmn_evoked.comment = 'MMN (Deviant - Standard)'
        
    #     # Calculate MMN peak amplitude and latency
    #     mmn_peak_window = (0.1, 0.25)  # 100-250 ms window for MMN
    #     mmn_stats = {}
        
    #     for ch in available_channels:
    #         ch_data = mmn_evoked.copy().pick(ch).data[0]
    #         ch_times = mmn_evoked.times
            
    #         # Find peak latency and amplitude within window
    #         peak_window_indices = np.where(
    #             (ch_times >= mmn_peak_window[0]) & (ch_times <= mmn_peak_window[1])
    #         )[0]
    #         if len(peak_window_indices) == 0:
    #             message("warning", f"No data points found in MMN window for channel {ch}")
    #             continue
                
    #         peak_data = ch_data[peak_window_indices]
    #         peak_times = ch_times[peak_window_indices]
            
    #         # Find the most negative peak (MMN is a negative component)
    #         peak_idx = np.argmin(peak_data)
    #         peak_latency = peak_times[peak_idx]
    #         peak_amplitude = peak_data[peak_idx]
            
    #         mmn_stats[ch] = {
    #             'peak_latency': float(peak_latency),
    #             'peak_amplitude': float(peak_amplitude)
    #         }
            
    #     # Create PDF report
    #     with PdfPages(target_pdf) as pdf:
    #         # 1. Title page
    #         fig_title = plt.figure(figsize=(8.5, 11))
    #         plt.axis('off')
    #         plt.text(0.5, 0.8, "MMN ERP Analysis Report", ha='center', fontsize=20, weight='bold')
    #         plt.text(0.5, 0.7, f"Subject: {autoclean_dict['bids_path'].subject}", ha='center', fontsize=14)
    #         plt.text(0.5, 0.65, f"Session: {autoclean_dict['bids_path'].session}", ha='center', fontsize=14)
    #         plt.text(0.5, 0.6, f"Run: {autoclean_dict['bids_path'].run}", ha='center', fontsize=14)
    #         plt.text(0.5, 0.55, f"Task: {autoclean_dict['bids_path'].task}", ha='center', fontsize=14)
    #         plt.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=12)
    #         plt.text(0.5, 0.45, f"Channels used: {', '.join(available_channels)}", ha='center', fontsize=12)
            
    #         # Add MMN stats to title page
    #         stats_text = "MMN Peak Statistics:\n"
    #         for ch, stats in mmn_stats.items():
    #             stats_text += f"\n{ch}: {stats['peak_amplitude']:.2f} µV at {stats['peak_latency']*1000:.0f} ms"
    #         plt.text(0.5, 0.35, stats_text, ha='center', fontsize=12)
            
    #         pdf.savefig(fig_title)
    #         plt.close(fig_title)
            
    #         # 2. ERP plots for key channels
    #         for ch in available_channels:
    #             fig_erp = plt.figure(figsize=(10, 8))
    #             plt.suptitle(f"MMN ERP Analysis - Channel {ch}", fontsize=16)
                
    #             gs = GridSpec(2, 2, height_ratios=[2, 1])
                
    #             # Plot ERPs
    #             ax_erp = fig_erp.add_subplot(gs[0, :])
    #             times = standard_evoked.times * 1000  # Convert to ms
    #             ax_erp.plot(times, standard_evoked.copy().pick(ch).data[0] * 1e6, 'b-', linewidth=2, label=standard_key)
    #             ax_erp.plot(times, deviant_evoked.copy().pick(ch).data[0] * 1e6, 'r-', linewidth=2, label=deviant_key)
    #             ax_erp.plot(times, mmn_evoked.copy().pick(ch).data[0] * 1e6, 'k-', linewidth=2, label='MMN Difference')
                
    #             # Highlight MMN window
    #             ax_erp.axvspan(mmn_peak_window[0]*1000, mmn_peak_window[1]*1000, color='gray', alpha=0.2)
                
    #             # Mark MMN peak
    #             if ch in mmn_stats:
    #                 peak_lat = mmn_stats[ch]['peak_latency'] * 1000  # Convert to ms
    #                 peak_amp = mmn_stats[ch]['peak_amplitude'] * 1e6  # Convert to µV
    #                 ax_erp.plot(peak_lat, peak_amp, 'ko', markersize=8)
    #                 ax_erp.annotate(f"{peak_amp:.1f} µV", xy=(peak_lat, peak_amp), xytext=(peak_lat+20, peak_amp-5),
    #                                  arrowprops=dict(arrowstyle="->", color='black'))
                
    #             ax_erp.set_xlabel('Time (ms)')
    #             ax_erp.set_ylabel('Amplitude (µV)')
    #             ax_erp.legend(loc='best')
    #             ax_erp.grid(True)
    #             ax_erp.set_title(f"Event-Related Potentials - Channel {ch}")
                
    #             # Plot topomaps at different time points
    #             time_points = [0.1, 0.15, 0.2, 0.25]  # 100, 150, 200, 250 ms
    #             for i, t in enumerate(time_points):
    #                 ax_topo = fig_erp.add_subplot(gs[1, i//2])
    #                 mne.viz.plot_topomap(mmn_evoked.copy().crop(t, t).data[:, 0], mmn_evoked.info,
    #                                      axes=ax_topo, show=False, times=t,
    #                                      cmap='RdBu_r', vmin=-3, vmax=3)
    #                 ax_topo.set_title(f"{int(t*1000)} ms")
                
    #             pdf.savefig(fig_erp)
    #             plt.close(fig_erp)
                
    #         # 3. Global field power plot
    #         fig_gfp = plt.figure(figsize=(10, 6))
    #         plt.suptitle("Global Field Power", fontsize=16)
            
    #         # Calculate GFP for each condition
    #         gfp_standard = np.std(standard_evoked.data, axis=0) * 1e6
    #         gfp_deviant = np.std(deviant_evoked.data, axis=0) * 1e6
    #         gfp_mmn = np.std(mmn_evoked.data, axis=0) * 1e6
            
    #         plt.plot(times, gfp_standard, 'b-', linewidth=2, label=standard_key)
    #         plt.plot(times, gfp_deviant, 'r-', linewidth=2, label=deviant_key)
    #         plt.plot(times, gfp_mmn, 'k-', linewidth=2, label='MMN Difference')
            
    #         plt.axvspan(mmn_peak_window[0]*1000, mmn_peak_window[1]*1000, color='gray', alpha=0.2)
    #         plt.xlabel('Time (ms)')
    #         plt.ylabel('Global Field Power (µV)')
    #         plt.legend(loc='best')
    #         plt.grid(True)
            
    #         pdf.savefig(fig_gfp)
    #         plt.close(fig_gfp)
            
    #     message("info", f"MMN ERP report saved to {target_pdf}")
        
    #     # Update metadata
    #     metadata = {
    #         "mmn_erp_report": {
    #             "creationDateTime": datetime.now().isoformat(),
    #             "report_file": Path(target_pdf).name,
    #             "channels_analyzed": available_channels,
    #             "standard_condition": standard_key,
    #             "deviant_condition": deviant_key,
    #             "mmn_peak_stats": mmn_stats
    #         }
    #     }
        
    #     self._update_metadata("mmn_erp_report", metadata)