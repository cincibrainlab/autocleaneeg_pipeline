"""Plotting functions for EEG data visualization.

This module provides standalone functions for creating plots and visualizations
of EEG data processing results.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.lines import Line2D

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")


def plot_raw_comparison(
    raw_original: mne.io.Raw,
    raw_cleaned: mne.io.Raw,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Raw Data Comparison: Original vs Cleaned",
    downsample_to: float = 100.0,
    scaling_factor: float = 2.0,
    spacing: float = 10.0,
    figsize: Optional[tuple] = None,
    verbose: Optional[bool] = None
) -> plt.Figure:
    """Plot raw data comparison showing original vs cleaned data overlay.

    This function creates a multi-channel plot comparing original and cleaned
    EEG data, with original data in red and cleaned data in black. The plot
    shows all channels over the full duration with proper scaling and spacing.

    Parameters
    ----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after preprocessing.
    output_path : str, Path, or None, default None
        Path to save the plot. If None, plot is not saved.
    title : str, default "Raw Data Comparison: Original vs Cleaned"
        Title for the plot.
    downsample_to : float, default 100.0
        Target sampling rate for plotting (Hz). Data is downsampled to reduce
        file size and improve rendering speed.
    scaling_factor : float, default 2.0
        Amplitude scaling factor for better visibility.
    spacing : float, default 10.0
        Vertical spacing between channels.
    figsize : tuple or None, default None
        Figure size (width, height). If None, calculated automatically.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> fig = plot_raw_comparison(raw_original, raw_cleaned)
    >>> fig = plot_raw_comparison(raw_original, raw_cleaned, output_path="comparison.png")
    
    See Also
    --------
    plot_ica_components : Visualize ICA components
    plot_psd_topography : Create PSD topography plots
    mne.viz.plot_raw : MNE raw data plotting functions
    """
    # Input validation
    if not isinstance(raw_original, mne.io.BaseRaw):
        raise TypeError(
            f"raw_original must be an MNE Raw object, got {type(raw_original).__name__}"
        )
    
    if not isinstance(raw_cleaned, mne.io.BaseRaw):
        raise TypeError(
            f"raw_cleaned must be an MNE Raw object, got {type(raw_cleaned).__name__}"
        )

    # Ensure compatibility
    if raw_original.ch_names != raw_cleaned.ch_names:
        raise ValueError("Channel names in raw_original and raw_cleaned do not match")
    
    if raw_original.times.shape != raw_cleaned.times.shape:
        raise ValueError("Time vectors in raw_original and raw_cleaned do not match")

    try:
        # Get data
        channel_labels = raw_original.ch_names
        n_channels = len(channel_labels)
        sfreq = raw_original.info["sfreq"]
        times = raw_original.times
        data_original = raw_original.get_data()
        data_cleaned = raw_cleaned.get_data()

        # Downsample for plotting
        downsample_factor = int(sfreq // downsample_to)
        if downsample_factor > 1:
            data_original = data_original[:, ::downsample_factor]
            data_cleaned = data_cleaned[:, ::downsample_factor]
            times = times[::downsample_factor]

        # Normalize each channel individually
        data_original_normalized = np.zeros_like(data_original)
        data_cleaned_normalized = np.zeros_like(data_cleaned)
        
        for idx in range(n_channels):
            # Original data
            channel_data_original = data_original[idx]
            channel_data_original = channel_data_original - np.mean(channel_data_original)
            std = np.std(channel_data_original)
            if std == 0:
                std = 1  # Avoid division by zero
            data_original_normalized[idx] = channel_data_original / std

            # Cleaned data (use same std for consistent scaling)
            channel_data_cleaned = data_cleaned[idx]
            channel_data_cleaned = channel_data_cleaned - np.mean(channel_data_cleaned)
            data_cleaned_normalized[idx] = channel_data_cleaned / std

        # Apply scaling factor
        data_original_scaled = data_original_normalized * scaling_factor
        data_cleaned_scaled = data_cleaned_normalized * scaling_factor

        # Calculate plotting offsets
        offsets = np.arange(n_channels) * spacing

        # Calculate figure size
        if figsize is None:
            total_duration = times[-1] - times[0]
            width_per_second = 0.1
            fig_width = min(total_duration * width_per_second, 50)
            fig_height = max(6, n_channels * 0.25)
            figsize = (fig_width, fig_height)

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

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

        # Customize plot
        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_labels, fontsize=8)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-spacing, offsets[-1] + spacing)
        ax.invert_yaxis()

        # Add legend
        legend_elements = [
            Line2D([0], [0], color="red", lw=0.5, linestyle="-", label="Original Data"),
            Line2D([0], [0], color="black", lw=0.5, linestyle="-", label="Cleaned Data"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        plt.tight_layout()

        # Save if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"Plot saved to: {output_path}")

        return fig

    except Exception as e:
        raise RuntimeError(f"Failed to create raw comparison plot: {str(e)}") from e


def plot_ica_components(
    ica: mne.preprocessing.ICA,
    raw: Optional[mne.io.Raw] = None,
    picks: Optional[list] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "ICA Components",
    verbose: Optional[bool] = None
) -> plt.Figure:
    """Plot ICA component topographies and properties.

    This function creates a comprehensive visualization of ICA components
    including topographical maps, time series, and power spectra.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object to visualize.
    raw : mne.io.Raw or None, default None
        Raw data used for ICA fitting. Required for time series and spectra.
    picks : list or None, default None
        Component indices to plot. If None, plots all components.
    output_path : str, Path, or None, default None
        Path to save the plot. If None, plot is not saved.
    title : str, default "ICA Components"
        Title for the plot.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> fig = plot_ica_components(ica, raw)
    >>> fig = plot_ica_components(ica, raw, picks=[0, 1, 2, 3])
    
    See Also
    --------
    plot_raw_comparison : Plot before/after data comparison
    mne.preprocessing.ICA.plot_components : MNE ICA component plotting
    """
    # Input validation
    if not isinstance(ica, mne.preprocessing.ICA):
        raise TypeError(f"ica must be an MNE ICA object, got {type(ica).__name__}")

    try:
        # Use MNE's built-in plotting with custom parameters
        if picks is None:
            picks = list(range(min(ica.n_components_, 25)))  # Limit to first 25 components
        
        # Create the plot using MNE's function
        fig = ica.plot_components(picks=picks, show=False)
        
        # Customize title
        fig.suptitle(title, fontsize=14)
        
        # Save if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"ICA components plot saved to: {output_path}")

        return fig

    except Exception as e:
        raise RuntimeError(f"Failed to create ICA components plot: {str(e)}") from e


def plot_psd_topography(
    raw: mne.io.Raw,
    freq_bands: Optional[dict] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Power Spectral Density Topography",
    verbose: Optional[bool] = None
) -> plt.Figure:
    """Plot power spectral density topographical maps for frequency bands.

    This function creates topographical maps showing the distribution of power
    across the scalp for different frequency bands.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data to analyze.
    freq_bands : dict or None, default None
        Dictionary of frequency bands with format {'band_name': (low_freq, high_freq)}.
        If None, uses standard EEG bands.
    output_path : str, Path, or None, default None
        Path to save the plot. If None, plot is not saved.
    title : str, default "Power Spectral Density Topography"
        Title for the plot.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> fig = plot_psd_topography(raw)
    >>> fig = plot_psd_topography(raw, freq_bands={'alpha': (8, 12)})
    
    See Also
    --------
    plot_raw_comparison : Plot before/after data comparison
    mne.io.Raw.compute_psd : Compute power spectral density
    """
    # Input validation
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError(f"raw must be an MNE Raw object, got {type(raw).__name__}")

    # Default frequency bands
    if freq_bands is None:
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8), 
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 50)
        }

    try:
        # Calculate PSD
        spectrum = raw.compute_psd(fmax=50, verbose=verbose)
        
        # Create the plot using MNE's function
        fig = spectrum.plot_topomap(
            bands=freq_bands,
            show=False
        )
        
        # Customize title
        fig.suptitle(title, fontsize=14)
        
        # Save if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            if verbose:
                print(f"PSD topography plot saved to: {output_path}")

        return fig

    except Exception as e:
        raise RuntimeError(f"Failed to create PSD topography plot: {str(e)}") from e