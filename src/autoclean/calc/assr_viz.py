import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mne
from matplotlib.colors import LinearSegmentedColormap
import os
import pandas as pd

# Example use
# To perform group analysis on exported data:
#    python assr_viz.py path/to/eeglab_file.set --group_analysis --data_dir path/to/exported_data --data_type itc
# To export heat maps for a single subject:
#    python assr_viz.py path/to/eeglab_file.set --export_data
# To export heat maps in CSV format:
#    python assr_viz.py path/to/eeglab_file.set --export_data --export_csv


def create_itc_colormap():
    """Create a custom colormap for ITC visualization"""
    colors = [(0, 0, 0.5),   # Dark blue for low values
              (0, 0, 0.8),   # Blue
              (0, 0.5, 0.8), # Blue-green
              (0.5, 0.8, 0), # Green-yellow
              (1, 1, 0)]     # Yellow for high values
    return LinearSegmentedColormap.from_list('assr_cmap', colors, N=100)


def create_ersp_colormap():
    """Create a custom colormap for ERSP visualization"""
    colors = [(0, 0, 0.8),    # Blue for negative values
              (1, 1, 1),      # White for zero
              (0.8, 0, 0)]    # Red for positive values
    return LinearSegmentedColormap.from_list('ersp_cmap', colors, N=100)


def plot_itc_channels(tf_data, epochs, output_dir=None, save_figures=True, file_basename=None):
    """
    Plot ITC for each channel
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with ITC plots
    """
    # Unpack time-frequency data
    itc = tf_data['itc']
    
    # Create a figure with subplots for each channel in a grid
    n_channels = len(epochs.ch_names)
    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Create a custom colormap for ITC
    cmap_assr = create_itc_colormap()
    
    # Plot each channel's ITC in its own subplot
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        if ch_idx < len(axes):
            # Unpack the tuple to get the actual TFR object (power and itc)
            power, itc_obj = itc
            
            # Plot single channel data using the correct itc object
            itc_obj.plot(
                picks=[ch_idx],
                title=f"Channel: {ch_name}",
                cmap=cmap_assr,
                vlim=(0, 0.4),
                axes=axes[ch_idx],
                colorbar=False,
                show=False
            )
            
            # Add a line to highlight the 40 Hz response
            axes[ch_idx].axhline(40, color='red', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add channel label in the corner of the plot
            axes[ch_idx].text(0.02, 0.98, ch_name, transform=axes[ch_idx].transAxes, 
                             fontsize=9, fontweight='bold', va='top', ha='left',
                             bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # Hide any unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)

    # Add a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('ITC Value')

    plt.suptitle('Inter-Trial Coherence (ITC) - Optimized for 40 Hz ASSR', fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_fig_itc_channels.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_fig_itc_channels.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved ITC channels plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_global_mean_itc(tf_data, output_dir=None, save_figures=True, epochs=None, file_basename=None):
    """
    Plot global mean ITC (average across all channels)
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    epochs : mne.Epochs, optional
        Epochs object for filename extraction
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with global mean ITC plot
    """
    # Unpack time-frequency data
    itc = tf_data['itc']
    
    # Create a custom colormap for ITC
    cmap_assr = create_itc_colormap()
    
    # Create figure for global mean ITC
    fig = plt.figure(figsize=(10, 6))
    
    # Fix the tuple unpacking issue - itc is a tuple, not an object
    power, itc_obj = itc
    
    # Create a copy of the original ITC object
    itc_avg = itc_obj.copy()
    
    # Average across all channels
    itc_avg.data = itc_obj.data.mean(axis=0, keepdims=True)
    
    # Keep only the first channel in the info
    itc_avg.pick([itc_avg.ch_names[0]])
    
    # Plot the global mean ITC
    itc_avg.plot(
        picks=[0],  # Only one channel exists now (the average)
        title='Global Mean ITC across all channels (40 Hz optimized)',
        cmap=cmap_assr,
        vlim=(0, 0.4),
        colorbar=True,
        show=False
    )
    
    plt.axhline(40, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='40 Hz')
    plt.text(0.02, 0.98, 'GLOBAL MEAN', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.7, pad=2))
    plt.legend()
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif epochs is not None and hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_summary_global_itc.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_summary_global_itc.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved global mean ITC plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_topomap(tf_data, epochs, time_point=0.3, output_dir=None, save_figures=True, file_basename=None):
    """
    Plot topographic map of ITC at 40 Hz
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    time_point : float, optional
        Time point for topographic map in seconds
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with topographic map
    """
    # Unpack time-frequency data
    itc = tf_data['itc']
    freqs = tf_data['freqs']
    
    # Create a custom colormap for ITC
    cmap_assr = create_itc_colormap()
    
    # Find the index of 40 Hz in our frequency array
    freq_idx = np.argmin(np.abs(freqs - 40))
    
    # Find time point for topo map
    time_idx = np.argmin(np.abs(itc[1].times - time_point))
    
    # Plot the topographic map
    fig, ax = plt.subplots(figsize=(8, 8))
    
    itc[1].plot_topomap(
        ch_type='eeg', 
        tmin=itc[1].times[time_idx], 
        tmax=itc[1].times[time_idx],
        fmin=freqs[freq_idx], 
        fmax=freqs[freq_idx],
        vlim=(0, 0.4),
        cmap=cmap_assr,
        axes=ax,
        show=False
    )
    
    # Set title separately using matplotlib
    ax.set_title(f'40 Hz ASSR Topography at t={itc[1].times[time_idx]:.2f}s')
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_topography_40hz_t{time_point:.2f}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_topography_40hz_t{time_point:.2f}.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved topography plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_ersp_channels(tf_data, epochs, output_dir=None, save_figures=True, file_basename=None):
    """
    Plot ERSP for each channel
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with ERSP plots
    """
    # Unpack time-frequency data
    ersp = tf_data['ersp']
    
    # Create a figure with subplots for each channel in a grid
    n_channels = len(epochs.ch_names)
    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Create a custom colormap for ERSP
    cmap_ersp = create_ersp_colormap()
    
    # Calculate color scale limits based on data percentiles
    # Use more extreme percentiles for better contrast if the data distribution is narrow
    vmin, vmax = np.percentile(ersp.data, 5), np.percentile(ersp.data, 95)
    
    # Ensure symmetrical limits for better visualization
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    # Plot each channel's ERSP in its own subplot
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        if ch_idx < len(axes):
            # Plot single channel data
            ersp.plot(
                picks=[ch_idx],
                title=f"Channel: {ch_name}",
                cmap=cmap_ersp,
                vlim=(vmin, vmax),
                axes=axes[ch_idx],
                colorbar=False,
                show=False
            )
            
            # Add a line to highlight the 40 Hz response
            axes[ch_idx].axhline(40, color='black', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add channel label in the corner of the plot
            axes[ch_idx].text(0.02, 0.98, ch_name, transform=axes[ch_idx].transAxes, 
                             fontsize=9, fontweight='bold', va='top', ha='left',
                             bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # Hide any unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)

    # Add a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('Power (dB)')

    plt.suptitle('Event-Related Spectral Perturbation (ERSP) - 40 Hz ASSR', fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_fig_ersp_channels.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_fig_ersp_channels.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved ERSP plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_global_mean_ersp(tf_data, output_dir=None, save_figures=True, epochs=None, file_basename=None):
    """
    Plot global mean ERSP (average across all channels)
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    epochs : mne.Epochs, optional
        Epochs object for filename extraction
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with global mean ERSP plot
    """
    # Unpack time-frequency data
    ersp = tf_data['ersp']
    
    # Create a custom colormap for ERSP
    cmap_ersp = create_ersp_colormap()
    
    # Create figure for global mean ERSP
    fig = plt.figure(figsize=(10, 6))
    
    # Create a copy of the original ERSP object
    ersp_avg = ersp.copy()
    
    # Average across all channels
    ersp_avg.data = ersp.data.mean(axis=0, keepdims=True)
    
    # Keep only the first channel in the info
    ersp_avg.pick([ersp_avg.ch_names[0]])
    
    # Calculate color scale limits based on data percentiles
    vmin, vmax = np.percentile(ersp_avg.data, 5), np.percentile(ersp_avg.data, 95)
    
    # Ensure symmetrical limits for better visualization
    abs_max = max(abs(vmin), abs(vmax))
    vmin, vmax = -abs_max, abs_max
    
    # Plot the global mean ERSP
    ersp_avg.plot(
        picks=[0],  # Only one channel exists now (the average)
        title='Global Mean ERSP across all channels',
        cmap=cmap_ersp,
        vlim=(vmin, vmax),
        colorbar=True,
        show=False
    )
    
    plt.axhline(40, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='40 Hz')
    plt.text(0.02, 0.98, 'GLOBAL MEAN', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.7, pad=2))
    plt.legend()
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif epochs is not None and hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_summary_global_ersp.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_summary_global_ersp.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved global mean ERSP plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_stp_channels(tf_data, epochs, output_dir=None, save_figures=True, file_basename=None):
    """
    Plot Single Trial Power (STP) for each channel
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with STP plots
    """
    # Unpack time-frequency data
    single_trial_power = tf_data['single_trial_power']
    freqs = tf_data['freqs']
    
    # Check if we're dealing with an EpochsTFR object
    is_epochs_tfr = hasattr(single_trial_power, 'average') and callable(getattr(single_trial_power, 'average'))
    
    if is_epochs_tfr:
        # If it's an EpochsTFR, we can use its average method to create an AverageTFR
        stp = single_trial_power.average()
    else:
        # First, average across trials to get a channel x frequency x time representation
        avg_power = np.mean(single_trial_power.data, axis=0)
        
        # Create a copy with the trial-averaged data
        stp = single_trial_power.copy()
        stp.data = avg_power
    
    # Create a figure with subplots for each channel in a grid
    n_channels = len(epochs.ch_names)
    n_rows = int(np.ceil(np.sqrt(n_channels)))
    n_cols = int(np.ceil(n_channels / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    # Create a custom colormap for STP
    cmap_stp = plt.cm.viridis  # Using viridis colormap for power
    
    # Get color scale limits based on data percentiles
    vmin, vmax = np.percentile(stp.data, 5), np.percentile(stp.data, 95)
    
    # Plot each channel's STP in its own subplot
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        if ch_idx < len(axes):
            # Plot single channel data
            stp.plot(
                picks=[ch_idx],
                title=f"Channel: {ch_name}",
                cmap=cmap_stp,
                vlim=(vmin, vmax),
                axes=axes[ch_idx],
                colorbar=False,
                show=False
            )
            
            # Add a line to highlight the 40 Hz response
            axes[ch_idx].axhline(40, color='white', linestyle='--', linewidth=1, alpha=0.7)
            
            # Add channel label in the corner of the plot
            axes[ch_idx].text(0.02, 0.98, ch_name, transform=axes[ch_idx].transAxes, 
                             fontsize=9, fontweight='bold', va='top', ha='left',
                             bbox=dict(facecolor='white', alpha=0.7, pad=1))

    # Hide any unused subplots
    for idx in range(n_channels, len(axes)):
        axes[idx].set_visible(False)

    # Add a colorbar for the entire figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(axes[0].images[0], cax=cbar_ax)
    cbar.set_label('Power (µV²)')

    plt.suptitle('Single Trial Power (STP) - 40 Hz ASSR', fontsize=16)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_fig_stp_channels.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_fig_stp_channels.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved STP plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_global_mean_stp(tf_data, epochs, output_dir=None, save_figures=True, file_basename=None):
    """
    Plot global mean STP (average across all channels)
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with global mean STP plot
    """
    # Unpack time-frequency data
    single_trial_power = tf_data['single_trial_power']
    freqs = tf_data['freqs']
    
    # Check if we're dealing with an EpochsTFR object
    is_epochs_tfr = hasattr(single_trial_power, 'average') and callable(getattr(single_trial_power, 'average'))
    
    if is_epochs_tfr:
        # If it's an EpochsTFR, we can use its average method to create an AverageTFR
        stp_avg = single_trial_power.average()
        
        # Now average across channels
        stp_avg.data = stp_avg.data.mean(axis=0, keepdims=True)
        
        # Keep only the first channel in the info
        stp_avg.pick([stp_avg.ch_names[0]])
    else:
        # First, average across trials to get a channel x frequency x time representation
        avg_power = np.mean(single_trial_power.data, axis=0)
        
        # Create a copy of the original object
        stp_avg = single_trial_power.copy()
        
        # Average across channels
        stp_avg.data = np.mean(avg_power, axis=0, keepdims=True)
        
        # Keep only the first channel in the info
        stp_avg.pick([stp_avg.ch_names[0]])
    
    # Create figure for global mean STP
    fig = plt.figure(figsize=(10, 6))
    
    # Create a custom colormap for STP
    cmap_stp = plt.cm.viridis  # Using viridis colormap for power
    
    # Get color scale limits based on data percentiles
    vmin, vmax = np.percentile(stp_avg.data, 5), np.percentile(stp_avg.data, 95)

    # Plot the global mean STP
    stp_avg.plot(
        picks=[0],  # Only one channel exists now (the average)
        title='Global Mean Single Trial Power across all channels',
        cmap=cmap_stp,
        vlim=(vmin, vmax),
        colorbar=True,
        show=False
    )
    
    plt.axhline(40, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='40 Hz')
    plt.text(0.02, 0.98, 'GLOBAL MEAN', transform=plt.gca().transAxes, 
             fontsize=12, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.7, pad=2))
    plt.legend()
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
    
    # Save figure if requested
    if save_figures and output_dir is not None:
        # Get the basename of the epochs file
        if file_basename is not None:
            # Use provided basename - takes precedence
            pass
        elif hasattr(epochs, 'filename') and epochs.filename is not None:
            file_basename = Path(epochs.filename).stem
        else:
            file_basename = "unknown"
            
        # Create figures subdirectory
        output_dir = Path(output_dir)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with descriptive filename including basename
        fig_path = figures_dir / f"{file_basename}_summary_global_stp.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = figures_dir / f"{file_basename}_summary_global_stp.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved global mean STP plot to {fig_path} and {fig_path_pdf}")
    
    return fig


def export_heatmap_data(tf_data, epochs, output_dir=None, file_basename=None):
    """
    Export heat map data for ITC, ERSP, and STP to numpy files for later group-level analysis.
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    output_dir : str or Path, optional
        Directory to save data files
    file_basename : str, optional
        Base filename to use for saving data, takes precedence over epoch filename
    
    Returns:
    --------
    export_paths : dict
        Dictionary containing paths to exported data files
    """
    # Get the basename of the epochs file
    if file_basename is not None:
        # Use provided basename - takes precedence
        pass
    elif hasattr(epochs, 'filename') and epochs.filename is not None:
        file_basename = Path(epochs.filename).stem
    else:
        file_basename = "unknown"
        
    # Create data export directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        data_dir = output_dir / "exported_data"
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Use current directory if no output_dir specified
        data_dir = Path("exported_data")
        data_dir.mkdir(exist_ok=True)
    
    export_paths = {}
    
    # Export ITC data
    if 'itc' in tf_data:
        power, itc_obj = tf_data['itc']
        
        # Extract data, times, and frequencies
        itc_data = itc_obj.data  # Shape: (channels, freqs, times)
        times = itc_obj.times
        freqs = tf_data['freqs']
        ch_names = epochs.ch_names
        
        # Create a dictionary with all necessary information
        itc_export = {
            'data': itc_data,
            'times': times,
            'freqs': freqs,
            'ch_names': ch_names,
            'subject_id': file_basename
        }
        
        # Save to numpy file
        itc_path = data_dir / f"{file_basename}_itc_data.npy"
        np.save(itc_path, itc_export)
        export_paths['itc'] = itc_path
        print(f"Exported ITC data to {itc_path}")
        
        # Also export raw data as CSV for use in other tools
        export_raw_data_as_csv(itc_data, times, freqs, ch_names, data_dir, f"{file_basename}_itc")
        export_paths['itc_csv'] = data_dir / f"{file_basename}_itc_data_info.csv"
    
    # Export ERSP data
    if 'ersp' in tf_data:
        ersp = tf_data['ersp']
        
        # Extract data, times, and frequencies
        ersp_data = ersp.data  # Shape: (channels, freqs, times)
        times = ersp.times
        freqs = tf_data['freqs']
        ch_names = epochs.ch_names
        
        # Create a dictionary with all necessary information
        ersp_export = {
            'data': ersp_data,
            'times': times,
            'freqs': freqs,
            'ch_names': ch_names,
            'subject_id': file_basename
        }
        
        # Save to numpy file
        ersp_path = data_dir / f"{file_basename}_ersp_data.npy"
        np.save(ersp_path, ersp_export)
        export_paths['ersp'] = ersp_path
        print(f"Exported ERSP data to {ersp_path}")
        
        # Also export raw data as CSV for use in other tools
        export_raw_data_as_csv(ersp_data, times, freqs, ch_names, data_dir, f"{file_basename}_ersp")
        export_paths['ersp_csv'] = data_dir / f"{file_basename}_ersp_data_info.csv"
    
    # Export STP data
    if 'single_trial_power' in tf_data:
        single_trial_power = tf_data['single_trial_power']
        
        # Check if we're dealing with an EpochsTFR object
        is_epochs_tfr = hasattr(single_trial_power, 'average') and callable(getattr(single_trial_power, 'average'))
        
        if is_epochs_tfr:
            # If it's an EpochsTFR, we can use its average method to create an AverageTFR
            stp = single_trial_power.average()
            stp_data = stp.data
            times = stp.times
        else:
            # First, average across trials to get a channel x frequency x time representation
            stp_data = np.mean(single_trial_power.data, axis=0)
            times = single_trial_power.times
        
        freqs = tf_data['freqs']
        ch_names = epochs.ch_names
        
        # Create a dictionary with all necessary information
        stp_export = {
            'data': stp_data,
            'times': times,
            'freqs': freqs,
            'ch_names': ch_names,
            'subject_id': file_basename
        }
        
        # Save to numpy file
        stp_path = data_dir / f"{file_basename}_stp_data.npy"
        np.save(stp_path, stp_export)
        export_paths['stp'] = stp_path
        print(f"Exported STP data to {stp_path}")
        
        # Also export raw data as CSV for use in other tools
        export_raw_data_as_csv(stp_data, times, freqs, ch_names, data_dir, f"{file_basename}_stp")
        export_paths['stp_csv'] = data_dir / f"{file_basename}_stp_data_info.csv"
    
    # Create a metadata file with information about the export
    metadata = {
        'subject_id': file_basename,
        'exported_data_types': list(export_paths.keys()),
        'ch_names': epochs.ch_names,
        'n_channels': len(epochs.ch_names),
        'n_freqs': len(tf_data['freqs']),
        'freq_range': [min(tf_data['freqs']), max(tf_data['freqs'])],
        'time_range': [min(times), max(times)]
    }
    
    metadata_path = data_dir / f"{file_basename}_export_metadata.npy"
    np.save(metadata_path, metadata)
    export_paths['metadata'] = metadata_path
    
    return export_paths


def export_raw_data_as_csv(data, times, freqs, ch_names, output_dir, base_filename):
    """
    Export raw data arrays to CSV files for use in other analysis tools.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data array with shape (channels, freqs, times)
    times : numpy.ndarray
        Array of time points
    freqs : numpy.ndarray
        Array of frequencies
    ch_names : list
        List of channel names
    output_dir : Path
        Directory to save CSV files
    base_filename : str
        Base filename for CSV files
    
    Returns:
    --------
    csv_paths : dict
        Dictionary containing paths to exported CSV files
    """
    output_dir = Path(output_dir)
    csv_paths = {}
    
    # Export data info (times, freqs, channel names)
    info_df = pd.DataFrame({
        'times': times,
        'freqs': freqs,
        'ch_names': ch_names + [''] * (max(len(times), len(freqs)) - len(ch_names))
    })
    info_path = output_dir / f"{base_filename}_data_info.csv"
    info_df.to_csv(info_path, index=False)
    csv_paths['info'] = info_path
    
    # Export data for each channel
    for ch_idx, ch_name in enumerate(ch_names):
        # Create a DataFrame with times as columns and freqs as rows
        ch_data = data[ch_idx]
        ch_df = pd.DataFrame(ch_data, index=freqs, columns=times)
        
        # Save to CSV
        ch_path = output_dir / f"{base_filename}_channel_{ch_name}.csv"
        ch_df.to_csv(ch_path)
        csv_paths[f'channel_{ch_name}'] = ch_path
    
    # Export global mean (average across channels)
    global_mean = np.mean(data, axis=0)
    global_df = pd.DataFrame(global_mean, index=freqs, columns=times)
    global_path = output_dir / f"{base_filename}_global_mean.csv"
    global_df.to_csv(global_path)
    csv_paths['global_mean'] = global_path
    
    print(f"Exported raw data as CSV to {output_dir}")
    return csv_paths


def group_analysis_heatmaps(data_dir, data_type='itc', output_dir=None, save_figures=True, group_name='group_average'):
    """
    Perform group-level analysis on exported heat map data.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing exported heat map data files
    data_type : str, optional
        Type of data to analyze ('itc', 'ersp', or 'stp')
    output_dir : str or Path, optional
        Directory to save group-level results
    save_figures : bool, optional
        Whether to save figures to disk
    group_name : str, optional
        Name to use for the group average files
    
    Returns:
    --------
    group_data : dict
        Dictionary containing group-level data and analysis results
    """
    data_dir = Path(data_dir)
    
    # Set up output directory
    if output_dir is None:
        output_dir = data_dir / "group_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files of the specified data type
    data_files = list(data_dir.glob(f"*_{data_type}_data.npy"))
    
    if not data_files:
        raise ValueError(f"No {data_type} data files found in {data_dir}")
    
    print(f"Found {len(data_files)} {data_type} data files for group analysis")
    
    # Load the first file to get dimensions and structure
    first_data = np.load(data_files[0], allow_pickle=True).item()
    
    # Initialize arrays for group data
    n_subjects = len(data_files)
    n_channels = len(first_data['ch_names'])
    n_freqs = len(first_data['freqs'])
    n_times = len(first_data['times'])
    
    # Create array to hold all subjects' data
    all_data = np.zeros((n_subjects, n_channels, n_freqs, n_times))
    subject_ids = []
    
    # Load data from all files
    for i, file_path in enumerate(data_files):
        data_dict = np.load(file_path, allow_pickle=True).item()
        all_data[i] = data_dict['data']
        subject_ids.append(data_dict['subject_id'])
    
    # Calculate group average
    group_avg = np.mean(all_data, axis=0)
    
    # Calculate standard error of the mean
    group_sem = np.std(all_data, axis=0) / np.sqrt(n_subjects)
    
    # Create a dictionary with group-level data
    group_data = {
        'data': group_avg,
        'sem': group_sem,
        'times': first_data['times'],
        'freqs': first_data['freqs'],
        'ch_names': first_data['ch_names'],
        'n_subjects': n_subjects,
        'subject_ids': subject_ids,
        'data_type': data_type
    }
    
    # Save group data
    group_data_path = output_dir / f"{group_name}_{data_type}_data.npy"
    np.save(group_data_path, group_data)
    print(f"Saved group {data_type} data to {group_data_path}")
    
    # Create and save a group-level figure
    if save_figures:
        # Create appropriate colormap based on data type
        if data_type == 'itc':
            cmap = create_itc_colormap()
            vmin, vmax = 0, 0.4
            cbar_label = 'ITC Value'
            title = f'Group Average ITC (n={n_subjects})'
        elif data_type == 'ersp':
            cmap = create_ersp_colormap()
            # Calculate color scale limits based on data percentiles
            vmin, vmax = np.percentile(group_avg, 5), np.percentile(group_avg, 95)
            # Ensure symmetrical limits for better visualization
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            cbar_label = 'Power (dB)'
            title = f'Group Average ERSP (n={n_subjects})'
        else:  # STP
            cmap = plt.cm.viridis
            vmin, vmax = np.percentile(group_avg, 5), np.percentile(group_avg, 95)
            cbar_label = 'Power (µV²)'
            title = f'Group Average STP (n={n_subjects})'
        
        # Create global mean figure (average across channels)
        global_avg = np.mean(group_avg, axis=0, keepdims=True)
        
        fig = plt.figure(figsize=(10, 6))
        
        # Create a fake AverageTFR object for plotting
        info = mne.create_info(ch_names=['Global'], sfreq=1000, ch_types='eeg')
        fake_tfr = mne.time_frequency.AverageTFR(
            info=info,
            data=global_avg,
            times=group_data['times'],
            freqs=group_data['freqs'],
            nave=n_subjects
        )
        
        # Plot the global mean
        fake_tfr.plot(
            picks=[0],
            title=title,
            cmap=cmap,
            vlim=(vmin, vmax),
            colorbar=True,
            show=False
        )
        
        plt.axhline(40, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='40 Hz')
        plt.text(0.02, 0.98, 'GROUP AVERAGE', transform=plt.gca().transAxes, 
                fontsize=12, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.7, pad=2))
        plt.legend()
        
        # Save figure
        fig_path = output_dir / f"{group_name}_{data_type}_global_average.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = output_dir / f"{group_name}_{data_type}_global_average.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved group {data_type} figure to {fig_path} and {fig_path_pdf}")
        
        plt.close(fig)
        
        # Create a topographic map at 40 Hz and a specific time point
        # Find the index of 40 Hz in our frequency array
        freq_idx = np.argmin(np.abs(group_data['freqs'] - 40))
        
        # Find time point for topo map (default to 0.3s or middle of time range if not available)
        time_point = 0.3
        if time_point < min(group_data['times']) or time_point > max(group_data['times']):
            time_point = np.mean([min(group_data['times']), max(group_data['times'])])
        time_idx = np.argmin(np.abs(group_data['times'] - time_point))
        
        # Create a topographic map
        plot_group_topomap(
            group_data, 
            freq_idx=freq_idx, 
            time_idx=time_idx, 
            output_dir=output_dir, 
            save_figures=save_figures, 
            group_name=group_name
        )
    
    return group_data


def plot_group_topomap(group_data, freq_idx=None, time_idx=None, freq=40, time_point=0.3, 
                      output_dir=None, save_figures=True, group_name='group_average'):
    """
    Plot topographic map of group-level data at a specific frequency and time point.
    
    Parameters:
    -----------
    group_data : dict
        Dictionary containing group-level data from group_analysis_heatmaps
    freq_idx : int, optional
        Index of frequency to plot. If None, will find closest to freq parameter.
    time_idx : int, optional
        Index of time point to plot. If None, will find closest to time_point parameter.
    freq : float, optional
        Frequency in Hz to plot (used if freq_idx is None)
    time_point : float, optional
        Time point in seconds to plot (used if time_idx is None)
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    group_name : str, optional
        Name to use for the group average files
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object with topographic map
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path("group_analysis")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find frequency index if not provided
    if freq_idx is None:
        freq_idx = np.argmin(np.abs(group_data['freqs'] - freq))
    
    # Find time index if not provided
    if time_idx is None:
        time_idx = np.argmin(np.abs(group_data['times'] - time_point))
    
    # Get actual frequency and time values
    actual_freq = group_data['freqs'][freq_idx]
    actual_time = group_data['times'][time_idx]
    
    # Get data for this frequency and time point
    data_type = group_data['data_type']
    topo_data = group_data['data'][:, freq_idx, time_idx]
    
    # Create appropriate colormap and limits based on data type
    if data_type == 'itc':
        cmap = create_itc_colormap()
        vmin, vmax = 0, 0.4
        title = f'Group Average ITC at {actual_freq:.1f} Hz, t={actual_time:.2f}s (n={group_data["n_subjects"]})'
    elif data_type == 'ersp':
        cmap = create_ersp_colormap()
        # Calculate color scale limits based on data percentiles
        vmin, vmax = np.percentile(group_data['data'], 5), np.percentile(group_data['data'], 95)
        # Ensure symmetrical limits for better visualization
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max
        title = f'Group Average ERSP at {actual_freq:.1f} Hz, t={actual_time:.2f}s (n={group_data["n_subjects"]})'
    else:  # STP
        cmap = plt.cm.viridis
        vmin, vmax = np.percentile(group_data['data'], 5), np.percentile(group_data['data'], 95)
        title = f'Group Average STP at {actual_freq:.1f} Hz, t={actual_time:.2f}s (n={group_data["n_subjects"]})'
    
    # Create a montage from channel names if possible
    # This is a simplified approach - in practice, you might need to match channel names to standard positions
    try:
        # Try to create a standard montage based on channel names
        montage = mne.channels.make_standard_montage('standard_1020')
        
        # Create info object with channel names
        info = mne.create_info(ch_names=group_data['ch_names'], sfreq=1000, ch_types='eeg')
        
        # Create a fake evoked object for topographic plotting
        evoked = mne.EvokedArray(topo_data.reshape(len(topo_data), 1), info)
        
        # Set montage
        evoked.set_montage(montage)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot topographic map
        im, cm = mne.viz.plot_topomap(
            topo_data, 
            evoked.info,
            axes=ax,
            show=False,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            outlines='head',
            contours=6
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if data_type == 'itc':
            cbar.set_label('ITC Value')
        elif data_type == 'ersp':
            cbar.set_label('Power (dB)')
        else:  # STP
            cbar.set_label('Power (µV²)')
        
        # Set title
        ax.set_title(title)
        
    except Exception as e:
        # If montage creation fails, create a simplified topographic plot
        print(f"Warning: Could not create standard montage. Using simplified topographic plot. Error: {e}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a grid of positions based on the number of channels
        n_channels = len(group_data['ch_names'])
        grid_size = int(np.ceil(np.sqrt(n_channels)))
        
        # Create x and y coordinates for each channel
        x = np.linspace(-0.8, 0.8, grid_size)
        y = np.linspace(-0.8, 0.8, grid_size)
        xx, yy = np.meshgrid(x, y)
        
        # Flatten coordinates
        pos = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Limit to the number of channels
        pos = pos[:n_channels]
        
        # Plot simplified topographic map
        im = ax.scatter(
            pos[:, 0], 
            pos[:, 1], 
            c=topo_data, 
            s=200, 
            cmap=cmap, 
            vmin=vmin, 
            vmax=vmax
        )
        
        # Add channel labels
        for i, ch_name in enumerate(group_data['ch_names']):
            ax.text(pos[i, 0], pos[i, 1], ch_name, ha='center', va='center', fontsize=8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if data_type == 'itc':
            cbar.set_label('ITC Value')
        elif data_type == 'ersp':
            cbar.set_label('Power (dB)')
        else:  # STP
            cbar.set_label('Power (µV²)')
        
        # Set title
        ax.set_title(title)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add a circle to represent the head
        circle = plt.Circle((0, 0), 0.9, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add nose
        ax.plot([0, 0], [0.9, 1.1], 'k-', linewidth=2)
        
        # Add ears
        ax.plot([-0.9, -1.1], [0, 0], 'k-', linewidth=2)
        ax.plot([0.9, 1.1], [0, 0], 'k-', linewidth=2)
    
    # Save figure if requested
    if save_figures:
        fig_path = output_dir / f"{group_name}_{data_type}_topomap_{actual_freq:.1f}Hz_t{actual_time:.2f}s.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        fig_path_pdf = output_dir / f"{group_name}_{data_type}_topomap_{actual_freq:.1f}Hz_t{actual_time:.2f}s.pdf"
        plt.savefig(fig_path_pdf, bbox_inches='tight')
        print(f"Saved group {data_type} topographic map to {fig_path} and {fig_path_pdf}")
    
    return fig


def plot_all_figures(tf_data, epochs, output_dir=None, save_figures=True, file_basename=None, export_data=False, export_csv=False):
    """
    Generate all plots from the analysis
    
    Parameters:
    -----------
    tf_data : dict
        Output from compute_time_frequency function
    epochs : mne.Epochs
        Epochs object used for computation
    output_dir : str or Path, optional
        Directory to save figures
    save_figures : bool, optional
        Whether to save figures to disk
    file_basename : str, optional
        Base filename to use for saving figures, takes precedence over epoch filename
    export_data : bool, optional
        Whether to export heat map data for group-level analysis
    export_csv : bool, optional
        Whether to export data as CSV files for use in other tools
        
    Returns:
    --------
    figs : dict
        Dictionary containing all figure objects
    """
    figs = {}
    
    # Plot ITC for all channels
    figs['itc_channels'] = plot_itc_channels(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Plot global mean ITC - pass epochs for filename extraction
    figs['global_mean_itc'] = plot_global_mean_itc(tf_data, output_dir, save_figures, epochs, file_basename)
    
    # Plot topographic map
    figs['topomap'] = plot_topomap(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Plot ERSP for all channels
    figs['ersp_channels'] = plot_ersp_channels(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Plot global mean ERSP - pass epochs for filename extraction
    figs['global_mean_ersp'] = plot_global_mean_ersp(tf_data, output_dir, save_figures, epochs, file_basename)
    
    # Plot STP for all channels
    figs['stp_channels'] = plot_stp_channels(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Plot global mean STP
    figs['global_mean_stp'] = plot_global_mean_stp(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Export heat map data if requested
    if export_data:
        export_paths = export_heatmap_data(tf_data, epochs, output_dir, file_basename)
        figs['export_paths'] = export_paths
    
    # Export data as CSV if requested
    if export_csv and not export_data:
        # If we haven't already exported the data through export_heatmap_data
        csv_paths = {}
        
        # Export ITC data
        if 'itc' in tf_data:
            power, itc_obj = tf_data['itc']
            itc_data = itc_obj.data
            times = itc_obj.times
            freqs = tf_data['freqs']
            ch_names = epochs.ch_names
            
            # Create data export directory
            if output_dir is not None:
                output_dir = Path(output_dir)
                data_dir = output_dir / "exported_data"
                data_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Use current directory if no output_dir specified
                data_dir = Path("exported_data")
                data_dir.mkdir(exist_ok=True)
            
            # Get the basename of the epochs file
            if file_basename is not None:
                # Use provided basename - takes precedence
                pass
            elif hasattr(epochs, 'filename') and epochs.filename is not None:
                file_basename = Path(epochs.filename).stem
            else:
                file_basename = "unknown"
            
            csv_paths['itc'] = export_raw_data_as_csv(itc_data, times, freqs, ch_names, data_dir, f"{file_basename}_itc")
        
        # Export ERSP data
        if 'ersp' in tf_data:
            ersp = tf_data['ersp']
            ersp_data = ersp.data
            times = ersp.times
            freqs = tf_data['freqs']
            ch_names = epochs.ch_names
            
            csv_paths['ersp'] = export_raw_data_as_csv(ersp_data, times, freqs, ch_names, data_dir, f"{file_basename}_ersp")
        
        # Export STP data
        if 'single_trial_power' in tf_data:
            single_trial_power = tf_data['single_trial_power']
            
            # Check if we're dealing with an EpochsTFR object
            is_epochs_tfr = hasattr(single_trial_power, 'average') and callable(getattr(single_trial_power, 'average'))
            
            if is_epochs_tfr:
                # If it's an EpochsTFR, we can use its average method to create an AverageTFR
                stp = single_trial_power.average()
                stp_data = stp.data
                times = stp.times
            else:
                # First, average across trials to get a channel x frequency x time representation
                stp_data = np.mean(single_trial_power.data, axis=0)
                times = single_trial_power.times
            
            freqs = tf_data['freqs']
            ch_names = epochs.ch_names
            
            csv_paths['stp'] = export_raw_data_as_csv(stp_data, times, freqs, ch_names, data_dir, f"{file_basename}_stp")
        
        figs['csv_paths'] = csv_paths
    
    return figs


if __name__ == "__main__":
    import argparse
    import sys
    import pandas as pd
    
    # Add parent directory to path for imports when running this file directly
    sys.path.append('..')
    from assr_analysis import analyze_assr
    
    parser = argparse.ArgumentParser(description='Generate visualizations for ASSR data')
    parser.add_argument('file_path', type=str, help='Path to the EEGLAB .set file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save figures')
    parser.add_argument('--no_save_figures', action='store_false', dest='save_figures', 
                        help='Do not save figures to disk')
    parser.add_argument('--export_data', action='store_true', 
                        help='Export heat map data for group-level analysis')
    parser.add_argument('--export_csv', action='store_true',
                        help='Export data as CSV files for use in other tools')
    parser.add_argument('--group_analysis', action='store_true',
                        help='Perform group analysis on exported data')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing exported data for group analysis')
    parser.add_argument('--data_type', type=str, default='itc', choices=['itc', 'ersp', 'stp'],
                        help='Type of data to analyze for group analysis')
    parser.add_argument('--group_name', type=str, default='group_average',
                        help='Name to use for group average files')
    parser.add_argument('--freq', type=float, default=40.0,
                        help='Frequency in Hz for topographic map (default: 40 Hz)')
    parser.add_argument('--time', type=float, default=0.3,
                        help='Time point in seconds for topographic map (default: 0.3s)')
    parser.add_argument('--plot_topomap_only', action='store_true',
                        help='Only plot topographic map from existing group data')
    
    args = parser.parse_args()
    
    # Check if we're only plotting a topographic map
    if args.plot_topomap_only:
        if args.data_dir is None:
            if args.output_dir is not None:
                # Use output_dir/exported_data/group_analysis as the default data_dir
                args.data_dir = os.path.join(args.output_dir, "exported_data", "group_analysis")
            else:
                # Use current directory/group_analysis as the default
                args.data_dir = "group_analysis"
        
        # Load group data
        group_data_path = os.path.join(args.data_dir, f"{args.group_name}_{args.data_type}_data.npy")
        if not os.path.exists(group_data_path):
            print(f"Error: Group data file not found at {group_data_path}")
            sys.exit(1)
        
        group_data = np.load(group_data_path, allow_pickle=True).item()
        
        # Plot topographic map
        plot_group_topomap(
            group_data,
            freq=args.freq,
            time_point=args.time,
            output_dir=args.output_dir,
            save_figures=args.save_figures,
            group_name=args.group_name
        )
    # Check if we're doing group analysis
    elif args.group_analysis:
        if args.data_dir is None:
            if args.output_dir is not None:
                # Use output_dir/exported_data as the default data_dir
                args.data_dir = os.path.join(args.output_dir, "exported_data")
            else:
                # Use current directory/exported_data as the default
                args.data_dir = "exported_data"
        
        # Perform group analysis
        group_data = group_analysis_heatmaps(
            args.data_dir,
            data_type=args.data_type,
            output_dir=args.output_dir,
            save_figures=args.save_figures,
            group_name=args.group_name
        )
        
        # Export group data as CSV if requested
        if args.export_csv:
            export_raw_data_as_csv(
                group_data['data'],
                group_data['times'],
                group_data['freqs'],
                group_data['ch_names'],
                Path(args.output_dir) if args.output_dir else Path("group_analysis"),
                f"{args.group_name}_{args.data_type}"
            )
    else:
        # Run analysis first
        analysis_results = analyze_assr(args.file_path, args.output_dir, save_results=True)
        
        # Then generate plots
        plot_all_figures(analysis_results['tf_data'], analysis_results['epochs'], 
                        args.output_dir, args.save_figures, export_data=args.export_data,
                        export_csv=args.export_csv)