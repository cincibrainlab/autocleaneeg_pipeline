import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import mne
from matplotlib.colors import LinearSegmentedColormap


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


def plot_all_figures(tf_data, epochs, output_dir=None, save_figures=True, file_basename=None):
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
    figs['topomap'] = plot_topomap(tf_data, epochs, output_dir=output_dir, save_figures=save_figures, file_basename=file_basename)
    
    # Plot ERSP for all channels
    figs['ersp_channels'] = plot_ersp_channels(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Plot global mean ERSP - pass epochs for filename extraction
    figs['global_mean_ersp'] = plot_global_mean_ersp(tf_data, output_dir, save_figures, epochs, file_basename)
    
    # Plot STP for all channels
    figs['stp_channels'] = plot_stp_channels(tf_data, epochs, output_dir, save_figures, file_basename)
    
    # Plot global mean STP
    figs['global_mean_stp'] = plot_global_mean_stp(tf_data, epochs, output_dir, save_figures, file_basename)
    
    return figs


if __name__ == "__main__":
    import argparse
    import sys
    
    # Add parent directory to path for imports when running this file directly
    sys.path.append('..')
    from assr_analysis import analyze_assr
    
    parser = argparse.ArgumentParser(description='Generate visualizations for ASSR data')
    parser.add_argument('file_path', type=str, help='Path to the EEGLAB .set file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save figures')
    parser.add_argument('--no_save_figures', action='store_false', dest='save_figures', 
                        help='Do not save figures to disk')
    
    args = parser.parse_args()
    
    # Run analysis first
    analysis_results = analyze_assr(args.file_path, args.output_dir, save_results=True)
    
    # Then generate plots
    plot_all_figures(analysis_results['tf_data'], analysis_results['epochs'], 
                    args.output_dir, args.save_figures)