# src/autoclean/step_functions/reports.py
"""Visualization and reporting functions.

This module provides functions for generating visualizations and reports
from EEG processing results. It includes:
- Run summary reports
- Data quality visualizations
- Artifact detection plots
- Processing stage comparisons

The functions generate clear, publication-ready figures and detailed
HTML reports documenting the processing pipeline results.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pylossless as ll
from matplotlib.gridspec import GridSpec
from mne_bids import BIDSPath

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages

# ReportLab imports for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)
from reportlab.platypus import Table as ReportLabTable
from reportlab.platypus import (
    TableStyle,
)

from autoclean.utils.database import get_run_record, manage_database
from autoclean.utils.logging import message
from autoclean.utils.montage import get_standard_set_in_montage, validate_channel_set


def step_plot_raw_vs_cleaned_overlay(
    raw_original: mne.io.Raw,
    raw_cleaned: mne.io.Raw,
    pipeline: Any,
    autoclean_dict: Dict[str, Any],
    suffix: str = "",
) -> None:
    """
    Plot raw data channels over the full duration, overlaying the original and cleaned data.
    Original data is plotted in red, cleaned data in black.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after preprocessing.
    pipeline : pylossless.Pipeline
        Pipeline object (can be None if not used).
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    suffix : str
        Suffix for the filename.
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np

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
    n_samples = len(times)
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
        ch_name = channel_labels[idx]
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

    # Independent Components
    target_figure = str(
        derivatives_path.copy().update(
            suffix="step_plot_raw_vs_cleaned_overlay", extension=".png", datatype="eeg"
        )
    )

    # Save as PNG with high DPI for quality
    fig.savefig(target_figure, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Raw channels overlay full duration plot saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_raw_vs_cleaned_overlay": Path(target_figure).name,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )


def step_plot_ica_full(pipeline: Any, autoclean_dict: Dict[str, Any]) -> None:
    """Plot ICA components."""
    """
    Plot ICA components over the full duration with their labels and probabilities.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        PyLossless pipeline object containing raw data and ICA
    autoclean_dict : dict
        Autoclean dictionary containing metadata
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # Get raw and ICA from pipeline
    raw = pipeline.raw
    ica = pipeline.ica2
    ic_labels = pipeline.flags["ic"]

    # Get ICA activations and create time vector
    ica_sources = ica.get_sources(raw)
    ica_data = ica_sources.get_data()
    sfreq = raw.info["sfreq"]
    times = raw.times
    n_components, n_samples = ica_data.shape

    # Normalize each component individually for better visibility
    for idx in range(n_components):
        component = ica_data[idx]
        # Scale to have a consistent peak-to-peak amplitude
        ptp = np.ptp(component)
        if ptp == 0:
            scaling_factor = 2.5  # Avoid division by zero
        else:
            scaling_factor = 2.5 / ptp
        ica_data[idx] = component * scaling_factor

    # Determine appropriate spacing
    spacing = 2  # Fixed spacing between components

    # Calculate figure size proportional to duration
    total_duration = times[-1] - times[0]
    width_per_second = 0.1  # Increased from 0.02 to 0.1 for wider view
    fig_width = total_duration * width_per_second
    max_fig_width = 200  # Doubled from 100 to allow wider figures
    fig_width = min(fig_width, max_fig_width)
    fig_height = max(6, n_components * 0.5)  # Ensure a minimum height

    # Create plot with wider figure
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create a colormap for the components
    cmap = plt.cm.get_cmap("tab20", n_components)
    line_colors = [cmap(i) for i in range(n_components)]

    # Plot components in original order
    for idx in range(n_components):
        offset = idx * spacing
        ax.plot(times, ica_data[idx] + offset, color=line_colors[idx], linewidth=0.5)

    # Set y-ticks and labels
    yticks = [idx * spacing for idx in range(n_components)]
    yticklabels = []
    for idx in range(n_components):
        label_text = f"IC{idx + 1}: {ic_labels['ic_type'][idx]} ({ic_labels['confidence'][idx]:.2f})"
        yticklabels.append(label_text)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)

    # Customize axes
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_title("ICA Component Activations (Full Duration)", fontsize=14)
    ax.set_xlim(times[0], times[-1])

    # Adjust y-axis limits
    ax.set_ylim(-spacing, (n_components - 1) * spacing + spacing)

    # Remove y-axis label as we have custom labels
    ax.set_ylabel("")

    # Invert y-axis to have the first component at the top
    ax.invert_yaxis()

    # Color the labels red or black based on component type
    artifact_types = ["eog", "muscle", "ecg", "other"]
    for ticklabel, idx in zip(ax.get_yticklabels(), range(n_components)):
        ic_type = ic_labels["ic_type"][idx]
        if ic_type in artifact_types:
            ticklabel.set_color("red")
        else:
            ticklabel.set_color("black")

    # Adjust layout
    plt.tight_layout()

    # Get output path for ICA components figure
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    target_figure = str(
        derivatives_path.copy().update(
            suffix="ica_components_full_duration", extension=".png", datatype="eeg"
        )
    )

    # Save figure with higher DPI for better resolution of wider plot
    fig.savefig(target_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "ica_components_full_duration": Path(target_figure).name,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return fig


def step_generate_ica_reports(
    pipeline: Any,
    cleaned_raw: mne.io.Raw,
    autoclean_dict: Dict[str, Any],
    duration: int = 60,
) -> None:
    """
    Generates two reports:
    1. All ICA components.
    2. Only the rejected ICA components.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        The pipeline object containing the ICA and raw data.
    autoclean_dict : dict
        Dictionary containing configuration and paths.
    duration : int
        Duration in seconds for plotting time series data.
    """
    # Generate report for all components
    report_filename = _plot_ica_components(
        pipeline, cleaned_raw, autoclean_dict, duration=duration, components="all"
    )

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "ica_all_components": report_filename,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    # Generate report for rejected components
    report_filename = _plot_ica_components(
        pipeline, cleaned_raw, autoclean_dict, duration=duration, components="rejected"
    )

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "ica_rejected_components": report_filename,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )


def plot_bad_channels_with_topography(
    raw_original, raw_cleaned, pipeline, autoclean_dict, zoom_duration=30, zoom_start=0
):
    """
    Plot bad channels with a topographical map and time series overlays for both full duration and a zoomed-in window.

    Parameters:
    -----------
    raw_original : mne.io.Raw
        Original raw EEG data before cleaning.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after interpolation of bad channels.
    pipeline : pylossless.Pipeline
        Pipeline object containing flags and raw data.
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    zoom_duration : float, optional
        Duration in seconds for the zoomed-in time series plot. Default is 30 seconds.
    zoom_start : float, optional
        Start time in seconds for the zoomed-in window. Default is 0 seconds.
    """
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # ----------------------------
    # 1. Collect Bad Channels
    # ----------------------------
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
        print("No bad channels were identified.")
        return

    # Debugging: Print bad channels
    print(f"Identified Bad Channels: {bad_channels}")

    # ----------------------------
    # 2. Identify Good Channels
    # ----------------------------
    all_channels = raw_original.ch_names
    good_channels = [ch for ch in all_channels if ch not in bad_channels]

    # Debugging: Print good channels count
    print(f"Number of Good Channels: {len(good_channels)}")

    # ----------------------------
    # 3. Extract Data for Bad Channels
    # ----------------------------
    picks_bad_original = mne.pick_channels(raw_original.ch_names, bad_channels)
    picks_bad_cleaned = mne.pick_channels(raw_cleaned.ch_names, bad_channels)

    if len(picks_bad_original) == 0:
        print("No bad channels found in original data.")
        return

    if len(picks_bad_cleaned) == 0:
        print("No bad channels found in cleaned data.")
        return

    data_original, times = raw_original.get_data(
        picks=picks_bad_original, return_times=True
    )
    data_cleaned = raw_cleaned.get_data(picks=picks_bad_cleaned)

    channel_labels = [raw_original.ch_names[i] for i in picks_bad_original]
    n_channels = len(channel_labels)

    # Debugging: Print number of bad channels being plotted
    print(f"Number of Bad Channels to Plot: {n_channels}")

    # ----------------------------
    # 4. Downsample Data if Necessary
    # ----------------------------
    sfreq = raw_original.info["sfreq"]
    desired_sfreq = 100  # Target sampling rate
    downsample_factor = int(sfreq // desired_sfreq)
    if downsample_factor > 1:
        data_original = data_original[:, ::downsample_factor]
        data_cleaned = data_cleaned[:, ::downsample_factor]
        times = times[::downsample_factor]
        print(
            f"Data downsampled by a factor of {downsample_factor} to {desired_sfreq} Hz."
        )

    # ----------------------------
    # 5. Normalize and Scale Data
    # ----------------------------
    data_original_normalized = np.zeros_like(data_original)
    data_cleaned_normalized = np.zeros_like(data_cleaned)
    # Dynamic spacing based on number of bad channels
    spacing = 10 + (n_channels * 2)  # Adjusted spacing

    for idx in range(n_channels):
        channel_data_original = data_original[idx]
        channel_data_cleaned = data_cleaned[idx]
        # Remove DC offset
        channel_data_original -= np.mean(channel_data_original)
        channel_data_cleaned -= np.mean(channel_data_cleaned)
        # Normalize by standard deviation
        std_orig = np.std(channel_data_original)
        std_clean = np.std(channel_data_cleaned)
        if std_orig == 0:
            std_orig = 1  # Prevent division by zero
        if std_clean == 0:
            std_clean = 1
        data_original_normalized[idx] = channel_data_original / std_orig
        data_cleaned_normalized[idx] = channel_data_cleaned / std_clean

    # Scaling factor for better visibility
    scaling_factor = 5  # Increased scaling factor
    data_original_scaled = data_original_normalized * scaling_factor
    data_cleaned_scaled = data_cleaned_normalized * scaling_factor

    # Calculate offsets
    offsets = np.arange(n_channels) * spacing

    # ----------------------------
    # 6. Define Zoom Window
    # ----------------------------
    zoom_end = zoom_start + zoom_duration
    if zoom_end > times[-1]:
        zoom_end = times[-1]
        zoom_start = max(zoom_end - zoom_duration, times[0])

    # ----------------------------
    # 7. Create Figure with GridSpec
    # ----------------------------
    fig_height = 10 + (n_channels * 0.3)
    fig = plt.figure(constrained_layout=True, figsize=(20, fig_height))
    gs = GridSpec(3, 2, figure=fig)

    # ----------------------------
    # 8. Topography Subplot
    # ----------------------------
    ax_topo = fig.add_subplot(gs[0, :])

    # Plot sensors with ch_groups for good and bad channels
    ch_groups = [
        [int(raw_original.ch_names.index(ch)) for ch in good_channels],
        [int(raw_original.ch_names.index(ch)) for ch in bad_channels],
    ]
    colors = "RdYlBu_r"

    # Plot again for the main figure subplot
    mne.viz.plot_sensors(
        raw_original.info,
        kind="topomap",
        ch_type="eeg",
        title="Sensor Topography: Good vs Bad Channels",
        show_names=True,
        ch_groups=ch_groups,
        pointsize=75,
        linewidth=0,
        cmap=colors,
        show=False,
        axes=ax_topo,
    )

    ax_topo.legend(["Good Channels", "Bad Channels"], loc="upper right", fontsize=12)
    ax_topo.set_title("Topography of Good and Bad Channels", fontsize=16)

    # ----------------------------
    # 9. Full Duration Time Series Subplot
    # ----------------------------
    ax_full = fig.add_subplot(gs[1, 0])
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
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, linestyle="-", label="Original Data"),
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Interpolated Data"),
    ]
    ax_full.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # ----------------------------
    # 10. Zoomed-In Time Series Subplot
    # ----------------------------
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

    ax_zoom.set_xlabel("Time (seconds)", fontsize=14)
    ax_zoom.set_title(
        f"Bad Channels: Original vs Interpolated (Zoom: {zoom_start}-{zoom_end} s)",
        fontsize=16,
    )
    ax_zoom.set_xlim(zoom_start, zoom_end)
    ax_zoom.set_ylim(-spacing, offsets[-1] + spacing)
    ax_zoom.set_yticks([])  # Hide y-ticks
    ax_zoom.invert_yaxis()

    # Add legend
    ax_zoom.legend(handles=legend_elements, loc="upper right", fontsize=12)

    # ----------------------------
    # 11. Add Channel Labels
    # ----------------------------
    for idx, ch in enumerate(channel_labels):
        label = f"{ch}\n({', '.join(bad_channels_info[ch])})"
        ax_full.text(
            times[0] - (0.05 * (times[-1] - times[0])),
            offsets[idx],
            label,
            horizontalalignment="right",
            fontsize=10,
            verticalalignment="center",
        )

    # ----------------------------
    # 12. Finalize and Save the Figure
    # ----------------------------
    plt.tight_layout()

    # Get output path for bad channels figure
    bids_path = autoclean_dict.get("bids_path", "")
    if bids_path:
        derivatives_path = pipeline.get_derivative_path(bids_path)
    else:
        derivatives_path = "."

    # Assuming pipeline.get_derivative_path returns a Path-like object with a copy method
    # and update method as per the initial code
    try:
        target_figure = str(
            derivatives_path.copy().update(
                suffix="step_bad_channels_with_map", extension=".png", datatype="eeg"
            )
        )
    except AttributeError:
        # Fallback if copy or update is not implemented
        target_figure = os.path.join(
            derivatives_path, "bad_channels_with_topography.png"
        )

    # Save the figure
    fig.savefig(target_figure, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Bad channels with topography plot saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_bad_channels_with_topography": Path(target_figure).name,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return fig


def _plot_ica_components(
    pipeline: Any,
    cleaned_raw: mne.io.Raw,
    autoclean_dict: Dict[str, Any],
    duration: int = 60,
    components: str = "all",
):
    """
    Plots ICA components with labels and saves reports.

    Parameters:
    -----------
    pipeline : pylossless.Pipeline
        Pipeline object containing raw data and ICA.
    autoclean_dict : dict
        Autoclean dictionary containing metadata.
    duration : int
        Duration in seconds to plot.
    components : str
        'all' to plot all components, 'rejected' to plot only rejected components.
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.gridspec import GridSpec

    # Get raw and ICA from pipeline
    raw = pipeline.raw
    ica = pipeline.ica2
    ic_labels = pipeline.flags["ic"]

    # Determine components to plot
    if components == "all":
        component_indices = range(ica.n_components_)
        report_name = "ica_components_all"
    elif components == "rejected":
        component_indices = ica.exclude
        report_name = "ica_components_rejected"
        if not component_indices:
            print("No components were rejected. Skipping rejected components report.")
            return
    else:
        raise ValueError("components parameter must be 'all' or 'rejected'.")

    # Get ICA activations
    ica_sources = ica.get_sources(raw)
    ica_data = ica_sources.get_data()

    # Limit data to specified duration
    sfreq = raw.info["sfreq"]
    n_samples = int(duration * sfreq)
    times = raw.times[:n_samples]

    # Create output path for the PDF report
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    pdf_path = str(derivatives_path.copy().update(suffix=report_name, extension=".pdf"))

    # Remove existing file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    with PdfPages(pdf_path) as pdf:
        # Calculate how many components to show per page
        components_per_page = 20
        num_pages = int(np.ceil(len(component_indices) / components_per_page))

        # Create summary tables split across pages
        for page in range(num_pages):
            start_idx = page * components_per_page
            end_idx = min((page + 1) * components_per_page, len(component_indices))
            page_components = component_indices[start_idx:end_idx]

            fig_table = plt.figure(figsize=(11, 8.5))
            ax_table = fig_table.add_subplot(111)
            ax_table.axis("off")

            # Prepare table data for this page
            table_data = []
            colors = []
            for idx in page_components:
                comp_info = ic_labels.iloc[idx]
                table_data.append(
                    [
                        f"IC{idx + 1}",
                        comp_info["ic_type"],
                        f"{comp_info['confidence']:.2f}",
                        "Yes" if idx in ica.exclude else "No",
                    ]
                )

                # Define colors for different IC types
                color_map = {
                    "brain": "#d4edda",  # Light green
                    "eog": "#f9e79f",  # Light yellow
                    "muscle": "#f5b7b1",  # Light red
                    "ecg": "#d7bde2",  # Light purple,
                    "ch_noise": "#ffd700",  # Light orange
                    "line_noise": "#add8e6",  # Light blue
                    "other": "#f0f0f0",  # Light grey
                }
                colors.append(
                    [color_map.get(comp_info["ic_type"].lower(), "white")] * 4
                )

            # Create and customize table
            table = ax_table.table(
                cellText=table_data,
                colLabels=["Component", "Type", "Confidence", "Rejected"],
                loc="center",
                cellLoc="center",
                cellColours=colors,
                colWidths=[0.2, 0.3, 0.25, 0.25],
            )

            # Customize table appearance
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)  # Reduced vertical scaling

            # Add title with page information, filename and timestamp
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig_table.suptitle(
                f"ICA Components Summary - {autoclean_dict['bids_path'].basename}\n"
                f"(Page {page + 1} of {num_pages})\n"
                f"Generated: {timestamp}",
                fontsize=12,
                y=0.95,
            )
            # Add legend for colors
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
                for color in color_map.values()
            ]
            ax_table.legend(
                legend_elements,
                color_map.keys(),
                loc="upper right",
                title="Component Types",
            )

            # Add margins
            plt.subplots_adjust(top=0.85, bottom=0.15)

            pdf.savefig(fig_table)
            plt.close(fig_table)

        # First page: Component topographies overview
        fig_topo = ica.plot_components(picks=component_indices, show=False)
        if isinstance(fig_topo, list):
            for f in fig_topo:
                pdf.savefig(f)
                plt.close(f)
        else:
            pdf.savefig(fig_topo)
            plt.close(fig_topo)

        # If rejected components, add overlay plot
        if components == "rejected":
            fig_overlay = plt.figure()
            end_time = min(30.0, pipeline.raw.times[-1])
            fig_overlay = pipeline.ica2.plot_overlay(
                pipeline.raw,
                start=0,
                stop=end_time,
                exclude=component_indices,
                show=False,
            )
            fig_overlay.set_size_inches(15, 10)  # Set size after creating figure

            pdf.savefig(fig_overlay)
            plt.close(fig_overlay)

        # For each component, create detailed plots
        for idx in component_indices:
            fig = plt.figure(constrained_layout=True, figsize=(12, 8))
            gs = GridSpec(nrows=3, ncols=3, figure=fig)

            # Axes for ica.plot_properties
            ax1 = fig.add_subplot(gs[0, 0])  # Data
            ax2 = fig.add_subplot(gs[0, 1])  # Epochs image
            ax3 = fig.add_subplot(gs[0, 2])  # ERP/ERF
            ax4 = fig.add_subplot(gs[1, 0])  # Spectrum
            ax5 = fig.add_subplot(gs[1, 1])  # Topomap
            ax_props = [ax1, ax2, ax3, ax4, ax5]

            # Plot properties
            ica.plot_properties(
                raw,
                picks=[idx],
                axes=ax_props,
                dB=True,
                plot_std=True,
                log_scale=False,
                reject="auto",
                show=False,
            )

            # Add time series plot
            ax_timeseries = fig.add_subplot(gs[2, :])  # Last row, all columns
            ax_timeseries.plot(times, ica_data[idx, :n_samples], linewidth=0.5)
            ax_timeseries.set_xlabel("Time (seconds)")
            ax_timeseries.set_ylabel("Amplitude")
            ax_timeseries.set_title(f"Component {idx + 1} Time Course ({duration}s)")

            # Add labels
            comp_info = ic_labels.iloc[idx]
            label_text = (
                f"Component {comp_info['component']}\n"
                f"Type: {comp_info['ic_type']}\n"
                f"Confidence: {comp_info['confidence']:.2f}"
            )

            fig.suptitle(
                label_text,
                fontsize=14,
                fontweight="bold",
                color=(
                    "red"
                    if comp_info["ic_type"]
                    in ["eog", "muscle", "ch_noise", "line_noise", "ecg"]
                    else "black"
                ),
            )

            # Save the figure
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Report saved to {pdf_path}")
        return Path(pdf_path).name


def step_psd_topo_figure(
    raw_original: mne.io.Raw,
    raw_cleaned: mne.io.Raw,
    pipeline: Any,
    autoclean_dict: Dict[str, Any],
    bands: Optional[List[Tuple[str, float, float]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate and save a single high-resolution image that includes:
    - Two PSD plots side by side: Absolute PSD (mV²) and Relative PSD (%).
    - Topographical maps for multiple EEG frequency bands arranged horizontally,
      showing both pre and post cleaning.
    - Annotations for average power and outlier channels.

    Parameters:
    -----------
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
    metadata : dict, optional
        Additional metadata to include in the JSON sidecar.

    Returns:
    --------
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
            suffix="step_psd_topo_figure", extension=".png", datatype="eeg"
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
        verbose=True,
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
    _plot_psd(
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
    _plot_topomaps(
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

    print(f"Combined figure saved to {target_figure}")

    metadata = {
        "artifact_reports": {
            "creationDateTime": datetime.now().isoformat(),
            "plot_psd_topo_figure": Path(target_figure).name,
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )

    return target_figure


def _plot_psd(
    fig,
    gs,
    freqs,
    psd_original_mean_mV2,
    psd_cleaned_mean_mV2,
    psd_original_rel,
    psd_cleaned_rel,
    num_bands,
):
    """Helper function to create PSD plots"""
    # First row: Two PSD plots side by side
    ax_abs_psd = fig.add_subplot(gs[0, : num_bands // 2])
    ax_rel_psd = fig.add_subplot(gs[0, num_bands // 2 :])

    # Plot Absolute PSD with log scale
    ax_abs_psd.plot(freqs, psd_original_mean_mV2, color="red", label="Original")
    ax_abs_psd.plot(freqs, psd_cleaned_mean_mV2, color="black", label="Cleaned")
    ax_abs_psd.set_yscale("log")  # Set y-axis to log scale
    ax_abs_psd.set_xlabel("Frequency (Hz)")
    ax_abs_psd.set_ylabel("Power Spectral Density (mV²/Hz)")
    ax_abs_psd.set_title("Absolute Power Spectral Density (mV²/Hz)")
    ax_abs_psd.legend()
    ax_abs_psd.grid(True, which="both")  # Grid lines for both major and minor ticks

    # Add vertical lines and annotations for power bands on both PSDs
    for ax in [ax_abs_psd, ax_rel_psd]:
        for band_name, (f_start, f_end) in {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta": (12, 30),
            "Gamma": (30, 80),
        }.items():
            ax.axvline(f_start, color="grey", linestyle="--", linewidth=1)
            ax.axvline(f_end, color="grey", linestyle="--", linewidth=1)
            ax.fill_betweenx(ax.get_ylim(), f_start, f_end, color="grey", alpha=0.1)
            ax.text(
                (f_start + f_end) / 2,
                ax.get_ylim()[1] * 0.95,
                band_name,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=9,
                color="grey",
            )

    # Plot Relative PSD with log scale
    ax_rel_psd.plot(freqs, psd_original_rel, color="red", label="Original")
    ax_rel_psd.plot(freqs, psd_cleaned_rel, color="black", label="Cleaned")
    ax_rel_psd.set_yscale("log")  # Set y-axis to log scale
    ax_rel_psd.set_xlabel("Frequency (Hz)")
    ax_rel_psd.set_ylabel("Relative Power (%)")
    ax_rel_psd.set_title("Relative Power Spectral Density (%)")
    ax_rel_psd.legend()
    ax_rel_psd.grid(True, which="both")  # Grid lines for both major and minor ticks


def _plot_topomaps(
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


def create_run_report(run_id: str, autoclean_dict: dict = None) -> None:
    """
    Creates a scientific report in PDF format using ReportLab based on the run metadata.

    Args:
        run_id (str): The run ID to generate a report for
        Optional: autoclean_dict (dict): The autoclean dictionary
    """
    if not run_id:
        message("error", "No run ID provided")
        return

    run_record = get_run_record(run_id)
    if not run_record or "metadata" not in run_record:
        message("error", "No metadata found for run ID")
        return

    # Early validation of required metadata sections
    required_sections = ["step_prepare_directories"]
    missing_sections = [
        section
        for section in required_sections
        if section not in run_record["metadata"]
    ]
    if missing_sections:
        message(
            "error",
            f"Missing required metadata sections: {', '.join(missing_sections)}",
        )
        return

    else:
        try:
            if autoclean_dict:
                try:
                    bids_path = None
                    bids_path = autoclean_dict["bids_path"]
                except Exception as e:
                    message(
                        "warning",
                        f"Failed to get BIDS path from autoclean_dict: Trying metadata",
                    )
            if not bids_path and "step_convert_to_bids" in run_record["metadata"]:
                bids_info = run_record["metadata"]["step_convert_to_bids"]
                if bids_info:
                    # Reconstruct BIDSPath object
                    bids_path = BIDSPath(
                        subject=bids_info["bids_subject"],
                        session=bids_info["bids_session"],
                        task=bids_info["bids_task"],
                        run=bids_info["bids_run"],
                        datatype=bids_info["bids_datatype"],
                        root=bids_info["bids_root"],
                        suffix=bids_info["bids_suffix"],
                        extension=bids_info["bids_extension"],
                    )

            task = run_record["task"]
            config_path = run_record["metadata"]["entrypoint"]["tasks"][task][
                "lossless_config"
            ]
            derivative_name = "pylossless"
            pipeline = ll.LosslessPipeline(config_path)
            derivatives_path = pipeline.get_derivative_path(bids_path, derivative_name)
            derivatives_dir = Path(derivatives_path.directory)
            derivatives_path = str(
                derivatives_path.copy().update(suffix="report", extension=".pdf")
            )
        except Exception as e:
            message(
                "warning",
                f"Failed to get BIDS path: {str(e)} : Saving only to metadata directory",
            )
            derivatives_path = None

    # Get metadata directory from step_prepare_directories
    metadata_dir = Path(run_record["metadata"]["step_prepare_directories"]["metadata"])
    if not metadata_dir.exists():
        metadata_dir.mkdir(parents=True, exist_ok=True)

    # Create PDF filename
    pdf_path = metadata_dir / f"{run_record['report_file']}"

    # Initialize the PDF document
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    # Get styles
    styles = getSampleStyleSheet()

    # Custom styles for better visual hierarchy
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=14,
        spaceAfter=6,
        textColor=colors.HexColor("#2C3E50"),
        alignment=1,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontSize=10,
        spaceAfter=4,
        textColor=colors.HexColor("#34495E"),
        alignment=1,
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=7,
        spaceAfter=2,
        textColor=colors.HexColor("#2C3E50"),
    )

    steps_style = ParagraphStyle(
        "Steps",
        parent=normal_style,
        fontSize=7,
        leading=10,
        spaceBefore=1,
        spaceAfter=1,
    )

    # Define frame style for main content
    frame_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECF0F1")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]
    )

    # Common table style
    table_style = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6FA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]
    )

    # Create story (content) for the PDF
    story = []

    # Title and Basic Info
    title = f"EEG Processing Report"
    story.append(Paragraph(title, title_style))

    # Add status-colored subtitle
    status_color = (
        colors.HexColor("#2ECC71")
        if run_record.get("success", False)
        else colors.HexColor("#E74C3C")
    )
    subtitle_style = ParagraphStyle(
        "CustomSubtitle",
        parent=heading_style,
        textColor=status_color,
        spaceAfter=2,
    )
    status_text = "SUCCESS" if run_record.get("success", False) else "FAILED"
    subtitle = f"Run ID: {run_id} - {status_text}"
    story.append(Paragraph(subtitle, subtitle_style))

    # Add timestamp
    timestamp_style = ParagraphStyle(
        "Timestamp",
        parent=normal_style,
        textColor=colors.HexColor("#7F8C8D"),
        alignment=1,
        spaceAfter=8,
    )
    timestamp = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    story.append(Paragraph(timestamp, timestamp_style))

    # Tables layout with better styling
    data = [
        [
            Paragraph("Import Information", heading_style),
            Paragraph("Preprocessing Parameters", heading_style),
            Paragraph("Lossless Configuration", heading_style),
        ]
    ]

    # Left column: Import info with colored background
    try:
        raw_info = run_record["metadata"].get("step_import", {})
        if not raw_info:
            raw_info = {"message": "Step import metadata not available"}

        import_info = []
        # Get values and format them safely
        duration = raw_info.get("durationSec")
        duration_str = (
            f"{duration:.1f} sec" if isinstance(duration, (int, float)) else "N/A"
        )

        sample_rate = raw_info.get("sampleRate")
        sample_rate_str = (
            f"{sample_rate} Hz" if isinstance(sample_rate, (int, float)) else "N/A"
        )

        import_info.extend(
            [
                ["File", raw_info.get("unprocessedFile", "N/A")],
                ["Duration", duration_str],
                ["Sample Rate", sample_rate_str],
                ["Channels", str(raw_info.get("channelCount", "N/A"))],
            ]
        )

        if not import_info:
            import_info = [["No import data available", "N/A"]]

    except Exception as e:
        message("warning", f"Error processing import information: {str(e)}")
        import_info = [["Error processing import data", "N/A"]]

    import_table = ReportLabTable(import_info, colWidths=[0.7 * inch, 1.3 * inch])
    import_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F8F9F9"),
                ),
            ]
        )
    )

    # Middle column: Preprocessing parameters
    preproc_info = []
    try:
        if "entrypoint" in run_record["metadata"]:
            task_config = run_record["metadata"]["entrypoint"]["tasks"][
                run_record["metadata"]["entrypoint"]["task"]
            ]["settings"]
            preproc_info.extend(
                [
                    [
                        "Resample",
                        (
                            f"{task_config['resample_step']['value']} Hz"
                            if task_config["resample_step"]["enabled"]
                            else "Disabled"
                        ),
                    ],
                    [
                        "Trim",
                        (
                            f"{task_config['trim_step']['value']} sec"
                            if task_config["trim_step"]["enabled"]
                            else "Disabled"
                        ),
                    ],
                    [
                        "Reference",
                        (
                            str(task_config["reference_step"]["value"])
                            if isinstance(task_config["reference_step"]["value"], str)
                            else (
                                ", ".join(task_config["reference_step"]["value"])
                                if isinstance(
                                    task_config["reference_step"]["value"], list
                                )
                                else (
                                    "Disabled"
                                    if task_config["reference_step"]["enabled"]
                                    else "Disabled"
                                )
                            )
                        ),
                    ],
                ]
            )
    except Exception as e:
        message("warning", f"Error processing preprocessing parameters: {str(e)}")
        preproc_info = [["Error processing parameters", "N/A"]]

    if not preproc_info:
        preproc_info = [["No preprocessing data available", "N/A"]]

    preproc_table = ReportLabTable(preproc_info, colWidths=[0.7 * inch, 1.3 * inch])
    preproc_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#EFF8F9"),
                ),
            ]
        )
    )

    # Right column: Lossless settings
    lossless_info = []
    try:
        if "step_run_pylossless" in run_record["metadata"]:
            lossless_config = run_record["metadata"]["step_run_pylossless"].get(
                "pylossless_config", {}
            )
            filter_args = lossless_config.get("filtering", {}).get("filter_args", {})
            ica_args = lossless_config.get("ica", {}).get("ica_args", {})
            lossless_info.extend(
                [
                    [
                        "Filter",
                        f"{filter_args.get('l_freq', 'N/A')}-{filter_args.get('h_freq', 'N/A')} Hz",
                    ],
                    [
                        "Notch",
                        f"{lossless_config.get('filtering', {}).get('notch_filter_args', {}).get('freqs', ['N/A'])[0]} Hz",
                    ],
                    ["ICA", ica_args.get("run2", {}).get("method", "N/A")],
                    [
                        "Components",
                        str(ica_args.get("run2", {}).get("n_components", "N/A")),
                    ],
                ]
            )
    except Exception as e:
        message("warning", f"Error processing lossless settings: {str(e)}")
        lossless_info = [["Error processing lossless data", "N/A"]]

    if not lossless_info:
        lossless_info = [["No lossless data available", "N/A"]]

    lossless_table = ReportLabTable(lossless_info, colWidths=[0.7 * inch, 1.3 * inch])
    lossless_table.setStyle(
        TableStyle(
            [
                *table_style._cmds,
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#F5EEF8"),
                ),
            ]
        )
    )

    # Add tables to main layout with spacing
    data.append([import_table, preproc_table, lossless_table])
    main_table = ReportLabTable(data, colWidths=[2 * inch, 2 * inch, 2 * inch])
    main_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ]
        )
    )

    # Add main content in a frame
    frame_data = [[main_table]]
    frame = ReportLabTable(frame_data, colWidths=[6.5 * inch])
    frame.setStyle(frame_style)
    story.append(frame)
    story.append(Spacer(1, 15))

    # Get final export details from step_gfp_clean_epochs and save_epochs_to_set
    export_info = []
    annotation_info = []

    try:
        metadata = run_record.get("metadata", {})
        gfp_info = metadata.get("step_gfp_clean_epochs", {})
        fixed_epochs_info = metadata.get("make_fixed_length_epochs", {})
        eventid_info = metadata.get("step_create_eventid_epochs", {})
        save_epochs_info = metadata.get("save_epochs_to_set", {})

        if any([gfp_info, fixed_epochs_info, eventid_info, save_epochs_info]):
            # Main export details
            export_info.extend(
                [
                    [
                        "Post-Epoch Duration",
                        (
                            f"{eventid_info.get('total_duration', 'N/A'):.1f} sec"
                            if isinstance(
                                eventid_info.get("total_duration"), (int, float)
                            )
                            else "N/A"
                        ),
                    ],
                    [
                        "Final Duration",
                        (
                            f"{gfp_info.get('total_duration_sec', 'N/A'):.1f} sec"
                            if isinstance(
                                gfp_info.get("total_duration_sec"), (int, float)
                            )
                            else "N/A"
                        ),
                    ],
                    [
                        "Sample Rate",
                        (
                            f"{int(gfp_info.get('total_samples', 0) / gfp_info.get('total_duration_sec', 1)):.0f} Hz"
                            if gfp_info.get("total_samples")
                            and gfp_info.get("total_duration_sec")
                            else "N/A"
                        ),
                    ],
                    ["Channels", str(gfp_info.get("channel_count", "N/A"))],
                    [
                        "Initial Epochs",
                        str(eventid_info.get("number_of_epochs", "N/A")),
                    ],
                    ["Final Epochs", str(save_epochs_info.get("n_epochs", "N/A"))],
                    [
                        "Epoch Length",
                        (
                            f"[{save_epochs_info.get('tmin', 'N/A')}, {save_epochs_info.get('tmax', 'N/A')}] sec"
                            if isinstance(save_epochs_info.get("tmin"), (int, float))
                            and isinstance(save_epochs_info.get("tmax"), (int, float))
                            else "N/A"
                        ),
                    ],
                ]
            )

            # Annotation counts
            if "annotation_types" in gfp_info:
                annotations = gfp_info["annotation_types"]
                for key, value in annotations.items():
                    annotation_info.append([f"Anno: {key}", str(value)])

    except Exception as e:
        message("warning", f"Error processing export details: {str(e)}")
        export_info = [["Error processing export details", "N/A"]]
        annotation_info = [["Error processing annotations", "N/A"]]

    # Ensure we have at least one row in each table
    if not export_info:
        export_info = [["No export data available", "N/A"]]
    if not annotation_info:
        annotation_info = [["No annotation data available", "N/A"]]

    # Create export details table with two columns
    export_data = [[Paragraph("Export Details", heading_style)]]

    # Create tables for each column
    export_left_table = ReportLabTable(export_info, colWidths=[1.5 * inch, 1.5 * inch])
    export_left_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )

    export_right_table = ReportLabTable(
        annotation_info, colWidths=[1.5 * inch, 1.5 * inch]
    )
    export_right_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )

    export_data.append([export_left_table, export_right_table])

    # Create the main export table
    export_table = ReportLabTable(export_data, colWidths=[3.25 * inch, 3.25 * inch])
    export_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("SPAN", (0, 0), (1, 0)),  # Make the header span both columns
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, 0), 0),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ]
        )
    )

    # Add export table in a frame
    export_frame = ReportLabTable([[export_table]], colWidths=[6.5 * inch])
    export_frame.setStyle(frame_style)
    story.append(export_frame)
    story.append(Spacer(1, 15))

    # Add processing steps and bad channels in a two-column table
    story.append(Paragraph("Processing Summary", heading_style))
    story.append(Spacer(1, 4))

    # Get steps from metadata and bad channels info
    steps = []
    for key in run_record["metadata"].keys():
        if key.startswith("step_"):
            steps.append(key)

    bad_channels = []
    if "step_clean_bad_channels" in run_record["metadata"]:
        try:
            bads = run_record["metadata"]["step_clean_bad_channels"].get("bads", [])
            if bads:
                # Sort bad channels by numerical value after "E" prefix
                bad_channels = sorted(
                    bads, key=lambda x: int(x.replace("E", "")), reverse=True
                )
        except (KeyError, ValueError) as e:
            message("warning", f"Error processing bad channels information: {str(e)}")

    # Create three-column table data
    summary_data = [["Processing Steps", "Bad Channels", "Artifact Reports"]]

    # Create a paragraph style with line breaks
    steps_style = ParagraphStyle(
        "Steps",
        parent=normal_style,
        fontSize=8,
        leading=12,  # Line spacing
        spaceBefore=2,
        spaceAfter=2,
    )

    # Format steps and bad channels as bullet points with <br/> tags
    steps_text = "<br/>".join([f"• {step}" for step in steps])
    channels_text = (
        "<br/>".join([f"• {ch}" for ch in bad_channels])
        if bad_channels
        else "No bad channels detected"
    )

    # Gather artifact report details
    artifact_details = []
    if "artifact_reports" in run_record["metadata"]:
        try:
            artifact_reports = run_record["metadata"]["artifact_reports"]
            if isinstance(artifact_reports, list):
                for report in artifact_reports:
                    for key, value in report.items():
                        if key != "creationDateTime":
                            artifact_details.append(f"• {key}")
            elif isinstance(artifact_reports, dict):
                for key, value in artifact_reports.items():
                    if key != "creationDateTime":
                        artifact_details.append(f"• {key}")
        except (KeyError, TypeError, AttributeError) as e:
            message("warning", f"Error reading artifact reports: {str(e)}")
            artifact_details.append("• Error reading artifact reports")

    artifact_text = (
        "<br/>".join(artifact_details)
        if artifact_details
        else "No artifact reports available"
    )

    summary_data.append(
        [
            Paragraph(steps_text, steps_style),
            Paragraph(channels_text, steps_style),
            Paragraph(artifact_text, steps_style),
        ]
    )

    # Create and style the summary table
    summary_table = ReportLabTable(
        summary_data, colWidths=[2.2 * inch, 2.2 * inch, 2.1 * inch]
    )
    summary_table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#BDC3C7"),
                ),  # Light gray border
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#F5F6FA"),
                ),  # Very light blue header
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#F8F9F9"),
                ),  # Very light gray rows
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("LEADING", (0, 1), (-1, -1), 12),  # Add spacing between lines
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Add Generated Files Section
    story.append(Paragraph("Generated Files", heading_style))
    story.append(Spacer(1, 4))

    try:
        files_data = [["Type", "File Name", "Size"]]

        if derivatives_dir.exists():
            files = list(derivatives_dir.glob("**/*"))

            if files:
                # Get and sort files
                sorted_files = sorted(
                    [f for f in files if f.is_file()], key=lambda x: (x.suffix, x.name)
                )

                # Process each file
                for file in sorted_files:
                    file_type = file.suffix.upper()[1:] if file.suffix else "Unknown"

                    # Format file size
                    size_bytes = file.stat().st_size
                    size_str = (
                        f"{size_bytes} B"
                        if size_bytes < 1024
                        else (
                            f"{size_bytes/1024:.1f} KB"
                            if size_bytes < 1024 * 1024
                            else f"{size_bytes/(1024*1024):.1f} MB"
                        )
                    )

                    files_data.append([file_type, file.name, size_str])

                # Create table if files were found
                if len(files_data) > 1:
                    story.append(
                        Paragraph(f"Directory: {derivatives_dir}", normal_style)
                    )
                    story.append(Spacer(1, 4))

                    files_table = ReportLabTable(
                        files_data, colWidths=[1.0 * inch, 4.0 * inch, 1.5 * inch]
                    )
                    files_table.setStyle(
                        TableStyle(
                            [
                                (
                                    "GRID",
                                    (0, 0),
                                    (-1, -1),
                                    0.5,
                                    colors.HexColor("#BDC3C7"),
                                ),
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#F5F6FA"),
                                ),
                                (
                                    "BACKGROUND",
                                    (0, 1),
                                    (-1, -1),
                                    colors.HexColor("#F8F9F9"),
                                ),
                                (
                                    "TEXTCOLOR",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#2C3E50"),
                                ),
                                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                            ]
                        )
                    )
                    story.append(files_table)
                else:
                    story.append(
                        Paragraph(
                            "No files found in derivatives directory", normal_style
                        )
                    )
            else:
                story.append(
                    Paragraph("No files found in derivatives directory", normal_style)
                )
        else:
            story.append(
                Paragraph(
                    f"Derivatives directory not found: {derivatives_dir}", normal_style
                )
            )
    except Exception as e:
        message(
            "warning", f"Error processing derivatives directory information: {str(e)}"
        )
        story.append(
            Paragraph(f"Error listing derivatives files: {str(e)}", normal_style)
        )

    story.append(Spacer(1, 15))

    # Build the PDF
    try:
        # First build in metadata directory
        doc.build(story)
        message("success", f"Run report generated in metadata directory: {pdf_path}")

        # Update metadata with pdf path
        metadata = {
            "create_run_report": {
                "creationDateTime": datetime.now().isoformat(),
                "metadata_report_path": str(pdf_path),
            }
        }

        # Create a copy in derivatives
        if derivatives_path:
            shutil.copy2(pdf_path, derivatives_path)
            message("success", f"Run report added to derivatives: {derivatives_path}")
            metadata["create_run_report"]["derivatives_report_path"] = str(
                derivatives_path
            )

        # Update database with metadata
        manage_database(
            operation="update", update_record={"run_id": run_id, "metadata": metadata}
        )

    except Exception as e:
        message("error", f"Failed to generate or copy report: {str(e)}")


def update_task_processing_log(summary_dict: Dict[str, Any]) -> None:
    """Update the task-specific processing log CSV file with details about the current file.

    Args:
        summary_dict: The summary dictionary containing processing details
    """
    try:

        # Define CSV path
        csv_path = (
            Path(summary_dict["output_dir"])
            / f"{summary_dict['task']}_processing_log.csv"
        )

        # Extract details from summary_dict
        details = {
            "timestamp": summary_dict["timestamp"],
            "study_user": os.getenv("USERNAME", "unknown"),
            "run_id": summary_dict["run_id"],
            "proc_state": summary_dict["proc_state"],
            "subj_basename": Path(summary_dict["basename"]).stem,
            "bids_subject": summary_dict["bids_subject"],
            "task": summary_dict["task"],
            "net_nbchan_orig": str(summary_dict["import_details"]["net_nbchan_orig"]),
            "net_nbchan_post": str(
                summary_dict["export_details"].get("net_nbchan_post", "")
            ),
            "proc_badchans": str(
                summary_dict["channel_dict"].get("removed_channels", "")
            ),
            "proc_filt_lowcutoff": str(summary_dict["processing_details"]["l_freq"]),
            "proc_filt_highcutoff": str(summary_dict["processing_details"]["h_freq"]),
            "proc_filt_notch": str(summary_dict["processing_details"]["notch_freqs"]),
            "proc_filt_notch_width": str(
                summary_dict["processing_details"]["notch_widths"]
            ),
            "proc_sRate_raw": str(summary_dict["import_details"]["sample_rate"]),
            "proc_sRate1": str(summary_dict["export_details"].get("srate_post", "")),
            "proc_xmax_raw": str(summary_dict["import_details"]["duration"]),
            "proc_xmax_post": str(
                summary_dict["export_details"].get("final_duration", "")
            ),
            "proc_xmax_percent": str(
                float(summary_dict["export_details"].get("final_duration", ""))
                / float(summary_dict["import_details"]["duration"])
            ),
            "epoch_length": str(summary_dict["export_details"].get("epoch_length", "")),
            "epoch_limits": (
                f"{summary_dict['export_details'].get('epoch_limits', '')}"
                if "epoch_limits" in summary_dict["export_details"]
                else ""
            ),
            "epoch_trials": str(
                summary_dict["export_details"].get("initial_n_epochs", "")
            ),
            "epoch_badtrials": str(
                float(summary_dict["export_details"].get("initial_n_epochs", ""))
                - float(summary_dict["export_details"].get("final_n_epochs", ""))
            ),
            "epoch_percent": str(
                float(summary_dict["export_details"].get("final_n_epochs", ""))
                / float(summary_dict["export_details"].get("initial_n_epochs", ""))
            ),
            "proc_nComps": str(summary_dict["ica_details"].get("proc_nComps", "")),
            "proc_removeComps": str(
                summary_dict["ica_details"].get("proc_removeComps", "")
            ),
            "exclude_category": summary_dict["exclude_category"],
        }

        if csv_path.exists():
            # Read existing CSV
            df = pd.read_csv(csv_path, dtype=str)  # Force all columns to be string type

            # Ensure all columns exist in DataFrame
            for col in details.keys():
                if col not in df.columns:
                    df[col] = ""

            # Update or append entry
            if details["subj_basename"] in df["subj_basename"].values:
                # Update existing row using dictionary update
                df.loc[
                    df["subj_basename"] == details["subj_basename"],
                    list(details.keys()),
                ] = pd.Series(details)
            else:
                # Append new entry
                df = pd.concat([df, pd.DataFrame([details])], ignore_index=True)
        else:
            # Create new DataFrame with all columns as string type
            df = pd.DataFrame([details], dtype=str)

        # Save updated CSV
        df.to_csv(csv_path, index=False)
        message(
            "info",
            f"Updated processing log for {details['subj_basename']} in {csv_path}",
        )

        # Update run record with CSV path
        metadata = {
            "processing_log": {
                "creationDateTime": datetime.now().isoformat(),
                "csv_path": str(csv_path),
            }
        }
        manage_database(
            operation="update",
            update_record={"run_id": summary_dict["run_id"], "metadata": metadata},
        )

    except Exception as e:
        message("error", f"Error updating processing log: {str(e)}")
        return


def create_json_summary(run_id: str) -> None:
    run_record = get_run_record(run_id)
    if not run_record:
        message("error", f"No run record found for run ID: {run_id}")
        return

    metadata = run_record.get("metadata", {})

    # Create a JSON summary of the metadata
    try:
        if "step_convert_to_bids" in run_record["metadata"]:
            bids_info = run_record["metadata"]["step_convert_to_bids"]
            if bids_info:
                # Reconstruct BIDSPath object
                bids_path = BIDSPath(
                    subject=bids_info["bids_subject"],
                    session=bids_info["bids_session"],
                    task=bids_info["bids_task"],
                    run=bids_info["bids_run"],
                    datatype=bids_info["bids_datatype"],
                    root=bids_info["bids_root"],
                    suffix=bids_info["bids_suffix"],
                    extension=bids_info["bids_extension"],
                )

        config_path = run_record["lossless_config"]
        derivative_name = "pylossless"
        pipeline = ll.LosslessPipeline(config_path)
        derivatives_path = pipeline.get_derivative_path(bids_path, derivative_name)
        derivatives_dir = Path(derivatives_path.directory)
        # derivatives_path = str(
        #     derivatives_path.copy().update(suffix="report", extension=".pdf")
        # )
    except Exception as e:
        message("error", f"Could not get derivatives path: {str(e)}")
        return

    outputs = [file.name for file in derivatives_dir.iterdir() if file.is_file()]

    # Determine processing state and exclusion category
    proc_state = "postcomps"
    exclude_category = ""
    if not run_record.get("success", False):
        error_msg = run_record.get("error", "").lower()
        if "line noise" in error_msg:
            proc_state = "LINE NOISE"
            exclude_category = "Excessive Line Noise"
        elif "insufficient data" in error_msg:
            proc_state = "INSUFFICIENT_DATA"
            exclude_category = "Insufficient Data"
        else:
            proc_state = "ERROR"
            exclude_category = f"Processing Error: {error_msg[:100]}"

    # FIND BAD CHANNELS
    channel_dict = {}
    if "step_clean_bad_channels" in metadata:
        channel_dict["step_clean_bad_channels"] = metadata["step_clean_bad_channels"][
            "bads"
        ]

    flagged_chs_file = None
    for file_name in outputs:
        if file_name.endswith("FlaggedChs.tsv"):
            flagged_chs_file = file_name
            break

    if flagged_chs_file:
        with open(derivatives_dir / flagged_chs_file, "r") as f:
            # Skip the header line
            next(f)
            # Read each line and extract the label and channel name
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    label, channel = parts
                    if label not in channel_dict:
                        channel_dict[label] = []
                    channel_dict[label].append(channel)

    # Get unique sorted list of all removed channels, sorting by numeric value
    bad_channels = sorted(
        set(channel for channels in channel_dict.values() for channel in channels),
        key=lambda x: int(x.replace("E", "")),
    )
    channel_dict["removed_channels"] = bad_channels

    if "step_prepare_directories" in metadata:
        output_dir = Path(metadata["step_prepare_directories"]["bids"]).parent

    # FIND IMPORT DETAILS
    import_details = {}
    if "step_import" in metadata:
        import_details["sample_rate"] = metadata["step_import"]["sampleRate"]
        import_details["net_nbchan_orig"] = metadata["step_import"]["channelCount"]
        import_details["duration"] = metadata["step_import"]["durationSec"]
        import_details["basename"] = metadata["step_import"]["unprocessedFile"]
        original_channel_count = int(metadata["step_import"]["channelCount"])
    else:
        message("error", "No import details found")
        return

    processing_details = {}
    if "step_run_pylossless" in metadata:
        pylossless_info = metadata["step_run_pylossless"]["pylossless_config"]
        processing_details["h_freq"] = pylossless_info["filtering"]["filter_args"][
            "h_freq"
        ]
        processing_details["l_freq"] = pylossless_info["filtering"]["filter_args"][
            "l_freq"
        ]
        processing_details["notch_freqs"] = pylossless_info["filtering"][
            "notch_filter_args"
        ]["freqs"]
        if "notch_widths" in pylossless_info["filtering"]["notch_filter_args"]:
            processing_details["notch_widths"] = pylossless_info["filtering"][
                "notch_filter_args"
            ]["notch_widths"]
        else:
            processing_details["notch_widths"] = "notch_freqs/200"

    # FIND EXPORT DETAILS
    export_details = {}
    if "save_epochs_to_set" in metadata:
        save_epochs_to_set = metadata["save_epochs_to_set"]
        epoch_length = save_epochs_to_set["tmax"] - save_epochs_to_set["tmin"]
        export_details["epoch_length"] = epoch_length
        export_details["final_n_epochs"] = save_epochs_to_set["n_epochs"]
        export_details["final_duration"] = epoch_length * save_epochs_to_set["n_epochs"]
        if original_channel_count and bad_channels:
            export_details["net_nbchan_post"] = original_channel_count - len(
                bad_channels
            )
        else:
            export_details["net_nbchan_post"] = original_channel_count

    if "step_create_regular_epochs" in metadata:
        make_fixed_length_epochs = metadata["step_create_regular_epochs"]
        export_details["initial_n_epochs"] = make_fixed_length_epochs[
            "initial_epoch_count"
        ]
        export_details["initial_duration"] = (
            make_fixed_length_epochs["initial_epoch_count"]
            * make_fixed_length_epochs["duration"]
        )
        export_details["srate_post"] = (
            make_fixed_length_epochs["single_epoch_samples"]
            // make_fixed_length_epochs["duration"]
        )
        export_details["epoch_limits"] = [
            make_fixed_length_epochs["tmin"],
            make_fixed_length_epochs["tmax"],
        ]

        # export_details["rejected_epochs"] = make_fixed_length_epochs["rejected_epochs"]
        # export_details["rejection_percent"] = make_fixed_length_epochs["rejection_percent"]

    elif "step_create_eventid_epochs" in metadata:
        create_eventid_epochs = metadata["step_create_eventid_epochs"]
        export_details["initial_n_epochs"] = create_eventid_epochs["number_of_events"]
        export_details["initial_duration"] = create_eventid_epochs["total_duration"]
        export_details["srate_post"] = (
            create_eventid_epochs["samples_per_epoch"] - 1
        ) / create_eventid_epochs["epoch_duration"]
        export_details["epoch_limits"] = [
            create_eventid_epochs["tmin"],
            create_eventid_epochs["tmax"],
        ]

    ica_details = {}
    if "step_run_ll_rejection_policy" in metadata:
        ll_rejection_policy = metadata["step_run_ll_rejection_policy"]
        ica_details["proc_removeComps"] = ll_rejection_policy["ica_components"]
        ica_details["proc_nComps"] = ll_rejection_policy["n_components"]

    summary_dict = {
        "run_id": run_id,
        "task": run_record["task"],
        "bids_subject": f"sub-{bids_path.subject}",
        "timestamp": run_record["timestamp"],
        "basename": import_details["basename"],
        "proc_state": proc_state,
        "exclude_category": exclude_category,
        "import_details": import_details,
        "processing_details": processing_details,
        "export_details": export_details,
        "ica_details": ica_details,
        "channel_dict": channel_dict,
        "outputs": outputs,
        "output_dir": output_dir,
        "derivatives_dir": derivatives_dir,
    }

    return summary_dict


def generate_mmn_erp(
    epochs: mne.Epochs, pipeline: Any, autoclean_dict: Dict[str, Any]
) -> None:
    """Analyze MMN data and create a PDF report with ERPs and topographies."""
    message("header", "step_analyze_mmn")

    # Get ROI channels
    task = autoclean_dict["task"]
    settings = autoclean_dict["tasks"][task]["settings"]
    roi_channels = get_standard_set_in_montage(
        "mmn_standard", settings["montage"]["value"]
    )
    roi_channels = validate_channel_set(roi_channels, epochs.ch_names)

    if not roi_channels:
        message("error", "No valid ROI channels found in data")
        return None

    # Get conditions
    event_id = settings["event_id"]["value"]
    conditions = {
        "standard": event_id.get("standard", "DIN2/1"),
        "predeviant": event_id.get("predeviant", "DIN2/2"),
        "deviant": event_id.get("deviant", "DIN2/3"),
    }

    # Create evoked objects
    try:
        evoked_dict = {
            "standard": epochs[conditions["standard"]].average(),
            "predeviant": epochs[conditions["predeviant"]].average(),
            "deviant": epochs[conditions["deviant"]].average(),
        }

        # Add difference waves
        evoked_dict["mmn"] = mne.combine_evoked(
            [evoked_dict["deviant"], evoked_dict["standard"]], weights=[1, -1]
        )
        evoked_dict["mmn_pre"] = mne.combine_evoked(
            [evoked_dict["deviant"], evoked_dict["predeviant"]], weights=[1, -1]
        )
    except KeyError as e:
        message("error", f"Could not find condition: {e}")
        message("info", f"Available conditions: {epochs.event_id}")
        return None

    # Create output paths
    derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
    pdf_path = str(
        derivatives_path.copy().update(suffix="erp-report", extension=".pdf")
    )

    # Remove existing file
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    with PdfPages(pdf_path) as pdf:
        # Plot ERPs - using non-interactive method
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        times = epochs.times * 1000  # Convert to milliseconds
        for condition, color in [
            ("standard", "black"),
            ("predeviant", "blue"),
            ("deviant", "red"),
        ]:
            data = evoked_dict[condition].get_data(picks=roi_channels).mean(axis=0)
            ax.plot(
                times,
                data * 1e6,
                color=color,
                linewidth=2,
                label=condition.capitalize(),
            )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_title("ERPs by Condition")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Plot difference waves - using non-interactive method
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        data = evoked_dict["mmn"].get_data(picks=roi_channels).mean(axis=0)
        ax.plot(times, data * 1e6, color="red", linewidth=2)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (μV)")
        ax.set_title("MMN (Deviant - Standard)")
        ax.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Plot topographies at key time points
        times = [0.100, 0.200, 0.300]  # N1, MMN, P3
        time_labels = ["N1 (100 ms)", "MMN (200 ms)", "P3 (300 ms)"]
        for condition in ["standard", "predeviant", "deviant", "mmn"]:
            fig = plt.figure(figsize=(12, 4))
            for idx, (time, label) in enumerate(zip(times, time_labels), 1):
                ax = fig.add_subplot(1, 3, idx)
                evoked_dict[condition].plot_topomap(
                    times=time, axes=ax, show=False, colorbar=False, time_unit="ms"
                )
                ax.set_title(label)
            plt.suptitle(f"{condition.upper()} Topography")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    # Save data
    np.savez(
        str(derivatives_path.copy().update(suffix="erp-data", extension=".csv")),
        times=epochs.times,
        standard=evoked_dict["standard"].data,
        predeviant=evoked_dict["predeviant"].data,
        deviant=evoked_dict["deviant"].data,
        mmn=evoked_dict["mmn"].data,
        mmn_pre=evoked_dict["mmn_pre"].data,
        channels=epochs.ch_names,
        roi_channels=roi_channels,
    )

    # Add metadata
    metadata = {
        "step_analyze_mmn": {
            "creationDateTime": datetime.now().isoformat(),
            "roi_channels": roi_channels,
            "epoch_counts": {
                "standard": len(epochs[conditions["standard"]]),
                "predeviant": len(epochs[conditions["predeviant"]]),
                "deviant": len(epochs[conditions["deviant"]]),
            },
            "report_path": pdf_path,
            "data_path": str(
                derivatives_path.copy().update(suffix="erp-data", extension=".csv")
            ),
        }
    }

    manage_database(
        operation="update",
        update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
    )
