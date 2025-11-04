"""AutoReject reporting utilities for comprehensive PDF generation.

This module provides visualization and reporting for the AutoReject artifact rejection
method, showing epoch quality, rejection statistics, and interpolation patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec


@dataclass
class AutoRejectReportResult:
    """Results from AutoReject report generation."""

    output_pdf: Path
    summary: Dict
    rejection_log: np.ndarray
    interpolated_channels: List[str]


def generate_autoreject_report(
    epochs_before: mne.Epochs,
    epochs_after: mne.Epochs,
    output_pdf: Union[str, Path],
    rejection_log: Optional[np.ndarray] = None,
    ar_params: Optional[Dict] = None,
) -> AutoRejectReportResult:
    """Generate a comprehensive PDF report for AutoReject epoch cleaning.

    Parameters
    ----------
    epochs_before : mne.Epochs
        Original epochs before AutoReject processing
    epochs_after : mne.Epochs
        Cleaned epochs after AutoReject processing
    output_pdf : str or Path
        Output path for the PDF report
    rejection_log : ndarray, optional
        AutoReject rejection log (n_epochs x n_channels) with codes:
        0 = good, 1 = interpolated, 2 = rejected
    ar_params : dict, optional
        AutoReject parameters used (n_interpolate, consensus, etc.)

    Returns
    -------
    AutoRejectReportResult
        Report results including summary statistics and file path
    """
    output_pdf_path = Path(output_pdf)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate rejection statistics
    n_before = len(epochs_before)
    n_after = len(epochs_after)
    n_rejected = n_before - n_after
    rejection_pct = (n_rejected / n_before) * 100 if n_before > 0 else 0.0

    # Identify interpolated channels
    interpolated_channels = []
    if rejection_log is not None:
        # Count interpolations per channel across all epochs
        n_interp_per_channel = np.sum(rejection_log == 1, axis=0)
        ch_names = epochs_before.ch_names
        interpolated_channels = [
            ch_names[i] for i in range(len(ch_names)) if n_interp_per_channel[i] > 0
        ]

    # Calculate PSDs (using welch method which supports fmax parameter)
    psd_before, freqs_before = epochs_before.compute_psd(
        method="welch", fmax=50.0
    ).get_data(return_freqs=True)
    psd_after, freqs_after = epochs_after.compute_psd(
        method="welch", fmax=50.0
    ).get_data(return_freqs=True)

    # Average across epochs
    psd_before_mean = psd_before.mean(axis=0)  # (n_channels, n_freqs)
    psd_after_mean = psd_after.mean(axis=0)

    # Build summary
    summary = {
        "epochs_before": n_before,
        "epochs_after": n_after,
        "rejected": n_rejected,
        "rejection_pct": float(rejection_pct),
        "interpolated_channels": interpolated_channels,
        "n_interpolated_channels": len(interpolated_channels),
        "sfreq": float(epochs_before.info["sfreq"]),
        "n_channels": len(epochs_before.ch_names),
        "params": ar_params or {},
    }

    # Create PDF report with multiple pages
    with PdfPages(output_pdf_path) as pdf:
        # Page 1: Overview and statistics
        fig1 = _create_overview_page(
            summary,
            epochs_before.ch_names,
            psd_before_mean,
            psd_after_mean,
            freqs_before,
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        # Page 2: Rejection visualization
        if rejection_log is not None:
            fig2 = _create_rejection_visualization(
                rejection_log, epochs_before.ch_names
            )
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

        # Page 3: Channel interpolation heatmap
        if rejection_log is not None:
            fig3 = _create_interpolation_heatmap(rejection_log, epochs_before.ch_names)
            pdf.savefig(fig3, bbox_inches="tight")
            plt.close(fig3)

        # Add metadata
        d = pdf.infodict()
        d["Title"] = "AutoReject Epoch Cleaning Report"
        d["Author"] = "AutoCleanEEG Pipeline"
        d["Subject"] = "Automated epoch artifact rejection and channel interpolation"

    return AutoRejectReportResult(
        output_pdf=output_pdf_path,
        summary=summary,
        rejection_log=rejection_log if rejection_log is not None else np.array([]),
        interpolated_channels=interpolated_channels,
    )


def _create_overview_page(
    summary: Dict,
    ch_names: List[str],
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    freqs: np.ndarray,
) -> plt.Figure:
    """Create overview page with statistics and PSD comparison."""
    fig = plt.figure(figsize=(11, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle(
        "AutoReject Epoch Cleaning Report",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Panel 1: Rejection Statistics (text box)
    ax_stats = fig.add_subplot(gs[0, :])
    ax_stats.axis("off")

    stats_text = f"""
REJECTION STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Epochs Before:        {summary['epochs_before']}
Epochs After:         {summary['epochs_after']}
Rejected:             {summary['rejected']} ({summary['rejection_pct']:.1f}%)

Channels:             {summary['n_channels']}
Sampling Rate:        {summary['sfreq']:.1f} Hz
Interpolated Channels: {summary['n_interpolated_channels']}

PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    if summary["params"]:
        for key, val in summary["params"].items():
            stats_text += f"{key:20s}: {val}\n"

    ax_stats.text(
        0.05,
        0.95,
        stats_text,
        transform=ax_stats.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # Panel 2: Rejection Rate Pie Chart
    ax_pie = fig.add_subplot(gs[1, 0])
    rejected = summary["rejected"]
    kept = summary["epochs_after"]
    colors = ["#ff6b6b", "#51cf66"]
    ax_pie.pie(
        [rejected, kept],
        labels=[f"Rejected\n{rejected}", f"Kept\n{kept}"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax_pie.set_title("Epoch Rejection Rate", fontweight="bold", pad=10)

    # Panel 3: Interpolated Channels Bar
    ax_bar = fig.add_subplot(gs[1, 1])
    if summary["interpolated_channels"]:
        interp_ch = summary["interpolated_channels"][:10]  # Top 10
        ax_bar.barh(range(len(interp_ch)), [1] * len(interp_ch), color="#4dabf7")
        ax_bar.set_yticks(range(len(interp_ch)))
        ax_bar.set_yticklabels(interp_ch)
        ax_bar.set_xlabel("Interpolated")
        ax_bar.set_title("Interpolated Channels (Top 10)", fontweight="bold", pad=10)
        ax_bar.invert_yaxis()
    else:
        ax_bar.text(
            0.5,
            0.5,
            "No channels interpolated",
            ha="center",
            va="center",
            transform=ax_bar.transAxes,
        )
        ax_bar.axis("off")

    # Panel 4: PSD Comparison (Grand Average)
    ax_psd = fig.add_subplot(gs[2:, :])

    # Average across channels
    psd_before_avg = 10 * np.log10(psd_before.mean(axis=0))
    psd_after_avg = 10 * np.log10(psd_after.mean(axis=0))

    ax_psd.plot(
        freqs, psd_before_avg, label="Before AutoReject", alpha=0.7, linewidth=2
    )
    ax_psd.plot(freqs, psd_after_avg, label="After AutoReject", alpha=0.7, linewidth=2)
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_psd.set_ylabel("Power (dB)", fontsize=11)
    ax_psd.set_title(
        "Power Spectral Density Comparison (Grand Average)",
        fontweight="bold",
        pad=15,
        fontsize=12,
    )
    ax_psd.legend(loc="upper right")
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_xlim([0, 50])

    return fig


def _create_rejection_visualization(
    rejection_log: np.ndarray, ch_names: List[str]
) -> plt.Figure:
    """Create visualization showing which epochs were rejected/interpolated."""
    fig = plt.figure(figsize=(11, 14))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3)

    fig.suptitle("Epoch Rejection Pattern", fontsize=16, fontweight="bold", y=0.98)

    # Panel 1: Heatmap of rejection log
    ax_heat = fig.add_subplot(gs[0, 0])

    # rejection_log: 0=good, 1=interpolated, 2=rejected
    # Create custom colormap
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#51cf66", "#ffd43b", "#ff6b6b"])  # green, yellow, red

    im = ax_heat.imshow(
        rejection_log.T,  # Transpose so channels are on y-axis
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=2,
    )

    ax_heat.set_xlabel("Epoch Number", fontsize=11)
    ax_heat.set_ylabel("Channel", fontsize=11)
    ax_heat.set_title(
        "Rejection Log (Green=Good, Yellow=Interpolated, Red=Rejected)",
        fontweight="bold",
        pad=10,
    )

    # Show subset of channel labels if too many
    if len(ch_names) > 30:
        step = len(ch_names) // 20
        yticks = range(0, len(ch_names), step)
        ax_heat.set_yticks(yticks)
        ax_heat.set_yticklabels([ch_names[i] for i in yticks], fontsize=8)
    else:
        ax_heat.set_yticks(range(len(ch_names)))
        ax_heat.set_yticklabels(ch_names, fontsize=8)

    plt.colorbar(im, ax=ax_heat, ticks=[0, 1, 2], label="Status")

    # Panel 2: Per-epoch rejection summary
    ax_summary = fig.add_subplot(gs[1, 0])

    # Count bad channels per epoch
    n_interpolated = np.sum(rejection_log == 1, axis=1)
    n_rejected = np.sum(rejection_log == 2, axis=1)

    epochs = np.arange(rejection_log.shape[0])

    ax_summary.bar(
        epochs,
        n_interpolated,
        label="Interpolated Channels",
        color="#ffd43b",
        alpha=0.7,
    )
    ax_summary.bar(
        epochs,
        n_rejected,
        bottom=n_interpolated,
        label="Rejected",
        color="#ff6b6b",
        alpha=0.7,
    )

    ax_summary.set_xlabel("Epoch Number", fontsize=11)
    ax_summary.set_ylabel("Number of Bad Channels", fontsize=11)
    ax_summary.set_title("Bad Channels per Epoch", fontweight="bold", pad=10)
    ax_summary.legend(loc="upper right")
    ax_summary.grid(True, alpha=0.3, axis="y")

    return fig


def _create_interpolation_heatmap(
    rejection_log: np.ndarray, ch_names: List[str]
) -> plt.Figure:
    """Create heatmap showing interpolation frequency per channel."""
    fig = plt.figure(figsize=(11, 14))

    fig.suptitle(
        "Channel Interpolation Summary", fontsize=16, fontweight="bold", y=0.98
    )

    # Count interpolations per channel
    n_interp_per_channel = np.sum(rejection_log == 1, axis=0)

    # Sort channels by interpolation frequency
    sorted_indices = np.argsort(n_interp_per_channel)[::-1]
    sorted_channels = [ch_names[i] for i in sorted_indices]
    sorted_counts = n_interp_per_channel[sorted_indices]

    # Plot top 30 channels
    n_show = min(30, len(ch_names))

    ax = fig.add_subplot(111)
    bars = ax.barh(range(n_show), sorted_counts[:n_show], color="#4dabf7")

    # Color bars by frequency
    max_count = sorted_counts[0] if len(sorted_counts) > 0 else 1
    for i, bar in enumerate(bars):
        intensity = sorted_counts[i] / max_count if max_count > 0 else 0
        bar.set_color(plt.cm.RdYlGn_r(intensity))

    ax.set_yticks(range(n_show))
    ax.set_yticklabels(sorted_channels[:n_show], fontsize=9)
    ax.set_xlabel("Number of Interpolations", fontsize=11)
    ax.set_ylabel("Channel", fontsize=11)
    ax.set_title(
        f"Interpolation Frequency (Top {n_show} Channels)",
        fontweight="bold",
        pad=15,
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Add count labels on bars
    for i, (count, bar) in enumerate(zip(sorted_counts[:n_show], bars)):
        if count > 0:
            ax.text(
                count,
                bar.get_y() + bar.get_height() / 2,
                f"  {int(count)}",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    return fig
