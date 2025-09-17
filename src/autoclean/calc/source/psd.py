"""Power spectral density helpers for source analysis."""
from __future__ import annotations

import os
import time
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from mne import SourceEstimate

from ._utils import ensure_stc_list
from scipy import signal


def calculate_source_psd(
    stc_or_stcs: SourceEstimate | Sequence[SourceEstimate],
    subjects_dir: str | None = None,
    subject: str = "fsaverage",
    n_jobs: int = 4,
    output_dir: str | None = None,
    subject_id: str | None = None,
    *,
    generate_plots: bool = True,
    segment_duration: float | None = 80,
):
    """Calculate ROI-averaged PSD from source estimates.

    The function now accepts either a single :class:`~mne.SourceEstimate` or a
    sequence of source estimates. When multiple epochs are provided they are
    forwarded to :func:`calculate_source_psd_list` so that the aggregation logic
    tailored for epoched data remains in effect.
    """

    stc_list, multiple = ensure_stc_list(stc_or_stcs)
    if multiple:
        return calculate_source_psd_list(
            stc_list,
            subjects_dir=subjects_dir,
            subject=subject,
            n_jobs=n_jobs,
            output_dir=output_dir,
            subject_id=subject_id,
            generate_plots=generate_plots,
            segment_duration=segment_duration,
        )

    return _calculate_source_psd_single(
        stc_list[0],
        subjects_dir=subjects_dir,
        subject=subject,
        n_jobs=n_jobs,
        output_dir=output_dir,
        subject_id=subject_id,
    )


def _calculate_source_psd_single(
    stc: SourceEstimate,
    subjects_dir: str | None = None,
    subject: str = "fsaverage",
    n_jobs: int = 4,
    output_dir: str | None = None,
    subject_id: str | None = None,
):
    """Calculate ROI-averaged PSD from a single source estimate."""
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    data = stc.data
    sfreq = stc.sfreq

    window_length = int(4 * sfreq)
    n_overlap = window_length // 2

    freqs = np.fft.rfftfreq(window_length, 1 / sfreq)
    freq_mask = (freqs >= 1.0) & (freqs <= 45.0)
    freqs = freqs[freq_mask]

    n_vertices = data.shape[0]
    psd = np.zeros((n_vertices, len(freqs)))

    def process_batch(vertices_idx: Iterable[int]) -> np.ndarray:
        indices = list(vertices_idx)
        batch_psd = np.zeros((len(indices), len(freqs)))
        for i, vertex_idx in enumerate(indices):
            f, Pxx = signal.welch(
                data[vertex_idx],
                fs=sfreq,
                window="hann",
                nperseg=window_length,
                noverlap=n_overlap,
                nfft=None,
                scaling="density",
            )
            batch_psd[i] = Pxx[freq_mask]
        return batch_psd

    batch_size = 1000
    for batch_idx in range(int(np.ceil(n_vertices / batch_size))):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)
        vertices_idx = range(start_idx, end_idx)
        psd[start_idx:end_idx] = process_batch(vertices_idx)

    labels = mne.read_labels_from_annot(
        subject, parc="aparc", subjects_dir=subjects_dir
    )
    labels = [label for label in labels if "unknown" not in label.name]

    roi_psds: list[dict[str, object]] = []
    for label in labels:
        label_verts = label.get_vertices_used()
        if label.hemi == "lh":
            stc_idx = np.where(np.in1d(stc.vertices[0], label_verts))[0]
        else:
            stc_idx = np.where(np.in1d(stc.vertices[1], label_verts))[0] + len(
                stc.vertices[0]
            )
        if len(stc_idx) == 0:
            continue
        roi_psd = np.mean(psd[stc_idx, :], axis=0)
        for freq_idx, freq in enumerate(freqs):
            roi_psds.append(
                {
                    "subject": subject_id,
                    "roi": label.name,
                    "hemisphere": label.hemi,
                    "frequency": freq,
                    "psd": roi_psd[freq_idx],
                }
            )

    psd_df = pd.DataFrame(roi_psds)

    file_path = os.path.join(output_dir, f"{subject_id}_roi_psd.parquet")
    psd_df.to_parquet(file_path)

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }

    band_psds: list[dict[str, object]] = []
    for roi in psd_df["roi"].unique():
        roi_data = psd_df[psd_df["roi"] == roi]
        for band_name, (band_min, band_max) in bands.items():
            band_data = roi_data[
                (roi_data["frequency"] >= band_min)
                & (roi_data["frequency"] < band_max)
            ]
            band_psds.append(
                {
                    "subject": subject_id,
                    "roi": roi,
                    "hemisphere": roi_data["hemisphere"].iloc[0],
                    "band": band_name,
                    "band_start_hz": band_min,
                    "band_end_hz": band_max,
                    "power": band_data["psd"].mean(),
                }
            )

    band_df = pd.DataFrame(band_psds)
    csv_path = os.path.join(output_dir, f"{subject_id}_roi_bands.csv")
    band_df.to_csv(csv_path, index=False)

    return psd_df, file_path


def calculate_source_psd_list(
    stc_list: Sequence[SourceEstimate] | SourceEstimate,
    subjects_dir: str | None = None,
    subject: str = "fsaverage",
    n_jobs: int = 4,
    output_dir: str | None = None,
    subject_id: str | None = None,
    generate_plots: bool = True,
    segment_duration: float | None = 80,
):
    """Calculate ROI-averaged PSD from a list of source estimates."""
    start_time = time.time()

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    if generate_plots:
        os.makedirs(os.path.join(output_dir, "psd_plots"), exist_ok=True)

    if subject_id is None:
        subject_id = "unknown_subject"

    stc_list, _ = ensure_stc_list(stc_list)

    epoch_duration = stc_list[0].times[-1] - stc_list[0].times[0]
    total_duration = epoch_duration * len(stc_list)
    sfreq = stc_list[0].sfreq

    selected_stcs = list(stc_list)
    if segment_duration is not None and segment_duration < total_duration:
        n_epochs_needed = int(np.ceil(segment_duration / epoch_duration))
        n_epochs_needed = min(n_epochs_needed, len(stc_list))
        start_idx = (len(stc_list) - n_epochs_needed) // 2
        selected_stcs = list(stc_list)[start_idx : start_idx + n_epochs_needed]

    fmin, fmax = 0.5, 45.0
    n_vertices = selected_stcs[0].data.shape[0]
    available_duration = len(selected_stcs) * epoch_duration
    window_length = int(min(4 * sfreq, available_duration * sfreq / 8))
    n_overlap = window_length // 2

    freqs = np.fft.rfftfreq(window_length, 1 / sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "lowalpha": (8, 10),
        "highalpha": (10, 13),
        "lowbeta": (13, 20),
        "highbeta": (20, 30),
        "gamma": (30, 45),
    }

    band_indices = {
        band_name: np.where((freqs >= band_min) & (freqs < band_max))[0]
        for band_name, (band_min, band_max) in bands.items()
    }

    sample_size = min(1000, n_vertices)
    sample_indices = np.linspace(0, n_vertices - 1, sample_size, dtype=int)
    vertex_variance = np.zeros(sample_size)
    for i, vertex_idx in enumerate(sample_indices):
        vertex_data = np.concatenate([stc.data[vertex_idx] for stc in selected_stcs])
        vertex_variance[i] = np.var(vertex_data)

    non_zero_variance = vertex_variance[vertex_variance > 0]
    if len(non_zero_variance) > 0:
        variance_threshold = np.percentile(non_zero_variance, 10)
    else:
        variance_threshold = 0

    labels = mne.read_labels_from_annot(
        subject, parc="aparc", subjects_dir=subjects_dir
    )
    labels = [label for label in labels if "unknown" not in label.name]

    def process_label(label_idx: int):
        label = labels[label_idx]
        label_verts = label.get_vertices_used()
        if label.hemi == "lh":
            stc_idx = np.where(np.in1d(selected_stcs[0].vertices[0], label_verts))[0]
        else:
            stc_idx = np.where(np.in1d(selected_stcs[0].vertices[1], label_verts))[0] + len(
                selected_stcs[0].vertices[0]
            )
        if len(stc_idx) == 0:
            return None

        psd = np.zeros((len(stc_idx), len(freqs)))
        for stc_i, stc in enumerate(selected_stcs):
            for i, vertex_idx in enumerate(stc_idx):
                vertex_data = stc.data[vertex_idx]
                if np.var(vertex_data) < variance_threshold:
                    continue
                f, Pxx = signal.welch(
                    vertex_data,
                    fs=sfreq,
                    window="hann",
                    nperseg=window_length,
                    noverlap=n_overlap,
                    nfft=None,
                    scaling="density",
                )
                psd[i] += Pxx[freq_mask]
        psd /= len(selected_stcs)

        roi_psd = np.mean(psd, axis=0)
        roi_psd_data = [
            {
                "subject": subject_id,
                "roi": label.name,
                "hemisphere": label.hemi,
                "frequency": freq,
                "psd": roi_psd[freq_idx],
            }
            for freq_idx, freq in enumerate(freqs)
        ]

        band_data = []
        for band_name, indices in band_indices.items():
            power = np.mean(roi_psd[indices]) if len(indices) > 0 else 0
            band_min, band_max = bands[band_name]
            band_data.append(
                {
                    "subject": subject_id,
                    "roi": label.name,
                    "hemisphere": label.hemi,
                    "band": band_name,
                    "band_start_hz": band_min,
                    "band_end_hz": band_max,
                    "power": power,
                }
            )
        return roi_psd_data, band_data

    roi_results = Parallel(n_jobs=n_jobs)(
        delayed(process_label)(i) for i in range(len(labels))
    )

    roi_psds: list[dict[str, object]] = []
    band_psds: list[dict[str, object]] = []
    for result in roi_results:
        if result is not None:
            roi_psd_data, band_data = result
            roi_psds.extend(roi_psd_data)
            band_psds.extend(band_data)

    psd_df = pd.DataFrame(roi_psds)
    band_df = pd.DataFrame(band_psds)

    file_path = os.path.join(output_dir, f"{subject_id}_roi_psd.parquet")
    psd_df.to_parquet(file_path)

    csv_path = os.path.join(output_dir, f"{subject_id}_roi_bands.csv")
    band_df.to_csv(csv_path, index=False)

    if generate_plots and not band_df.empty:
        band_summary = (
            band_df.groupby(["band", "hemisphere"])["power"].mean().reset_index()
        )
        pivot_df = band_summary.pivot(
            index="band", columns="hemisphere", values="power"
        )
        band_order = [
            "delta",
            "theta",
            "lowalpha",
            "highalpha",
            "alpha",
            "lowbeta",
            "highbeta",
            "gamma",
        ]
        pivot_df = pivot_df.reindex(band_order)

        plt.figure(figsize=(12, 8))
        pivot_df.plot(kind="bar", ax=plt.gca())
        plt.title(f"Mean Band Power by Hemisphere - {subject_id}")
        plt.ylabel("Power (µV²/Hz)")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject_id}_hemisphere_bands.png"))

        plt.figure(figsize=(12, 8))
        np.log10(pivot_df).plot(kind="bar", ax=plt.gca())
        plt.title(f"Log10 Mean Band Power by Hemisphere - {subject_id}")
        plt.ylabel("Log10 Power")
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject_id}_hemisphere_bands_log.png"))

    total_time = time.time() - start_time
    print(
        f"Total processing time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)"
    )

    return psd_df, file_path


def visualize_psd_results(
    psd_df: pd.DataFrame, output_dir: str | None = None, subject_id: str | None = None
):
    """Create diagnostic plots for PSD data."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    if output_dir is None:
        output_dir = os.getcwd()

    if subject_id is None:
        subject_id = psd_df["subject"].iloc[0]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)

    bands = {
        "delta": (1, 4, "#1f77b4"),
        "theta": (4, 8, "#ff7f0e"),
        "alpha": (8, 13, "#2ca02c"),
        "beta": (13, 30, "#d62728"),
        "gamma": (30, 45, "#9467bd"),
    }

    ax1 = fig.add_subplot(gs[0, 0])
    regions_to_plot = [
        "precentral-lh",
        "postcentral-lh",
        "superiorparietal-lh",
        "lateraloccipital-lh",
        "superiorfrontal-lh",
    ]
    available_rois = psd_df["roi"].unique()
    regions_to_plot = [r for r in regions_to_plot if r in available_rois]
    if not regions_to_plot:
        regions_to_plot = list(available_rois)[:5]

    for roi in regions_to_plot:
        roi_data = psd_df[psd_df["roi"] == roi]
        ax1.plot(
            roi_data["frequency"],
            roi_data["psd"],
            linewidth=2,
            alpha=0.8,
            label=roi.split("-")[0],
        )

    for band_name, (fmin, fmax, color) in bands.items():
        ax1.axvspan(fmin, fmax, color=color, alpha=0.1)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Spectral Density")
    ax1.set_title("PSD for Selected Regions (Left Hemisphere)")
    ax1.legend(loc="upper right")
    ax1.set_xlim(1, 45)
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    for hemi, color, label in zip(
        ["lh", "rh"], ["#1f77b4", "#d62728"], ["Left Hemisphere", "Right Hemisphere"]
    ):
        hemi_data = (
            psd_df[psd_df["hemisphere"] == hemi]
            .groupby("frequency")["psd"]
            .mean()
            .reset_index()
        )
        ax2.plot(
            hemi_data["frequency"],
            hemi_data["psd"],
            linewidth=2.5,
            color=color,
            label=label,
        )
        for band_name, (fmin, fmax, band_color) in bands.items():
            ax2.axvspan(fmin, fmax, color=band_color, alpha=0.1)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power Spectral Density")
    ax2.set_title("Left vs Right Hemisphere Average")
    ax2.legend(loc="upper right")
    ax2.set_xlim(1, 45)
    ax2.grid(True, which="both", ls="--", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    summary_df = (
        psd_df.groupby(["roi", "hemisphere"])["psd"].mean().reset_index()
    )
    pivot = summary_df.pivot_table(
        index="roi", columns="hemisphere", values="psd"
    ).fillna(0)
    pivot_sorted = pivot.sort_values(by=["lh", "rh"], ascending=False).head(20)
    pivot_sorted.plot(kind="bar", ax=ax3)
    ax3.set_title("Top 20 Regions by Mean PSD")
    ax3.set_ylabel("Mean PSD")
    ax3.grid(True, axis="y")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")

    ax4 = fig.add_subplot(gs[1, 1])
    for band_name, (fmin, fmax, color) in bands.items():
        band_data = psd_df[
            (psd_df["frequency"] >= fmin) & (psd_df["frequency"] <= fmax)
        ]
        sns.kdeplot(
            data=band_data,
            x="psd",
            hue="hemisphere",
            fill=True,
            common_norm=False,
            alpha=0.5,
            ax=ax4,
            label=band_name,
        )
    ax4.set_xlabel("PSD Value")
    ax4.set_title("Distribution of PSD Values by Band and Hemisphere")
    ax4.grid(True, axis="x")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, f"{subject_id}_psd_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )

    return fig


__all__ = [
    "calculate_source_psd",
    "calculate_source_psd_list",
    "visualize_psd_results",
]
