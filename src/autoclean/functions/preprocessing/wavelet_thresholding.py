"""Wavelet thresholding for EEG data and reporting utilities.

This module implements wavelet-based denoising identical in spirit to the
HAPPE MATLAB pipeline. It performs a discrete wavelet transform on each channel
and applies universal soft-thresholding to attenuate high-amplitude
transients. The module also contains PDF reporting helpers so the
thresholding logic can be used standalone without importing additional
subpackages.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pywt
from mne.time_frequency import psd_array_welch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as ReportImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table as ReportTable,
    TableStyle,
)

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Try to use a pleasant matplotlib style for the report figures
try:  # pragma: no cover - style availability depends on matplotlib version
    plt.style.use("seaborn-v0_8-whitegrid")
except (OSError, ValueError):  # Fallback if style is missing
    plt.style.use("ggplot")


class WaveletReportResult:
    """Container for generated wavelet report artifacts."""

    def __init__(
        self,
        pdf_path: Path,
        metrics: pd.DataFrame,
        psd_metrics: pd.DataFrame,
        summary: Dict[str, Union[str, float, int, Dict[str, float]]],
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.metrics = metrics
        self.psd_metrics = psd_metrics
        self.summary = summary


def _resolve_decomposition_level(
    signal_length: int,
    wavelet: str,
    level: int,
) -> int:
    """Return a safe decomposition level for the requested wavelet."""

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(signal_length, wavelet_obj.dec_len)
    if max_level <= 0:
        return 0
    return min(level, max_level)


def _denoise_signal(
    signal: np.ndarray,
    wavelet: str,
    level: int,
) -> np.ndarray:
    """Denoise a 1D signal using wavelet thresholding."""

    signal_array = np.asarray(signal)
    effective_level = _resolve_decomposition_level(signal_array.size, wavelet, level)
    if effective_level == 0:
        return signal_array.copy()

    coeffs = pywt.wavedec(signal_array, wavelet, level=effective_level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(signal_array.size)) if sigma else 0.0
    coeffs_thresh = [coeffs[0]]
    for coeff in coeffs[1:]:
        coeffs_thresh.append(pywt.threshold(coeff, uthresh, mode="soft"))
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    if denoised.shape[0] != signal_array.shape[0]:
        denoised = denoised[: signal_array.shape[0]]
    return denoised.astype(signal_array.dtype, copy=False)


def wavelet_threshold(
    data: Union[mne.io.BaseRaw, mne.Epochs],
    wavelet: str = "sym4",
    level: int = 5,
) -> Union[mne.io.BaseRaw, mne.Epochs]:
    """Apply wavelet thresholding to EEG data."""

    cleaned = data.copy()
    if isinstance(cleaned, mne.io.BaseRaw):
        arr = cleaned.get_data()
        for idx in range(arr.shape[0]):
            arr[idx] = _denoise_signal(arr[idx], wavelet, level)
        cleaned._data = arr
    elif isinstance(cleaned, mne.Epochs):
        arr = cleaned.get_data()
        for epoch in range(arr.shape[0]):
            for channel in range(arr.shape[1]):
                arr[epoch, channel] = _denoise_signal(arr[epoch, channel], wavelet, level)
        cleaned._data = arr
    else:
        raise TypeError("data must be mne.io.BaseRaw or mne.Epochs")
    return cleaned


def _load_raw_object(
    source: Union[str, Path, mne.io.BaseRaw],
    preload: bool = True,
) -> Tuple[mne.io.BaseRaw, Optional[Path]]:
    """Load an MNE Raw object from a path or return a copy if already provided."""

    if isinstance(source, mne.io.BaseRaw):
        return source.copy(), None

    path = Path(source)
    suffix = "".join(path.suffixes).lower()

    if ".fif" in suffix:
        raw = mne.io.read_raw_fif(path, preload=preload, verbose=False)
    elif path.suffix.lower() == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=preload, verbose=False)
    elif path.suffix.lower() in {".edf", ".bdf"}:
        raw = mne.io.read_raw_edf(path, preload=preload, verbose=False)
    else:
        raise ValueError(
            f"Unsupported file type for wavelet report: '{path.suffix}'."
        )

    return raw, path


def _compute_channel_metrics(
    baseline: np.ndarray,
    cleaned: np.ndarray,
    ch_names: Sequence[str],
    scaling: float = 1e6,
) -> pd.DataFrame:
    """Compute peak-to-peak and standard deviation metrics per channel."""

    ptp_before = np.ptp(baseline, axis=1) * scaling
    ptp_after = np.ptp(cleaned, axis=1) * scaling
    std_before = baseline.std(axis=1) * scaling
    std_after = cleaned.std(axis=1) * scaling

    reduction_ptp = np.zeros_like(ptp_before)
    np.divide(
        ptp_before - ptp_after,
        ptp_before,
        out=reduction_ptp,
        where=ptp_before != 0,
    )
    reduction_ptp *= 100

    reduction_std = np.zeros_like(std_before)
    np.divide(
        std_before - std_after,
        std_before,
        out=reduction_std,
        where=std_before != 0,
    )
    reduction_std *= 100

    metrics = pd.DataFrame(
        {
            "channel": ch_names,
            "ptp_before_uv": ptp_before,
            "ptp_after_uv": ptp_after,
            "ptp_reduction_pct": reduction_ptp,
            "std_before_uv": std_before,
            "std_after_uv": std_after,
            "std_reduction_pct": reduction_std,
        }
    )

    return metrics


def _compute_psd_metrics(
    baseline: np.ndarray,
    cleaned: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    bands: Dict[str, Tuple[float, float]] = FREQUENCY_BANDS,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Welch PSDs and band power reductions."""

    n_times = baseline.shape[1]
    n_fft = min(1024, n_times)
    n_overlap = max(n_fft // 2, 0)
    psd_kwargs = dict(fmin=1.0, fmax=45.0, n_fft=n_fft, n_overlap=n_overlap, verbose=False)
    psd_before, freqs = psd_array_welch(baseline, sfreq=sfreq, **psd_kwargs)
    psd_after, _ = psd_array_welch(cleaned, sfreq=sfreq, **psd_kwargs)

    records = []
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            continue
        band_power_before = psd_before[:, mask].mean(axis=1)
        band_power_after = psd_after[:, mask].mean(axis=1)
        reduction = np.zeros_like(band_power_before)
        np.divide(
            band_power_before - band_power_after,
            band_power_before,
            out=reduction,
            where=band_power_before != 0,
        )
        reduction *= 100
        for idx, channel in enumerate(ch_names):
            records.append(
                {
                    "channel": channel,
                    "band": band,
                    "power_before": band_power_before[idx],
                    "power_after": band_power_after[idx],
                    "power_reduction_pct": reduction[idx],
                }
            )

    psd_metrics = pd.DataFrame(records)
    return psd_metrics, freqs, psd_before, psd_after


def _build_overview_figure(
    baseline: np.ndarray,
    cleaned: np.ndarray,
    ch_names: Sequence[str],
    sfreq: float,
    snippet_duration: float,
    metrics: pd.DataFrame,
    scaling: float = 1e6,
) -> io.BytesIO:
    """Create a matplotlib figure summarizing wavelet effects."""

    if metrics.empty:
        raise ValueError("Metrics dataframe cannot be empty when building figures")

    num_samples = baseline.shape[1]
    snippet_samples = min(int(snippet_duration * sfreq), num_samples)
    if snippet_samples <= 0:
        snippet_samples = num_samples

    time_axis = np.arange(snippet_samples) / sfreq

    top_channels = (
        metrics.sort_values("ptp_reduction_pct", ascending=False)["channel"].tolist()
    )
    if not top_channels:
        top_channels = [ch_names[0]]

    top_idx = ch_names.index(top_channels[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    axes[0].plot(
        time_axis,
        baseline[top_idx, :snippet_samples] * scaling,
        label="Original",
        linewidth=1.2,
        color="#1f77b4",
    )
    axes[0].plot(
        time_axis,
        cleaned[top_idx, :snippet_samples] * scaling,
        label="Wavelet-cleaned",
        linewidth=1.2,
        color="#d62728",
    )
    axes[0].set_title(f"Channel {ch_names[top_idx]} (top reduction)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend(frameon=False)

    top_n = metrics.sort_values("ptp_reduction_pct", ascending=False).head(10)
    axes[1].barh(
        top_n["channel"],
        top_n["ptp_reduction_pct"],
        color="#2ca02c",
        alpha=0.8,
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Peak-to-peak reduction (%)")
    axes[1].set_title("Top 10 channels by reduction")

    fig.suptitle("Wavelet Thresholding Overview", fontsize=14)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _build_psd_figure(
    freqs: np.ndarray,
    psd_before: np.ndarray,
    psd_after: np.ndarray,
    psd_metrics: pd.DataFrame,
) -> io.BytesIO:
    """Visualize mean PSD and band reductions."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    mean_before = psd_before.mean(axis=0)
    mean_after = psd_after.mean(axis=0)
    axes[0].plot(freqs, mean_before, label="Original", color="#1f77b4", linewidth=1.2)
    axes[0].plot(freqs, mean_after, label="Wavelet-cleaned", color="#d62728", linewidth=1.2)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (V²/Hz)")
    axes[0].set_title("Mean PSD across channels")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)

    band_summary = (
        psd_metrics.groupby("band")["power_reduction_pct"].mean().reindex(FREQUENCY_BANDS.keys())
    )
    axes[1].bar(
        band_summary.index,
        band_summary.values,
        color="#9467bd",
        alpha=0.85,
    )
    axes[1].axhline(0, color="#444", linewidth=0.8)
    axes[1].set_ylabel("Mean band power change (%)")
    axes[1].set_title("Band power reductions")

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _create_summary_table(summary: Dict[str, Union[str, float, int, Dict[str, float]]]) -> ReportTable:
    """Create a styled summary table for the PDF report."""

    table_data = [
        ["Channels analyzed", f"{summary['channels']}"]
    ]
    table_data.extend(
        [
            ["Sampling rate (Hz)", f"{summary['sfreq']:.2f}"],
            ["Duration (s)", f"{summary['duration_sec']:.2f}"],
            [
                "Effective wavelet level",
                f"{summary['effective_level']} (requested {summary['requested_level']})",
            ],
            [
                "Mean peak-to-peak reduction (%)",
                f"{summary['ptp_mean']:.2f}",
            ],
            [
                "Median peak-to-peak reduction (%)",
                f"{summary['ptp_median']:.2f}",
            ],
            [
                "Maximum reduction channel",
                f"{summary['ptp_max_channel']} ({summary['ptp_max']:.2f}%)",
            ],
        ]
    )

    band_reductions = summary.get("band_reductions", {})
    for band in FREQUENCY_BANDS.keys():
        if band in band_reductions:
            table_data.append(
                [
                    f"{band.title()} band change (%)",
                    f"{band_reductions[band]:.2f}",
                ]
            )

    table = ReportTable(table_data, colWidths=[2.8 * inch, 3.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ECF0F1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#2C3E50")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FBFC")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BDC3C7")),
            ]
        )
    )
    return table


def _create_psd_table(psd_metrics: pd.DataFrame) -> ReportTable:
    """Create a table summarizing band power reductions."""

    band_summary = (
        psd_metrics.groupby(["band"])["power_reduction_pct"].mean().reindex(FREQUENCY_BANDS.keys())
    )
    table_data = [["Band", "Mean power change (%)"]]
    for band, value in band_summary.items():
        table_data.append([band.title(), f"{value:.2f}"])

    table = ReportTable(table_data, colWidths=[2.2 * inch, 2.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#2C3E50")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DBDB")),
            ]
        )
    )

    return table


def _create_top_channel_table(metrics: pd.DataFrame, top_n: int = 10) -> ReportTable:
    """Render a table of the top channels ranked by reduction."""

    top_channels = metrics.sort_values("ptp_reduction_pct", ascending=False).head(top_n)
    table_data = [["Channel", "P2P before (µV)", "P2P after (µV)", "Reduction (%)", "STD reduction (%)"]]

    for _, row in top_channels.iterrows():
        table_data.append(
            [
                row["channel"],
                f"{row['ptp_before_uv']:.3f}",
                f"{row['ptp_after_uv']:.3f}",
                f"{row['ptp_reduction_pct']:.2f}",
                f"{row['std_reduction_pct']:.2f}",
            ]
        )

    table = ReportTable(table_data, colWidths=[1.4 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F9")]),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#2C3E50")),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DBDB")),
            ]
        )
    )

    return table


def _build_pdf_report(
    pdf_path: Path,
    source_name: str,
    figure_buffer: io.BytesIO,
    psd_buffer: io.BytesIO,
    summary_table: ReportTable,
    psd_table: ReportTable,
    channel_table: ReportTable,
    summary: Dict[str, Union[str, float, int, Dict[str, float]]],
) -> None:
    """Assemble and write the PDF report."""

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "WaveletTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#2C3E50"),
        alignment=1,
    )
    heading_style = ParagraphStyle(
        "WaveletHeading",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#34495E"),
    )
    normal_style = ParagraphStyle(
        "WaveletNormal",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2C3E50"),
    )

    story = []
    story.append(Paragraph("Wavelet Thresholding Report", title_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Source file: {source_name}", normal_style))
    story.append(
        Paragraph(
            f"Channels analysed: {summary['channels']} (picks applied)", normal_style
        )
    )
    story.append(Spacer(1, 0.25 * inch))

    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Overview", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    overview_img = ReportImage(figure_buffer, width=6.5 * inch, height=4.0 * inch)
    story.append(overview_img)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Power spectral density", heading_style))
    story.append(Spacer(1, 0.1 * inch))
    psd_img = ReportImage(psd_buffer, width=6.5 * inch, height=4.0 * inch)
    story.append(psd_img)
    story.append(Spacer(1, 0.2 * inch))
    story.append(psd_table)
    story.append(Spacer(1, 0.3 * inch))

    story.append(Paragraph("Top channels", heading_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(channel_table)

    doc.build(story)


def generate_wavelet_report(
    source: Union[str, Path, mne.io.BaseRaw],
    output_pdf: Union[str, Path],
    wavelet: str = "sym4",
    level: int = 5,
    picks: Union[str, Iterable[str]] = "eeg",
    snippet_duration: float = 5.0,
    top_n_channels: int = 10,
) -> WaveletReportResult:
    """Generate a PDF report comparing pre/post wavelet thresholding."""

    raw, source_path = _load_raw_object(source)
    source_name = source_path.name if source_path else getattr(raw, "filenames", ["Raw data"])[0]
    output_pdf_path = Path(output_pdf)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    raw_subset = raw.copy()
    if picks:
        raw_subset.pick(picks)

    baseline = raw_subset.get_data()
    cleaned = wavelet_threshold(raw_subset, wavelet=wavelet, level=level).get_data()

    metrics = _compute_channel_metrics(baseline, cleaned, raw_subset.ch_names)
    psd_metrics, freqs, psd_before, psd_after = _compute_psd_metrics(
        baseline, cleaned, sfreq=float(raw_subset.info["sfreq"]), ch_names=raw_subset.ch_names
    )

    effective_level = _resolve_decomposition_level(
        baseline.shape[1], wavelet, level
    )
    sfreq = float(raw_subset.info["sfreq"])
    duration = baseline.shape[1] / sfreq if sfreq else 0.0

    band_reductions = (
        psd_metrics.groupby("band")["power_reduction_pct"].mean().to_dict()
        if not psd_metrics.empty
        else {}
    )

    summary = {
        "channels": int(len(raw_subset.ch_names)),
        "sfreq": sfreq,
        "duration_sec": duration,
        "effective_level": effective_level,
        "requested_level": level,
        "ptp_mean": float(metrics["ptp_reduction_pct"].mean()),
        "ptp_median": float(metrics["ptp_reduction_pct"].median()),
        "ptp_max": float(metrics["ptp_reduction_pct"].max()),
        "ptp_max_channel": str(
            metrics.loc[metrics["ptp_reduction_pct"].idxmax(), "channel"]
            if not metrics.empty
            else "N/A"
        ),
        "band_reductions": {band: float(band_reductions.get(band, 0.0)) for band in FREQUENCY_BANDS.keys()},
    }

    figure_buffer = _build_overview_figure(
        baseline,
        cleaned,
        raw_subset.ch_names,
        sfreq,
        snippet_duration,
        metrics,
    )
    psd_buffer = _build_psd_figure(freqs, psd_before, psd_after, psd_metrics)

    summary_table = _create_summary_table(summary)
    psd_table = _create_psd_table(psd_metrics)
    channel_table = _create_top_channel_table(metrics, top_n=top_n_channels)

    _build_pdf_report(
        output_pdf_path,
        source_name,
        figure_buffer,
        psd_buffer,
        summary_table,
        psd_table,
        channel_table,
        summary,
    )

    return WaveletReportResult(
        pdf_path=output_pdf_path,
        metrics=metrics,
        psd_metrics=psd_metrics,
        summary=summary,
    )


__all__ = [
    "wavelet_threshold",
    "WaveletReportResult",
    "generate_wavelet_report",
]
