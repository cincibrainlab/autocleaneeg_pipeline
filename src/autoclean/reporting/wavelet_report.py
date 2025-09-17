"""Wavelet thresholding reporting utilities."""

from __future__ import annotations

import importlib.util
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import pandas as pd

import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Try to use a pleasant matplotlib style for the report figures
try:  # pragma: no cover - style availability depends on matplotlib version
    plt.style.use("seaborn-v0_8-whitegrid")
except (OSError, ValueError):  # Fallback if style is missing
    plt.style.use("ggplot")


@dataclass
class WaveletReportResult:
    """Container for generated wavelet report artifacts."""

    pdf_path: Path
    metrics: pd.DataFrame
    summary: Dict[str, Union[str, float, int]]


@lru_cache(maxsize=1)
def _get_wavelet_module():
    """Load the wavelet thresholding module without importing the whole package."""

    module_path = (
        Path(__file__).resolve().parents[1]
        / "functions"
        / "preprocessing"
        / "wavelet_thresholding.py"
    )
    spec = importlib.util.spec_from_file_location(
        "autoclean.functions.preprocessing.wavelet_thresholding",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load wavelet_thresholding module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _create_summary_table(summary: Dict[str, Union[str, float, int]]) -> ReportTable:
    """Create a styled summary table for the PDF report."""

    table_data = [
        ["Channels analyzed", f"{summary['channels']}"],
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
    summary_table: ReportTable,
    channel_table: ReportTable,
    summary: Dict[str, Union[str, float, int]],
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

    img = ReportImage(figure_buffer, width=6.5 * inch, height=4.0 * inch)
    story.append(img)
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
    """Generate a PDF report comparing pre/post wavelet thresholding.

    Parameters
    ----------
    source : path-like or Raw
        Input EEG dataset. If a path is provided, it is loaded with appropriate
        MNE reader based on the extension.
    output_pdf : path-like
        Destination for the generated PDF report.
    wavelet : str
        Mother wavelet passed to ``wavelet_threshold``.
    level : int
        Requested decomposition level. The effective level is clamped per
        channel based on data length.
    picks : str or iterable of str
        Channels to include (passed to ``Raw.pick``). Defaults to EEG channels.
    snippet_duration : float
        Duration in seconds used for waveform visualisation of the top channel.
    top_n_channels : int
        Number of top channels to display in the summary table.

    Returns
    -------
    WaveletReportResult
        Dataclass containing the PDF path, metrics dataframe, and summary stats.
    """

    raw, source_path = _load_raw_object(source)
    source_name = source_path.name if source_path else getattr(raw, "filenames", ["Raw data"])[0]
    output_pdf_path = Path(output_pdf)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    raw_subset = raw.copy()
    if picks:
        raw_subset.pick(picks)

    baseline = raw_subset.get_data()
    wavelet_module = _get_wavelet_module()
    cleaned = wavelet_module.wavelet_threshold(
        raw_subset, wavelet=wavelet, level=level
    ).get_data()

    metrics = _compute_channel_metrics(baseline, cleaned, raw_subset.ch_names)

    effective_level = wavelet_module._resolve_decomposition_level(
        baseline.shape[1], wavelet, level
    )
    sfreq = float(raw_subset.info["sfreq"])
    duration = baseline.shape[1] / sfreq if sfreq else 0.0

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
    }

    figure_buffer = _build_overview_figure(
        baseline,
        cleaned,
        raw_subset.ch_names,
        sfreq,
        snippet_duration,
        metrics,
    )

    summary_table = _create_summary_table(summary)
    channel_table = _create_top_channel_table(metrics, top_n=top_n_channels)

    _build_pdf_report(
        output_pdf_path,
        source_name,
        figure_buffer,
        summary_table,
        channel_table,
        summary,
    )

    return WaveletReportResult(
        pdf_path=output_pdf_path,
        metrics=metrics,
        summary=summary,
    )
