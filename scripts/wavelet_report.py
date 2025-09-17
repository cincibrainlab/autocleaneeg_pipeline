#!/usr/bin/env python3
"""CLI for generating wavelet thresholding PDF reports."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Optional, Union


def _load_generate_wavelet_report():
    """Import generate_wavelet_report without importing the full package tree."""

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "autoclean"
        / "reporting"
        / "wavelet_report.py"
    )
    spec = importlib.util.spec_from_file_location(
        "autoclean.reporting.wavelet_report_cli",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load wavelet_report module") from None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.generate_wavelet_report


def _parse_picks(picks: Iterable[str]) -> Optional[Union[str, Iterable[str]]]:
    """Normalize channel pick arguments."""

    picks = list(picks)
    if not picks:
        return "eeg"

    if len(picks) == 1:
        entry = picks[0].lower()
        if entry == "all":
            return None
        if entry == "eeg":
            return "eeg"
    return picks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a PDF report comparing pre/post wavelet thresholding.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an EEG dataset (e.g., .set, .fif, .edf, .bdf).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination PDF report. Defaults to '<input>_wavelet_report.pdf'.",
    )
    parser.add_argument(
        "--wavelet",
        default="sym4",
        help="Mother wavelet to use (default: sym4).",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=5,
        help="Requested decomposition level (auto-clamped per channel).",
    )
    parser.add_argument(
        "--picks",
        nargs="+",
        default=["eeg"],
        help="Channels to include (default: EEG channels). Use 'all' for every channel.",
    )
    parser.add_argument(
        "--snippet-duration",
        type=float,
        default=5.0,
        help="Duration in seconds for waveform preview (default: 5.0).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top channels to include in the summary table (default: 10).",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        help="Optional path to export channel metrics as CSV.",
    )
    parser.add_argument(
        "--psd-csv",
        type=Path,
        help="Optional path to export band power metrics as CSV.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    output_pdf = (
        args.output
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_wavelet_report.pdf")
    )

    picks = _parse_picks(args.picks)

    generate_wavelet_report = _load_generate_wavelet_report()

    result = generate_wavelet_report(
        input_path,
        output_pdf,
        wavelet=args.wavelet,
        level=args.level,
        picks=picks,
        snippet_duration=args.snippet_duration,
        top_n_channels=args.top_n,
    )

    if args.metrics_csv:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        result.metrics.to_csv(args.metrics_csv, index=False)
    if args.psd_csv:
        args.psd_csv.parent.mkdir(parents=True, exist_ok=True)
        result.psd_metrics.to_csv(args.psd_csv, index=False)

    band_reductions = result.summary.get("band_reductions", {})
    alpha_change = band_reductions.get("alpha")

    print(f"Report written to: {result.pdf_path}")
    message = (
        f"Mean peak-to-peak reduction: {result.summary['ptp_mean']:.2f}%"
        f" | Max channel: {result.summary['ptp_max_channel']}"
        f" ({result.summary['ptp_max']:.2f}%)"
    )
    if alpha_change is not None:
        message += f" | Alpha band change: {alpha_change:.2f}%"
    print(message)


if __name__ == "__main__":
    main()
