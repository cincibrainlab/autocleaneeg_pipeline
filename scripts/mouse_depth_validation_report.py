#!/usr/bin/env python
"""Build a PDF that shows the mouse depth analysis blocks recover known truth.

A seeded synthetic cortex + hippocampus LFP recording
(``create_synthetic_mouse_depth_raw``) is analysed THROUGH
``Task.run_postprocessing_analysis`` (the lab code path). Every number in the
summary table is read back from the CSV/parquet files the runner wrote. A few
figure panels need data the runner does not output (comodulogram grid, phase-bin
amplitudes, surrogate histogram, windowed band-power series, filtered traces);
those call ``autoclean.calc.oscillation_analysis`` or scipy directly and say so
in their captions.

Usage:
    uv run --extra dev python scripts/mouse_depth_validation_report.py \
        [--out PATH] [--seed 42] [--duration 120]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import pprint
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from reportlab.lib import colors as rl_colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy import signal  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mne  # noqa: E402

from autoclean.calc.oscillation_analysis import (  # noqa: E402
    _analytic,
    compute_phase_amplitude_coupling,
    compute_power_power_coupling,
)
from autoclean.core.task import Task  # noqa: E402
from tests.fixtures import synthetic_data as fixture  # noqa: E402
from tests.fixtures.synthetic_data import (  # noqa: E402
    MOUSE_DEPTH_CHANNELS,
    create_synthetic_mouse_depth_raw,
)

warnings.filterwarnings("ignore")

TITLE = "AutoCleanEEG — Mouse Depth LFP Analysis Validation"
CHANNELS = list(MOUSE_DEPTH_CHANNELS)
CTX_C, HPC_C = "#2b5c8a", "#d1793a"
TRUTH_C, PASS_C, FAIL_C = "#555555", "#2e7d32", "#c62828"
GRID_C = "#e3e3e3"
PAGE_W, PAGE_H = letter
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 0.9 * inch, 0.95 * inch, 0.85 * inch
TEXT_W_IN = (PAGE_W - 2 * MARGIN_X) / inch
PSD_BANDS = {
    "delta\n1–4": (1, 4),
    "theta\n5–10": (5, 10),
    "beta\n15–25": (15, 25),
    "low gamma\n30–45": (30, 45),
    "mid gamma\n70–100": (70, 100),
    "ripple\n120–250": (120, 250),
}
PREFIX = "mouse"
ALL_BLOCKS = ("sensor_psd", "fooof", "power_coupling", "pac", "hfo", "connectivity")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID_C,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 100,
        "savefig.dpi": 200,
    }
)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class _DepthTask(Task):
    """Minimal Task that runs postprocessing blocks on an in-memory Raw."""

    def __init__(self, config):
        self.settings = {}
        super().__init__(config)

    def run(self):
        pass

    def _update_metadata(self, *args, **kwargs):
        pass


def block_settings(truth, n_surrogates=200):
    """Block configs, identical to tests/unit/core/test_mouse_depth_analysis.py."""
    regions = truth["regions"]
    pc = truth["power_coupling"]
    return {
        "sensor_psd": {
            "enabled": True,
            "input": "clean_raw",
            "fmin": 1,
            "fmax": 300,
            "n_fft": 8000,
            "freq_bands": None,
        },
        "fooof": {
            "enabled": True,
            "input": "sensor_psd",
            "freq_range": [2, 55],
            "aperiodic_mode": "fixed",
        },
        "power_coupling": {
            "enabled": True,
            "pairs": [pc["coupled"], pc["control"]],
            "window_s": 1.0,
            "regions": regions,
        },
        "pac": {
            "enabled": True,
            "phase_bands": [truth["pac"]["phase_band"]],
            "amp_bands": [truth["pac"]["amp_band"], [150, 200]],
            "n_surrogates": n_surrogates,
            "seed": 1,
            "regions": regions,
        },
        "hfo": {"enabled": True, "band": truth["hfo"]["band"], "regions": regions},
        "connectivity": {
            "enabled": True,
            "bands": {"beta": truth["connectivity"]["band"]},
            "methods": ["wpli", "wpli2_debiased"],
            "regions": regions,
        },
    }


def run_runner(raw, truth, work_dir, blocks=ALL_BLOCKS, overrides=None, epochs=None):
    """Run the selected blocks through the task and return the reports dir."""
    settings = {b: s for b, s in block_settings(truth).items() if b in blocks}
    for block, extra in (overrides or {}).items():
        settings[block].update(extra)
    reports = Path(work_dir) / "reports"
    task = _DepthTask(
        {
            "run_id": "run",
            "unprocessed_file": Path(work_dir) / f"{PREFIX}.set",
            "task": "MouseDepth",
            "reports_dir": reports,
        }
    )
    task.raw = raw
    task.epochs = epochs
    task.settings = {"postprocessing_analysis": {"enabled": True, "value": settings}}
    task.results = task.run_postprocessing_analysis()
    return reports, settings


def read_back(reports):
    """Read every table the runner wrote from disk."""
    r = Path(reports)
    files = {
        "psd": r / "sensor_psd" / f"{PREFIX}_postprocessing_sensor_psd_spectra.parquet",
        "ap": r / "fooof" / f"{PREFIX}_postprocessing_fooof_aperiodic.csv",
        "per": r / "fooof" / f"{PREFIX}_postprocessing_fooof_periodic.csv",
        "ppc": r / "power_coupling" / f"{PREFIX}_postprocessing_power_coupling.csv",
        "pac": r / "pac" / f"{PREFIX}_postprocessing_pac.csv",
        "hfo_ev": r / "hfo" / f"{PREFIX}_postprocessing_hfo_events.csv",
        "hfo_sum": r / "hfo" / f"{PREFIX}_postprocessing_hfo_summary.csv",
        "conn": r / "connectivity" / f"{PREFIX}_postprocessing_connectivity.csv",
    }
    out = {}
    for key, path in files.items():
        if path.exists():
            out[key] = (
                pd.read_parquet(path)
                if path.suffix == ".parquet"
                else pd.read_csv(path)
            )
    return out


# --------------------------------------------------------------------------
# Metrics (same definitions and thresholds as the acceptance tests)
# --------------------------------------------------------------------------

PEAK_TOL_HZ = 0.5
MIN_DURATION_S = 60.0


def psd_peak(psd, channel, lo, hi):
    d = psd[(psd["channel"] == channel) & psd["frequency"].between(lo, hi)]
    return float(d.loc[d["psd"].idxmax(), "frequency"])


def pair_row(df, a, b):
    rows = df[(df["channel_a"] == a) & (df["channel_b"] == b)]
    assert len(rows) == 1, (a, b, len(rows))
    return rows.iloc[0]


def pac_row(df, channel, amp_low):
    rows = df[(df["phase_channel"] == channel) & (df["amp_low"] == amp_low)]
    assert len(rows) == 1
    return rows.iloc[0]


def match_flags(events, truth_events, pad=0.05):
    return np.array(
        [
            any(
                e.onset_s <= t1 + pad and e.offset_s >= t0 - pad
                for t0, t1, _ in truth_events
            )
            for e in events.itertuples()
        ],
        dtype=bool,
    )


def hfo_metrics(events, truth_events, duration, channel="HPC_2"):
    ev = events[events["channel"] == channel]
    hit = match_flags(ev, truth_events)
    detected = sum(
        any(e.onset_s <= t1 + 0.05 and e.offset_s >= t0 - 0.05 for e in ev.itertuples())
        for t0, t1, _ in truth_events
    )
    sens = detected / len(truth_events) if truth_events else float("nan")
    return sens, int((~hit).sum()) / duration * 60.0, detected, int((~hit).sum())


PRIMARY = "wpli2_debiased"
CONN_BANDS = {"beta": [15, 25], "low_gamma": [30, 45], "high_gamma": [60, 90]}


def conn_of(df, band="beta", method=PRIMARY):
    return df[(df["band"] == band) & (df["method"] == method)]


def is_connected(row, want):
    return {row["channel_a"], row["channel_b"]} == want


def connected_vs_other(df, want, band, method=PRIMARY):
    """Return (connected value, max other value, max |other|) for one band."""
    d = conn_of(df, band, method)
    mask = d.apply(lambda r: is_connected(r, want), axis=1)
    return (
        float(d[mask]["value"].iloc[0]),
        float(d[~mask]["value"].max()),
    )


def transient_raw(raw, annotate):
    """Copy raw; add 10 large 20 ms transients on HPC_3, optionally BAD-annotated."""
    out = raw.copy()
    sf = out.info["sfreq"]
    ch = out.ch_names.index("HPC_3")
    duration = out.times[-1]
    times = np.linspace(0.1 * duration, 0.9 * duration, 10)
    n_t = int(round(0.02 * sf))
    win = np.hanning(n_t)
    for t in times:
        i0 = int(round(t * sf))
        out._data[ch, i0 : i0 + n_t] += (
            2e-3 * win * np.sin(2 * np.pi * 200.0 * np.arange(n_t) / sf)
        )
    if annotate:
        out.set_annotations(
            out.annotations
            + mne.Annotations(
                onset=times - 0.1,
                duration=[0.22] * len(times),
                description=["BAD_artifact"] * len(times),
            )
        )
    return out, times


def top_pair(conn):
    top = conn.sort_values("value", ascending=False).iloc[0]
    return top["channel_a"], top["channel_b"], float(top["value"])


# --------------------------------------------------------------------------
# Figure helpers
# --------------------------------------------------------------------------


def shades(base, n=4, top=0.55):
    b = np.array(mcolors.to_rgb(base))
    return [tuple(b + (1 - b) * t) for t in np.linspace(0, top, n)]


CH_COLOR = {}
for _i, _c in enumerate(shades(CTX_C)):
    CH_COLOR[f"CTX_{_i + 1}"] = _c
for _i, _c in enumerate(shades(HPC_C)):
    CH_COLOR[f"HPC_{_i + 1}"] = _c


def region_color(ch):
    return CTX_C if ch.startswith("CTX") else HPC_C


def panel(ax, letter_, dx=-0.11, dy=1.06):
    ax.text(
        dx,
        dy,
        letter_,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def new_fig(height, **kw):
    return plt.figure(figsize=(TEXT_W_IN, height), layout="constrained", **kw)


def fig_to_flowable(fig, width_in=TEXT_W_IN):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    w, h = fig.get_size_inches()
    return Image(buf, width=width_in * inch, height=width_in * inch * h / w)


def matrix(df, value="value", method=None, order=CHANNELS):
    d = df if method is None else df[df["method"] == method]
    m = np.full((len(order), len(order)), np.nan)
    idx = {c: i for i, c in enumerate(order)}
    for r in d.itertuples():
        i, j = idx[r.channel_a], idx[r.channel_b]
        m[i, j] = m[j, i] = getattr(r, value)
    return m


def draw_matrix(ax, m, cmap, vmin, vmax, fmt="{:.2f}", fs=6.5):
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(8), CHANNELS, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(8), CHANNELS, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(3.5, color="white", lw=2.5)
    ax.axvline(3.5, color="white", lw=2.5)
    ax.axhline(3.5, color="#222222", lw=0.8)
    ax.axvline(3.5, color="#222222", lw=0.8)
    norm = mcolors.Normalize(vmin, vmax)
    for i in range(8):
        for j in range(8):
            if np.isfinite(m[i, j]):
                rgb = plt.get_cmap(cmap)(norm(m[i, j]))[:3]
                lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                ax.text(
                    j,
                    i,
                    fmt.format(m[i, j]),
                    ha="center",
                    va="center",
                    fontsize=fs,
                    color="white" if lum < 0.5 else "#222222",
                )
    return im


def trim_edges(x, n):
    return x[n : x.size - n]


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_data(raw, truth):
    sf = raw.info["sfreq"]
    data = raw.get_data() * 1e6
    events = truth["hfo"]["events"]
    first = events[2][0]
    t0 = max(first - 2.0, 0.0)
    t1 = t0 + 5.0
    s0, s1 = int(t0 * sf), int(t1 * sf)
    t = np.arange(s0, s1) / sf
    step = 3.0 * data.std(axis=1).max()
    fig = new_fig(4.6)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1, 1], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    for i, ch in enumerate(CHANNELS):
        y = data[i, s0:s1]
        ax.plot(t, y - y.mean() - i * step, color=region_color(ch), lw=0.5)
        ax.text(
            t0 - 0.05,
            -i * step,
            ch,
            ha="right",
            va="center",
            fontsize=8,
            color=region_color(ch),
        )
    for ev in events:
        if t0 <= ev[0] <= t1:
            ax.axvspan(ev[0], ev[1], color=TRUTH_C, alpha=0.15, lw=0)
    ax.plot(
        [t1 + 0.06] * 2,
        [-7 * step + 0.0, -7 * step + 200],
        color="k",
        lw=2,
        clip_on=False,
    )
    ax.text(t1 + 0.1, -7 * step + 100, "200 µV", fontsize=7.5, va="center")
    ax.set_xlim(t0, t1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(False)
    ax.set_title("Eight-channel LFP, 5 s window (gray shading = truth ripple)")
    ax.set_xlabel("Time (s)")
    panel(ax, "A", dx=-0.04)

    hp = CHANNELS.index("HPC_2")
    sos = signal.butter(4, [120, 250], "bandpass", fs=sf, output="sos")
    filt = signal.sosfiltfilt(sos, data[hp])
    centre = (events[2][0] + events[2][1]) / 2
    w0, w1 = centre - 0.15, centre + 0.15
    a, b = int(w0 * sf), int(w1 * sf)
    tz = np.arange(a, b) / sf
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(tz, data[hp, a:b], color=HPC_C, lw=0.8)
    ax1.set_ylabel("raw (µV)")
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2.plot(tz, filt[a:b], color=HPC_C, lw=0.8)
    ax2.set_ylabel("filtered (µV)")
    ax2.set_xlabel("Time (s)")
    for axx in (ax1, ax2):
        axx.axvspan(events[2][0], events[2][1], color=TRUTH_C, alpha=0.15, lw=0)
    ax1.tick_params(labelbottom=False)
    ax1.set_title(
        f"Ripple zoom on HPC_2 (truth {events[2][2]:.0f} Hz, "
        f"{(events[2][1] - events[2][0]) * 1e3:.0f} ms window)"
    )
    panel(ax1, "B", dx=-0.04)
    return fig_to_flowable(fig)


def fig_psd(d, truth):
    psd = d["psd"]
    fig = new_fig(7.2)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    peaks = {p["channel"]: p["freq_hz"] for p in truth["peaks"]}
    for k, (reg, pref, title) in enumerate(
        (("CTX", "CTX_", "Cortex"), ("HPC", "HPC_", "Hippocampus"))
    ):
        ax = fig.add_subplot(gs[0, k])
        for ch in CHANNELS:
            if not ch.startswith(pref):
                continue
            s = psd[psd["channel"] == ch]
            ax.loglog(
                s["frequency"], s["psd"] * 1e12, color=CH_COLOR[ch], lw=1.1, label=ch
            )
        f0 = peaks[pref + "1"]
        ax.axvline(f0, color=TRUTH_C, ls="--", lw=0.9)
        ax.text(
            f0 / 1.08 if k == 0 else f0 * 1.08,
            0.97,
            f"truth {f0:g} Hz",
            transform=ax.get_xaxis_transform(),
            fontsize=7.5,
            color=TRUTH_C,
            va="top",
            ha="right" if k == 0 else "left",
        )
        ax.axvline(60, color="#999999", ls=":", lw=0.9)
        ax.text(
            64,
            0.97,
            "60 Hz line",
            transform=ax.get_xaxis_transform(),
            fontsize=7,
            color="#777777",
            va="top",
        )
        ax.set_xlim(1, 300)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (µV²/Hz)" if k == 0 else "")
        ax.set_title(title)
        ax.legend(loc="lower left", ncol=2, handlelength=1.2)
        panel(ax, "AB"[k])
    ax = fig.add_subplot(gs[1, :])
    f = psd["frequency"].to_numpy()
    mat = np.zeros((8, len(PSD_BANDS)))
    for i, ch in enumerate(CHANNELS):
        s = psd[psd["channel"] == ch].sort_values("frequency")
        for j, (lo, hi) in enumerate(PSD_BANDS.values()):
            m = s["frequency"].between(lo, hi)
            mat[i, j] = np.log10(np.trapz(s["psd"][m], s["frequency"][m]) * 1e12)
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(PSD_BANDS)), list(PSD_BANDS), fontsize=7.5)
    ax.set_yticks(range(8), CHANNELS)
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.axhline(3.5, color="white", lw=3)
    norm = mcolors.Normalize(mat.min(), mat.max())
    for i in range(8):
        for j in range(mat.shape[1]):
            rgb = plt.get_cmap("YlGnBu")(norm(mat[i, j]))[:3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if lum < 0.5 else "#222222",
            )
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
    cb.set_label("log₁₀ band power (µV²)")
    ax.set_title("Band power by channel and band (Hz)")
    panel(ax, "C", dx=-0.07)
    del f
    return fig_to_flowable(fig)


def specparam_model(ap_row, per_rows, freqs):
    """Aperiodic line and full model rebuilt from the fooof CSV rows."""
    logap = ap_row["offset"] - ap_row["exponent"] * np.log10(freqs)
    full = logap.copy()
    for p in per_rows.itertuples():
        sigma = p.bandwidth / 2.0
        full += p.power * np.exp(-((freqs - p.center_frequency) ** 2) / (2 * sigma**2))
    return 10**logap, 10**full


def polyfit_exponent(psd, ch, lo=2, hi=55):
    d = psd[(psd["channel"] == ch) & psd["frequency"].between(lo, hi)]
    slope = np.polyfit(np.log10(d["frequency"]), np.log10(d["psd"]), 1)[0]
    return -float(slope)


def fig_aperiodic(d, truth):
    psd, ap, per = d["psd"], d["ap"].set_index("channel"), d["per"]
    fig = new_fig(5.6)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    fr = np.linspace(2, 55, 400)
    for k, ch in enumerate(("HPC_1", "CTX_1")):
        ax = fig.add_subplot(gs[0, k])
        s = psd[(psd["channel"] == ch) & psd["frequency"].between(1, 100)]
        ax.loglog(s["frequency"], s["psd"] * 1e12, color="#9a9a9a", lw=1.0, label="PSD")
        line, full = specparam_model(ap.loc[ch], per[per["channel"] == ch], fr)
        ax.loglog(
            fr,
            line * 1e12,
            color=region_color(ch),
            lw=1.8,
            label=f"aperiodic fit (exp {ap.loc[ch, 'exponent']:.2f})",
        )
        ax.loglog(fr, full * 1e12, color="#222222", lw=0.9, ls="-", label="full model")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (µV²/Hz)" if k == 0 else "")
        ax.set_title(f"{ch}  (truth exponent {truth['aperiodic']['exponent'][ch]:g})")
        ax.legend(loc="lower left")
        panel(ax, "AB"[k])
    tru = np.array([truth["aperiodic"]["exponent"][c] for c in CHANNELS])
    rec = np.array([ap.loc[c, "exponent"] for c in CHANNELS])
    pol = np.array([polyfit_exponent(psd, c) for c in CHANNELS])
    ax = fig.add_subplot(gs[1, 0])
    lim = (1.5, 2.6)
    ax.fill_between(
        lim,
        [lim[0] - 0.2, lim[1] - 0.2],
        [lim[0] + 0.2, lim[1] + 0.2],
        color="#dddddd",
        alpha=0.7,
        lw=0,
        label="±0.2",
    )
    ax.plot(lim, lim, color=TRUTH_C, ls="--", lw=0.9, label="identity")
    for i, ch in enumerate(CHANNELS):
        ax.scatter(
            tru[i],
            rec[i],
            s=34,
            color=region_color(ch),
            zorder=3,
            label="specparam" if i == 0 else None,
            edgecolor="white",
            lw=0.5,
        )
        ax.scatter(
            tru[i],
            pol[i],
            s=28,
            marker="v",
            facecolor="none",
            edgecolor=region_color(ch),
            lw=1.0,
            zorder=3,
            label="polyfit (old)" if i == 0 else None,
        )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True exponent")
    ax.set_ylabel("Recovered exponent")
    ax.set_title("Recovery, all 8 channels")
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[2:] + h[:2], lab[2:] + lab[:2], loc="upper left")
    panel(ax, "C")
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(8)
    ax.bar(
        x - 0.2,
        rec - tru,
        0.38,
        color=[region_color(c) for c in CHANNELS],
        label="specparam",
    )
    ax.bar(
        x + 0.2,
        pol - tru,
        0.38,
        color="white",
        edgecolor=[region_color(c) for c in CHANNELS],
        hatch="////",
        lw=0.8,
        label="polyfit (old)",
    )
    ax.axhline(0, color="#333333", lw=0.8)
    ax.axhspan(-0.2, 0.2, color="#dddddd", alpha=0.7, lw=0)
    ax.set_xticks(x, CHANNELS, rotation=60, ha="right")
    ax.set_ylabel("Recovered − true exponent")
    ax.set_title("Bias per channel")
    ax.set_ylim(-0.3, 0.38)
    ax.legend(loc="upper left", ncol=2)
    panel(ax, "D")
    return fig_to_flowable(fig), rec, pol


def windowed_log_power(x, band, sf, win_s=1.0):
    env = np.abs(_analytic(x, band, sf)) ** 2
    edge = int(round(min(1.0, 0.1 * env.size / sf) * sf))  # same trim as the runner
    env = env[edge : env.size - edge]
    w = int(round(win_s * sf))
    n = env.size // w
    return np.log10(env[: n * w].reshape(n, w).mean(axis=1))


def fig_power(raw, truth, d):
    sf = raw.info["sfreq"]
    pc = truth["power_coupling"]
    ppc = d["ppc"]
    mats = compute_power_power_coupling(
        raw, bands={"gamma": pc["coupled"]["band_a"]}, window_s=1.0
    )
    mr = matrix(mats, "pearson_r")
    fig = new_fig(6.5)
    gs = fig.add_gridspec(
        3, 3, height_ratios=[1.35, 0.8, 0.8], width_ratios=[1, 1, 1.3]
    )
    diffs = []
    series = {}
    for k, (key, name) in enumerate(
        (("coupled", "Coupled pair"), ("control", "Control pair"))
    ):
        p = pc[key]
        row = pair_row(ppc, p["channel_a"], p["channel_b"])
        a = windowed_log_power(raw.get_data(p["channel_a"])[0], p["band_a"], sf)
        b = windowed_log_power(raw.get_data(p["channel_b"])[0], p["band_b"], sf)
        diffs.append(abs(np.corrcoef(a, b)[0, 1] - row["pearson_r"]))
        series[key] = (a, b)
        ax = fig.add_subplot(gs[0, k])
        col = PASS_C if key == "coupled" else "#777777"
        ax.scatter(a, b, s=11, color=col, alpha=0.7, edgecolor="none")
        m, c0 = np.polyfit(a, b, 1)
        xs = np.array([a.min(), a.max()])
        ax.plot(xs, m * xs + c0, color="#222222", lw=1.0)
        ax.set_xlabel(f"{p['channel_a']} log₁₀ power (V²)")
        ax.set_ylabel(f"{p['channel_b']} log₁₀ power" if k == 0 else "")
        ptxt = "p < 0.001" if row["p_value"] < 1e-3 else f"p = {row['p_value']:.2f}"
        ax.set_title(f"{name}")
        ax.text(
            0.04 if key == "coupled" else 0.96,
            0.95,
            f"r = {row['pearson_r']:.2f}\n{ptxt}\nn = {int(row['n_windows'])}",
            transform=ax.transAxes,
            va="top",
            ha="left" if key == "coupled" else "right",
            fontsize=8,
        )
        panel(ax, "AB"[k], dx=-0.14)
    ax = fig.add_subplot(gs[0, 2])
    im = draw_matrix(ax, mr, "RdBu_r", -0.8, 0.8, fmt="{:.1f}", fs=5.5)
    ax.set_title("r, 30–45 Hz")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    panel(ax, "C", dx=-0.2)
    for k, (key, name) in enumerate((("coupled", "Coupled"), ("control", "Control"))):
        p = pc[key]
        a, b = series[key]
        ax = fig.add_subplot(gs[1 + k, :])
        t = np.arange(a.size) + 0.5
        za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
        ax.plot(t, za, color=CTX_C, lw=1.0, label=p["channel_a"])
        ax.plot(t, zb, color=HPC_C, lw=1.0, label=p["channel_b"])
        ax.set_xlim(0, a.size)
        ax.set_ylabel("z-scored\nlog power")
        ax.set_title(f"{name} 30–45 Hz power, 1 s windows", fontsize=9)
        ax.legend(loc="upper right", ncol=2)
        if k == 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.tick_params(labelbottom=False)
        panel(ax, "DE"[k], dx=-0.045)
    odd = [
        f"{CHANNELS[i]}\u2013{CHANNELS[j]} ({mr[i, j]:+.2f})"
        for i in range(8)
        for j in range(i + 1, 8)
        if abs(mr[i, j]) > 0.25
        and {CHANNELS[i], CHANNELS[j]}
        != {pc["coupled"]["channel_a"], pc["coupled"]["channel_b"]}
    ]
    coupled = {pc["coupled"]["channel_a"], pc["coupled"]["channel_b"]}
    others = [
        (abs(mr[i, j]), mr[i, j], CHANNELS[i], CHANNELS[j])
        for i in range(8)
        for j in range(i + 1, 8)
        if {CHANNELS[i], CHANNELS[j]} != coupled
    ]
    ci, cj = (CHANNELS.index(c) for c in sorted(coupled))
    coupled_r = float(mr[ci, cj])
    runner_up = max(others, key=lambda o: o[1])
    stats = {
        "coupled_r": coupled_r,
        "runner_up_r": float(runner_up[1]),
        "runner_up": f"{runner_up[2]}\u2013{runner_up[3]}",
        "margin": coupled_r - float(runner_up[1]),
        "max_abs_other": float(max(o[0] for o in others)),
    }
    return fig_to_flowable(fig), max(diffs), odd, stats


def pac_phase_bins(raw, ch, pb, ab, n_bins=18):
    sf = raw.info["sfreq"]
    x = raw.get_data(ch)[0]
    edge = int(round(min(1.0, 0.1 * x.size / sf) * sf))
    ph = trim_edges(np.angle(_analytic(x, tuple(pb), sf)), edge)
    amp = trim_edges(np.abs(_analytic(x, tuple(ab), sf)), edge)
    idx = np.minimum(((ph + np.pi) / (2 * np.pi) * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    mean = np.bincount(idx, weights=amp, minlength=n_bins) / counts
    p = mean / mean.sum()
    mi = (np.log(n_bins) + np.sum(p * np.log(p))) / np.log(n_bins)
    return mean, mi, idx, amp


def comodulogram(raw, channels):
    phase_c = np.arange(2, 15, 1.0)
    amp_c = np.arange(20, 201, 10.0)
    # The function raises when the amplitude band is narrower than 2 x the
    # phase high edge, so use one fixed width of 2 x the largest phase high edge.
    amp_width = 2.0 * (phase_c[-1] + 1)
    grids = {c: np.full((len(amp_c), len(phase_c)), np.nan) for c in channels}
    pairs = [(c, c) for c in channels]
    for j, ac in enumerate(amp_c):
        ab = [ac - amp_width / 2, ac + amp_width / 2]
        valid = [i for i, pc_ in enumerate(phase_c) if ab[0] > pc_ + 1]
        if not valid:
            continue
        df = compute_phase_amplitude_coupling(
            raw,
            phase_bands=[[phase_c[i] - 1, phase_c[i] + 1] for i in valid],
            amp_bands=[ab],
            channel_pairs=pairs,
            n_surrogates=0,
        )
        for r in df.itertuples():
            i = int(round((r.phase_low + r.phase_high) / 2 - phase_c[0]))
            grids[r.phase_channel][j, i] = r.mi
    return phase_c, amp_c, grids


def fig_pac(raw, truth, d):
    sf = raw.info["sfreq"]
    pcfg = truth["pac"]
    hit, ctl = pcfg["channel"], pcfg["control_channel"]
    phase_c, amp_c, grids = comodulogram(raw, [hit, ctl])
    vmax = np.nanmax(grids[hit])
    fig = new_fig(5.8)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    axes = []
    mappable = None
    for k, ch in enumerate((hit, ctl)):
        ax = fig.add_subplot(gs[0, k])
        mappable = ax.pcolormesh(
            phase_c,
            amp_c,
            np.ma.masked_invalid(grids[ch]),
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            shading="nearest",
        )
        ax.add_patch(
            Rectangle(
                (pcfg["phase_band"][0], pcfg["amp_band"][0]),
                pcfg["phase_band"][1] - pcfg["phase_band"][0],
                pcfg["amp_band"][1] - pcfg["amp_band"][0],
                fill=False,
                ec="white",
                ls="--",
                lw=1.2,
            )
        )
        ax.grid(False)
        ax.set_xlabel("Phase frequency (Hz)")
        ax.set_ylabel("Amplitude frequency (Hz)" if k == 0 else "")
        ax.set_title(f"{ch}" + ("  (coupled)" if k == 0 else "  (control)"))
        panel(ax, "AB"[k], dx=-0.13)
        axes.append(ax)
    cb = fig.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("Modulation index (shared scale)")
    axes[0].text(
        0.03,
        0.04,
        "dashed box = truth",
        transform=axes[0].transAxes,
        color="white",
        fontsize=7.5,
    )
    row = pac_row(d["pac"], hit, pcfg["amp_band"][0])
    crow = pac_row(d["pac"], ctl, pcfg["amp_band"][0])
    # C: phase-bin amplitude
    ax = fig.add_subplot(gs[1, 0])
    mean_h, mi_h, idx, amp_h = pac_phase_bins(
        raw, hit, pcfg["phase_band"], pcfg["amp_band"]
    )
    mean_c, mi_c, _, _ = pac_phase_bins(raw, ctl, pcfg["phase_band"], pcfg["amp_band"])
    centers = (np.arange(18) + 0.5) * 20 - 180
    xs = np.concatenate([centers, centers + 360])
    ax.bar(xs, np.tile(mean_h, 2) * 1e6, width=18, color=HPC_C, label=f"{hit}")
    ax.plot(
        xs,
        np.tile(mean_c, 2) * 1e6,
        color="#555555",
        lw=1.2,
        marker="o",
        ms=2.5,
        label=f"{ctl} (control)",
    )
    ax.set_xlabel("Theta phase (°), two cycles")
    ax.set_ylabel("Mean 70–100 Hz\namplitude (µV)")
    ax.set_xticks([-180, 0, 180, 360, 540])
    ax.set_xlim(-190, 550)
    ax.set_title("Amplitude by phase bin (18 bins)")
    ax.set_ylim(0, max(mean_h.max(), mean_c.max()) * 1e6 * 1.3)
    ax.legend(loc="upper center", ncol=2, fontsize=7)
    panel(ax, "C", dx=-0.13)
    # D: surrogates
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(1)
    shift = int(round(1.0 * sf))
    shifts = rng.integers(shift, amp_h.size - shift + 1, size=1000)
    counts = np.bincount(idx, minlength=18)

    def mi_of(a):
        m = np.bincount(idx, weights=a, minlength=18) / counts
        p = m / m.sum()
        return (np.log(18) + np.sum(p * np.log(p))) / np.log(18)

    surr = np.array([mi_of(np.roll(amp_h, int(s))) for s in shifts])
    bins = np.logspace(np.log10(surr.min() * 0.8), np.log10(row["mi"] * 2), 50)
    ax.hist(surr, bins=bins, color="#aaaaaa")
    ax.axvline(row["mi"], color=HPC_C, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Modulation index (log scale)")
    ax.set_ylabel("Surrogates")
    ax.set_title("Shifted-amplitude surrogates")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)
    ax.text(
        0.04,
        0.96,
        f"observed MI = {row['mi']:.4f}\nrunner z = {row['z_score']:.0f}\n"
        f"runner p = {row['p_value']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color=HPC_C,
    )
    panel(ax, "D")
    diff = abs(mi_h - row["mi"]) / row["mi"]
    return fig_to_flowable(fig), diff, (row, crow)


def fig_hfo(raw, truth, d):
    sf = raw.info["sfreq"]
    ev, summ = d["hfo_ev"], d["hfo_sum"].set_index("channel")
    tev = truth["hfo"]["events"]
    dur = truth["duration"]
    fig = new_fig(5.9)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, :])
    rows = ["truth"] + CHANNELS
    ypos = {r: len(rows) - 1 - i for i, r in enumerate(rows)}
    ax.vlines(
        [(a + b) / 2 for a, b, _ in tev],
        ypos["truth"] - 0.38,
        ypos["truth"] + 0.38,
        color=TRUTH_C,
        lw=1.0,
    )
    n_fp = 0
    for ch in CHANNELS:
        e = ev[ev["channel"] == ch]
        if ch == "HPC_2":
            hit = match_flags(e, tev)
        else:
            hit = np.zeros(len(e), dtype=bool)
        mid = ((e["onset_s"] + e["offset_s"]) / 2).to_numpy()
        ax.vlines(mid[hit], ypos[ch] - 0.38, ypos[ch] + 0.38, color=HPC_C, lw=1.0)
        ax.vlines(mid[~hit], ypos[ch] - 0.38, ypos[ch] + 0.38, color=FAIL_C, lw=1.6)
        n_fp += int((~hit).sum())
    ax.set_yticks([ypos[r] for r in rows], ["Truth (HPC_2)"] + CHANNELS)
    for lab in ax.get_yticklabels()[1:]:
        lab.set_color(region_color(lab.get_text()))
    ax.set_xlim(0, dur)
    ax.set_ylim(-0.7, len(rows) + 0.4)
    ax.set_xlabel("Time (s)")
    ax.set_title("Ripple truth vs detections on all channels")
    ax.plot([], [], color=HPC_C, lw=1.0, label="detection matching truth")
    ax.plot([], [], color=FAIL_C, lw=1.6, label="false positive")
    ax.legend(loc="upper right", ncol=2)
    panel(ax, "A", dx=-0.07)
    # B: one detected ripple
    e2 = ev[ev["channel"] == "HPC_2"].copy()
    e2["hit"] = match_flags(e2, tev)
    tp = e2[e2["hit"]]
    pick = tp.iloc[
        (tp["peak_amplitude"] - tp["peak_amplitude"].median()).abs().argmin()
    ]
    hp = raw.get_data("HPC_2")[0] * 1e6
    sos = signal.butter(4, [120, 250], "bandpass", fs=sf, output="sos")
    filt = signal.sosfiltfilt(sos, hp)
    k = max(1, int(round(0.003 * sf)))
    rms = np.sqrt(np.convolve(filt**2, np.ones(k) / k, mode="same"))
    thr = summ.loc["HPC_2", "threshold"] * 1e6
    c = (pick["onset_s"] + pick["offset_s"]) / 2
    a, b = int((c - 0.12) * sf), int((c + 0.12) * sf)
    tz = np.arange(a, b) / sf
    gsb = gs[1, 0].subgridspec(3, 1, hspace=0.12)
    labels = ["raw (µV)", "120–250 Hz (µV)", "RMS (µV)"]
    traces = [hp[a:b], filt[a:b], rms[a:b]]
    tm = [t for t in tev if t[0] <= c + 0.1 and t[1] >= c - 0.1]
    for i in range(3):
        axb = fig.add_subplot(gsb[i])
        axb.plot(tz, traces[i], color=HPC_C if i < 2 else "#333333", lw=0.8)
        axb.axvspan(pick["onset_s"], pick["offset_s"], color=PASS_C, alpha=0.18, lw=0)
        for t0, t1, _ in tm:
            axb.axvline(t0, color=TRUTH_C, ls="--", lw=0.8)
            axb.axvline(t1, color=TRUTH_C, ls="--", lw=0.8)
        axb.set_ylabel(labels[i], fontsize=7.5)
        axb.set_xlim(tz[0], tz[-1])
        if i == 2:
            axb.axhline(thr, color=FAIL_C, lw=1.0)
            axb.text(tz[0], thr, " threshold", color=FAIL_C, fontsize=7, va="bottom")
            axb.set_xlabel("Time (s)")
        else:
            axb.tick_params(labelbottom=False)
        if i == 0:
            axb.set_title("One detected ripple", fontsize=9)
            panel(axb, "B", dx=-0.13)
    # D rates
    ax = fig.add_subplot(gs[1, 1])
    rates = [summ.loc[ch, "rate_per_min"] for ch in CHANNELS]
    ax.bar(range(8), rates, color=[region_color(ch) for ch in CHANNELS])
    truth_rate = len(tev) / dur * 60
    for i, r in enumerate(rates):
        if r <= 0:
            continue
        if r > 5:
            ax.text(
                i, r - 0.8, f"{r:.1f}", ha="center", va="top", fontsize=7, color="white"
            )
        else:
            ax.text(i, r + 0.4, f"{r:.1f}", ha="center", fontsize=7)
    ax.axhline(
        truth_rate, color=TRUTH_C, ls="--", lw=0.9, label=f"truth {truth_rate:.1f}/min"
    )
    ax.axhline(1.0, color=FAIL_C, ls=":", lw=1.0, label="FP limit 1/min")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xticks(range(8), CHANNELS, rotation=60, ha="right")
    ax.set_ylabel("Detections per minute")
    ax.set_ylim(0, max(rates + [truth_rate]) * 1.3)
    ax.set_title("Rate per channel")
    panel(ax, "C", dx=-0.17)
    return fig_to_flowable(fig), n_fp


def fig_connectivity(d, d_coh, d_band, want):
    wp = conn_of(d["conn"])
    coh = d_coh["conn"]
    coh = coh[coh["method"] == "coh"]
    fig = new_fig(5.9)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    im = draw_matrix(ax, matrix(wp), "YlGnBu", 0, 1)
    ax.set_title("wPLI² debiased, 15–25 Hz")
    panel(ax, "A", dx=-0.2)
    ax = fig.add_subplot(gs[0, 1])
    im = draw_matrix(ax, matrix(coh), "YlGnBu", 0, 1)
    ax.set_title("Coherence, 15–25 Hz")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    panel(ax, "B", dx=-0.13)
    ax = fig.add_subplot(gs[1, 0])
    top = wp.sort_values("value", ascending=False).head(8).iloc[::-1]
    cohv = {(r.channel_a, r.channel_b): r.value for r in coh.itertuples()}

    def kind(r):
        if r.region_a != r.region_b:
            return "#7b6fa8"
        return CTX_C if r.region_a == "cortex" else HPC_C

    cols = [kind(r) for r in top.itertuples()]
    y = np.arange(len(top))
    ax.barh(y, top["value"], color=cols, height=0.62)
    ax.scatter(
        [cohv[(r.channel_a, r.channel_b)] for r in top.itertuples()],
        y,
        marker="D",
        s=22,
        color="white",
        edgecolor="#222222",
        zorder=3,
        label="coherence",
    )
    ax.set_yticks(y, [f"{r.channel_a} – {r.channel_b}" for r in top.itertuples()])
    ax.set_xlabel("wPLI² debiased (bars) / coherence (◇)")
    ax.set_xlim(0, 1)
    ax.set_title("Eight highest pairs")
    ax.legend(loc="lower right")
    panel(ax, "C", dx=-0.3)
    # D: band specificity
    ax = fig.add_subplot(gs[1, 1])
    bands = list(CONN_BANDS)
    vals = [connected_vs_other(d_band["conn"], want, b) for b in bands]
    x = np.arange(len(bands))
    ax.bar(x - 0.2, [v[0] for v in vals], 0.38, color="#7b6fa8", label="CTX_2–HPC_1")
    ax.bar(
        x + 0.2, [v[1] for v in vals], 0.38, color="#c0c6ce", label="highest other pair"
    )
    for i, v in enumerate(vals):
        ax.text(i - 0.2, v[0] + 0.02, f"{v[0]:.2f}", ha="center", fontsize=7.5)
        ax.text(i + 0.2, v[1] + 0.02, f"{v[1]:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(
        x,
        [
            f"{b.replace('_', ' ')}\n{CONN_BANDS[b][0]}–{CONN_BANDS[b][1]}"
            for b in bands
        ],
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("wPLI² debiased")
    ax.set_title("Band specificity")
    ax.legend(loc="upper right", fontsize=7)
    panel(ax, "D", dx=-0.17)
    return fig_to_flowable(fig), coh


def fig_controls(ctrl):
    fig = new_fig(4.9)
    gs = fig.add_gridspec(2, 3)
    for k, c in enumerate(ctrl):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        vals = [c["on"], c["off"]]
        ax.bar([0, 1], vals, color=["#44546a", "#c0c6ce"], width=0.6)
        top = max(max(vals), c["line"])
        for i, v in enumerate(vals):
            ax.text(
                i,
                v + top * 0.03,
                c["fmt"].format(v),
                ha="center",
                fontsize=8,
                bbox={"fc": "white", "ec": "none", "pad": 0.8},
            )
        ax.axhline(c["line"], color=FAIL_C, ls="--", lw=1.0, label=c["line_label"])
        ax.set_xticks([0, 1], ["effect on", "effect off"])
        ax.set_ylim(min(0, min(vals) * 1.2), top * 1.25)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(c["title"])
        ax.set_ylabel(c["ylabel"])
        ax.legend(loc="upper right", fontsize=7)
        panel(ax, "ABCDE"[k], dx=-0.2, dy=1.06)
    return fig_to_flowable(fig)


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

BODY = ParagraphStyle(
    "body",
    fontName="DejaVu",
    fontSize=8.6,
    leading=12.2,
    alignment=TA_LEFT,
    textColor=rl_colors.HexColor("#222222"),
    spaceAfter=5,
)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=10.2, spaceAfter=3)
CAPTION = ParagraphStyle(
    "caption",
    parent=BODY,
    fontSize=7.7,
    leading=10.4,
    textColor=rl_colors.HexColor("#444444"),
    spaceBefore=3,
    spaceAfter=8,
)
H1 = ParagraphStyle(
    "h1",
    parent=BODY,
    fontName="DejaVu-Bold",
    fontSize=13.5,
    leading=17,
    spaceBefore=2,
    spaceAfter=6,
    textColor=rl_colors.HexColor("#1f2d3d"),
)
H2 = ParagraphStyle(
    "h2",
    parent=BODY,
    fontName="DejaVu-Bold",
    fontSize=9.5,
    leading=13,
    spaceBefore=6,
    spaceAfter=3,
    textColor=rl_colors.HexColor("#1f2d3d"),
)
TITLE_S = ParagraphStyle(
    "title",
    parent=BODY,
    fontName="DejaVu-Bold",
    fontSize=20,
    leading=25,
    spaceAfter=3,
    textColor=rl_colors.HexColor("#1f2d3d"),
)
SUB = ParagraphStyle(
    "sub",
    parent=BODY,
    fontSize=9.5,
    leading=13,
    textColor=rl_colors.HexColor("#555555"),
    spaceAfter=8,
)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.6, leading=9.8, spaceAfter=0)
CELL_B = ParagraphStyle("cellb", parent=CELL, fontName="DejaVu-Bold")
MONO = ParagraphStyle(
    "mono",
    fontName="DejaVu-Mono",
    fontSize=6.3,
    leading=7.8,
    textColor=rl_colors.HexColor("#222222"),
)


def register_fonts():
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(
        TTFont("DejaVu-Oblique", str(base / "DejaVuSans-Oblique.ttf"))
    )
    pdfmetrics.registerFont(
        TTFont("DejaVu-BoldOblique", str(base / "DejaVuSans-BoldOblique.ttf"))
    )
    pdfmetrics.registerFont(TTFont("DejaVu-Mono", str(base / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily(
        "DejaVu",
        normal="DejaVu",
        bold="DejaVu-Bold",
        italic="DejaVu-Oblique",
        boldItalic="DejaVu-BoldOblique",
    )


class NumberedCanvas(rl_canvas.Canvas):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total):
        self.setFont("DejaVu", 7.5)
        self.setFillColor(rl_colors.HexColor("#666666"))
        y = PAGE_H - 0.55 * inch
        self.drawString(MARGIN_X, y, TITLE)
        self.drawRightString(
            PAGE_W - MARGIN_X, y, f"Page {self._pageNumber} of {total}"
        )
        self.setStrokeColor(rl_colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(MARGIN_X, y - 5, PAGE_W - MARGIN_X, y - 5)
        self.line(MARGIN_X, 0.55 * inch, PAGE_W - MARGIN_X, 0.55 * inch)
        self.drawString(
            MARGIN_X, 0.4 * inch, "Synthetic data, seeded; values read from runner CSVs"
        )
        self.drawRightString(
            PAGE_W - MARGIN_X, 0.4 * inch, f"Page {self._pageNumber} of {total}"
        )


def P(text, style=BODY):
    return Paragraph(text, style)


def make_table(rows, widths, header=True, extra=None, font=CELL):
    data = [
        [c if not isinstance(c, str) else Paragraph(c, font) for c in r] for r in rows
    ]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#d5d5d5")),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#eef1f5")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, rl_colors.HexColor("#555555")),
        ]
    t.setStyle(TableStyle(style + (extra or [])))
    return t


fig_counter = [0]


def figure_block(flow, caption):
    fig_counter[0] += 1
    return KeepTogether(
        [flow, P(f"<b>Figure {fig_counter[0]}.</b> {caption}", CAPTION)]
    )


def git_info():
    def run(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"

    dirty = bool(run("status", "--porcelain", "--", "src", "tests"))
    return (
        run("rev-parse", "--short", "HEAD"),
        run("rev-parse", "--abbrev-ref", "HEAD"),
        dirty,
    )


def result_cell(ok):
    return "PASS" if ok else "FAIL"


def build(out_path, seed, duration):
    t_start = time.time()
    register_fonts()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Simulating mouse depth recording (seed={seed}, duration={duration:g} s)")
    raw, truth = create_synthetic_mouse_depth_raw(duration=duration, seed=seed)
    tmp = Path(tempfile.mkdtemp(prefix="mdr_"))

    print("Running all blocks through Task.run_postprocessing_analysis")
    reports, settings = run_runner(raw, truth, tmp / "full")
    d = read_back(reports)
    print("Running connectivity with coherence for comparison")
    rep_c, _ = run_runner(
        raw,
        truth,
        tmp / "coh",
        blocks=("connectivity",),
        overrides={"connectivity": {"methods": ["wpli2_debiased", "coh"]}},
    )
    d_coh = read_back(rep_c)
    wp_same = np.allclose(
        conn_of(d_coh["conn"])["value"].to_numpy(),
        conn_of(d["conn"])["value"].to_numpy(),
    )

    print("Running connectivity in beta, low gamma and high gamma")
    rep_b, _ = run_runner(
        raw,
        truth,
        tmp / "bands",
        blocks=("connectivity",),
        overrides={"connectivity": {"bands": CONN_BANDS}},
    )
    d_band = read_back(rep_b)

    print("Running HFO with BAD-annotated and unannotated transients")
    marked, t_times = transient_raw(raw, annotate=True)
    rep_m, _ = run_runner(marked, truth, tmp / "bad", blocks=("hfo",))
    d_bad = read_back(rep_m)
    unmarked, _ = transient_raw(raw, annotate=False)
    rep_u, _ = run_runner(unmarked, truth, tmp / "unbad", blocks=("hfo",))
    d_unbad = read_back(rep_u)

    print("Running pac, hfo and connectivity on 2 s Epochs (clean_epochs)")
    epochs = mne.make_fixed_length_epochs(
        raw, duration=2.0, preload=True, verbose=False
    )
    ep_blocks = ("pac", "hfo", "connectivity")
    rep_e, _ = run_runner(
        raw,
        truth,
        tmp / "epochs",
        blocks=ep_blocks,
        overrides={b: {"input": "clean_epochs"} for b in ep_blocks},
        epochs=epochs,
    )
    d_ep = read_back(rep_e)

    print("Running connectivity with the common reference removed")
    r_nr, t_nr = create_synthetic_mouse_depth_raw(
        duration=duration, seed=seed, effects={"common_reference": False}
    )
    rep_nr, _ = run_runner(
        r_nr,
        t_nr,
        tmp / "noref",
        blocks=("connectivity",),
        overrides={"connectivity": {"methods": ["wpli2_debiased", "coh"]}},
    )
    d_nr = read_back(rep_nr)

    # ---- measurements from read-back CSVs ----
    psd, ap = d["psd"], d["ap"].set_index("channel")
    theta = psd_peak(psd, "HPC_1", 5, 10)
    beta = psd_peak(psd, "CTX_1", 15, 25)
    t_theta = next(p for p in truth["peaks"] if p["channel"] == "HPC_1")["freq_hz"]
    t_beta = next(p for p in truth["peaks"] if p["channel"] == "CTX_1")["freq_hz"]
    all_ok = bool((d["ap"]["status"] == "SUCCESS").all())
    err = {
        c: abs(ap.loc[c, "exponent"] - e)
        for c, e in truth["aperiodic"]["exponent"].items()
    }
    worst = max(err, key=err.get)
    poly_err = abs(
        polyfit_exponent(psd, "HPC_1") - truth["aperiodic"]["exponent"]["HPC_1"]
    )
    spec_err = err["HPC_1"]
    pc = truth["power_coupling"]
    c_row = pair_row(d["ppc"], pc["coupled"]["channel_a"], pc["coupled"]["channel_b"])
    k_row = pair_row(d["ppc"], pc["control"]["channel_a"], pc["control"]["channel_b"])
    pcfg = truth["pac"]
    pac_hit = pac_row(d["pac"], pcfg["channel"], 70.0)
    pac_ctl = pac_row(d["pac"], pcfg["control_channel"], 70.0)
    pac_high = pac_row(d["pac"], pcfg["channel"], 150.0)
    sens, fp, n_det, n_fp = hfo_metrics(d["hfo_ev"], truth["hfo"]["events"], duration)
    others = d["hfo_sum"][d["hfo_sum"]["channel"] != "HPC_2"]
    max_other = float(others["rate_per_min"].max())
    conn_all = d["conn"]
    conn = conn_of(conn_all)
    n_epochs = int(conn["n_epochs"].iloc[0])
    epoch_len = float(conn["epoch_length_s"].iloc[0])
    ta, tb, tv = top_pair(conn)
    want = set(truth["connectivity"]["connected"])
    conn_row = pair_row(conn, *sorted(want))
    unconn = conn[~conn.apply(lambda r: {r.channel_a, r.channel_b} == want, axis=1)]
    max_unconn = float(unconn["value"].max())
    wpli_top = top_pair(conn_of(conn_all, "beta", "wpli"))
    band_vals = {b: connected_vs_other(d_band["conn"], want, b) for b in CONN_BANDS}
    # BAD annotations
    bad_s = 0.22 * len(t_times)
    ev_bad = d_bad["hfo_ev"]
    ev3 = ev_bad[ev_bad["channel"] == "HPC_3"]
    n_overlap_bad = int(
        sum(
            bool(((ev3["onset_s"] <= t + 0.1) & (ev3["offset_s"] >= t - 0.1)).any())
            for t in t_times
        )
    )
    sens_bad = hfo_metrics(ev_bad, truth["hfo"]["events"], duration)[0]
    dur3 = float(
        d_bad["hfo_sum"].set_index("channel").loc["HPC_3", "analyzed_duration_s"]
    )
    dur_drop = duration - dur3
    ev3u = d_unbad["hfo_ev"]
    ev3u = ev3u[ev3u["channel"] == "HPC_3"]
    n_hit_unbad = int(
        sum(
            bool(((ev3u["onset_s"] <= t + 0.03) & (ev3u["offset_s"] >= t - 0.01)).any())
            for t in t_times
        )
    )
    # Epochs path
    ep_pac = pac_row(d_ep["pac"], pcfg["channel"], 70.0)
    ep_sens, _, ep_det, _ = hfo_metrics(
        d_ep["hfo_ev"], truth["hfo"]["events"], duration
    )
    ep_top = top_pair(conn_of(d_ep["conn"]))
    # PAC preferred phase (truth 0 rad: amplitude = 1 + depth * cos(theta phase))
    pref = float(pac_hit["preferred_phase_rad"])
    pref_err = float(abs(np.angle(np.exp(1j * pref))))

    # ---- effect-removal controls (re-run the runner on regenerated data) ----
    print("Running effect-removal controls")
    ctl = {}
    for eff, blocks in (
        ("peaks", ("sensor_psd", "fooof")),
        ("power_coupling", ("power_coupling",)),
        ("pac", ("pac",)),
        ("hfo", ("hfo",)),
        ("connectivity", ("connectivity",)),
    ):
        r_off, t_off = create_synthetic_mouse_depth_raw(
            duration=duration, seed=seed, effects={eff: False}
        )
        rep, _ = run_runner(r_off, t_off, tmp / f"off_{eff}", blocks=blocks)
        ctl[eff] = (read_back(rep), t_off)
    per_on = d["per"]
    n_theta_on = int(
        (
            (per_on["channel"] == "HPC_1") & per_on["center_frequency"].between(6, 9)
        ).sum()
    )
    per_off = ctl["peaks"][0]["per"]
    n_theta_off = int(
        (
            (per_off["channel"] == "HPC_1") & per_off["center_frequency"].between(6, 9)
        ).sum()
    )
    cd = ctl["power_coupling"][0]["ppc"]
    off_ppc = float(
        pair_row(cd, pc["coupled"]["channel_a"], pc["coupled"]["channel_b"])[
            "pearson_r"
        ]
    )
    off_pac = float(pac_row(ctl["pac"][0]["pac"], pcfg["channel"], 70.0)["z_score"])
    off_hfo_n = int((ctl["hfo"][0]["hfo_ev"]["channel"] == "HPC_2").sum())
    off_hfo_rate = off_hfo_n / duration * 60.0
    cdf = conn_of(ctl["connectivity"][0]["conn"])
    off_top = top_pair(cdf)
    off_conn = float(pair_row(cdf, *sorted(want))["value"])

    checks = [
        (
            "PSD",
            f"HPC_1 theta peak within {PEAK_TOL_HZ} Hz of truth {t_theta:g} Hz",
            f"|error| ≤ {PEAK_TOL_HZ} Hz",
            f"{theta:.2f} Hz (err {abs(theta - t_theta):.2f})",
            abs(theta - t_theta) <= PEAK_TOL_HZ,
        ),
        (
            "PSD",
            f"CTX_1 beta peak within {PEAK_TOL_HZ} Hz of truth {t_beta:g} Hz",
            f"|error| ≤ {PEAK_TOL_HZ} Hz",
            f"{beta:.2f} Hz (err {abs(beta - t_beta):.2f})",
            abs(beta - t_beta) <= PEAK_TOL_HZ,
        ),
        (
            "Aperiodic",
            "specparam fit succeeded on all 8 channels",
            "all SUCCESS",
            "8/8 SUCCESS" if all_ok else "failures",
            all_ok,
        ),
        (
            "Aperiodic",
            "largest exponent error over 8 channels",
            "≤ 0.20",
            f"{err[worst]:.3f} ({worst})",
            max(err.values()) <= 0.2,
        ),
        (
            "Aperiodic",
            "HPC_1 specparam error below polyfit error",
            "specparam < polyfit",
            f"{spec_err:.3f} vs {poly_err:.3f}",
            spec_err < poly_err,
        ),
        (
            "Power–power",
            "coupled CTX_1–HPC_2 Pearson r (30–45 Hz)",
            "> 0.5",
            f"r = {c_row['pearson_r']:.3f}",
            c_row["pearson_r"] > 0.5,
        ),
        (
            "Power–power",
            "control CTX_3–HPC_4 |r| and p",
            "|r| < 0.2 and p > 0.01",
            f"r = {k_row['pearson_r']:.3f}, p = {k_row['p_value']:.2f}",
            abs(k_row["pearson_r"]) < 0.2 and k_row["p_value"] > 0.01,
        ),
        (
            "PAC",
            "HPC_1 theta→70–100 Hz z-score and p",
            "z > 3 and p < 0.05",
            f"z = {pac_hit['z_score']:.1f}, p = {pac_hit['p_value']:.3f}",
            pac_hit["z_score"] > 3 and pac_hit["p_value"] < 0.05,
        ),
        (
            "PAC",
            "HPC_3 control (70–100 Hz) z-score",
            "z < 3",
            f"z = {pac_ctl['z_score']:.2f}",
            pac_ctl["z_score"] < 3,
        ),
        (
            "PAC",
            "HPC_1 theta→150–200 Hz z-score (no truth)",
            "z < 3",
            f"z = {pac_high['z_score']:.2f}",
            pac_high["z_score"] < 3,
        ),
        (
            "HFO",
            f"sensitivity on HPC_2 ({n_det}/{len(truth['hfo']['events'])} truth ripples)",
            "≥ 0.90",
            f"{sens:.3f}",
            sens >= 0.9,
        ),
        (
            "HFO",
            "false positives on HPC_2",
            "≤ 1.0 /min",
            f"{fp:.2f} /min ({n_fp})",
            fp <= 1.0,
        ),
        (
            "HFO",
            "highest detection rate on the 7 other channels",
            "≤ 1.0 /min",
            f"{max_other:.2f} /min",
            max_other <= 1.0,
        ),
        (
            "Connectivity",
            "highest wPLI² debiased pair (beta) is CTX_2–HPC_1",
            "top pair = truth",
            f"{ta}–{tb} ({tv:.3f})",
            {ta, tb} == want,
        ),
        (
            "Connectivity",
            "highest raw wPLI pair (beta) is CTX_2–HPC_1",
            "top pair = truth",
            f"{wpli_top[0]}–{wpli_top[1]} ({wpli_top[2]:.3f})",
            {wpli_top[0], wpli_top[1]} == want,
        ),
        (
            "Connectivity",
            "table has 28 pairs x 2 methods",
            "= 56",
            f"{len(conn_all)} rows",
            len(conn_all) == 56,
        ),
        (
            "Conn. noise floor",
            "beta wPLI² debiased: unconnected max / connected",
            "< 0.1 / > 0.5",
            f"{max_unconn:.3f} / {float(conn_row['value']):.3f}",
            max_unconn < 0.1 and float(conn_row["value"]) > 0.5,
        ),
        (
            "Conn. specificity",
            "low gamma 30–45 Hz: connected ≤ max other + 0.05",
            "≤ max other + 0.05",
            f"{band_vals['low_gamma'][0]:.3f} vs {band_vals['low_gamma'][1]:.3f}",
            band_vals["low_gamma"][0] <= band_vals["low_gamma"][1] + 0.05,
        ),
        (
            "Conn. specificity",
            "high gamma 60–90 Hz: connected ≤ max other + 0.05",
            "≤ max other + 0.05",
            f"{band_vals['high_gamma'][0]:.3f} vs {band_vals['high_gamma'][1]:.3f}",
            band_vals["high_gamma"][0] <= band_vals["high_gamma"][1] + 0.05,
        ),
        (
            "PAC",
            "HPC_1 preferred phase vs truth 0 rad",
            "|error| ≤ 0.6 rad",
            f"{pref:.2f} rad",
            pref_err <= 0.6,
        ),
        (
            "BAD annotations",
            "HPC_3 HFO events overlapping the 10 BAD-marked transients",
            "= 0",
            f"{n_overlap_bad}",
            n_overlap_bad == 0,
        ),
        (
            "BAD annotations",
            "HPC_2 sensitivity with BAD spans present",
            "≥ 0.90",
            f"{sens_bad:.3f}",
            sens_bad >= 0.9,
        ),
        (
            "BAD annotations",
            f"HPC_3 analyzed duration drop (BAD time {bad_s:.1f} s)",
            "within 0.1 s of BAD time",
            f"{dur_drop:.2f} s",
            abs(dur_drop - bad_s) < 0.1,
        ),
        (
            "BAD control",
            "same transients, no annotations: detected on HPC_3",
            "≥ 8 of 10",
            f"{n_hit_unbad} of 10",
            n_hit_unbad >= 8,
        ),
        (
            "Epochs input",
            "PAC z-score on HPC_1 (clean_epochs, 2 s)",
            "> 3",
            f"z = {ep_pac['z_score']:.1f}",
            ep_pac["z_score"] > 3,
        ),
        (
            "Epochs input",
            "HFO sensitivity on HPC_2 (ripples at epoch edges may be cut)",
            "≥ 0.80",
            f"{ep_sens:.3f} ({ep_det}/{len(truth['hfo']['events'])})",
            ep_sens >= 0.8,
        ),
        (
            "Epochs input",
            "connectivity: connected pair first (beta)",
            "top pair = truth",
            f"{ep_top[0]}–{ep_top[1]} ({ep_top[2]:.3f})",
            {ep_top[0], ep_top[1]} == want,
        ),
        (
            "Control: peaks off",
            "HPC_1 periodic peaks in 6–9 Hz",
            "= 0",
            f"{n_theta_off} (on: {n_theta_on})",
            n_theta_off == 0 and n_theta_on > 0,
        ),
        (
            "Control: power coupling off",
            "coupled-pair Pearson r",
            "< 0.3",
            f"r = {off_ppc:.3f}",
            off_ppc < 0.3,
        ),
        (
            "Control: PAC off",
            "HPC_1 70–100 Hz z-score",
            "< 3",
            f"z = {off_pac:.2f}",
            off_pac < 3,
        ),
        (
            "Control: HFO off",
            "HPC_2 detections",
            "≤ 1.0 /min",
            f"{off_hfo_rate:.2f} /min ({off_hfo_n})",
            off_hfo_rate <= 1.0,
        ),
        (
            "Control: connectivity off",
            "wPLI² debiased of CTX_2–HPC_1",
            "< 0.1",
            f"{off_conn:.3f}",
            off_conn < 0.1,
        ),
    ]
    n_pass = sum(c[4] for c in checks)

    # ---- figures ----
    print("Drawing figures")
    f_data = fig_data(raw, truth)
    f_psd = fig_psd(d, truth)
    f_ap, rec, pol = fig_aperiodic(d, truth)
    f_pp, ppc_diff, odd_pairs, ppc_stats = fig_power(raw, truth, d)
    f_pac, pac_diff, _ = fig_pac(raw, truth, d)
    f_hfo, _ = fig_hfo(raw, truth, d)
    f_conn, coh = fig_connectivity(d, d_coh, d_band, want)
    ctrl = [
        {
            "title": "Theta peak (HPC_1)",
            "ylabel": "periodic peaks 6–9 Hz",
            "on": n_theta_on,
            "off": n_theta_off,
            "line": 0.5,
            "line_label": "off must be 0",
            "fmt": "{:.0f}",
        },
        {
            "title": "Power coupling",
            "ylabel": "Pearson r (coupled pair)",
            "on": float(c_row["pearson_r"]),
            "off": off_ppc,
            "line": 0.3,
            "line_label": "off < 0.3",
            "fmt": "{:.2f}",
        },
        {
            "title": "PAC",
            "ylabel": "z-score (HPC_1)",
            "on": float(pac_hit["z_score"]),
            "off": off_pac,
            "line": 3.0,
            "line_label": "z = 3",
            "fmt": "{:.1f}",
        },
        {
            "title": "HFO",
            "ylabel": "HPC_2 detections / min",
            "on": float(d["hfo_sum"].set_index("channel").loc["HPC_2", "rate_per_min"]),
            "off": off_hfo_rate,
            "line": 1.0,
            "line_label": "off ≤ 1/min",
            "fmt": "{:.1f}",
        },
        {
            "title": "Connectivity",
            "ylabel": "wPLI² deb. CTX_2–HPC_1",
            "on": float(conn_row["value"]),
            "off": off_conn,
            "line": 0.1,
            "line_label": "off < 0.1",
            "fmt": "{:.2f}",
        },
    ]
    f_ctl = fig_controls(ctrl)

    # ---- document ----
    sha, branch, dirty = git_info()
    story = []

    # Simulation parameters, read from truth (or the fixture's constants) so the
    # prose cannot drift from the data.
    peak_of = {p_["channel"]: p_ for p_ in truth["peaks"]}
    theta_pk, beta_pk = peak_of["HPC_1"], peak_of["CTX_1"]

    def half_width(pk):
        return (pk["band"][1] - pk["band"][0]) / 2.0

    chi_ctx = truth["aperiodic"]["exponent"]["CTX_1"]
    chi_hpc = truth["aperiodic"]["exponent"]["HPC_1"]
    rms_ctx = fixture._APERIODIC_RMS_V["CTX"] * 1e6
    rms_hpc = fixture._APERIODIC_RMS_V["HPC"] * 1e6
    hfo_events = truth["hfo"]["events"]
    ev_dur_s = [e[1] - e[0] for e in hfo_events]
    ev_freq = [e[2] for e in hfo_events]
    ev_dur_txt = (
        f"{min(ev_dur_s) * 1e3:.0f}–{max(ev_dur_s) * 1e3:.0f} ms"
        if hfo_events
        else "no"
    )
    ev_freq_txt = f"{min(ev_freq):.0f}–{max(ev_freq):.0f} Hz" if hfo_events else "no"
    pac_t = truth["pac"]
    pac_phase_txt = f"{pac_t['phase_band'][0]}–{pac_t['phase_band'][1]} Hz"
    pac_amp_txt = f"{pac_t['amp_band'][0]}–{pac_t['amp_band'][1]} Hz"
    conn_band_txt = (
        f"{truth['connectivity']['band'][0]}–{truth['connectivity']['band'][1]} Hz"
    )
    lag_ms = truth["connectivity"]["lag_s"] * 1e3
    line_hz = truth["line_noise_hz"]
    line_uv = fixture._LINE_AMP_V * 1e6
    story.append(P(TITLE, TITLE_S))
    story.append(
        P(
            "Visual and numerical evidence that six analysis deliverables recover "
            "known ground truth on simulated mouse cortical and hippocampal depth recordings.",
            SUB,
        )
    )
    meta = [
        [
            "Date",
            dt.date.today().isoformat(),
            "Git",
            f"{sha} on {branch}" + (" (+ uncommitted changes)" if dirty else ""),
        ],
        [
            "Seed",
            str(seed),
            "Data",
            f"8 channels, {duration:g} s, {truth['sfreq']:g} Hz, synthetic",
        ],
    ]
    story.append(make_table(meta, [0.6, 2.1, 0.6, 3.4], header=False, font=CELL))
    story.append(Spacer(1, 8))
    story.append(
        P(
            f"<b>Result: {n_pass} of {len(checks)} criteria pass.</b> "
            + (
                "All criteria are met. "
                if n_pass == len(checks)
                else "<font color='#c62828'><b>Failed criteria are marked FAIL below.</b></font> "
            )
            + "Each analysis ran through <font face='DejaVu-Mono'>Task.run_postprocessing_analysis</font>, "
            "the same path labs use. Measured values come from the CSV and parquet files that run "
            "wrote. Thresholds are the ones in the repository acceptance tests; none was changed "
            "for this report. Each effect was then switched off in the simulator and the analysis "
            "re-run; the measure must fail without its effect (controls)."
        )
    )
    rows = [
        [
            P("<b>Deliverable</b>", CELL),
            P("<b>Criterion</b>", CELL),
            P("<b>Threshold</b>", CELL),
            P("<b>Measured</b>", CELL),
            P("<b>Result</b>", CELL),
        ]
    ]
    extra = []
    for i, (dlv, crit, thr_, meas, ok) in enumerate(checks, start=1):
        rows.append([dlv, crit, thr_, meas, ""])
        extra.append(
            ("BACKGROUND", (4, i), (4, i), rl_colors.HexColor(PASS_C if ok else FAIL_C))
        )
        rows[-1][4] = Paragraph(
            f"<font color='white'><b>{result_cell(ok)}</b></font>",
            ParagraphStyle("c", parent=CELL, alignment=1),
        )
    story.append(make_table(rows, [1.2, 2.35, 1.25, 1.4, 0.5], extra=extra))
    story.append(Spacer(1, 4))
    story.append(
        P(
            "Table 1. Summary of acceptance criteria. The last five rows are the effect-removal "
            "controls (Figure 8). The BAD, Epochs and band-specificity rows re-run the runner on "
            "modified data or settings, as in the acceptance tests.",
            CAPTION,
        )
    )

    # 1. data
    story.append(PageBreak())
    story.append(P("1. Simulated data", H1))
    story.append(
        P(
            "The recording is synthetic and seeded, so the right answer is known for every analysis. "
            "It is meant to look like a mouse depth LFP: 1/f<super>χ</super> background with "
            f"χ = {chi_ctx:g} (cortex) and {chi_hpc:g} (hippocampus) at {rms_ctx:g} and {rms_hpc:g} µV RMS, "
            "a hippocampal theta rhythm, "
            f"cortical beta, gamma bursts, theta-gamma coupling, sharp-wave ripples at {fixture._HFO_PER_MIN:g} per minute, "
            f"a small shared reference signal and {line_hz:g} Hz line noise. Public datasets (for example CRCNS hc-* "
            "from the Buzsáki lab) were not used because they need registration and large downloads."
        )
    )
    story.append(
        figure_block(
            f_data,
            "Simulated recording. <b>A</b>: all eight channels (blue cortex, orange hippocampus) for a 5 s window; "
            "slow theta waves are visible on HPC channels and gray bands mark truth ripples on HPC_2. "
            "<b>B</b>: one ripple on HPC_2, raw and 120–250 Hz band-passed (scipy, for display only); the "
            "gray band is the ground-truth window and the filtered trace should show a burst inside it.",
        )
    )
    comp = [
        ["Component", "Where", "Ground truth"],
        [
            "1/f background",
            "all channels",
            f"exponent {chi_ctx:g} (CTX), {chi_hpc:g} (HPC); RMS {rms_ctx:g} / {rms_hpc:g} µV",
        ],
        [
            "Spectral peaks",
            "theta HPC_1–4; beta CTX_1–4",
            f"theta {theta_pk['freq_hz']:g} Hz (±{half_width(theta_pk):g} Hz noise band); "
            f"beta {beta_pk['freq_hz']:g} Hz (±{half_width(beta_pk):g} Hz noise band)",
        ],
        [
            "Power–power coupling",
            "CTX_1–HPC_2 (coupled); CTX_3–HPC_4 (control)",
            "shared slow envelope on 30–45 Hz bursts",
        ],
        [
            "Theta–gamma PAC",
            "HPC_1 (HPC_3 control)",
            f"theta phase ({pac_phase_txt}) modulates {pac_amp_txt}, depth {fixture._PAC_DEPTH:g}",
        ],
        [
            "Sharp-wave ripples",
            "HPC_2",
            f"{len(hfo_events)} events, {ev_freq_txt}, {ev_dur_txt} (observed ranges), "
            f"{fixture._HFO_FACTOR[0]:g}–{fixture._HFO_FACTOR[1]:g}× background",
        ],
        [
            "Lagged shared source",
            "CTX_2 → HPC_1",
            f"{conn_band_txt}, lag {lag_ms:g} ms",
        ],
        [
            "Reference and line noise",
            "all channels",
            f"common 1/f<super>1.5</super> at {fixture._COMMON_FRACTION:.0%} of RMS; "
            f"{line_hz:g} Hz at {line_uv:g} µV",
        ],
    ]
    story.append(make_table(comp, [1.5, 2.4, 2.8]))
    story.append(
        P("Table 2. Injected components and their ground-truth parameters.", CAPTION)
    )

    # 2. PSD
    story.append(PageBreak())
    story.append(P("2. Power spectral density", H1))
    story.append(
        P(
            "Welch PSD from the <font face='DejaVu-Mono'>sensor_psd</font> block (1–300 Hz, "
            "n_fft 8000, 0.25 Hz bins)."
        )
    )
    story.append(
        figure_block(
            f_psd,
            "<b>A, B</b>: PSD of every cortex and hippocampus channel. Expect a straight 1/f slope with a bump at "
            f"{beta_pk['freq_hz']:g} Hz in cortex and {theta_pk['freq_hz']:g} Hz in hippocampus; dashed gray lines are the truth frequencies. "
            f"The runner's peak frequencies were {beta:.2f} Hz (CTX_1) and {theta:.2f} Hz (HPC_1). "
            f"The {line_hz:g} Hz spike is the injected line noise. <b>C</b>: log band power for each channel "
            "(band powers integrated from the saved PSD table); the values are printed in each cell.",
        )
    )

    # 3. aperiodic
    story.append(PageBreak())
    story.append(P("3. Aperiodic activity", H1))
    story.append(
        P("specparam fits the 2–55 Hz range (fixed mode) to the sensor PSD table.")
    )
    story.append(
        figure_block(
            f_ap,
            "<b>A, B</b>: HPC_1 and CTX_1 spectra with the specparam aperiodic fit and full model (aperiodic "
            "plus the peaks in the periodic CSV, rebuilt for display). The fit should follow the 1/f slope and the "
            "full model should capture the peak. <b>C</b>: true vs recovered exponent for all 8 channels; "
            "filled circles (specparam) should sit on the dashed identity line inside the ±0.2 band, "
            "open triangles (the old log-log polyfit) are shown for comparison. <b>D</b>: signed error per "
            f"channel. Largest specparam error: {max(err.values()):.3f}; "
            f"largest polyfit error: {np.max(np.abs(pol - np.array([truth['aperiodic']['exponent'][c] for c in CHANNELS]))):.3f}.",
        )
    )

    # 4. power-power
    story.append(PageBreak())
    story.append(P("4. Power–power coupling", H1))
    n_windows = int(c_row["n_windows"])
    story.append(
        P(
            "Log 30–45 Hz band power in non-overlapping 1 s windows, correlated across channels "
            f"({n_windows} windows after discarding filter edges at each end)."
        )
    )
    story.append(
        figure_block(
            f_pp,
            "<b>A, B</b>: window-by-window log band power for the coupled pair (CTX_1–HPC_2) and a control "
            "pair (CTX_3–HPC_4). The coupled cloud should slope upward; the control should be a round blob. "
            "r and p are read from the runner CSV. <b>C</b>: Pearson r for all channel pairs (direct call to "
            "<font face='DejaVu-Mono'>compute_power_power_coupling</font>); CTX_1–HPC_2 is the designed pair "
            f"(r = {ppc_stats['coupled_r']:.2f}). The highest other pair is {ppc_stats['runner_up']} "
            f"(r = {ppc_stats['runner_up_r']:.2f}), a margin of {ppc_stats['margin']:.2f}. "
            f"The largest undesigned |r| is {ppc_stats['max_abs_other']:.2f}. Undesigned pairs above 0.25 in magnitude: "
            + (", ".join(odd_pairs) or "none")
            + ". Each channel's 1/f background is an independent random draw, so these pairs are "
            "unrelated by construction. <b>D, E</b>: z-scored power time courses; the coupled pair rises and falls "
            f"together. Window series are recomputed for display and match the CSV r to {ppc_diff:.1e}.",
        )
    )

    # 5. PAC
    story.append(PageBreak())
    story.append(P("5. Phase–amplitude coupling", H1))
    story.append(
        P(
            f"Tort modulation index with {settings['pac'].get('n_bins', 18)} phase bins and "
            f"{settings['pac']['n_surrogates']} shifted-amplitude surrogates "
            f"(runner, seed {settings['pac']['seed']})."
        )
    )
    story.append(
        figure_block(
            f_pac,
            "<b>A, B</b>: comodulograms (phase centers 2–14 Hz, 2 Hz wide; amplitude centers 20–200 Hz, all 30 Hz wide, "
            "which is 2 × the largest phase high edge, the minimum the function accepts) for the coupled "
            "channel and the control, on one color scale. A bright spot should appear only in A, inside the dashed truth "
            f"box (theta phase, {pac_amp_txt}). Gray cells have overlapping phase and amplitude bands. "
            f"<b>C</b>: mean {pac_amp_txt} amplitude per theta phase bin, shown for two cycles; HPC_1 should show a "
            "clear peak near phase 0 and the control a flat profile. "
            "<b>D</b>: observed MI against 1000 circularly shifted surrogates (own histogram, same procedure as the "
            "runner); the observed value should be far to the right. The comodulogram (n_surrogates=0), "
            "phase-bin profile and histogram are direct calls to "
            f"<font face='DejaVu-Mono'>oscillation_analysis</font>; the MI in C matches the runner CSV to {pac_diff:.1e} (relative).",
        )
    )

    # 6. HFO
    story.append(PageBreak())
    story.append(P("6. High-frequency oscillations (ripples)", H1))
    story.append(
        P(
            "Robust RMS threshold detector on 120–250 Hz (median + 3 × 1.4826 × MAD, "
            "at least 4 cycles)."
        )
    )
    story.append(
        figure_block(
            f_hfo,
            "<b>A</b>: truth ripple times (top row) against detections on every channel. Orange ticks match a truth "
            "event (±50 ms); red ticks are false positives. Detections should line up with the truth row only on "
            "HPC_2. <b>B</b>: one detected ripple (median amplitude among matches): raw, filtered, and RMS envelope "
            "(3 ms) with the detector threshold from the summary CSV; the green band is the detection, dashed lines the "
            "truth window. (Filtered and RMS traces are recomputed for display.) <b>C</b>: detections per minute per channel.",
        )
    )
    hrows = [
        ["Measure", "Value"],
        ["Truth ripples (HPC_2)", str(len(truth["hfo"]["events"]))],
        [
            "Detected on HPC_2",
            f"{int((d['hfo_ev']['channel'] == 'HPC_2').sum())} events",
        ],
        [
            "Truth ripples detected (sensitivity)",
            f"{n_det} / {len(truth['hfo']['events'])} = {sens:.3f}",
        ],
        ["False positives on HPC_2", f"{fp:.2f} /min ({n_fp})"],
        ["Highest rate on other channels", f"{max_other:.2f} /min"],
    ]
    story.append(make_table(hrows, [3.0, 2.2]))
    story.append(P("Table 3. Detection statistics.", CAPTION))

    # 7. connectivity
    story.append(PageBreak())
    story.append(P("7. Functional connectivity", H1))
    story.append(
        P(
            f"Across-epoch multitaper estimates (mne-connectivity, 2 Hz multitaper bandwidth) from "
            f"{n_epochs} fixed-length {epoch_len:g} s epochs. The primary measure is debiased wPLI² "
            "(<font face='DejaVu-Mono'>wpli2_debiased</font>), which removes the small-sample bias of "
            "plain wPLI, so unconnected pairs sit near zero. Coherence comes from a second runner "
            "call on the same data. Spans annotated <font face='DejaVu-Mono'>BAD_</font> are excluded."
        )
    )
    unc_mean = float(unconn["value"].mean())

    def unconnected_mean(df, method):
        m = df[df["method"] == method]
        keep = ~m.apply(lambda r: {r.channel_a, r.channel_b} == want, axis=1)
        return float(m[keep]["value"].mean())

    coh_unc = unconnected_mean(d_coh["conn"], "coh")
    coh_unc_nr = unconnected_mean(d_nr["conn"], "coh")
    wp_unc_nr = unconnected_mean(d_nr["conn"], PRIMARY)
    story.append(
        figure_block(
            f_conn,
            "<b>A</b>: wPLI² debiased matrix (primary); white and black lines separate cortex (first four) from "
            f"hippocampus. Only CTX_2–HPC_1 ({float(conn_row['value']):.2f}) should stand out; the other pairs "
            f"average {unc_mean:.3f} with a maximum of {max_unconn:.3f} (noise floor, limit 0.1). "
            f"<b>B</b>: coherence for the same pairs; "
            f"unconnected pairs average {coh_unc:.2f}. Coherence counts zero-lag shared signal such as a common "
            "reference; wPLI ignores it. Re-running with the common reference removed, mean unconnected "
            f"coherence changes from {coh_unc:.3f} to {coh_unc_nr:.3f} while mean unconnected wPLI² debiased changes from "
            f"{unc_mean:.3f} to {wp_unc_nr:.3f}. "
            + (
                "At this reference strength (10% of RMS) the change is negligible, so these data do not "
                "demonstrate common-reference inflation of coherence. "
                if abs(coh_unc - coh_unc_nr) < 0.01
                else ""
            )
            + "<b>C</b>: the eight highest wPLI² debiased pairs (purple = cortex–hippocampus), with coherence as "
            "white diamonds. The connected pair ranks first. <b>D</b>: the connected pair against the "
            "highest other pair in beta (15–25 Hz, where the shared source lives), low gamma (30–45 Hz) and "
            "high gamma (60–90 Hz). Only beta should show the connected pair; the gamma bands should not "
            "elevate it.",
        )
    )

    # 8. controls
    story.append(PageBreak())
    story.append(P("8. Effect-removal controls", H1))
    story.append(
        P(
            "Each effect was switched off in the simulator (everything else, including the random "
            "numbers, unchanged) and the relevant blocks were re-run through the runner."
        )
    )
    story.append(
        figure_block(
            f_ctl,
            "Each measure with its effect on (dark) and off (light). Red dashed lines mark the limit that the "
            "off condition must stay under; the on bar should be well above it. Values are read from the runner "
            "outputs of the original run and of the regenerated control runs. "
            "Panel A uses the count of specparam peaks in 6–9 Hz on HPC_1.",
        )
    )
    crow = [
        ["Control", "Measure", "On", "Off", "Limit (off)"],
        ["Peaks", "HPC_1 peaks in 6–9 Hz", f"{n_theta_on}", f"{n_theta_off}", "0"],
        [
            "Power coupling",
            "Pearson r, CTX_1–HPC_2",
            f"{c_row['pearson_r']:.3f}",
            f"{off_ppc:.3f}",
            "< 0.3",
        ],
        [
            "PAC",
            "z-score, HPC_1 70–100 Hz",
            f"{pac_hit['z_score']:.1f}",
            f"{off_pac:.2f}",
            "< 3",
        ],
        [
            "HFO",
            "HPC_2 detections /min",
            f"{d['hfo_sum'].set_index('channel').loc['HPC_2', 'rate_per_min']:.1f}",
            f"{off_hfo_rate:.2f}",
            "≤ 1.0",
        ],
        [
            "Connectivity",
            "wPLI² debiased CTX_2–HPC_1",
            f"{float(conn_row['value']):.3f}",
            f"{off_conn:.3f}",
            "< 0.1",
        ],
    ]
    story.append(make_table(crow, [1.2, 2.3, 0.9, 0.9, 1.0]))
    story.append(
        P(
            "Table 4. Control values. With connectivity off, the top wPLI² debiased pair is "
            f"{off_top[0]}–{off_top[1]} ({off_top[2]:.3f}).",
            CAPTION,
        )
    )

    story.append(Spacer(1, 6))
    story.append(
        P(
            "<b>BAD annotations and Epochs input.</b> Ten 2 mV, 20 ms transients were added to HPC_3 and "
            "each was annotated <font face='DejaVu-Mono'>BAD_artifact</font> (±0.1 s). With the annotations "
            f"the HFO block reported {n_overlap_bad} events overlapping them, kept HPC_2 sensitivity at "
            f"{sens_bad:.2f} and cut HPC_3 analyzed duration by {dur_drop:.2f} s (BAD time {bad_s:.2f} s). "
            f"Without the annotations the same transients were detected {n_hit_unbad} of 10 times, so the "
            "check can fail. Running pac, hfo and connectivity on 2 s Epochs "
            f"(<font face='DejaVu-Mono'>clean_epochs</font>) gave PAC z = {ep_pac['z_score']:.1f}, HFO "
            f"sensitivity {ep_sens:.2f} and the connected pair first. Ripples that cross an epoch edge can be "
            "cut or lost, so the Epochs HFO limit is 0.80 instead of 0.90.",
            BODY,
        )
    )

    # 9. methods
    story.append(CondPageBreak(3.5 * inch))
    story.append(P("9. Methods, lab use and limitations", H1))
    story.append(
        P(
            "<b>Method.</b> A seeded recording is generated by <font face='DejaVu-Mono'>create_synthetic_mouse_depth_raw</font>. "
            "A minimal Task holds it as <font face='DejaVu-Mono'>clean_raw</font> and calls "
            "<font face='DejaVu-Mono'>run_postprocessing_analysis</font> with the configuration below. "
            "The report reads the CSV and parquet outputs back from disk and measures them with the same definitions as "
            "<font face='DejaVu-Mono'>tests/unit/core/test_mouse_depth_analysis.py</font>. Extra runner calls on the same "
            "data or on modified copies add a coherence comparison, the three connectivity bands, the BAD-annotated and "
            "unannotated transient copies, and the Epochs input. The coherence run uses methods wPLI² debiased and coherence"
            + (
                " (its wPLI² debiased values equal the main run)."
                if wp_same
                else " (<b>its wPLI² debiased values differ from the main run</b>)."
            )
        )
    )
    story.append(
        P(
            "<b>How to run in the lab.</b> This is the configuration the main run used. Put this dictionary in the task's "
            "<font face='DejaVu-Mono'>postprocessing_analysis</font> setting, then regenerate this report with "
            "<font face='DejaVu-Mono'>uv run --extra dev python scripts/mouse_depth_validation_report.py</font>.",
            BODY,
        )
    )
    cfg = pprint.pformat(settings, width=100, sort_dicts=False)
    story.append(Preformatted(cfg, MONO))
    story.append(Spacer(1, 6))
    story.append(P("<b>Limitations.</b>", BODY))
    for item in (
        "The data are synthetic only. Real recordings need a reference choice, artifact rejection, and tuning of "
        "the HFO threshold; these results do not show performance on real LFP.",
        f"Ripples are simulated with {ev_dur_txt} windows (±2σ of the Gaussian envelope). In testing, events "
        "with a window of 31–47 ms were partly missed because fewer than 4 cycles exceeded the threshold, "
        "so the simulation uses longer windows. Real short ripples may be under-detected.",
        "PSD peak tolerance is 0.5 Hz because the injected peaks are narrowband noise, not pure tones.",
        "Power-coupling p-values use Pearson's test, which assumes independent windows. Slow envelopes are "
        "autocorrelated, so these p-values are too optimistic; in this run the largest undesigned |r| was "
        f"{ppc_stats['max_abs_other']:.2f}. "
        "Judge coupling by rank against all pairs (as tested), or add a surrogate test before using p-values "
        "on real data.",
        f"Connectivity refuses a band whose low edge is below 5 / epoch_length ({5 / epoch_len:g} Hz for "
        f"{epoch_len:g} s epochs), so delta needs longer epochs (fewer than 5 cycles per epoch is too few).",
        "Connectivity is across-epoch, so it needs many epochs; the estimate is only as good as the number of "
        f"good {epoch_len:g} s epochs left after BAD exclusion. Bands narrower than 2 Hz are refused.",
        "The BAD-annotation and Epochs checks use one set of 10 injected transients and one epoching; they show "
        "the code path works, not that thresholds suit real artifacts.",
        "Single seed, single recording; the report does not estimate variability across seeds.",
        *(
            [
                "The simulated common reference (10% of RMS) does not measurably change coherence in "
                "15\u201325 Hz, so wPLI's insensitivity to zero-lag inflation is not demonstrated by "
                "these data."
            ]
            if abs(coh_unc - coh_unc_nr) < 0.01
            else []
        ),
    ):
        story.append(P("• " + item, SMALL))

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title=TITLE,
        author="AutoCleanEEG",
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Wrote {out_path} in {time.time() - t_start:.0f} s")
    for dlv, crit, thr_, meas, ok in checks:
        print(f"{result_cell(ok):4s} {dlv:28s} {crit[:58]:58s} {thr_:22s} {meas}")
    print(f"{n_pass}/{len(checks)} pass")
    return n_pass == len(checks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out",
        default=str(
            REPO / "build" / "mouse_depth_report" / "mouse_depth_validation_report.pdf"
        ),
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--duration", type=float, default=120.0)
    args = ap.parse_args()
    if args.duration < MIN_DURATION_S:
        ap.error(
            f"--duration must be at least {MIN_DURATION_S:g} s; got {args.duration:g}"
        )
    build(args.out, args.seed, args.duration)


if __name__ == "__main__":
    main()
