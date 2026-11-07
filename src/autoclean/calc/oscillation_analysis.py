"""Oscillation analysis helpers: power-power coupling, PAC, and HFO detection."""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Mapping, NamedTuple, Sequence

import mne
import numpy as np
import pandas as pd
from scipy import fft as sp_fft
from scipy import signal, stats

_PPC_COLUMNS = [
    "channel_a",
    "band_a_low",
    "band_a_high",
    "channel_b",
    "band_b_low",
    "band_b_high",
    "n_windows",
    "pearson_r",
    "spearman_r",
    "p_value",
    "window_s",
    "status",
]
_PAC_COLUMNS = [
    "phase_channel",
    "amp_channel",
    "phase_low",
    "phase_high",
    "amp_low",
    "amp_high",
    "mi",
    "surrogate_mean",
    "surrogate_std",
    "z_score",
    "p_value",
    "n_surrogates",
    "preferred_phase_rad",
    "seed",
    "status",
]
_STATUS_OK = "ok"
_STATUS_INSUFFICIENT = "insufficient_data"
_HFO_EVENT_COLUMNS = [
    "channel",
    "onset_s",
    "offset_s",
    "duration_s",
    "peak_freq_hz",
    "peak_amplitude",
    "band_low",
    "band_high",
]
# Added for Epochs input only (after _HFO_EVENT_COLUMNS).
_HFO_EPOCH_COLUMNS = ["epoch_index", "epoch_onset_s", "onset_in_epoch_s"]
_HFO_SUMMARY_COLUMNS = [
    "channel",
    "n_events",
    "rate_per_min",
    "analyzed_duration_s",
    "threshold",
    "band_low",
    "band_high",
]
_FILTER_ORDER = 4
# sosfiltfilt needs more samples than its pad length (3 * (2 * n_sections + 1)).
_MIN_SEGMENT_SAMPLES = 3 * (2 * _FILTER_ORDER + 1) + 1
_Picks = str | Sequence[str] | Sequence[int] | None


class _Segment(NamedTuple):
    """One contiguous analysis span: data (n_channels, n_times) plus timing."""

    data: np.ndarray
    start_s: float  # Raw: seconds from recording start; Epochs: epoch start time
    index: int | None  # Epochs: epoch index; Raw: None


def compute_power_power_coupling(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    pairs: Sequence[Mapping[str, Any]] | None = None,
    bands: Mapping[str, Sequence[float]] | None = None,
    picks: _Picks = None,
    window_s: float = 1.0,
) -> pd.DataFrame:
    """Correlate log band-power envelopes between channel/band pairs.

    Band power is ``|hilbert(bandpass(x))|**2`` averaged in consecutive
    non-overlapping windows of ``window_s`` (per epoch for Epochs), then
    log10-transformed. Filter edges (1 s, or 10% of the segment if shorter, and
    never less than one cycle of the band's low edge) are discarded first; spans
    that keep less than one window afterwards are dropped. This is decided per
    band: a pair of different bands uses the larger of the two edges, so a slow
    band never trims or drops data for a fast one. ``pairs`` entries are dicts
    with ``channel_a``, ``band_a``, ``channel_b``, ``band_b``. If ``pairs`` is
    None, every unordered pair of picked channels is combined with each band in
    ``bands``.

    ``picks`` follows MNE (None = data channels, bads excluded; a type such as
    ``"eeg"``, channel names, or indices). Raw spans annotated ``BAD*`` are
    skipped: each contiguous good span is analyzed like one epoch. A pair with
    fewer than 3 usable windows gets a row with NaN statistics and
    ``status="insufficient_data"`` (``"ok"`` otherwise); ValueError is raised
    only if no pair is computable. ``df.attrs["n_segments_dropped"]`` counts
    spans too short for any analysis; ``df.attrs["n_segments_dropped_by_band"]``
    maps each ``"low-high"`` band to all spans unusable for that band.
    """
    _check_data(data)
    sfreq = float(data.info["sfreq"])
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    win = int(round(window_s * sfreq))
    if win < 1:
        raise ValueError("window_s is shorter than one sample")

    specs: list[tuple[str, tuple[float, float], str, tuple[float, float]]] = []
    if pairs is None:
        if not bands:
            raise ValueError("bands is required when pairs is None")
        names = _resolve_channels(data, picks)
        for label, band in bands.items():
            checked = _validate_band(band, data, str(label))
            for ch_a, ch_b in itertools.combinations(names, 2):
                specs.append((ch_a, checked, ch_b, checked))
    else:
        for i, pair in enumerate(pairs):
            missing_keys = {"channel_a", "band_a", "channel_b", "band_b"} - set(pair)
            if missing_keys:
                raise ValueError(f"pairs[{i}] is missing keys: {sorted(missing_keys)}")
            specs.append(
                (
                    str(pair["channel_a"]),
                    _validate_band(pair["band_a"], data, f"pairs[{i}].band_a"),
                    str(pair["channel_b"]),
                    _validate_band(pair["band_b"], data, f"pairs[{i}].band_b"),
                )
            )

    names_needed = list(dict.fromkeys(n for s in specs for n in (s[0], s[2])))
    _check_channels_exist(data, names_needed)
    _check_not_bad(data, names_needed)
    segments, n_dropped = _select_channel_data(data, names_needed)
    index = {name: i for i, name in enumerate(names_needed)}

    seg_cache: dict[float, list[_Segment]] = {}

    def _usable(low_hz: float) -> list[_Segment]:
        if low_hz not in seg_cache:
            seg_cache[low_hz] = _usable_segments(segments, sfreq, low_hz, window_s)
        return seg_cache[low_hz]

    dropped_by_band: dict[str, int] = {}
    minimums: dict[str, float] = {}
    for _, band_a, _, band_b in specs:
        for band in (band_a, band_b):
            key = _band_key(band)
            if key not in dropped_by_band:
                dropped_by_band[key] = n_dropped + len(segments) - len(_usable(band[0]))
                minimums[f"{key} Hz"] = _min_span_seconds(sfreq, band[0], window_s)

    cache: dict[tuple[str, tuple[float, float], float], np.ndarray] = {}

    def _log_power(ch: str, band: tuple[float, float], low_hz: float) -> np.ndarray:
        key = (ch, band, low_hz)
        if key not in cache:
            chunks = []
            for seg in _usable(low_hz):
                env = np.abs(_analytic(seg.data[index[ch]], band, sfreq)) ** 2
                # Drop filter edges (1 s, or 10% if shorter), as in PAC.
                edge = _edge_samples(env.size, sfreq, low_hz)
                env = env[edge : env.size - edge]
                n_win = env.size // win
                if n_win:
                    chunks.append(env[: n_win * win].reshape(n_win, win).mean(axis=1))
            power = np.concatenate(chunks) if chunks else np.empty(0)
            cache[key] = np.log10(np.maximum(power, np.finfo(float).tiny))
        return cache[key]

    rows: list[dict[str, Any]] = []
    any_segments = False
    for ch_a, band_a, ch_b, band_b in specs:
        # The larger edge belongs to the lower low-frequency edge.
        low_pair = min(band_a[0], band_b[0])
        any_segments = any_segments or bool(_usable(low_pair))
        pa = _log_power(ch_a, band_a, low_pair)
        pb = _log_power(ch_b, band_b, low_pair)
        row: dict[str, Any] = {
            "channel_a": ch_a,
            "band_a_low": band_a[0],
            "band_a_high": band_a[1],
            "channel_b": ch_b,
            "band_b_low": band_b[0],
            "band_b_high": band_b[1],
            "n_windows": int(pa.size),
            "pearson_r": float("nan"),
            "spearman_r": float("nan"),
            "p_value": float("nan"),
            "window_s": float(window_s),
            "status": _STATUS_INSUFFICIENT,
        }
        if pa.size >= 3:
            pearson = stats.pearsonr(pa, pb)
            spearman = stats.spearmanr(pa, pb)
            row.update(
                pearson_r=float(pearson[0]),
                spearman_r=float(spearman[0]),
                p_value=float(pearson[1]),
                status=_STATUS_OK,
            )
        rows.append(row)
    if not any(r["status"] == _STATUS_OK for r in rows):
        if not any_segments:
            raise _no_usable_data_error(data, minimums, n_dropped)
        raise ValueError(
            "Data too short for power-power coupling: need at least 3 windows "
            f"of {window_s:g} s per pair (best pair has "
            f"{max(r['n_windows'] for r in rows)})."
        )
    out = pd.DataFrame(rows, columns=_PPC_COLUMNS)
    out.attrs["n_segments_dropped"] = n_dropped
    out.attrs["n_segments_dropped_by_band"] = dropped_by_band
    return out


def compute_phase_amplitude_coupling(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    phase_bands: Sequence[Sequence[float]],
    amp_bands: Sequence[Sequence[float]],
    picks: _Picks = None,
    channel_pairs: Sequence[Sequence[str]] | None = None,
    n_bins: int = 18,
    n_surrogates: int = 200,
    min_shift_s: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Tort modulation index phase-amplitude coupling.

    Filter edges (1 s, or 10% of the segment if shorter, and never less than
    one cycle of the phase band's low edge) are discarded. Segments that keep
    less than 3 cycles of that phase band afterwards are dropped. This is
    decided per phase band, so a slow phase band never trims or drops data for
    a faster one. A (phase band, amp band) pair whose phase band has too little
    data (fewer than 10 cycles in total, or too short for ``min_shift_s``) gets
    a row with NaN results and ``status="insufficient_data"`` (``"ok"``
    otherwise); ValueError is raised only if no pair is computable.
    ``attrs["n_segments_dropped"]`` counts spans too short for any analysis and
    ``attrs["n_segments_dropped_by_band"]`` maps each ``"low-high"`` phase band
    to all spans unusable for it. Surrogates circularly shift the amplitude
    series by a random offset in ``[min_shift_s, T - min_shift_s]``. With ``n_surrogates=0`` the surrogate
    columns are NaN. ``channel_pairs`` is a list of ``(phase_ch, amp_ch)``;
    by default each picked channel is paired with itself. ``picks`` and
    ``BAD*`` annotations are handled as in :func:`compute_power_power_coupling`.
    The amplitude band must be at least 2 x the phase band high edge wide.
    """
    _check_data(data)
    sfreq = float(data.info["sfreq"])
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if n_surrogates < 0:
        raise ValueError("n_surrogates must be >= 0")
    if min_shift_s <= 0:
        raise ValueError("min_shift_s must be positive")
    if not len(phase_bands) or not len(amp_bands):
        raise ValueError("phase_bands and amp_bands must not be empty")

    p_bands = [
        _validate_band(b, data, f"phase_bands[{i}]") for i, b in enumerate(phase_bands)
    ]
    a_bands = [
        _validate_band(b, data, f"amp_bands[{i}]") for i, b in enumerate(amp_bands)
    ]
    for pb in p_bands:
        for ab in a_bands:
            if ab[0] <= pb[1]:
                raise ValueError(
                    f"Amplitude band low edge ({ab[0]:g} Hz) must be above the "
                    f"phase band high edge ({pb[1]:g} Hz)."
                )
            if ab[1] - ab[0] < 2 * pb[1]:
                raise ValueError(
                    f"Amplitude band {ab[0]:g}-{ab[1]:g} Hz is {ab[1] - ab[0]:g} Hz "
                    f"wide; it must be at least 2 x the phase band high edge "
                    f"({2 * pb[1]:g} Hz) to resolve the modulation (Aru et al. 2015)."
                )

    if channel_pairs is None:
        pairs = [(n, n) for n in _resolve_channels(data, picks)]
    else:
        pairs = []
        for i, cp in enumerate(channel_pairs):
            if len(cp) != 2:
                raise ValueError(f"channel_pairs[{i}] must be (phase_ch, amp_ch)")
            pairs.append((str(cp[0]), str(cp[1])))
    names_needed = list(dict.fromkeys(n for p in pairs for n in p))
    _check_channels_exist(data, names_needed)
    _check_not_bad(data, names_needed)

    segments, n_dropped = _select_channel_data(data, names_needed)
    index = {name: i for i, name in enumerate(names_needed)}
    min_shift = int(round(min_shift_s * sfreq))

    # Per phase band: usable segments and whether the band can be analyzed.
    # The amp band sits above the phase band, so the phase band's one-cycle
    # edge is always the larger one and sets the trimming for the pair.
    band_segments: dict[tuple[float, float], list[_Segment]] = {}
    band_reason: dict[tuple[float, float], str | None] = {}
    dropped_by_band: dict[str, int] = {}
    minimums: dict[str, float] = {}
    for pb in p_bands:
        if pb in band_segments:
            continue
        low = pb[0]
        usable = _usable_segments(segments, sfreq, low, 3.0 / low)
        band_segments[pb] = usable
        dropped_by_band[_band_key(pb)] = n_dropped + len(segments) - len(usable)
        minimums[f"{_band_key(pb)} Hz"] = _min_span_seconds(sfreq, low, 3.0 / low)
        duration = sum(seg.data.shape[1] for seg in usable) / sfreq
        min_duration = max(10.0 / low, 2 * min_shift_s + 1.0)
        n_kept = sum(
            seg.data.shape[1] - 2 * _edge_samples(seg.data.shape[1], sfreq, low)
            for seg in usable
        )
        reason = None
        if usable and duration < min_duration:
            reason = (
                f"Data too short for phase-amplitude coupling: {duration:.2f} s, "
                f"need at least {min_duration:.2f} s (10 cycles of the lowest "
                f"phase frequency and 2 x min_shift_s + 1 s)."
            )
        elif usable and n_surrogates > 0 and n_kept < 2 * min_shift + 1:
            reason = (
                "Data too short after discarding filter edges for the requested "
                f"min_shift_s={min_shift_s:g}."
            )
        elif not usable:
            reason = "no usable segments"
        band_reason[pb] = reason

    phase_cache: dict[tuple[str, tuple[float, float]], tuple[np.ndarray, ...]] = {}
    amp_cache: dict[tuple[str, tuple[float, float], float], np.ndarray] = {}

    def _trimmed(
        ch: str, band: tuple[float, float], low: float, pb: tuple[float, float], fn: Any
    ) -> np.ndarray:
        parts = []
        for seg in band_segments[pb]:
            val = fn(_analytic(seg.data[index[ch]], band, sfreq))
            edge = _edge_samples(val.size, sfreq, low)
            parts.append(val[edge : val.size - edge])
        return np.concatenate(parts)

    def _phase(ch: str, pb: tuple[float, float]) -> tuple[np.ndarray, ...]:
        key = (ch, pb)
        if key not in phase_cache:
            ph = _trimmed(ch, pb, pb[0], pb, np.angle)
            idx = np.minimum(
                ((ph + np.pi) / (2 * np.pi) * n_bins).astype(np.intp), n_bins - 1
            )
            phase_cache[key] = (ph, idx, np.bincount(idx, minlength=n_bins))
        return phase_cache[key]

    def _amp(ch: str, ab: tuple[float, float], pb: tuple[float, float]) -> np.ndarray:
        key = (ch, ab, pb[0])
        if key not in amp_cache:
            amp_cache[key] = _trimmed(ch, ab, pb[0], pb, np.abs)
        return amp_cache[key]

    log_bins = np.log(n_bins)
    rows: list[dict[str, Any]] = []

    def _mi(idx: np.ndarray, counts: np.ndarray, amp: np.ndarray) -> float:
        sums = np.bincount(idx, weights=amp, minlength=n_bins)
        mean = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
        total = mean.sum()
        if total <= 0:
            return float("nan")
        p = mean / total
        p = p[p > 0]
        return float((log_bins + np.sum(p * np.log(p))) / log_bins)

    for ph_ch, amp_ch in pairs:
        for pb in p_bands:
            if band_reason[pb] is not None:
                for ab in a_bands:
                    rows.append(
                        {
                            "phase_channel": ph_ch,
                            "amp_channel": amp_ch,
                            "phase_low": pb[0],
                            "phase_high": pb[1],
                            "amp_low": ab[0],
                            "amp_high": ab[1],
                            "mi": float("nan"),
                            "surrogate_mean": float("nan"),
                            "surrogate_std": float("nan"),
                            "z_score": float("nan"),
                            "p_value": float("nan"),
                            "n_surrogates": int(n_surrogates),
                            "preferred_phase_rad": float("nan"),
                            "seed": int(seed),
                            "status": _STATUS_INSUFFICIENT,
                        }
                    )
                continue
            ph, idx, counts = _phase(ph_ch, pb)
            for ab in a_bands:
                amp = _amp(amp_ch, ab, pb)
                mi = _mi(idx, counts, amp)
                pref = float(np.angle(np.sum(amp * np.exp(1j * ph))))
                if n_surrogates > 0:
                    rng = np.random.default_rng(
                        [int(seed) & 0xFFFFFFFF, *_stable_key(ph_ch, amp_ch, pb, ab)]
                    )
                    shifts = rng.integers(
                        min_shift, amp.size - min_shift + 1, size=n_surrogates
                    )
                    surr = np.array(
                        [_mi(idx, counts, np.roll(amp, int(k))) for k in shifts]
                    )
                    s_mean = float(np.mean(surr))
                    s_std = float(np.std(surr))
                    z = (mi - s_mean) / s_std if s_std > 0 else float("nan")
                    p_val = float((1 + np.sum(surr >= mi)) / (1 + n_surrogates))
                else:
                    s_mean = s_std = z = p_val = float("nan")
                rows.append(
                    {
                        "phase_channel": ph_ch,
                        "amp_channel": amp_ch,
                        "phase_low": pb[0],
                        "phase_high": pb[1],
                        "amp_low": ab[0],
                        "amp_high": ab[1],
                        "mi": mi,
                        "surrogate_mean": s_mean,
                        "surrogate_std": s_std,
                        "z_score": z,
                        "p_value": p_val,
                        "n_surrogates": int(n_surrogates),
                        "preferred_phase_rad": pref,
                        "seed": int(seed),
                        "status": _STATUS_OK,
                    }
                )
    if all(r is not None for r in band_reason.values()):
        if not any(band_segments.values()):
            raise _no_usable_data_error(data, minimums, n_dropped)
        raise ValueError(
            next(r for r in band_reason.values() if r != "no usable segments")
        )
    out = pd.DataFrame(rows, columns=_PAC_COLUMNS)
    out.attrs["n_segments_dropped"] = n_dropped
    out.attrs["n_segments_dropped_by_band"] = dropped_by_band
    return out


def detect_hfos(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    band: Sequence[float] = (120.0, 250.0),
    picks: _Picks = None,
    rms_window_s: float = 0.003,
    threshold_sd: float = 3.0,
    min_duration_s: float = 0.015,
    max_duration_s: float = 0.3,
    min_cycles: int = 4,
    merge_gap_s: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect high-frequency oscillations with a robust RMS threshold.

    The band-passed signal's moving RMS is thresholded at
    ``median + threshold_sd * 1.4826 * MAD``. Nearby candidates are merged and
    kept if they satisfy the duration limits and contain at least
    ``min_cycles`` positive peaks above the threshold level. Returns
    ``(events_df, summary_df)``; Epochs are detected per epoch and events get
    ``epoch_index``, ``epoch_onset_s`` and ``onset_in_epoch_s`` columns.

    ``BAD*``-annotated Raw spans are skipped and excluded from
    ``analyzed_duration_s``. Each good span is band-passed separately, but one
    robust threshold per channel is computed from the RMS pooled across all
    spans (or epochs). Spans shorter than ``2 * min_duration_s`` are dropped. Raw
    ``onset_s`` is seconds from the recording start. For Epochs,
    ``epoch_onset_s`` is ``events[e, 0] / sfreq + tmin`` (the epoch start in
    event-sample time, which includes the raw ``first_samp`` offset),
    ``onset_in_epoch_s`` is the offset from the epoch start, and ``onset_s`` is
    their sum. ``picks`` is handled as in :func:`compute_power_power_coupling`.
    The number of dropped spans is stored in ``attrs["n_segments_dropped"]``.
    """
    _check_data(data)
    sfreq = float(data.info["sfreq"])
    low, high = _validate_band(band, data, "band")
    if rms_window_s <= 0 or min_duration_s <= 0 or max_duration_s < min_duration_s:
        raise ValueError("rms_window_s and duration limits must be positive/ordered")
    names = _resolve_channels(data, picks)
    segments, n_dropped = _select_channel_data(
        data, names, min_span_s=2 * min_duration_s
    )
    is_epochs = isinstance(data, mne.BaseEpochs)
    rms_n = max(1, int(round(rms_window_s * sfreq)))
    merge_gap = int(round(merge_gap_s * sfreq))
    analyzed = sum(seg.data.shape[1] for seg in segments) / sfreq
    kernel = np.ones(rms_n) / rms_n
    sos = _bandpass_sos((low, high), sfreq)
    nfft = int(sp_fft.next_fast_len(int(2 ** np.ceil(np.log2(2 * sfreq)))))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sfreq)
    in_band = (freqs >= low) & (freqs <= high)

    event_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for c, name in enumerate(names):
        filt = [signal.sosfiltfilt(sos, seg.data[c]) for seg in segments]
        rms = [np.sqrt(np.convolve(f**2, kernel, mode="same")) for f in filt]
        all_rms = np.concatenate(rms)
        med = float(np.median(all_rms))
        thr = med + threshold_sd * 1.4826 * float(np.median(np.abs(all_rms - med)))
        n_events = 0
        for seg, f, r in zip(segments, filt, rms):
            above = np.concatenate(([False], r > thr, [False]))
            edges = np.flatnonzero(np.diff(above.astype(np.int8)))
            runs = [[int(s), int(t)] for s, t in zip(edges[::2], edges[1::2])]
            merged: list[list[int]] = []
            for run in runs:
                if merged and run[0] - merged[-1][1] <= merge_gap:
                    merged[-1][1] = run[1]
                else:
                    merged.append(run)
            for start, stop in merged:
                dur = (stop - start) / sfreq
                if dur < min_duration_s or dur > max_duration_s:
                    continue
                seg_f = f[start:stop]
                mid = seg_f[1:-1]
                peaks = (
                    (mid > seg_f[:-2]) & (mid >= seg_f[2:]) & (mid > thr)
                    if seg_f.size > 2
                    else np.zeros(0, dtype=bool)
                )
                if int(peaks.sum()) < min_cycles:
                    continue
                spec = np.abs(np.fft.rfft(seg_f * np.hanning(seg_f.size), n=nfft))
                spec[~in_band] = 0.0
                offset = seg.start_s
                row: dict[str, Any] = {
                    "channel": name,
                    "onset_s": offset + start / sfreq,
                    "offset_s": offset + stop / sfreq,
                    "duration_s": dur,
                    "peak_freq_hz": float(freqs[int(np.argmax(spec))]),
                    "peak_amplitude": float(np.max(np.abs(seg_f))),
                    "band_low": low,
                    "band_high": high,
                }
                if is_epochs:
                    row["epoch_index"] = seg.index
                    row["epoch_onset_s"] = seg.start_s
                    row["onset_in_epoch_s"] = start / sfreq
                event_rows.append(row)
                n_events += 1
        summary_rows.append(
            {
                "channel": name,
                "n_events": n_events,
                "rate_per_min": n_events / analyzed * 60.0,
                "analyzed_duration_s": analyzed,
                "threshold": thr,
                "band_low": low,
                "band_high": high,
            }
        )
    columns = _HFO_EVENT_COLUMNS + (_HFO_EPOCH_COLUMNS if is_epochs else [])
    events_df = pd.DataFrame(event_rows, columns=columns)
    summary_df = pd.DataFrame(summary_rows, columns=_HFO_SUMMARY_COLUMNS)
    events_df.attrs["n_segments_dropped"] = n_dropped
    summary_df.attrs["n_segments_dropped"] = n_dropped
    return events_df, summary_df


_REGION_COLUMN_NAMES = {
    "channel": "region",
    "channel_a": "region_a",
    "channel_b": "region_b",
    "phase_channel": "phase_region",
    "amp_channel": "amp_region",
}


def annotate_regions(
    df: pd.DataFrame,
    regions: Mapping[str, str],
    channel_columns: Sequence[str],
    ch_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with a region column for each channel column.

    ``regions`` maps region name to a channel-name prefix. Channels matching no
    prefix get None. Added columns: ``region`` (for ``channel``), ``region_a`` /
    ``region_b``, ``phase_region`` / ``amp_region``. If ``ch_names`` is given,
    every region must match at least one of them, else ValueError.
    """
    if ch_names is not None:
        for region, prefix in regions.items():
            if not any(str(n).startswith(str(prefix)) for n in ch_names):
                raise ValueError(
                    f"Region '{region}' with channel prefix '{prefix}' matches no "
                    "channels in the data."
                )

    def _region(channel: Any) -> str | None:
        for region, prefix in regions.items():
            if str(channel).startswith(str(prefix)):
                return str(region)
        return None

    out = df.copy()
    for col in channel_columns:
        out[_REGION_COLUMN_NAMES.get(col, f"{col}_region")] = out[col].map(_region)
    return out


def _check_data(data: Any) -> None:
    if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
        raise TypeError("data must be an MNE Raw or Epochs object")


def _validate_band(
    band: Sequence[float],
    data: mne.io.BaseRaw | mne.BaseEpochs,
    name: str = "band",
) -> tuple[float, float]:
    if len(band) != 2:
        raise ValueError(f"{name} must contain exactly two values (low, high)")
    low, high = float(band[0]), float(band[1])
    if low <= 0 or low >= high:
        raise ValueError(f"{name} must satisfy 0 < low < high, got ({low:g}, {high:g})")
    sfreq = float(data.info["sfreq"])
    nyq = sfreq / 2.0
    if high >= nyq:
        raise ValueError(
            f"{name} high edge {high:g} Hz is not below the Nyquist frequency: "
            f"data sfreq is {sfreq:g} Hz (Nyquist {nyq:g} Hz). "
            f"Raise or disable resample_step so sfreq > 2 x {high:g} Hz."
        )
    lowpass = data.info["lowpass"]
    if lowpass is not None and lowpass < nyq and high > lowpass:
        raise ValueError(
            f"{name} high edge {high:g} Hz exceeds the data low-pass: data was "
            f"low-pass filtered at {float(lowpass):g} Hz; raise the filtering "
            f"h_freq above {high:g} Hz."
        )
    return low, high


def _resolve_channels(
    data: mne.io.BaseRaw | mne.BaseEpochs, picks: _Picks
) -> list[str]:
    """Resolve picks the MNE way (None = data channels); bad channels dropped."""
    info = data.info
    if isinstance(picks, str):
        spec: Any = picks
    elif picks is None:
        spec = None
    else:
        spec = list(picks)
        if not all(isinstance(p, (str, int, np.integer)) for p in spec):
            raise TypeError(
                "picks must be None, a channel name/type, or a sequence of "
                "names/indices"
            )
        if any(isinstance(p, str) for p in spec) and not all(
            isinstance(p, str) for p in spec
        ):
            raise TypeError("picks must not mix channel names and indices")
    # Zero-length stand-in with the same info: public MNE pick() resolves the
    # picks without copying the data.
    stub = mne.io.RawArray(
        np.zeros((len(info.ch_names), 1)), info.copy(), verbose=False
    )
    try:
        stub.pick("data" if spec is None else spec, exclude="bads")
    except (ValueError, IndexError, TypeError) as exc:
        words = [spec] if isinstance(spec, str) else (spec or [])
        known = (
            set(info.ch_names)
            | set(info.get_channel_types(unique=True))
            | {"data", "all"}
        )
        missing = [p for p in words if isinstance(p, str) and p not in known]
        if missing:
            raise ValueError(f"Channels not found in data: {missing}") from exc
        raise ValueError(str(exc)) from exc
    bads = set(info["bads"])
    names = [n for n in stub.ch_names if n not in bads]
    if not names:
        raise ValueError("picks selected no good channels")
    return names


def _check_channels_exist(
    data: mne.io.BaseRaw | mne.BaseEpochs, names: Sequence[str]
) -> None:
    missing = [n for n in names if n not in data.ch_names]
    if missing:
        raise ValueError(f"Channels not found in data: {missing}")


def _check_not_bad(data: mne.io.BaseRaw | mne.BaseEpochs, names: Sequence[str]) -> None:
    bads = [n for n in names if n in data.info["bads"]]
    if bads:
        raise ValueError(f"Channels are marked bad in info['bads']: {bads}")


def _edge_samples(n_times: int, sfreq: float, low_hz: float | None = None) -> int:
    """Filter-edge samples to discard at each end.

    1 s, or 10% of the segment if shorter; never less than one cycle of
    ``low_hz`` (the band's low edge), so short spans still lose the filter
    transient of a slow band.
    """
    edge_s = min(1.0, 0.1 * n_times / sfreq)
    if low_hz:
        edge_s = max(edge_s, 1.0 / low_hz)
    return int(round(edge_s * sfreq))


def _band_key(band: tuple[float, float]) -> str:
    return f"{band[0]:g}-{band[1]:g}"


def _min_span_seconds(sfreq: float, low_hz: float, min_kept_s: float) -> float:
    """Shortest span length (s) that keeps ``min_kept_s`` after edge trimming."""
    lo, hi = 0.0, max(10.0 * (min_kept_s + 2.0 / low_hz), 10.0)
    for _ in range(50):
        mid = (lo + hi) / 2
        n = int(round(mid * sfreq))
        if n - 2 * _edge_samples(n, sfreq, low_hz) >= min_kept_s * sfreq:
            hi = mid
        else:
            lo = mid
    return max(_MIN_SEGMENT_SAMPLES / sfreq, hi)


def _usable_segments(
    segments: Sequence[_Segment], sfreq: float, low_hz: float, min_kept_s: float
) -> list[_Segment]:
    """Segments that keep at least ``min_kept_s`` after trimming for ``low_hz``."""
    need = max(1, int(np.ceil(min_kept_s * sfreq - 1e-9)))
    out = []
    for seg in segments:
        n_t = seg.data.shape[1]
        if n_t - 2 * _edge_samples(n_t, sfreq, low_hz) >= need:
            out.append(seg)
    return out


def _no_usable_data_error(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    minimums: Mapping[str, float],
    n_dropped: int,
) -> ValueError:
    mins = "; ".join(f"{k}: {v:.2f} s" for k, v in minimums.items())
    if isinstance(data, mne.BaseEpochs):
        where = "epochs shorter than the minimum are dropped"
    else:
        where = "Raw spans annotated BAD* are excluded"
    return ValueError(
        "No usable data: every segment is too short to analyze "
        f"(minimum usable length: {mins}; {n_dropped} dropped; {where})."
    )


def _stable_key(
    ph_ch: str, amp_ch: str, pb: tuple[float, float], ab: tuple[float, float]
) -> list[int]:
    """Seed words from names and band edges, independent of what else is requested."""
    text = f"{ph_ch}\x00{amp_ch}\x00{pb[0]!r},{pb[1]!r}\x00{ab[0]!r},{ab[1]!r}"
    digest = hashlib.sha256(text.encode()).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


def _bad_sample_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """Boolean mask (n_times,) of samples inside ``BAD*`` annotations.

    Mirrors ``raw.get_data(reject_by_annotation="NaN")``: description prefix
    match is case-insensitive, annotation onsets are shifted by the raw's
    ``first_time`` and converted with rounding. No data is copied.
    """
    n = len(raw.times)
    mask = np.zeros(n, bool)
    ann = raw.annotations
    idx = [i for i, d in enumerate(ann.description) if str(d).upper().startswith("BAD")]
    if not idx:
        return mask
    onsets = np.asarray(ann.onset)[idx] - raw.first_time
    ends = onsets + np.asarray(ann.duration)[idx]
    starts_i = np.clip(raw.time_as_index(onsets, use_rounding=True), 0, n)
    ends_i = np.clip(raw.time_as_index(ends, use_rounding=True), 0, n)
    for s0, e0 in zip(starts_i, ends_i):
        mask[s0:e0] = True
    return mask


def _select_channel_data(
    data: mne.io.BaseRaw | mne.BaseEpochs,
    names: Sequence[str],
    min_span_s: float = 0.0,
) -> tuple[list[_Segment], int]:
    """Return analysis segments and the number of too-short spans dropped.

    Epochs: one segment per epoch. Raw: one segment per contiguous span not
    covered by a ``BAD*`` annotation. A span must have at least
    ``min_span_s`` seconds (and the filter pad length). Band-specific edge
    trimming is applied by the caller (see :func:`_usable_segments`).
    Raises ValueError, naming the minimum, if none remain.
    """
    _check_channels_exist(data, names)
    sfreq = float(data.info["sfreq"])
    arr = np.asarray(data.get_data(picks=list(names)), dtype=float)
    if isinstance(data, mne.BaseEpochs):
        starts = data.events[:, 0] / sfreq + float(data.tmin)
        raw_spans: list[_Segment] = [
            _Segment(arr[e], float(starts[e]), e) for e in range(arr.shape[0])
        ]
        check = arr.transpose(1, 0, 2).reshape(len(names), -1)
    else:
        # BAD mask from the annotations alone (no data copy). NaNs outside BAD
        # spans are real and raise below; NaNs inside BAD spans are excluded.
        good = ~_bad_sample_mask(data)
        check = arr[:, good]
        edges = np.flatnonzero(
            np.diff(np.concatenate(([0], good.astype(np.int8), [0])))
        )
        raw_spans = [
            _Segment(arr[:, s:t], s / sfreq, None)
            for s, t in zip(edges[::2], edges[1::2])
        ]
    bad = [n for i, n in enumerate(names) if not np.all(np.isfinite(check[i]))]
    if bad:
        raise ValueError(f"Non-finite values (NaN/inf) in channels: {bad}")

    min_samples = max(_MIN_SEGMENT_SAMPLES, int(np.ceil(min_span_s * sfreq)))
    segments = [seg for seg in raw_spans if seg.data.shape[1] >= min_samples]
    n_dropped = len(raw_spans) - len(segments)
    if not segments:
        raise _no_usable_data_error(
            data, {"any analysis": min_samples / sfreq}, n_dropped
        )
    return segments, n_dropped


def _bandpass_sos(band: tuple[float, float], sfreq: float) -> np.ndarray:
    sos: np.ndarray = signal.butter(
        _FILTER_ORDER, band, btype="bandpass", fs=sfreq, output="sos"
    )
    return sos


def _analytic(x: np.ndarray, band: tuple[float, float], sfreq: float) -> np.ndarray:
    """Band-pass (zero phase) and return the analytic signal."""
    filtered = signal.sosfiltfilt(_bandpass_sos(band, sfreq), x)
    n = filtered.size
    return np.asarray(signal.hilbert(filtered, N=sp_fft.next_fast_len(n))[:n])
