"""Utilities for generating synthetic EEG data for testing."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mne
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy import signal as sps


def create_test_montage_info(montage: str, n_channels: int, sfreq: float) -> mne.Info:
    """Create MNE Info object with specified montage.

    Parameters
    ----------
    montage : str
        Montage name (e.g., 'GSN-HydroCel-129', 'standard_1020')
    n_channels : int
        Number of EEG channels
    sfreq : float
        Sampling frequency in Hz

    Returns
    -------
    mne.Info
        MNE Info object with montage and channel information
    """
    # Create channel names based on montage type
    if "GSN-HydroCel" in montage:
        if "129" in montage:
            ch_names = [f"E{i}" for i in range(1, 129)] + ["Cz"]  # E1-E128 + Cz
        elif "128" in montage:
            ch_names = [f"E{i}" for i in range(1, 129)]  # E1-E128
        elif "124" in montage:
            ch_names = [f"E{i}" for i in range(1, 125)]  # E1-E124
        else:
            ch_names = [f"E{i}" for i in range(1, n_channels + 1)]
    elif "standard_1020" in montage:
        # Standard 10-20 channel names
        standard_names = [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "FC5",
            "FC1",
            "FC2",
            "FC6",
            "T7",
            "C3",
            "Cz",
            "C4",
            "T8",
            "TP9",
            "CP5",
            "CP1",
            "CP2",
            "CP6",
            "TP10",
            "P7",
            "P3",
            "Pz",
            "P4",
            "P8",
            "PO9",
            "O1",
            "Oz",
            "O2",
            "PO10",
        ]
        ch_names = standard_names[:n_channels]
    elif "MEA30" in montage:
        ch_names = [f"CH{i}" for i in range(1, n_channels + 1)]
    else:
        ch_names = [f"EEG{i:03d}" for i in range(1, n_channels + 1)]

    # Ensure we have the right number of channels
    if len(ch_names) != n_channels:
        ch_names = (
            ch_names[:n_channels]
            if len(ch_names) > n_channels
            else ch_names + [f"EEG{i:03d}" for i in range(len(ch_names), n_channels)]
        )

    # Create channel types (all EEG)
    ch_types = ["eeg"] * n_channels

    # Create info object
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Add montage if it's a standard MNE montage
    try:
        montage_obj = make_standard_montage(montage)
        info.set_montage(montage_obj, match_case=False, on_missing="ignore")
    except Exception:
        # If montage not found, continue without it
        pass

    # Note: highpass/lowpass are set automatically by MNE and cannot be set directly
    # They will be updated when filtering is applied to the Raw object

    return info


def create_synthetic_eeg_signal(
    n_channels: int, n_samples: int, sfreq: float, seed: Optional[int] = 42
) -> np.ndarray:
    """Create realistic synthetic EEG signal data.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    n_samples : int
        Number of time samples
    sfreq : float
        Sampling frequency in Hz
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Synthetic EEG data (n_channels x n_samples)
    """
    if seed is not None:
        np.random.seed(seed)

    # Create time vector
    times = np.arange(n_samples) / sfreq

    # Initialize data array
    data = np.zeros((n_channels, n_samples))

    # Add realistic EEG components
    for ch in range(n_channels):
        # Base random noise (pink noise-like)
        noise = np.random.randn(n_samples) * 10e-6  # 10 μV baseline noise

        # Add alpha rhythm (8-12 Hz) - more prominent in posterior channels
        alpha_power = 20e-6 if ch > n_channels * 0.6 else 10e-6  # Higher in posterior
        alpha_freq = np.random.uniform(8, 12)
        alpha = alpha_power * np.sin(
            2 * np.pi * alpha_freq * times + np.random.uniform(0, 2 * np.pi)
        )

        # Add beta rhythm (13-30 Hz)
        beta_power = 5e-6
        beta_freq = np.random.uniform(15, 25)
        beta = beta_power * np.sin(
            2 * np.pi * beta_freq * times + np.random.uniform(0, 2 * np.pi)
        )

        # Add theta rhythm (4-8 Hz) - more prominent in frontal channels
        theta_power = 15e-6 if ch < n_channels * 0.3 else 5e-6  # Higher in frontal
        theta_freq = np.random.uniform(4, 8)
        theta = theta_power * np.sin(
            2 * np.pi * theta_freq * times + np.random.uniform(0, 2 * np.pi)
        )

        # Add slow waves (1-4 Hz)
        slow_power = 8e-6
        slow_freq = np.random.uniform(1, 4)
        slow = slow_power * np.sin(
            2 * np.pi * slow_freq * times + np.random.uniform(0, 2 * np.pi)
        )

        # Add occasional artifacts
        # Eye blinks (if frontal channel)
        if ch < 3:  # First few channels get eye blinks
            blink_times = np.random.choice(
                n_samples, size=int(n_samples / sfreq / 5), replace=False
            )  # Every 5 seconds average
            for blink_time in blink_times:
                if blink_time + int(0.2 * sfreq) < n_samples:  # 200ms blink
                    blink_profile = (
                        np.exp(-np.linspace(0, 5, int(0.2 * sfreq)) ** 2) * 100e-6
                    )
                    noise[blink_time : blink_time + len(blink_profile)] += blink_profile

        # Muscle artifacts (if temporal channel)
        if n_channels * 0.3 < ch < n_channels * 0.7:  # Temporal channels
            muscle_times = np.random.choice(
                n_samples, size=int(n_samples / sfreq / 10), replace=False
            )
            for muscle_time in muscle_times:
                if muscle_time + int(0.1 * sfreq) < n_samples:  # 100ms muscle burst
                    muscle_profile = np.random.randn(int(0.1 * sfreq)) * 30e-6
                    noise[
                        muscle_time : muscle_time + len(muscle_profile)
                    ] += muscle_profile

        # Combine all components
        data[ch, :] = noise + alpha + beta + theta + slow

    # Add common reference signal across all channels
    common_signal = np.random.randn(n_samples) * 5e-6
    data += common_signal[np.newaxis, :]

    return data


def create_synthetic_raw(
    montage: str = "GSN-HydroCel-129",
    n_channels: int = 129,
    duration: float = 60.0,
    sfreq: float = 1000.0,
    seed: Optional[int] = 42,
) -> mne.io.Raw:
    """Create synthetic Raw object with realistic EEG data.

    Parameters
    ----------
    montage : str
        Montage name
    n_channels : int
        Number of EEG channels
    duration : float
        Duration in seconds
    sfreq : float
        Sampling frequency in Hz
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    mne.io.Raw
        Synthetic Raw object
    """
    # Calculate number of samples
    n_samples = int(duration * sfreq)

    # Create info object
    info = create_test_montage_info(montage, n_channels, sfreq)

    # Generate synthetic data
    data = create_synthetic_eeg_signal(n_channels, n_samples, sfreq, seed)

    # Create Raw object
    raw = RawArray(data, info, verbose=False)

    # Add some realistic metadata
    raw.info["description"] = f"Synthetic_{montage}_{n_channels}ch_{duration}s"

    return raw


def create_synthetic_events(
    paradigm: str, duration: float, sfreq: float, seed: Optional[int] = 42
) -> np.ndarray:
    """Create synthetic event arrays for different paradigms.

    Parameters
    ----------
    paradigm : str
        Type of paradigm ('resting', 'chirp', 'mmn', 'assr')
    duration : float
        Duration in seconds
    sfreq : float
        Sampling frequency in Hz
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Events array (n_events x 3) with [sample, prev_id, event_id]
    """
    if seed is not None:
        np.random.seed(seed)

    if paradigm == "resting":
        # No events for resting state
        return np.array([]).reshape(0, 3).astype(int)

    elif paradigm == "chirp":
        # Chirp stimuli every 2 seconds
        event_times = np.arange(2, duration - 1, 2) * sfreq  # Start at 2s, every 2s
        events = np.zeros((len(event_times), 3), dtype=int)
        events[:, 0] = event_times.astype(int)
        events[:, 2] = 1  # Chirp stimulus event ID
        return events

    elif paradigm == "mmn":
        # MMN paradigm: 80% standard (event_id=1), 20% deviant (event_id=2)
        # Inter-stimulus interval: 500ms
        event_times = (
            np.arange(1, duration - 0.5, 0.5) * sfreq
        )  # Every 500ms starting at 1s
        n_events = len(event_times)

        # Create event IDs (80% standard, 20% deviant)
        event_ids = np.ones(n_events, dtype=int)  # Start with all standard
        deviant_indices = np.random.choice(
            n_events, size=int(0.2 * n_events), replace=False
        )
        event_ids[deviant_indices] = 2  # Set deviants

        events = np.zeros((n_events, 3), dtype=int)
        events[:, 0] = event_times.astype(int)
        events[:, 2] = event_ids
        return events

    elif paradigm == "assr":
        # ASSR stimuli at 40 Hz for blocks
        # Create stimulus blocks: 2s on, 1s off
        events = []
        current_time = 1.0  # Start at 1s

        while current_time < duration - 3:
            # Start of stimulus block
            events.append([int(current_time * sfreq), 0, 1])

            # End of stimulus block (2s later)
            current_time += 2.0
            events.append([int(current_time * sfreq), 0, 2])

            # Gap between blocks (1s)
            current_time += 1.0

        return np.array(events, dtype=int)

    else:
        raise ValueError(f"Unknown paradigm: {paradigm}")


def save_synthetic_data_files(
    test_data_dir: Path, overwrite: bool = False
) -> Dict[str, Path]:
    """Generate and save synthetic data files for testing.

    Parameters
    ----------
    test_data_dir : Path
        Directory to save test data files
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping file descriptions to file paths
    """
    test_data_dir.mkdir(parents=True, exist_ok=True)
    file_paths = {}

    # Create different montage and paradigm combinations
    test_configs = [
        {
            "name": "resting_129ch",
            "montage": "GSN-HydroCel-129",
            "n_channels": 129,
            "paradigm": "resting",
        },
        {
            "name": "resting_128ch",
            "montage": "GSN-HydroCel-128",
            "n_channels": 128,
            "paradigm": "resting",
        },
        {
            "name": "chirp_129ch",
            "montage": "GSN-HydroCel-129",
            "n_channels": 129,
            "paradigm": "chirp",
        },
        {
            "name": "mmn_129ch",
            "montage": "GSN-HydroCel-129",
            "n_channels": 129,
            "paradigm": "mmn",
        },
        {
            "name": "resting_1020",
            "montage": "standard_1020",
            "n_channels": 32,
            "paradigm": "resting",
        },
        {
            "name": "bad_channels",
            "montage": "GSN-HydroCel-129",
            "n_channels": 129,
            "paradigm": "resting",
        },  # Will add many bad channels
    ]

    for config in test_configs:
        # Raw file
        raw_file = test_data_dir / f"{config['name']}_raw.fif"
        if overwrite or not raw_file.exists():
            raw = create_synthetic_raw(
                montage=config["montage"],
                n_channels=config["n_channels"],
                duration=30.0,  # Shorter for CI
                sfreq=1000.0,
            )

            # Special case: add many bad channels for testing quality control
            if config["name"] == "bad_channels":
                # Mark half the channels as bad
                raw.info["bads"] = raw.ch_names[::2]  # Every other channel

            # Save as .fif (MNE raw format)
            raw.save(raw_file, overwrite=overwrite, verbose=False)
            file_paths[f"{config['name']}_raw"] = raw_file

        # Events file
        events_file = test_data_dir / f"{config['name']}_events.eve"
        if overwrite or not events_file.exists():
            events = create_synthetic_events(config["paradigm"], 30.0, 1000.0)
            if len(events) > 0:
                mne.write_events(
                    events_file, events, overwrite=overwrite, verbose=False
                )
                file_paths[f"{config['name']}_events"] = events_file

    return file_paths


def create_corrupted_data_samples(test_data_dir: Path) -> Dict[str, Path]:
    """Create data samples with various corruption patterns for testing error handling.

    Parameters
    ----------
    test_data_dir : Path
        Directory to save corrupted test data

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping corruption types to file paths
    """
    corrupted_dir = test_data_dir / "corrupted"
    corrupted_dir.mkdir(exist_ok=True)
    file_paths = {}

    # 1. Extremely noisy data
    raw_noisy = create_synthetic_raw(duration=10.0)
    # Add extreme noise
    raw_noisy._data += np.random.randn(*raw_noisy._data.shape) * 500e-6  # 500 μV noise
    noisy_file = corrupted_dir / "extremely_noisy.fif"
    raw_noisy.save(noisy_file, overwrite=True, verbose=False)
    file_paths["extremely_noisy"] = noisy_file

    # 2. Data with many bad channels (should trigger quality flags)
    raw_bad_channels = create_synthetic_raw(duration=10.0)
    raw_bad_channels.info["bads"] = raw_bad_channels.ch_names[
        :100
    ]  # Mark most channels as bad
    bad_channels_file = corrupted_dir / "many_bad_channels.fif"
    raw_bad_channels.save(bad_channels_file, overwrite=True, verbose=False)
    file_paths["many_bad_channels"] = bad_channels_file

    # 3. Very short duration (should trigger minimum length checks)
    raw_short = create_synthetic_raw(duration=1.0)  # Only 1 second
    short_file = corrupted_dir / "too_short.fif"
    raw_short.save(short_file, overwrite=True, verbose=False)
    file_paths["too_short"] = short_file

    return file_paths


def generate_all_test_data(
    test_data_dir: Optional[Path] = None, overwrite: bool = False
) -> Dict[str, Path]:
    """Generate all synthetic test data files.

    Parameters
    ----------
    test_data_dir : Path, optional
        Directory to save test data. If None, uses tests/fixtures/data
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping file descriptions to file paths
    """
    if test_data_dir is None:
        # Default to tests/fixtures/data directory
        test_data_dir = Path(__file__).parent / "data"

    print(f"Generating synthetic test data in {test_data_dir}")

    # Generate main test data files
    file_paths = save_synthetic_data_files(test_data_dir, overwrite=overwrite)

    # Generate corrupted data samples
    corrupted_paths = create_corrupted_data_samples(test_data_dir)
    file_paths.update(corrupted_paths)

    print(f"Generated {len(file_paths)} test data files:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")

    return file_paths


# ---------------------------------------------------------------------------
# Synthetic mouse depth (LFP) recording with ground truth
# ---------------------------------------------------------------------------

MOUSE_DEPTH_CHANNELS = [
    "CTX_1",
    "CTX_2",
    "CTX_3",
    "CTX_4",
    "HPC_1",
    "HPC_2",
    "HPC_3",
    "HPC_4",
]
MOUSE_DEPTH_EFFECTS = (
    "aperiodic",
    "peaks",
    "power_coupling",
    "pac",
    "hfo",
    "connectivity",
    "common_reference",
    "line_noise",
)

# Default amplitudes (volts RMS unless noted)
_APERIODIC_RMS_V = {"CTX": 80e-6, "HPC": 100e-6}
_APERIODIC_CHI = {"CTX": 1.8, "HPC": 2.2}
_THETA_HZ = 7.5
_BETA_HZ = 20.0
# Peak band power as a multiple of the local 1/f density (x bandwidth)
_THETA_RATIO = {"HPC_1": 12.0, "HPC_2": 6.0, "HPC_3": 6.0, "HPC_4": 6.0}
_BETA_RATIO = 6.0
_GAMMA_RATIO = 6.0  # 30-45 Hz burst power vs. local 1/f power
_PAC_GAMMA_RATIO = 3.0  # 70-100 Hz power vs. local 1/f power
_PAC_DEPTH = 0.8
_CONNECTIVITY_RMS_V = 25e-6
_CONNECTIVITY_LAG_S = 0.008
_COMMON_FRACTION = 0.10
_LINE_AMP_V = 2e-6
_HFO_PER_MIN = 20.0
_HFO_FACTOR = (5.0, 8.0)


def _bandpass(x, lo, hi, sfreq, order=4):
    sos = sps.butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sps.sosfiltfilt(sos, x, axis=-1)


def _fft_bandpass(x, lo, hi, sfreq):
    """Zero-phase brick-wall band-pass in the frequency domain (circular, no edges).

    Used for very slow envelopes, where an IIR filter's edge transient would
    last seconds and show up as spurious power at the start and end.
    """
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    keep = (freqs >= lo) & (freqs <= hi)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * keep, n=n, axis=-1)


def _unit(x):
    """Scale each row to zero mean and unit standard deviation."""
    x = x - x.mean(axis=-1, keepdims=True)
    return x / x.std(axis=-1, keepdims=True)


def _shape_spectrum(white, sfreq, chi):
    """Shape white noise to 1/f^chi. Returns (unit-std noise, one-sided |H|^2, freqs)."""
    n = white.shape[-1]
    freqs = np.fft.rfftfreq(n, 1.0 / sfreq)
    h = np.maximum(freqs, 0.5) ** (-chi / 2.0)
    h[0] = 0.0
    out = np.fft.irfft(np.fft.rfft(white, axis=-1) * h, n=n, axis=-1)
    return _unit(out), h**2, freqs


def _band_density(h2, freqs, rms, f0):
    """One-sided PSD (V^2/Hz) at f0 for a unit-shaped signal with given RMS."""
    df = freqs[1] - freqs[0]
    return rms**2 * h2[np.argmin(np.abs(freqs - f0))] / (h2.sum() * df)


def _band_rms(h2, freqs, rms, lo, hi):
    mask = (freqs >= lo) & (freqs <= hi)
    return rms * np.sqrt(h2[mask].sum() / h2.sum())


def create_synthetic_mouse_depth_raw(
    duration: float = 120.0,
    sfreq: float = 2000.0,
    seed: int = 42,
    effects: Optional[Dict[str, bool]] = None,
) -> Tuple[mne.io.RawArray, Dict[str, Any]]:
    """Create a seeded synthetic mouse depth (cortex + hippocampus) LFP recording.

    Eight channels (CTX_1-4, HPC_1-4) are built from independent 1/f noise plus
    optional injected structure: spectral peaks, gamma power-envelope coupling,
    theta-gamma phase-amplitude coupling, sharp-wave ripples, a lagged shared
    source, common reference noise and line noise. All random numbers are drawn
    up front from one generator in a fixed order, so turning an effect off only
    removes that component.

    Parameters
    ----------
    duration : float
        Recording length in seconds.
    sfreq : float
        Sampling frequency in Hz.
    seed : int
        Seed for ``np.random.default_rng``.
    effects : dict of str to bool, optional
        Switches for "aperiodic", "peaks", "power_coupling", "pac", "hfo",
        "connectivity", "common_reference" and "line_noise" (all default True).

    Returns
    -------
    raw : mne.io.RawArray
        Data in volts, channel type "eeg".
    truth : dict
        JSON-serializable ground truth.
    """
    eff = {k: True for k in MOUSE_DEPTH_EFFECTS}
    if effects:
        unknown = set(effects) - set(MOUSE_DEPTH_EFFECTS)
        if unknown:
            raise ValueError(f"Unknown effects: {sorted(unknown)}")
        eff.update({k: bool(v) for k, v in effects.items()})

    ch_names = list(MOUSE_DEPTH_CHANNELS)
    n_ch = len(ch_names)
    n = int(round(duration * sfreq))
    rng = np.random.default_rng(seed)

    # ---- draw every random number up front, in a fixed order ----
    white_ap = rng.standard_normal((n_ch, n))
    white_theta = rng.standard_normal((4, n))
    white_beta = rng.standard_normal((4, n))
    white_env = rng.standard_normal((3, n))  # shared, CTX_3, HPC_4
    white_gamma = rng.standard_normal((4, n))  # CTX_1, HPC_2, CTX_3, HPC_4
    white_pac = rng.standard_normal((2, n))  # HPC_1, HPC_3
    white_conn = rng.standard_normal(n)
    white_common = rng.standard_normal(n)
    line_phase = rng.uniform(0, 2 * np.pi, n_ch)
    n_ev = max(int(round(_HFO_PER_MIN * duration / 60.0)), 0)
    ev_u = rng.uniform(0.0, 1.0, n_ev)
    ev_freq = rng.uniform(130.0, 180.0, n_ev)
    ev_dur = rng.uniform(0.05, 0.10, n_ev)
    ev_factor = rng.uniform(_HFO_FACTOR[0], _HFO_FACTOR[1], n_ev)
    ev_phase = rng.uniform(0, 2 * np.pi, n_ev)

    # ---- 1. aperiodic background (always computed; used as scale reference) ----
    chis = [_APERIODIC_CHI[c[:3]] for c in ch_names]
    rms_ap = [_APERIODIC_RMS_V[c[:3]] for c in ch_names]
    ap = np.zeros((n_ch, n))
    h2s = []
    freqs = None
    for i in range(n_ch):
        ap[i], h2, freqs = _shape_spectrum(white_ap[i], sfreq, chis[i])
        ap[i] *= rms_ap[i]
        h2s.append(h2)

    def density(i, f0):
        return _band_density(h2s[i], freqs, rms_ap[i], f0)

    data = np.zeros((n_ch, n))
    if eff["aperiodic"]:
        data += ap

    idx = {c: i for i, c in enumerate(ch_names)}
    peaks_truth = []

    # ---- 2. spectral peaks ----
    theta_src = _unit(_bandpass(white_theta, 6.5, 8.5, sfreq))
    beta_src = _unit(_bandpass(white_beta, 18.5, 21.5, sfreq))
    theta_amp = {}
    for k in range(4):
        ch = f"HPC_{k + 1}"
        i = idx[ch]
        amp = np.sqrt(_THETA_RATIO[ch] * density(i, _THETA_HZ) * 2.0)
        theta_amp[ch] = amp
        peaks_truth.append(
            {
                "channel": ch,
                "freq_hz": _THETA_HZ,
                "band": [6.5, 8.5],
                "rms_v": float(amp),
            }
        )
        if eff["peaks"]:
            data[i] += amp * theta_src[k]
    for k in range(4):
        ch = f"CTX_{k + 1}"
        i = idx[ch]
        amp = np.sqrt(_BETA_RATIO * density(i, _BETA_HZ) * 3.0)
        peaks_truth.append(
            {
                "channel": ch,
                "freq_hz": _BETA_HZ,
                "band": [18.5, 21.5],
                "rms_v": float(amp),
            }
        )
        if eff["peaks"]:
            data[i] += amp * beta_src[k]

    # ---- 3. gamma power-envelope coupling ----
    slow = _unit(_fft_bandpass(white_env, 0.05, 0.5, sfreq))
    env = np.exp(0.8 * slow)
    env /= np.sqrt(np.mean(env**2, axis=-1, keepdims=True))
    gamma = _unit(_bandpass(white_gamma, 30.0, 45.0, sfreq))
    # (channel, envelope row, carrier row)
    for ch, e_row, g_row in (
        ("CTX_1", 0, 0),
        ("HPC_2", 0, 1),
        ("CTX_3", 1, 2),
        ("HPC_4", 2, 3),
    ):
        i = idx[ch]
        amp = np.sqrt(_GAMMA_RATIO * density(i, 37.5) * 15.0)
        if eff["power_coupling"]:
            data[i] += amp * env[e_row] * gamma[g_row]

    # ---- 4. theta-gamma PAC ----
    theta_phase = np.angle(sps.hilbert(theta_src[0]))
    mod = 1.0 + _PAC_DEPTH * np.cos(theta_phase)
    mod /= np.sqrt(np.mean(mod**2))
    pac_carrier = _unit(_bandpass(white_pac, 70.0, 100.0, sfreq))
    for ch, row in (("HPC_1", 0), ("HPC_3", 1)):
        i = idx[ch]
        amp = np.sqrt(_PAC_GAMMA_RATIO * density(i, 85.0) * 30.0)
        if ch == "HPC_1" and eff["pac"]:
            data[i] += amp * mod * pac_carrier[row]
        else:
            data[i] += amp * pac_carrier[row]

    # ---- 5. sharp-wave ripples on HPC_2 ----
    events = []
    ripple = np.zeros(n)
    if n_ev > 0:
        i = idx["HPC_2"]
        bg_rms = _band_rms(h2s[i], freqs, rms_ap[i], 120.0, 250.0)
        slot = (duration - 1.0) / n_ev
        t = np.arange(n) / sfreq
        for e in range(n_ev):
            centre = 0.5 + (e + 0.1 + 0.8 * ev_u[e]) * slot
            sigma = ev_dur[e] / 4.0
            lo, hi = int((centre - 4 * sigma) * sfreq), int(
                (centre + 4 * sigma) * sfreq
            )
            seg = t[lo:hi] - centre
            ripple[lo:hi] += (
                ev_factor[e]
                * bg_rms
                * np.exp(-(seg**2) / (2 * sigma**2))
                * np.cos(2 * np.pi * ev_freq[e] * seg + ev_phase[e])
            )
            events.append(
                [
                    float(centre - 2 * sigma),
                    float(centre + 2 * sigma),
                    float(ev_freq[e]),
                ]
            )
        if eff["hfo"]:
            data[i] += ripple

    # ---- 6. lagged shared source CTX_2 -> HPC_1 ----
    lag = int(round(_CONNECTIVITY_LAG_S * sfreq))
    src = _CONNECTIVITY_RMS_V * _unit(_bandpass(white_conn, 15.0, 25.0, sfreq))
    if eff["connectivity"]:
        data[idx["CTX_2"]] += src
        data[idx["HPC_1"], lag:] += src[: n - lag]

    # ---- 7. common reference noise (1/f^1.5 shaped, ~10% of channel RMS) ----
    if eff["common_reference"]:
        common, _, _ = _shape_spectrum(white_common, sfreq, 1.5)
        for i in range(n_ch):
            data[i] += _COMMON_FRACTION * rms_ap[i] * common

    # ---- 8. line noise ----
    if eff["line_noise"]:
        t = np.arange(n) / sfreq
        for i in range(n_ch):
            data[i] += _LINE_AMP_V * np.sin(2 * np.pi * 60.0 * t + line_phase[i])

    # ---- build Raw ----
    info = create_info(ch_names, sfreq, ch_types="eeg")
    info["description"] = f"Synthetic mouse depth (LFP) recording, seed={seed}"
    raw = RawArray(data, info, verbose=False)
    if eff["hfo"] and events:
        raw.set_annotations(
            mne.Annotations(
                onset=[e[0] for e in events],
                duration=[e[1] - e[0] for e in events],
                description=["HFO_truth"] * len(events),
            )
        )

    truth: Dict[str, Any] = {
        "seed": int(seed),
        "sfreq": float(sfreq),
        "duration": float(duration),
        "effects": dict(eff),
        "regions": {"cortex": "CTX_", "hippocampus": "HPC_"},
        "aperiodic": {"exponent": {c: float(chis[i]) for i, c in enumerate(ch_names)}},
        "peaks": peaks_truth,
        "power_coupling": {
            "coupled": {
                "channel_a": "CTX_1",
                "band_a": [30, 45],
                "channel_b": "HPC_2",
                "band_b": [30, 45],
            },
            "control": {
                "channel_a": "CTX_3",
                "band_a": [30, 45],
                "channel_b": "HPC_4",
                "band_b": [30, 45],
            },
        },
        "pac": {
            "channel": "HPC_1",
            "phase_band": [6, 9],
            "amp_band": [70, 100],
            "control_channel": "HPC_3",
        },
        "hfo": {"channel": "HPC_2", "band": [120, 250], "events": events},
        "connectivity": {
            "connected": ["CTX_2", "HPC_1"],
            "band": [15, 25],
            "lag_s": _CONNECTIVITY_LAG_S,
        },
        "line_noise_hz": 60.0,
    }
    return raw, truth
