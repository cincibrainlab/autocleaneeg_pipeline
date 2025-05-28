"""Utilities for generating synthetic EEG data for testing."""

from pathlib import Path
from typing import Dict, Optional

import mne
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray


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

    n_samples = int(duration * sfreq)

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
