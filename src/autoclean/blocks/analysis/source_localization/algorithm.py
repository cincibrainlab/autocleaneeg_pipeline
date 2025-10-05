"""Source localization algorithms - using autocleaneeg-eeg2source package.

This module maintains backward compatibility by providing the same function signatures,
but all processing is now delegated to the autocleaneeg-eeg2source PyPI package.

IMPORTANT: These functions are maintained for backward compatibility only.
New code should use the SourceLocalizationMixin.apply_source_localization() method.

References
----------
Hämäläinen MS & Ilmoniemi RJ (1994). Interpreting magnetic fields of the brain:
minimum norm estimates. Medical & Biological Engineering & Computing, 32(1), 35-42.
"""

import tempfile
import os
from pathlib import Path

import mne

try:
    from autoclean_eeg2source.core.converter import SequentialProcessor
    from autoclean_eeg2source.core.memory_manager import MemoryManager
except ImportError:
    raise ImportError(
        "autocleaneeg-eeg2source package not found. Install with:\n"
        "  pip install autocleaneeg-eeg2source"
    )


def estimate_source_function_raw(raw: mne.io.Raw, config: dict = None, save_stc: bool = False):
    """
    Perform source localization on continuous EEG data.

    NOTE: This function now uses autocleaneeg-eeg2source package and returns
    68-channel DK region data, not raw source estimates.

    Args:
        raw: MNE Raw object
        config: Configuration dictionary (optional)
        save_stc: Ignored (kept for backward compatibility)

    Returns:
        mne.io.Raw: Raw object with 68 DK atlas regions as channels
    """
    print("WARNING: Using autocleaneeg-eeg2source package for source localization")
    print("Output is 68-channel DK region data, not raw source estimates")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to temp file
        tmp_input = os.path.join(tmpdir, "temp_raw.set")
        raw.export(tmp_input, fmt='eeglab', overwrite=True)

        # Process with package
        processor = SequentialProcessor(
            memory_manager=MemoryManager(),
            montage=config.get("montage", "GSN-HydroCel-129") if config else "GSN-HydroCel-129",
            resample_freq=raw.info['sfreq'],
            lambda2=config.get("lambda2", 1.0/9.0) if config else 1.0/9.0
        )

        result = processor.process_file(tmp_input, tmpdir)

        if result['status'] != 'success':
            raise RuntimeError(f"Processing failed: {result.get('error')}")

        # Load 68-region output
        output_raw = mne.io.read_raw_eeglab(result['output_file'], preload=True)

    return output_raw


def estimate_source_function_epochs(epochs: mne.Epochs, config: dict = None):
    """
    Perform source localization on epoched EEG data.

    NOTE: This function now uses autocleaneeg-eeg2source package and returns
    68-channel DK region data, not raw source estimates.

    Args:
        epochs: MNE Epochs object
        config: Configuration dictionary (optional)

    Returns:
        mne.Epochs: Epochs object with 68 DK atlas regions as channels
    """
    print("WARNING: Using autocleaneeg-eeg2source package for source localization")
    print("Output is 68-channel DK region data, not raw source estimates")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to temp file
        tmp_input = os.path.join(tmpdir, "temp_epochs.set")
        epochs.export(tmp_input, fmt='eeglab', overwrite=True)

        # Process with package
        processor = SequentialProcessor(
            memory_manager=MemoryManager(),
            montage=config.get("montage", "GSN-HydroCel-129") if config else "GSN-HydroCel-129",
            resample_freq=epochs.info['sfreq'],
            lambda2=config.get("lambda2", 1.0/9.0) if config else 1.0/9.0
        )

        result = processor.process_file(tmp_input, tmpdir)

        if result['status'] != 'success':
            raise RuntimeError(f"Processing failed: {result.get('error')}")

        # Load 68-region output
        output_epochs = mne.read_epochs_eeglab(result['output_file'])

    return output_epochs


def convert_stc_to_eeg(*args, **kwargs):
    """
    DEPRECATED: This function is no longer needed.

    Source localization now always outputs 68-channel DK region data.
    Use estimate_source_function_raw() or estimate_source_function_epochs() directly.
    """
    raise DeprecationWarning(
        "convert_stc_to_eeg() is deprecated. "
        "Source localization now always returns 68-channel DK region data. "
        "Use estimate_source_function_raw() or estimate_source_function_epochs()."
    )


def convert_stc_list_to_eeg(*args, **kwargs):
    """
    DEPRECATED: This function is no longer needed.

    Source localization now always outputs 68-channel DK region data.
    Use estimate_source_function_epochs() directly.
    """
    raise DeprecationWarning(
        "convert_stc_list_to_eeg() is deprecated. "
        "Source localization now always returns 68-channel DK region data. "
        "Use estimate_source_function_epochs()."
    )
