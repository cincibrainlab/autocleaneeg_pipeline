"""Simple Statistical Learning EEG processing using standalone functions."""

import mne
from autoclean.functions import (
    resample_data, filter_data, rereference_data, trim_edges,
    detect_bad_channels, interpolate_bad_channels,
    detect_dense_oscillatory_artifacts, annotate_noisy_segments, annotate_uncorrelated_segments,
    fit_ica, classify_ica_components, apply_iclabel_rejection,
    create_sl_epochs, detect_outlier_epochs, gfp_clean_epochs
)

# Load data
raw = mne.io.read_raw_egi("C:/Users/Gam9LG/Documents/DATA/stat_learning/1037_slstructured.raw", preload=True)
raw.set_montage("GSN-HydroCel-129")

# Basic preprocessing
raw = resample_data(raw, sfreq=500)
raw = filter_data(raw, l_freq=0.1, h_freq=100.0, notch_freqs=[60,120], notch_widths=5)
raw = trim_edges(raw, duration=4.0)

# Fix bad channels
bad_channels = detect_bad_channels(raw)
if bad_channels:
    raw = interpolate_bad_channels(raw, bad_channels)

# Re-reference
raw = rereference_data(raw)

# Mark bad segments
raw = detect_dense_oscillatory_artifacts(raw)
raw = annotate_noisy_segments(raw)
raw = annotate_uncorrelated_segments(raw)

# ICA processing
ica = fit_ica(raw, n_components=20)
labels = classify_ica_components(raw, ica)
raw, rejected_components = apply_iclabel_rejection(
    raw, ica, labels,
    ic_flags_to_reject=["eog", "muscle", "ecg"],
    ic_rejection_threshold=0.8,
    verbose=True
)

# Create and clean epochs
epochs = create_sl_epochs(raw)
epochs = detect_outlier_epochs(epochs)
epochs = gfp_clean_epochs(epochs)

# Save results
raw.export("C:/Users/Gam9LG/Documents/DATA/stat_learning/clean_raw.fif", fmt='eeglab', overwrite=True)
epochs.export("C:/Users/Gam9LG/Documents/DATA/stat_learning/clean_epochs-epo.fif", fmt='eeglab', overwrite=True)

print(f"Done. Final epochs: {len(epochs)}")