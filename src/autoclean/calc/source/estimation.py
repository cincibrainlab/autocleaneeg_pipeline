"""Source estimation utilities split from the legacy monolithic module."""
from __future__ import annotations

import matplotlib
import mne
from mne.datasets import fetch_fsaverage

from autoclean.io.export import save_stc_to_file


def _switch_backend(name: str) -> str:
    """Switch the Matplotlib backend and return the previous backend."""
    previous = matplotlib.get_backend()
    try:
        matplotlib.use(name)
    except Exception:  # pragma: no cover - backend failures depend on environment
        return previous
    return previous


def estimate_source_function_raw(raw: mne.io.Raw, config: dict | None = None):
    """Perform source localization on continuous resting-state EEG data."""
    previous_backend = _switch_backend("Qt5Agg")

    raw.set_eeg_reference("average", projection=True)
    noise_cov = mne.make_ad_hoc_cov(raw.info)

    fs_dir = fetch_fsaverage()
    trans = "fsaverage"
    src = mne.read_source_spaces(f"{fs_dir}/bem/fsaverage-ico-5-src.fif")
    bem = mne.read_bem_solution(f"{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=10
    )

    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
    stc = mne.minimum_norm.apply_inverse_raw(
        raw, inv, lambda2=1.0 / 9.0, method="MNE", pick_ori="normal", verbose=True
    )

    if config is not None:
        save_stc_to_file(stc, config, stage="post_source_localization")

    _switch_backend(previous_backend or "Agg")
    return stc


def estimate_source_function_epochs(
    epochs: mne.Epochs, config: dict | None = None
):
    """Perform source localization on epoched EEG data."""
    previous_backend = _switch_backend("Qt5Agg")

    epochs.set_eeg_reference("average", projection=True)
    noise_cov = mne.make_ad_hoc_cov(epochs.info)

    fs_dir = fetch_fsaverage()
    trans = "fsaverage"
    src = mne.read_source_spaces(f"{fs_dir}/bem/fsaverage-ico-5-src.fif")
    bem = mne.read_bem_solution(f"{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif")

    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=10
    )

    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
    stc = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, lambda2=1.0 / 9.0, method="MNE", pick_ori="normal", verbose=True
    )

    if config is not None:
        if isinstance(stc, list):
            for i, stc_epoch in enumerate(stc[: min(3, len(stc))]):
                save_stc_to_file(
                    stc_epoch, config, stage=f"post_source_localization_epoch_{i}"
                )
        else:
            save_stc_to_file(stc, config, stage="post_source_localization")

    _switch_backend(previous_backend or "Agg")
    return stc


__all__ = [
    "estimate_source_function_raw",
    "estimate_source_function_epochs",
]
