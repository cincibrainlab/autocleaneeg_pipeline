"""AutoReject cleaning helpers packaged as a plugin block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, TYPE_CHECKING

import mne

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    import autoreject


class RawBuilder(Protocol):
    """Build an epochs object from raw EEG data."""

    def __call__(self, raw: mne.io.BaseRaw) -> mne.BaseEpochs:
        ...


@dataclass
class AutorejectResult:
    """Container bundling the AutoReject output."""

    epochs: mne.BaseEpochs
    reject_log: "autoreject.RejectLog"


def _import_autoreject() -> "autoreject":
    """Import AutoReject with a helpful error message if it is missing."""

    try:
        import autoreject
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The 'autoreject' package is required for this plugin. "
            "Install it with `pip install autoreject`."
        ) from exc
    return autoreject


def _default_epoch_builder(raw: mne.io.BaseRaw) -> mne.BaseEpochs:
    """Create 1-second non-overlapping epochs from a Raw recording."""

    return mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.0, preload=True)


def run_autoreject(epochs: mne.BaseEpochs, **kwargs) -> AutorejectResult:
    """Apply AutoReject to an ``mne.Epochs`` object."""

    autoreject = _import_autoreject()
    ar = autoreject.AutoReject(random_state=11, n_jobs=30, verbose=True, **kwargs)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    return AutorejectResult(epochs=epochs_ar, reject_log=reject_log)


def run_autoreject_raw(
    raw: mne.io.BaseRaw,
    epoch_builder: Optional[Callable[[mne.io.BaseRaw], mne.BaseEpochs]] = None,
    **autoreject_kwargs,
) -> AutorejectResult:
    """Build epochs from ``raw`` and run AutoReject using :func:`run_autoreject`."""

    builder = epoch_builder or _default_epoch_builder
    epochs = builder(raw)
    return run_autoreject(epochs, **autoreject_kwargs)


__all__ = [
    "AutorejectResult",
    "RawBuilder",
    "run_autoreject",
    "run_autoreject_raw",
]
