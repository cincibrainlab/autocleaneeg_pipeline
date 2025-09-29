"""GFP-based epoch cleaning helpers packaged for plugin reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import mne
import numpy as np
import pandas as pd

DEFAULT_NON_SCALP_CHANNELS: Sequence[str] = (
    "E17",
    "E38",
    "E43",
    "E44",
    "E48",
    "E49",
    "E56",
    "E73",
    "E81",
    "E88",
    "E94",
    "E107",
    "E113",
    "E114",
    "E119",
    "E120",
    "E121",
    "E125",
    "E126",
    "E127",
    "E128",
)


@dataclass
class GFPCleaningResult:
    """Bundle the cleaned epochs and summary statistics."""

    epochs: mne.BaseEpochs
    stats: pd.DataFrame
    cleaned_stats: pd.DataFrame
    removed_count: int
    scalp_channels: list[str]
    requested_epochs_exceeded: bool


def clean_epochs_by_gfp(
    epochs: mne.BaseEpochs,
    *,
    gfp_threshold: float = 3.0,
    number_of_epochs: Optional[int] = None,
    random_seed: Optional[int] = None,
    non_scalp_channels: Sequence[str] = DEFAULT_NON_SCALP_CHANNELS,
) -> GFPCleaningResult:
    """Return a copy of ``epochs`` with GFP outliers removed."""

    if not isinstance(epochs, mne.BaseEpochs):
        raise TypeError("GFP cleaning requires an MNE Epochs object")

    if number_of_epochs is not None and number_of_epochs < 0:
        raise ValueError("number_of_epochs must be non-negative when provided")

    if not epochs.preload:
        epochs.load_data()

    epochs_clean = epochs.copy()

    scalp_channels = [
        ch for ch in epochs_clean.ch_names if ch not in set(non_scalp_channels)
    ]
    if not scalp_channels:
        raise ValueError("No scalp channels available for GFP calculation")

    scalp_indices = [epochs_clean.ch_names.index(ch) for ch in scalp_channels]

    data = epochs_clean.get_data()[:, scalp_indices, :]
    gfp = np.sqrt(np.mean(data**2, axis=(1, 2)))

    stats = pd.DataFrame(
        {
            "epoch": np.arange(len(gfp)),
            "gfp": gfp,
            "mean_amplitude": data.mean(axis=(1, 2)),
            "max_amplitude": data.max(axis=(1, 2)),
            "min_amplitude": data.min(axis=(1, 2)),
            "std_amplitude": data.std(axis=(1, 2)),
        }
    )

    gfp_mean = float(stats["gfp"].mean())
    gfp_std = float(stats["gfp"].std())
    if np.isclose(gfp_std, 0.0):
        z_scores = np.zeros_like(stats["gfp"].to_numpy())
    else:
        z_scores = np.abs((stats["gfp"].to_numpy() - gfp_mean) / gfp_std)

    good_mask = z_scores < gfp_threshold
    removed_count = int((~good_mask).sum())

    cleaned_epochs = epochs_clean[good_mask]
    cleaned_stats = stats.loc[good_mask].reset_index(drop=True)

    requested_epochs_exceeded = False
    if number_of_epochs is not None and len(cleaned_epochs) > 0:
        if len(cleaned_epochs) < number_of_epochs:
            requested_epochs_exceeded = True
            number_of_epochs = len(cleaned_epochs)

        if number_of_epochs:
            rng = np.random.default_rng(random_seed)
            selected_indices = np.sort(
                rng.choice(len(cleaned_epochs), size=number_of_epochs, replace=False)
            )
            cleaned_epochs = cleaned_epochs[selected_indices]
            cleaned_stats = cleaned_stats.iloc[selected_indices].reset_index(drop=True)

    return GFPCleaningResult(
        epochs=cleaned_epochs,
        stats=stats,
        cleaned_stats=cleaned_stats,
        removed_count=removed_count,
        scalp_channels=list(scalp_channels),
        requested_epochs_exceeded=requested_epochs_exceeded,
    )


def render_gfp_plots(
    derivatives_path,
    stats: pd.DataFrame,
    cleaned_stats: pd.DataFrame,
) -> None:
    """Render GFP bar and heatmap plots next to the derivatives directory."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional visualization dependency
        raise ImportError(
            "matplotlib is required to render GFP summary plots. "
            "Install it with `pip install matplotlib`."
        ) from exc

    plot_fname = derivatives_path.copy().update(
        suffix="gfp", extension="png", check=False
    )
    plt.figure(figsize=(12, 4))
    plt.bar(stats.index, stats["gfp"], width=0.8, color="red", alpha=0.3)
    plt.bar(cleaned_stats.index, cleaned_stats["gfp"], width=0.8, color="blue")
    plt.xlabel("Epoch Number")
    plt.ylabel("Global Field Power (GFP)")
    plt.title("GFP Values by Epoch (Red = Removed, Blue = Kept)")
    plt.savefig(plot_fname, dpi=150, bbox_inches="tight")
    plt.close()

    heatmap_fname = derivatives_path.copy().update(
        suffix="gfp-heatmap", extension="png", check=False
    )
    plt.figure(figsize=(30, 18))
    n_epochs = len(stats)
    n_cols = 8
    n_rows = int(np.ceil(n_epochs / n_cols))
    grid = np.full((n_rows, n_cols), np.nan)
    for idx, (_, gfp_value) in enumerate(stats["gfp"].items()):
        row, col = divmod(idx, n_cols)
        grid[row, col] = gfp_value

    im = plt.imshow(grid, cmap="RdYlBu_r", aspect="auto")
    plt.colorbar(im, label="GFP Value (×10⁻⁶)", fraction=0.02, pad=0.04)
    for idx, (epoch_id, gfp_value) in enumerate(stats["gfp"].items()):
        row, col = divmod(idx, n_cols)
        kept = epoch_id in cleaned_stats.index
        color = "black" if kept else "red"
        plt.text(
            col,
            row,
            f"ID: {epoch_id}\nGFP: {gfp_value:.1e}",
            ha="center",
            va="center",
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8, pad=0.8),
        )

    plt.title("GFP Heatmap by Epoch (Red = Removed, Black = Kept)", fontsize=14, pad=20)
    plt.xlabel("Column", fontsize=12, labelpad=10)
    plt.ylabel("Row", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(heatmap_fname, dpi=300, bbox_inches="tight")
    plt.close()


__all__ = [
    "DEFAULT_NON_SCALP_CHANNELS",
    "GFPCleaningResult",
    "clean_epochs_by_gfp",
    "render_gfp_plots",
]
