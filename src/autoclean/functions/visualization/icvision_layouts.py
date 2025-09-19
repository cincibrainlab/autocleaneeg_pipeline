"""Custom ICA plotting utilities with a faster, PDF-friendly layout.

This module adapts the ICVision visualization helpers so that AutoClean can
render ICA component plots without relying on the slower ``ICA.plot_properties``
stack.  The helpers provide a consistent layout that combines component
topography, short time-series previews, ERP-style image segments, and power
spectra.  Each helper is designed to work in batch-mode contexts (for
classification snapshots) as well as in richer PDF reports.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.gridspec import GridSpec
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
from scipy.ndimage import uniform_filter1d

# Import caching functions (optional to avoid circular imports)
try:
    from autoclean.mixins.viz._ica_sources_cache import get_cached_ica_sources
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Force non-interactive backend for batch environments
matplotlib.use("Agg", force=True)

logger = logging.getLogger(__name__)


CLASSIFICATION_COLOR_MAP: Dict[str, str] = {
    "brain": "#1b9e77",
    "muscle": "#d95f02",
    "eog": "#7570b3",
    "ecg": "#e7298a",
    "line_noise": "#66a61e",
    "ch_noise": "#e6ab02",
    "other": "#a6761d",
}


def _resolve_component_index(component_idx: int, ica_obj: ICA) -> int:
    """Normalize the requested component index and validate the bounds."""

    if component_idx < 0:
        raise ValueError("component_idx must be non-negative")

    if ica_obj.n_components_ is not None and component_idx >= ica_obj.n_components_:
        raise ValueError(
            f"component_idx={component_idx} is outside the fitted ICA range "
            f"(n_components_={ica_obj.n_components_})"
        )

    return component_idx


def plot_component_for_classification(
    ica_obj: ICA,
    raw_obj: mne.io.Raw,
    component_idx: int,
    output_dir: Optional[Path] = None,
    *,
    classification_label: Optional[str] = None,
    classification_confidence: Optional[float] = None,
    classification_reason: Optional[str] = None,
    classification_method: Optional[str] = None,
    raw_full: Optional[mne.io.Raw] = None,
    return_fig_object: bool = False,
    source_filename: Optional[str] = None,
    psd_fmax: Optional[float] = None,
) -> Union[Path, plt.Figure, None]:
    """Render the custom layout for a single ICA component.

    Parameters
    ----------
    ica_obj
        The fitted :class:`~mne.preprocessing.ICA` instance.
    raw_obj
        The :class:`~mne.io.Raw` object that supplied the data for ICA.
    component_idx
        Zero-based index of the component to visualize.
    output_dir
        Directory where the ``.webp`` snapshot will be written when
        ``return_fig_object`` is ``False``.  The directory is required in the
        saving mode.
    classification_label
        Optional label (``"brain"``, ``"eog"``, etc.) for annotating the
        figure headers when building PDF reports.
    classification_confidence
        Optional confidence score paired with ``classification_label``.
    classification_reason
        Optional textual rationale to include in PDF reports.
    classification_method
        Optional classifier identifier (e.g., ``"iclabel"``, ``"icvision"``, ``"hybrid"``)
        to embed in figure titles.
    raw_full
        Optional full-duration raw object used strictly for PSD estimation when
        the visualized snippet (``raw_obj``) is cropped.
    return_fig_object
        When ``True`` the matplotlib :class:`~matplotlib.figure.Figure` is
        returned directly instead of saving to disk.
    source_filename
        Optional filename footer used in PDF mode.
    psd_fmax
        Optional upper frequency bound for the power spectrum plot.  The value
        is clamped to the Nyquist frequency of the data.

    Returns
    -------
    Path | matplotlib.figure.Figure | None
        Depending on ``return_fig_object`` either a path to the saved file or a
        figure object is returned.  ``None`` signals that plotting failed and a
        placeholder should be used instead.
    """

    component_idx = _resolve_component_index(component_idx, ica_obj)

    if not return_fig_object and output_dir is None:
        raise ValueError("output_dir must be provided when saving figures to disk")

    # Always start from a clean canvas to keep batch jobs deterministic
    plt.close("all")

    fig_height = 9.5
    gridspec_bottom = 0.05

    if return_fig_object and classification_reason:
        fig_height = 11
        gridspec_bottom = 0.18

    fig = plt.figure(figsize=(12, fig_height), dpi=120)
    method_display = None
    if classification_method:
        method_key = classification_method.lower()
        method_display = {
            "iclabel": "ICLabel",
            "icvision": "ICVision",
            "hybrid": "Hybrid",
        }.get(method_key, classification_method)
    method_suffix = f" [{method_display}]" if method_display else ""
    main_title = f"ICA Component IC{component_idx} Analysis{method_suffix}"
    gridspec_top = 0.95
    suptitle_y_pos = 0.98

    if return_fig_object and classification_label is not None:
        gridspec_top = 0.90
        suptitle_y_pos = 0.96

    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[0.915, 0.572, 2.213],
        width_ratios=[0.9, 1.0],
        hspace=0.7,
        wspace=0.35,
        left=0.05,
        right=0.95,
        top=gridspec_top,
        bottom=gridspec_bottom,
    )

    ax_topo = fig.add_subplot(gs[0:2, 0])
    ax_cont_data = fig.add_subplot(gs[2, 0])
    ax_ts_scroll = fig.add_subplot(gs[0, 1])
    ax_psd = fig.add_subplot(gs[2, 1])

    try:
        # Use cached sources for better performance
        if CACHE_AVAILABLE:
            sources = get_cached_ica_sources(ica_obj, raw_obj)
        else:
            sources = ica_obj.get_sources(raw_obj)
            
        sfreq = sources.info["sfreq"]
        component_data = sources.get_data(picks=[component_idx])
        component_data = np.asarray(component_data)
        if component_data.size == 0:
            raise ValueError("empty component data")
        component_data_array = component_data[0]
    except Exception as exc:  # pragma: no cover - safety net for production runs
        logger.error("Failed to pull ICA sources for IC%s: %s", component_idx, exc)
        plt.close(fig)
        return None

    psd_data = component_data_array
    psd_sfreq = sfreq
    if raw_full is not None:
        try:
            # Use cached sources for full-duration data as well
            if CACHE_AVAILABLE:
                sources_full = get_cached_ica_sources(ica_obj, raw_full)
            else:
                sources_full = ica_obj.get_sources(raw_full)
                
            psd_sfreq = sources_full.info["sfreq"]
            component_data_full = sources_full.get_data(picks=[component_idx])
            component_data_full = np.asarray(component_data_full)
            if component_data_full.size:
                psd_data = component_data_full[0]
            else:
                psd_sfreq = sfreq
        except Exception as exc:  # pragma: no cover - PSD fallback
            logger.warning(
                "Falling back to cropped data for IC%s PSD due to: %s",
                component_idx,
                exc,
            )
            psd_data = component_data_array
            psd_sfreq = sfreq

    # --- Topography -----------------------------------------------------
    try:
        ica_obj.plot_components(
            picks=component_idx,
            axes=ax_topo,
            ch_type="eeg",
            show=False,
            colorbar=False,
            cmap="jet",
            outlines="head",
            sensors=True,
            contours=6,
        )
        ax_topo.set_title(f"IC{component_idx} Topography", fontsize=12)
        ax_topo.set_xlabel("")
        ax_topo.set_ylabel("")
        ax_topo.set_xticks([])
        ax_topo.set_yticks([])
    except Exception as exc:  # pragma: no cover - defensive path
        logger.error("Topography plotting failed for IC%s: %s", component_idx, exc)
        ax_topo.text(0.5, 0.5, "Topography plot failed", ha="center", va="center")

    # --- Short scrolling trace -----------------------------------------
    try:
        duration_segment_ts = 2.5
        max_samples_ts = min(
            int(duration_segment_ts * sfreq), len(component_data_array)
        )
        times_ts_ms = (np.arange(max_samples_ts) / sfreq) * 1000

        ax_ts_scroll.plot(
            times_ts_ms,
            component_data_array[:max_samples_ts],
            linewidth=0.8,
            color="dodgerblue",
        )
        ax_ts_scroll.set_title("Scrolling IC Activity (First 2.5s)", fontsize=10)
        ax_ts_scroll.set_xlabel("Time (ms)", fontsize=9)
        ax_ts_scroll.set_ylabel("Amplitude (a.u.)", fontsize=9)
        if max_samples_ts > 0 and times_ts_ms.size > 0:
            ax_ts_scroll.set_xlim(times_ts_ms[0], times_ts_ms[-1])
        ax_ts_scroll.grid(True, linestyle=":", alpha=0.6)
        ax_ts_scroll.tick_params(axis="both", which="major", labelsize=8)
    except Exception as exc:  # pragma: no cover
        logger.error("Time-series plotting failed for IC%s: %s", component_idx, exc)
        ax_ts_scroll.text(0.5, 0.5, "Time series plot failed", ha="center", va="center")

    # --- ERP-style image ------------------------------------------------
    try:
        continuous_data_array = psd_data
        continuous_sfreq = psd_sfreq
        comp_data_centered = continuous_data_array - np.mean(continuous_data_array)
        target_segment_duration_s = 1.5
        target_max_segments = 200
        segment_len_samples = int(target_segment_duration_s * continuous_sfreq) or 1

        available_samples = comp_data_centered.shape[0]
        segment_sfreq = continuous_sfreq
        max_total_samples = int(target_max_segments * segment_len_samples)
        samples_to_use = min(available_samples, max_total_samples)

        if segment_len_samples > 0 and samples_to_use >= segment_len_samples:
            n_segments = math.floor(samples_to_use / segment_len_samples)
            final_samples = n_segments * segment_len_samples
            erp_image_data = comp_data_centered[:final_samples].reshape(
                n_segments, segment_len_samples
            )
        elif samples_to_use > 0:
            n_segments = 1
            segment_len_samples = samples_to_use
            erp_image_data = comp_data_centered[:segment_len_samples].reshape(
                1, segment_len_samples
            )
        else:
            n_segments = 0
            segment_len_samples = 1
            erp_image_data = np.zeros((1, 1))

        if n_segments >= 3 and erp_image_data.shape[0] >= 3:
            erp_image_smoothed = uniform_filter1d(
                erp_image_data, size=3, axis=0, mode="nearest"
            )
        else:
            erp_image_smoothed = erp_image_data

        if erp_image_smoothed.size > 0:
            max_abs_val = float(np.max(np.abs(erp_image_smoothed)))
            clim_val = (2.0 / 3.0) * max_abs_val if max_abs_val > 1e-9 else 1.0
        else:
            clim_val = 1.0
        clim_val = max(clim_val, 1e-9)

        im = ax_cont_data.imshow(
            erp_image_smoothed,
            aspect="auto",
            cmap="jet",
            interpolation="nearest",
            vmin=-clim_val,
            vmax=clim_val,
        )

        ax_cont_data.set_title(
            f"Continuous Data Segments (Max {target_max_segments})", fontsize=10
        )
        ax_cont_data.set_xlabel("Time (ms)", fontsize=9)
        if segment_len_samples > 1:
            num_xticks = min(4, segment_len_samples)
            xtick_positions = np.linspace(0, segment_len_samples - 1, num_xticks)
            xtick_labels = (xtick_positions / segment_sfreq * 1000).astype(int)
            ax_cont_data.set_xticks(xtick_positions)
            ax_cont_data.set_xticklabels(xtick_labels)
        else:
            ax_cont_data.set_xticks([])

        ax_cont_data.set_ylabel("Trials (Segments)", fontsize=9)
        if n_segments > 1:
            num_yticks = min(5, n_segments)
            ytick_positions = np.linspace(0, n_segments - 1, num_yticks).astype(int)
            ax_cont_data.set_yticks(ytick_positions)
            ax_cont_data.set_yticklabels(ytick_positions)
        elif n_segments == 1:
            ax_cont_data.set_yticks([0])
            ax_cont_data.set_yticklabels(["0"])
        else:
            ax_cont_data.set_yticks([])

        if n_segments > 0:
            ax_cont_data.invert_yaxis()

        cbar = fig.colorbar(
            im, ax=ax_cont_data, orientation="vertical", fraction=0.046, pad=0.1
        )
        cbar.set_label("Activation (a.u.)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    except Exception as exc:  # pragma: no cover
        logger.error("Continuous data plotting failed for IC%s: %s", component_idx, exc)
        ax_cont_data.text(
            0.5, 0.5, "Continuous data plot failed", ha="center", va="center"
        )

    # --- Power spectral density ----------------------------------------
    try:
        fmin_psd = 1.0
        nyquist = psd_sfreq / 2.0
        if psd_fmax is not None:
            fmax_psd = min(psd_fmax, nyquist - 0.51)
        else:
            fmax_psd = min(80.0, nyquist - 0.51)

        n_fft_psd = int(psd_sfreq * 2.0)
        n_fft_psd = min(n_fft_psd, len(psd_data))
        if len(psd_data) >= 256:
            n_fft_psd = max(n_fft_psd, 256)
        elif len(psd_data) > 0:
            n_fft_psd = max(n_fft_psd, len(psd_data))
        else:
            n_fft_psd = 1

        if n_fft_psd <= 0 or fmax_psd <= fmin_psd:
            raise ValueError("Invalid parameters for PSD computation")

        psds, freqs = psd_array_welch(
            psd_data,
            sfreq=psd_sfreq,
            fmin=fmin_psd,
            fmax=fmax_psd,
            n_fft=n_fft_psd,
            n_overlap=int(n_fft_psd * 0.5),
            verbose=False,
            average="mean",
        )
        if psds.size == 0:
            raise ValueError("PSD computation returned empty array")

        psds_db = 10 * np.log10(np.maximum(psds, 1e-20))
        ax_psd.plot(freqs, psds_db, color="red", linewidth=1.2)
        ax_psd.set_title(
            f"IC{component_idx} Power Spectrum (1-{int(fmax_psd)}Hz)", fontsize=10
        )
        ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_psd.set_ylabel("Power (dB)", fontsize=9)
        if len(freqs) > 0:
            ax_psd.set_xlim(freqs[0], freqs[-1])
        ax_psd.grid(True, linestyle="--", alpha=0.5)
        ax_psd.tick_params(axis="both", which="major", labelsize=8)
    except Exception as exc:  # pragma: no cover
        logger.error("PSD plotting failed for IC%s: %s", component_idx, exc)
        ax_psd.text(0.5, 0.5, "PSD plot failed", ha="center", va="center")

    if return_fig_object:
        if classification_label is not None and classification_confidence is not None:
            subtitle_color = CLASSIFICATION_COLOR_MAP.get(
                classification_label.lower(), "black"
            )
            label_prefix = (
                f"{method_display} Classification"
                if method_display
                else "Classification"
            )
            fig.text(
                0.05,
                suptitle_y_pos,
                main_title,
                ha="left",
                va="top",
                fontsize=14,
                fontweight="bold",
                transform=fig.transFigure,
            )
            fig.text(
                0.95,
                suptitle_y_pos,
                (
                    f"{label_prefix}: {classification_label.title()} "
                    f"(Confidence: {classification_confidence:.2f})"
                ),
                ha="right",
                va="top",
                fontsize=13,
                fontweight="bold",
                color=subtitle_color,
                transform=fig.transFigure,
            )
        else:
            fig.suptitle(main_title, fontsize=14, y=suptitle_y_pos)

        if classification_reason:
            fig.text(
                0.05,
                gridspec_bottom - 0.03,
                f"Rationale: {classification_reason}",
                ha="left",
                va="top",
                fontsize=8,
                wrap=True,
                transform=fig.transFigure,
                bbox=dict(
                    boxstyle="round,pad=0.4", fc="aliceblue", alpha=0.75, ec="lightgrey"
                ),
            )

        if source_filename:
            fig.text(
                0.5,
                0.01,
                (
                    "AutocleanEEG Pipeline | https://github.com/cincibrainlab/autoclean_pipeline "
                    f"| Source: {source_filename}"
                ),
                ha="center",
                va="bottom",
                fontsize=8,
                style="italic",
                color="gray",
                transform=fig.transFigure,
            )

        return fig

    # Saving mode
    assert output_dir is not None  # appease type-checkers
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"component_IC{component_idx}_vision_analysis.webp"
    filepath = output_dir / filename
    try:
        fig.subplots_adjust(
            left=0.05, right=0.95, bottom=0.05, top=0.93, hspace=0.7, wspace=0.35
        )
        plt.savefig(filepath, format="webp", bbox_inches="tight", pad_inches=0.1)
        logger.debug("Saved component plot for IC%s to %s", component_idx, filepath)
    except Exception as exc:  # pragma: no cover
        logger.error("Saving figure failed for IC%s: %s", component_idx, exc)
        plt.close(fig)
        return None
    finally:
        plt.close(fig)

    return filepath


def plot_components_batch(
    ica_obj: ICA,
    raw_obj: mne.io.Raw,
    component_indices: Iterable[int],
    output_dir: Path,
    *,
    batch_size: int = 1,
    psd_fmax: Optional[float] = None,
) -> Dict[int, Optional[Path]]:
    """Generate component plots sequentially with periodic cleanup."""

    component_indices = list(component_indices)
    if not component_indices:
        logger.info("No ICA components provided for plotting.")
        return {}

    start_time = time.time()
    logger.info(
        "Starting ICA component plotting for %d components (batch_size=%d)",
        len(component_indices),
        batch_size,
    )

    results: Dict[int, Optional[Path]] = {}

    for counter, component_idx in enumerate(component_indices, start=1):
        try:
            plt.close("all")
            image_path = plot_component_for_classification(
                ica_obj,
                raw_obj,
                component_idx,
                output_dir,
                return_fig_object=False,
                psd_fmax=psd_fmax,
            )
            results[component_idx] = image_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to render IC%s: %s", component_idx, exc)
            results[component_idx] = None
        finally:
            if counter % max(1, batch_size) == 0:
                plt.close("all")

    elapsed = time.time() - start_time
    logger.info(
        "Completed ICA plotting: %d/%d successful in %.2fs (%.2fs/component)",
        sum(path is not None for path in results.values()),
        len(component_indices),
        elapsed,
        (elapsed / len(component_indices)) if component_indices else 0.0,
    )

    return results


def save_ica_data(
    ica_obj: ICA,
    output_dir: Path,
    *,
    input_basename: Optional[str] = None,
    filename_prefix: Optional[str] = None,
) -> Path:
    """Persist the ICA decomposition to disk for downstream review."""

    if filename_prefix is None:
        if input_basename is None:
            filename_prefix = "icvision_classified"
        else:
            filename_prefix = f"{input_basename}_icvision_classified"

    output_path = Path(output_dir) / f"{filename_prefix}_ica.fif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ica_obj.save(output_path, overwrite=True)
        logger.info("Saved ICA object to %s", output_path)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to save ICA object: %s", exc)
        raise RuntimeError(f"Failed to save ICA object: {exc}") from exc

    return output_path


def plot_ica_topographies_overview(
    ica_obj: ICA,
    indices_to_plot: Optional[Iterable[int]] = None,
    *,
    max_plots_per_fig: int = 25,
) -> List[plt.Figure]:
    """Create overview figures showing batches of ICA component topographies."""

    if indices_to_plot is None:
        indices = list(range(ica_obj.n_components_ or 0))
    else:
        indices = list(indices_to_plot)

    if not indices:
        logger.info("No ICA components available for the topography overview.")
        return []

    figures: List[plt.Figure] = []

    for start in range(0, len(indices), max_plots_per_fig):
        batch = indices[start : start + max_plots_per_fig]
        if not batch:
            continue

        n_batch = len(batch)
        ncols = max(1, math.ceil(math.sqrt(n_batch / 1.5)))
        nrows = math.ceil(n_batch / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(min(ncols * 2.5, 14), min(nrows * 2.5, 18)),
            squeeze=False,
        )
        fig.suptitle(
            f"ICA Topographies Overview (Batch {start // max_plots_per_fig + 1})",
            fontsize=14,
        )

        for ax_idx, comp_idx in enumerate(batch):
            row, col = divmod(ax_idx, ncols)
            ax = axes[row, col]
            try:
                ica_obj.plot_components(
                    picks=comp_idx,
                    axes=ax,
                    show=False,
                    colorbar=False,
                    cmap="jet",
                    outlines="head",
                    sensors=False,
                    contours=4,
                )
                ax.set_title(f"IC{comp_idx}", fontsize=9)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to plot topography for IC%s: %s", comp_idx, exc)
                ax.text(0.5, 0.5, "Error", ha="center", va="center")
                ax.set_title(f"IC{comp_idx} (Err)", fontsize=9)

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])

        total_axes = nrows * ncols
        for idx in range(n_batch, total_axes):
            row, col = divmod(idx, ncols)
            fig.delaxes(axes[row, col])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        figures.append(fig)

    return figures
