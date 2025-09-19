"""Isolated helpers for experimenting with ICA component layout tweaks.

The production plotting helper ``plot_component_for_classification`` inside
``src/autoclean/functions/visualization/icvision_layouts.py`` still performs
all of the data munging and rendering.  The layout (figure size, GridSpec
configuration, and axis creation) is now delegated to the sandbox function
defined here so that we can iterate on spacing experiments without touching
the high-traffic visualization module.  The sandbox location keeps the helper
out of the public API while we benchmark alternative arrangements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


@dataclass
class ICALayoutSpec:
    """Container describing the figure and axes produced by the layout helper."""

    fig: plt.Figure
    ax_topo: plt.Axes
    ax_cont_data: plt.Axes
    ax_ts_scroll: plt.Axes
    ax_psd: plt.Axes
    main_title: str
    method_display: Optional[str]
    gridspec_bottom: float
    suptitle_y_pos: float


def _resolve_method_display(classification_method: Optional[str]) -> Optional[str]:
    """Normalize the classifier name for consistent labelling."""

    if not classification_method:
        return None

    method_key = classification_method.lower()
    return {
        "iclabel": "ICLabel",
        "icvision": "ICVision",
        "hybrid": "Hybrid",
    }.get(method_key, classification_method)


def build_ic_classification_layout(
    component_idx: int,
    *,
    classification_label: Optional[str] = None,
    classification_reason: Optional[str] = None,
    classification_method: Optional[str] = None,
    return_fig_object: bool = False,
) -> ICALayoutSpec:
    """Create the Matplotlib layout used by the ICA classification plots.

    Parameters
    ----------
    component_idx:
        Zero-based ICA component index for title construction.
    classification_label:
        Optional label shown in figure headers when rendering to a Figure.
    classification_reason:
        Optional rationale string.  When supplied alongside ``return_fig_object``
        a taller canvas is requested to make room for the footer textbox.
    classification_method:
        Name of the classifier (``"iclabel"``, ``"icvision"``, etc.).  The value
        is prettified for display in the figure title.
    return_fig_object:
        Whether the caller intends to keep the Figure in-memory.  This affects
        the subplot spacing so that headers and reasoning text fit comfortably.

    Returns
    -------
    ICALayoutSpec
        Dataclass containing the figure, axes, and metadata used by the caller
        to complete the rendering pipeline.
    """

    fig_height = 9.5
    gridspec_bottom = 0.05

    if return_fig_object and classification_reason:
        fig_height = 11
        gridspec_bottom = 0.18

    fig = plt.figure(figsize=(12, fig_height), dpi=120)
    method_display = _resolve_method_display(classification_method)
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

    return ICALayoutSpec(
        fig=fig,
        ax_topo=ax_topo,
        ax_cont_data=ax_cont_data,
        ax_ts_scroll=ax_ts_scroll,
        ax_psd=ax_psd,
        main_title=main_title,
        method_display=method_display,
        gridspec_bottom=gridspec_bottom,
        suptitle_y_pos=suptitle_y_pos,
    )
