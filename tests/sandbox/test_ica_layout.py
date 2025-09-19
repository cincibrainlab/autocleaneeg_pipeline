"""Tests for the sandbox ICA layout helper."""

import matplotlib.pyplot as plt

from autoclean.sandbox import build_ic_classification_layout


def test_build_layout_returns_expected_spec():
    layout = build_ic_classification_layout(
        2,
        classification_label="brain",
        classification_reason="Example rationale",
        classification_method="iclabel",
        return_fig_object=True,
    )

    assert layout.fig is not None
    assert layout.ax_topo.get_figure() is layout.fig
    assert layout.ax_cont_data.get_figure() is layout.fig
    assert layout.ax_ts_scroll.get_figure() is layout.fig
    assert layout.ax_psd.get_figure() is layout.fig
    assert "IC2" in layout.main_title
    assert layout.method_display == "ICLabel"
    assert layout.gridspec_bottom > 0.1  # taller canvas for the rationale box
    assert layout.suptitle_y_pos == 0.96

    plt.close(layout.fig)


def test_build_layout_saving_mode_stays_compact():
    layout = build_ic_classification_layout(
        5,
        classification_label=None,
        classification_reason="ignored unless returning fig",
        classification_method="Hybrid",
        return_fig_object=False,
    )

    assert layout.gridspec_bottom == 0.05
    assert layout.suptitle_y_pos == 0.98
    assert layout.method_display == "Hybrid"
    assert layout.main_title.endswith("[Hybrid]")

    plt.close(layout.fig)
