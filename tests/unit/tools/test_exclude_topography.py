"""Tests for Qt Exclude clicked-timepoint mapping helpers."""

from autoclean.tools.exclude_topography import (
    epoch_sample_from_time,
    raw_sample_from_time,
    review_shortcut_action,
    should_open_topography_from_click,
)


def test_raw_sample_from_time_bounds_edge_clicks() -> None:
    assert raw_sample_from_time(-0.1, sfreq=100.0, n_samples=8) == 0
    assert raw_sample_from_time(0.025, sfreq=100.0, n_samples=8) == 2
    assert raw_sample_from_time(99.0, sfreq=100.0, n_samples=8) == 7


def test_epoch_sample_from_time_maps_epoch_edges() -> None:
    times = (-0.2, -0.1, 0.0, 0.1)

    assert epoch_sample_from_time(-0.2, epoch_times=times, n_epochs=3).sample_index == 0
    assert epoch_sample_from_time(-0.1, epoch_times=times, n_epochs=3).sample_index == 1
    assert epoch_sample_from_time(0.19, epoch_times=times, n_epochs=3).sample_index == 3
    second_epoch_start = epoch_sample_from_time(0.2, epoch_times=times, n_epochs=3)
    assert second_epoch_start.epoch_index == 1
    assert second_epoch_start.sample_index == 0
    assert epoch_sample_from_time(99.0, epoch_times=times, n_epochs=3).epoch_index == 2


def test_topography_click_gate_only_accepts_secondary_target_release() -> None:
    base_event = {
        "is_target": True,
        "is_mouse_release": True,
        "is_secondary_button": True,
        "is_control_click": False,
        "widget_width": 400,
        "widget_height": 200,
    }

    assert should_open_topography_from_click(**base_event)
    assert should_open_topography_from_click(
        **{**base_event, "is_secondary_button": False, "is_control_click": True}
    )
    assert not should_open_topography_from_click(**{**base_event, "is_target": False})
    assert not should_open_topography_from_click(
        **{**base_event, "is_mouse_release": False}
    )
    assert not should_open_topography_from_click(
        **{**base_event, "is_secondary_button": False}
    )
    assert not should_open_topography_from_click(**{**base_event, "widget_width": 100})


def test_review_shortcut_action_maps_decision_keys() -> None:
    assert (
        review_shortcut_action(
            text="p", key=0, up_key=1, down_key=2, text_input_has_focus=False
        )
        == "PASS"
    )
    assert (
        review_shortcut_action(
            text="F", key=0, up_key=1, down_key=2, text_input_has_focus=False
        )
        == "FAIL"
    )
    assert (
        review_shortcut_action(
            text="r", key=0, up_key=1, down_key=2, text_input_has_focus=False
        )
        == "REVIEW"
    )
    assert (
        review_shortcut_action(
            text="c", key=0, up_key=1, down_key=2, text_input_has_focus=False
        )
        == "UNSET"
    )


def test_review_shortcut_action_preserves_text_inputs_and_navigation() -> None:
    assert (
        review_shortcut_action(
            text="p", key=0, up_key=1, down_key=2, text_input_has_focus=True
        )
        is None
    )
    assert (
        review_shortcut_action(
            text="", key=1, up_key=1, down_key=2, text_input_has_focus=False
        )
        == "UP"
    )
    assert (
        review_shortcut_action(
            text="", key=2, up_key=1, down_key=2, text_input_has_focus=False
        )
        == "DOWN"
    )
