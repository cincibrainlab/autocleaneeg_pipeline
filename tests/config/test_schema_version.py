"""Schema version enforcement tests."""

from autoclean.configkit.schema import SCHEMA_VERSION, _build_task_settings_schema


def test_schema_accepts_current_version():
    schema = _build_task_settings_schema()
    minimal = {
        "schema_version": SCHEMA_VERSION,
        "montage": {"enabled": True, "value": None},
        "resample_step": {"enabled": False, "value": None},
        "filtering": {
            "enabled": False,
            "value": {
                "l_freq": None,
                "h_freq": None,
                "notch_freqs": None,
                "notch_widths": None,
            },
        },
        "drop_outerlayer": {"enabled": False, "value": None},
        "eog_step": {"enabled": False, "value": None},
        "trim_step": {"enabled": False, "value": 0},
        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
        "reference_step": {"enabled": False, "value": None},
        "ICA": {"enabled": False, "value": {"method": "fastica"}},
        "component_rejection": {
            "enabled": False,
            "method": "iclabel",
            "value": {
                "ic_flags_to_reject": [],
                "ic_rejection_threshold": 0,
            },
        },
        "epoch_settings": {
            "enabled": False,
            "value": {"tmin": None, "tmax": None},
            "event_id": None,
            "remove_baseline": {"enabled": False, "window": None},
            "threshold_rejection": {"enabled": False, "volt_threshold": {}},
        },
    }

    schema.validate(minimal)


def test_schema_rejects_mismatched_version():
    schema = _build_task_settings_schema()
    invalid = {
        "schema_version": "1900.01",
        "montage": {"enabled": True, "value": None},
        "resample_step": {"enabled": False, "value": None},
        "filtering": {
            "enabled": False,
            "value": {
                "l_freq": None,
                "h_freq": None,
                "notch_freqs": None,
                "notch_widths": None,
            },
        },
        "drop_outerlayer": {"enabled": False, "value": None},
        "eog_step": {"enabled": False, "value": None},
        "trim_step": {"enabled": False, "value": 0},
        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
        "reference_step": {"enabled": False, "value": None},
        "ICA": {"enabled": False, "value": {"method": "fastica"}},
        "component_rejection": {
            "enabled": False,
            "method": "iclabel",
            "value": {
                "ic_flags_to_reject": [],
                "ic_rejection_threshold": 0,
            },
        },
        "epoch_settings": {
            "enabled": False,
            "value": {"tmin": None, "tmax": None},
            "event_id": None,
            "remove_baseline": {"enabled": False, "window": None},
            "threshold_rejection": {"enabled": False, "volt_threshold": {}},
        },
    }

    try:
        schema.validate(invalid)
    except Exception:
        return

    raise AssertionError("schema accepted mismatched schema_version")


def test_schema_accepts_bad_channel_log_block():
    schema = _build_task_settings_schema()
    config = {
        "schema_version": SCHEMA_VERSION,
        "montage": {"enabled": True, "value": None},
        "resample_step": {"enabled": False, "value": None},
        "filtering": {
            "enabled": False,
            "value": {
                "l_freq": None,
                "h_freq": None,
                "notch_freqs": None,
                "notch_widths": None,
            },
        },
        "drop_outerlayer": {"enabled": False, "value": None},
        "bad_channel_log": {
            "enabled": True,
            "value": {
                "path": "bad_channels.csv",
                "file_column": "file",
                "channels_column": "bad_channels",
                "action": "mark",
                "strict": False,
            },
        },
        "eog_step": {"enabled": False, "value": None},
        "trim_step": {"enabled": False, "value": 0},
        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
        "reference_step": {"enabled": False, "value": None},
        "ICA": {"enabled": False, "value": {"method": "fastica"}},
        "component_rejection": {
            "enabled": False,
            "method": "iclabel",
            "value": {
                "ic_flags_to_reject": [],
                "ic_rejection_threshold": 0,
            },
        },
        "epoch_settings": {
            "enabled": False,
            "value": {"tmin": None, "tmax": None},
            "event_id": None,
            "remove_baseline": {"enabled": False, "window": None},
            "threshold_rejection": {"enabled": False, "volt_threshold": {}},
        },
    }

    schema.validate(config)
