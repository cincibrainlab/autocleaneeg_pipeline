"""ASSR task configuration schema tests."""

from copy import deepcopy

import pytest
from schema import SchemaError

from autoclean.configkit.schema import validate_task_module_config
from autoclean.data.builtins.tasks.auditory.ASSR_40Hz import config as assr_config


def test_schema_accepts_assr_analysis_overrides() -> None:
    task_config = deepcopy(assr_config)
    task_config["assr_analysis"] = {
        "enabled": True,
        "value": {
            "profile": "assr_epochs",
            "baseline": (-0.2, 0.0),
            "time_windows": {"response": [0.0, 0.5]},
            "freq_bands": {"assr": [39.0, 41.0]},
            "combined_bands": {"broad_assr": [[30.0, 45.0], [65.0, 80.0]]},
            "exclude_channel_types": ("eog",),
            "save_tfr": True,
        },
    }

    validated = validate_task_module_config(task_config)

    assert validated["assr_analysis"] == task_config["assr_analysis"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("profile", "unknown_profile"),
        ("save_tfr", "yes"),
    ],
)
def test_schema_rejects_invalid_assr_analysis_values(key: str, value: object) -> None:
    task_config = deepcopy(assr_config)
    task_config["assr_analysis"] = {
        "enabled": True,
        "value": {
            "profile": "assr_epochs",
            "save_tfr": False,
            key: value,
        },
    }

    with pytest.raises(SchemaError):
        validate_task_module_config(task_config)
