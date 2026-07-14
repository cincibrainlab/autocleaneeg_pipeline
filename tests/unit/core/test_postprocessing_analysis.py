from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from schema import SchemaError

from autoclean.configkit.schema import (
    SCHEMA_VERSION,
    export_task_schema_layout,
    validate_task_module_config,
)
from autoclean.core.task import Task


def _minimal_postprocessing_schema_config() -> dict:
    return {
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


class _PostprocessingTask(Task):
    def __init__(self, config):
        self.settings = {}
        super().__init__(config)
        self.epochs = object()
        self.raw = object()
        self.calls = []

    def run(self):
        pass

    def apply_sensor_psd(self, **kwargs):
        enabled, _ = self._check_step_enabled("apply_sensor_psd")
        self.calls.append(("sensor_psd", kwargs, enabled))
        if not enabled:
            return None, None, {}
        psd_df = pd.DataFrame(
            {
                "channel": ["Fz", "Fz", "Fz"],
                "frequency": [2.0, 4.0, 8.0],
                "psd": [10.0, 5.0, 2.5],
            }
        )
        return psd_df, [2], {"sensor_psd_bands": "sensor_psd/test_bands.csv"}

    def apply_source_localization(self, **kwargs):
        enabled, _ = self._check_step_enabled("apply_source_localization")
        self.calls.append(("source_localization", kwargs, enabled))
        if not enabled:
            return None
        return "source_epochs"

    def apply_source_psd(self, **kwargs):
        enabled, _ = self._check_step_enabled("apply_source_psd")
        self.calls.append(("source_psd", kwargs, enabled))
        if not enabled:
            return None, None
        return [1, 2, 3], "source_psd/test.parquet"


@pytest.fixture
def task(tmp_path: Path):
    config = {
        "run_id": "run",
        "unprocessed_file": tmp_path / "subject.set",
        "task": "ExampleTask",
        "reports_dir": tmp_path / "reports",
    }
    return _PostprocessingTask(config)


def test_postprocessing_analysis_runs_enabled_blocks_in_documented_order(task):
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "source_psd": {"enabled": True, "input": "source_epochs"},
                "sensor_psd": {
                    "enabled": True,
                    "input": "clean_epochs",
                    "freq_bands": {"alpha": [8, 13], "gamma": None},
                },
                "source_localization": {"enabled": True, "input": "clean_epochs"},
            },
        }
    }

    with patch.object(task, "_update_metadata") as update_metadata:
        results = task.run_postprocessing_analysis()

    assert [call[0] for call in task.calls] == [
        "sensor_psd",
        "source_localization",
        "source_psd",
    ]
    assert all(call[2] for call in task.calls)
    assert [result["block"] for result in results] == [
        "sensor_psd",
        "source_localization",
        "source_psd",
    ]
    update_metadata.assert_called_once()
    assert (
        task.config["reports_dir"]
        / "postprocessing_analysis"
        / "resolved_settings.json"
    ).exists()


def test_postprocessing_analysis_disabled_returns_empty_list(task):
    task.settings = {"postprocessing_analysis": {"enabled": False, "value": {}}}

    results = task.run_postprocessing_analysis()

    assert results == []
    assert task.calls == []


def test_postprocessing_analysis_rejects_non_dict_value(task):
    task.settings = {
        "postprocessing_analysis": {"enabled": True, "value": "not-a-dict"}
    }

    with pytest.raises(ValueError, match="must be a dictionary"):
        task.run_postprocessing_analysis()


def test_postprocessing_analysis_propagates_block_errors(task):
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "sensor_psd": {"enabled": True, "input": "clean_epochs"},
            },
        }
    }

    def _raise(**kwargs):
        raise RuntimeError("boom")

    task.apply_sensor_psd = _raise

    with pytest.raises(RuntimeError, match="boom"):
        task.run_postprocessing_analysis()


def test_postprocessing_analysis_rejects_missing_input(task):
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "sensor_psd": {"enabled": True, "input": "source_psd"},
            },
        }
    }

    with pytest.raises(ValueError, match="source_psd.*not available"):

        task.run_postprocessing_analysis()


def test_postprocessing_input_preserves_objects_without_truth_testing(task):
    class NoTruthValue:
        def __bool__(self):
            raise AssertionError("postprocessing inputs must not be truth-tested")

    imported = NoTruthValue()
    source = NoTruthValue()
    task.original_raw = imported
    task.source_eeg = source

    assert task._resolve_postprocessing_input("imported_raw") is imported
    assert task._resolve_postprocessing_input("source_epochs") is source


def test_postprocessing_input_falls_back_only_for_none(task):
    imported_fallback = object()
    source_fallback = object()
    task.original_raw = None
    task.source_eeg = None
    task.raw = imported_fallback
    task.source_epochs = source_fallback

    assert task._resolve_postprocessing_input("imported_raw") is imported_fallback
    assert task._resolve_postprocessing_input("source_epochs") is source_fallback


def test_postprocessing_fooof_consumes_sensor_psd_table(task):
    task.sensor_psd_result = {
        "spectra": pd.DataFrame(
            {
                "subject": ["s1", "s1", "s1"],
                "channel": ["Fz", "Fz", "Fz"],
                "frequency": [2.0, 4.0, 8.0],
                "psd": [10.0, 5.0, 2.5],
            }
        )
    }
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "fooof": {
                    "enabled": True,
                    "input": "sensor_psd",
                    "freq_range": [2, 8],
                },
            },
        }
    }

    with patch.object(task, "_update_metadata"):
        results = task.run_postprocessing_analysis()

    assert results[0]["block"] == "fooof"
    assert results[0]["method"] == "tabular_psd_parameterization"
    assert task.fooof_aperiodic_df.iloc[0]["status"] == "SUCCESS"
    assert not task.fooof_periodic_df.empty
    assert (task.config["reports_dir"] / "fooof").exists()


def test_postprocessing_tabular_fooof_records_fixed_model(task):
    task.sensor_psd_result = {
        "spectra": pd.DataFrame({"frequency": [2.0, 4.0, 8.0], "psd": [10.0, 5.0, 2.5]})
    }
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {"fooof": {"enabled": True, "input": "sensor_psd"}},
        }
    }

    with patch.object(task, "_update_metadata") as update_metadata:
        task.run_postprocessing_analysis()

    assert set(task.fooof_aperiodic_df["aperiodic_mode"]) == {"fixed"}
    fooof_metadata = next(
        call.args[1]
        for call in update_metadata.call_args_list
        if call.args[0] == "step_postprocessing_fooof"
    )
    assert fooof_metadata["aperiodic_mode"] == "fixed"


def test_postprocessing_tabular_fooof_rejects_knee_model(task):
    task.sensor_psd_result = {
        "spectra": pd.DataFrame({"frequency": [2.0, 4.0, 8.0], "psd": [10.0, 5.0, 2.5]})
    }
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "fooof": {
                    "enabled": True,
                    "input": "sensor_psd",
                    "aperiodic_mode": "knee",
                }
            },
        }
    }

    with pytest.raises(ValueError, match="supports only aperiodic_mode='fixed'"):
        task.run_postprocessing_analysis()


def test_postprocessing_fooof_uses_output_alias_from_prior_block(task):
    task.epochs = object()
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "sensor_psd": {
                    "enabled": True,
                    "input": "clean_epochs",
                    "output": "my_sensor_psd",
                },
                "fooof": {
                    "enabled": True,
                    "input": "my_sensor_psd",
                    "freq_range": [2, 8],
                    "run_periodic": False,
                },
            },
        }
    }

    with patch.object(task, "_update_metadata"):
        results = task.run_postprocessing_analysis()

    assert [result["block"] for result in results] == ["sensor_psd", "fooof"]
    assert results[1]["method"] == "tabular_psd_parameterization"
    assert task.fooof_aperiodic_df.iloc[0]["status"] == "SUCCESS"


def test_postprocessing_fooof_rejects_non_tabular_source_alias(task):
    alias_object = object()
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "source_localization": {
                    "enabled": True,
                    "input": "clean_epochs",
                    "output": "configured_source",
                },
                "fooof": {
                    "enabled": True,
                    "input": "configured_source",
                    "run_periodic": False,
                },
            },
        }
    }

    with (
        patch.object(task, "_update_metadata"),
        patch.object(task, "apply_source_localization", return_value=alias_object),
        patch.object(task, "apply_fooof_aperiodic") as apply_fooof,
    ):
        with pytest.raises(ValueError, match="requires a PSD table input"):
            task.run_postprocessing_analysis()

    apply_fooof.assert_not_called()


def test_postprocessing_resolves_legacy_sensor_psd_dataframe(task):
    task.sensor_psd_df = pd.DataFrame(
        {
            "channel": ["Fz", "Fz", "Fz"],
            "frequency": [2.0, 4.0, 8.0],
            "psd": [10.0, 5.0, 2.5],
        }
    )
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "fooof": {
                    "enabled": True,
                    "input": "sensor_psd",
                    "freq_range": [2, 8],
                    "run_periodic": False,
                },
            },
        }
    }

    with patch.object(task, "_update_metadata"):
        results = task.run_postprocessing_analysis()

    assert results[0]["method"] == "tabular_psd_parameterization"
    assert task.fooof_aperiodic_df.iloc[0]["status"] == "SUCCESS"


def test_postprocessing_analysis_is_exported_in_schema_layout():
    layout = export_task_schema_layout()

    assert "postprocessing_analysis" in layout["tasks"]


def test_sensor_psd_schema_accepts_default_frequency_bands():
    config = _minimal_postprocessing_schema_config()
    config["postprocessing_analysis"] = {
        "enabled": True,
        "value": {
            "sensor_psd": {
                "enabled": True,
                "freq_bands": "default",
            }
        },
    }

    validated = validate_task_module_config(config)

    assert (
        validated["postprocessing_analysis"]["value"]["sensor_psd"]["freq_bands"]
        == "default"
    )


def test_postprocessing_schema_rejects_unknown_blocks_and_bad_types():
    with pytest.raises(SchemaError):
        validate_task_module_config(
            {
                "postprocessing_analysis": {
                    "enabled": True,
                    "value": {"unknown_block": {"enabled": True}},
                }
            }
        )

    with pytest.raises(SchemaError):
        validate_task_module_config(
            {
                "postprocessing_analysis": {
                    "enabled": True,
                    "value": {"fooof": {"enabled": "yes"}},
                }
            }
        )
