from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from autoclean.configkit.schema import export_task_schema_layout
from autoclean.core.task import Task


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


def test_postprocessing_fooof_passes_resolved_source_alias_to_legacy_method(task):
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

    output_file = task.config["reports_dir"] / "fooof_aperiodic.parquet"
    with (
        patch.object(task, "_update_metadata"),
        patch.object(
            task, "apply_source_localization", return_value=alias_object
        ),
        patch.object(
            task, "apply_fooof_aperiodic", return_value=([], output_file)
        ) as apply_fooof,
    ):
        task.run_postprocessing_analysis()

    assert apply_fooof.call_args.kwargs["stc"] is alias_object


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
