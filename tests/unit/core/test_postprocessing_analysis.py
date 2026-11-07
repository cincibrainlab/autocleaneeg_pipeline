from pathlib import Path
from unittest.mock import patch

import numpy as np
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


LARGE_PEAKS = ((8.0, 8.0, 1.5), (20.0, 6.0, 2.5))


def _synthetic_psd(offset=1.0, chi=2.0, peaks=((8.0, 3.0, 1.5), (20.0, 2.0, 2.0))):
    """Return 1-45 Hz PSD = 10**(offset - chi*log10 f) * (1 + Gaussian peaks)."""
    freqs = np.arange(1.0, 45.0001, 0.25)
    gain = np.ones_like(freqs)
    for center, height, sigma in peaks:
        gain += height * np.exp(-((freqs - center) ** 2) / (2 * sigma**2))
    return freqs, 10 ** (offset - chi * np.log10(freqs)) * gain


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
        freqs, psd = _synthetic_psd()
        psd_df = pd.DataFrame({"channel": "Fz", "frequency": freqs, "psd": psd})
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


def test_postprocessing_analysis_writes_metadata_before_reraising(task):
    import json

    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "sensor_psd": {"enabled": True, "input": "clean_epochs"},
                "source_localization": {"enabled": True, "input": "clean_epochs"},
            },
        }
    }

    def _raise(**kwargs):
        raise RuntimeError("boom")

    task.apply_source_localization = _raise

    with patch.object(task, "_update_metadata") as update_metadata:
        with pytest.raises(RuntimeError, match="boom"):
            task.run_postprocessing_analysis()

    update_metadata.assert_called_once()
    name, payload = update_metadata.call_args[0]
    assert name == "step_postprocessing_analysis"
    assert [(b["block"], b["status"]) for b in payload["blocks"]] == [
        ("sensor_psd", "completed"),
        ("source_localization", "failed"),
    ]
    assert "boom" in payload["blocks"][1]["error"]
    path = (
        task.config["reports_dir"]
        / "postprocessing_analysis"
        / "resolved_settings.json"
    )
    saved = json.loads(path.read_text())
    assert "failed" in json.dumps(saved)


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
    freqs, psd = _synthetic_psd()
    task.sensor_psd_result = {
        "spectra": pd.DataFrame(
            {"subject": "s1", "channel": "Fz", "frequency": freqs, "psd": psd}
        )
    }
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "fooof": {
                    "enabled": True,
                    "input": "sensor_psd",
                    "freq_range": [1, 45],
                },
            },
        }
    }

    with patch.object(task, "_update_metadata"):
        results = task.run_postprocessing_analysis()

    assert results[0]["block"] == "fooof"
    assert results[0]["method"] == "specparam_tabular"
    assert task.fooof_aperiodic_df.iloc[0]["status"] == "SUCCESS"
    assert not task.fooof_periodic_df.empty
    assert (task.config["reports_dir"] / "fooof").exists()


def test_postprocessing_tabular_fooof_records_fixed_model(task):
    freqs, psd = _synthetic_psd()
    task.sensor_psd_result = {"spectra": pd.DataFrame({"frequency": freqs, "psd": psd})}
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


def test_postprocessing_tabular_fooof_accepts_knee_model(task):
    freqs, psd = _synthetic_psd()
    task.sensor_psd_result = {"spectra": pd.DataFrame({"frequency": freqs, "psd": psd})}
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

    with patch.object(task, "_update_metadata"):
        task.run_postprocessing_analysis()

    row = task.fooof_aperiodic_df.iloc[0]
    assert row["status"] == "SUCCESS"
    assert row["aperiodic_mode"] == "knee"
    assert row["knee"] is not None and np.isfinite(row["knee"])


def test_postprocessing_tabular_fooof_rejects_unknown_aperiodic_mode(task):
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
                    "aperiodic_mode": "bogus",
                }
            },
        }
    }

    with pytest.raises(ValueError, match="must be 'fixed' or 'knee'"):
        task.run_postprocessing_analysis()


def test_postprocessing_tabular_fooof_recovers_exponent_despite_peaks(task):
    chi = 2.0
    freqs, psd = _synthetic_psd(chi=chi, peaks=LARGE_PEAKS)
    # Regression check: the previous method was a plain log-log line fit over
    # the whole spectrum, which oscillatory peaks bias.
    naive_exponent = -np.polyfit(np.log10(freqs), np.log10(psd), deg=1)[0]
    assert abs(naive_exponent - chi) > 0.2

    task.sensor_psd_result = {"spectra": pd.DataFrame({"frequency": freqs, "psd": psd})}
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {"fooof": {"enabled": True, "input": "sensor_psd"}},
        }
    }
    with patch.object(task, "_update_metadata"):
        task.run_postprocessing_analysis()

    aperiodic = task.fooof_aperiodic_df.iloc[0]
    assert aperiodic["status"] == "SUCCESS"
    assert abs(aperiodic["exponent"] - chi) < 0.1
    assert aperiodic["knee"] is None or pd.isna(aperiodic["knee"])
    print(
        f"specparam exponent={aperiodic['exponent']:.3f} polyfit={naive_exponent:.3f}"
    )

    periodic = task.fooof_periodic_df
    for expected in (8.0, 20.0):
        near = periodic[(periodic["center_frequency"] - expected).abs() < 1.0]
        assert len(near) == 1
        assert near.iloc[0]["bandwidth"] is not None
        assert near.iloc[0]["bandwidth"] > 0


def test_postprocessing_tabular_fooof_marks_failed_fit(task):
    freqs, psd = _synthetic_psd()
    task.sensor_psd_result = {"spectra": pd.DataFrame({"frequency": freqs, "psd": psd})}
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {"fooof": {"enabled": True, "input": "sensor_psd"}},
        }
    }
    with patch("specparam.SpectralModel.fit", side_effect=RuntimeError("boom")):
        with patch.object(task, "_update_metadata"):
            task.run_postprocessing_analysis()

    row = task.fooof_aperiodic_df.iloc[0]
    assert row["status"] == "FIT_FAILED"
    assert pd.isna(row["exponent"])
    assert task.fooof_periodic_df.empty


def _run_fooof(task, psd_df, **settings):
    task.sensor_psd_result = {"spectra": psd_df}
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {"fooof": {"enabled": True, "input": "sensor_psd", **settings}},
        }
    }
    with patch.object(task, "_update_metadata"):
        task.run_postprocessing_analysis()


def test_postprocessing_tabular_fooof_invalid_settings_raise(task):
    freqs, psd = _synthetic_psd()
    with pytest.raises(ValueError, match="invalid fooof settings"):
        _run_fooof(
            task,
            pd.DataFrame({"frequency": freqs, "psd": psd}),
            peak_width_limits=[12, 1],
        )


def test_postprocessing_tabular_fooof_marks_nonpositive_psd_invalid(task):
    freqs, psd = _synthetic_psd()
    psd = psd.copy()
    psd[10] = 0.0
    _run_fooof(task, pd.DataFrame({"frequency": freqs, "psd": psd}))

    row = task.fooof_aperiodic_df.iloc[0]
    assert row["status"] == "INVALID_PSD"
    assert "non-positive" in row["fit_message"]
    assert pd.isna(row["exponent"])


def test_postprocessing_tabular_fooof_failed_fit_records_message(task):
    freqs, psd = _synthetic_psd()
    with patch("specparam.SpectralModel.fit", side_effect=RuntimeError("boom")):
        _run_fooof(task, pd.DataFrame({"frequency": freqs, "psd": psd}))
    row = task.fooof_aperiodic_df.iloc[0]
    assert row["status"] == "FIT_FAILED"
    assert "boom" in row["fit_message"]


def test_postprocessing_tabular_fooof_no_peaks_writes_readable_periodic_csv(task):
    freqs = np.arange(1.0, 45.0001, 0.25)
    psd = 10 ** (1.0 - 2.0 * np.log10(freqs))
    _run_fooof(
        task,
        pd.DataFrame({"frequency": freqs, "psd": psd}),
        min_peak_height=50.0,
    )
    assert task.fooof_aperiodic_df.iloc[0]["status"] == "SUCCESS"
    assert task.fooof_aperiodic_df.iloc[0]["fit_message"] is None
    periodic = pd.read_csv(task.fooof_periodic_file)
    assert periodic.empty
    assert {"center_frequency", "power", "bandwidth", "status"} <= set(periodic.columns)
    aperiodic = pd.read_csv(task.fooof_aperiodic_file)
    assert "fit_message" in aperiodic.columns


@pytest.mark.parametrize(
    ("alias_key", "alias_value"),
    [
        ("output", "my_sensor_psd"),
        ("output", ["first_sensor_psd", "my_sensor_psd"]),
        ("output", ("first_sensor_psd", "my_sensor_psd")),
        ("aliases", "my_sensor_psd"),
        ("aliases", ["first_sensor_psd", "my_sensor_psd"]),
        ("aliases", ("first_sensor_psd", "my_sensor_psd")),
    ],
)
def test_postprocessing_fooof_uses_configured_alias_from_prior_block(
    task, alias_key, alias_value
):
    task.epochs = object()
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "sensor_psd": {
                    "enabled": True,
                    "input": "clean_epochs",
                    alias_key: alias_value,
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
    assert results[1]["method"] == "specparam_tabular"
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
    freqs, psd = _synthetic_psd()
    task.sensor_psd_df = pd.DataFrame({"channel": "Fz", "frequency": freqs, "psd": psd})
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

    assert results[0]["method"] == "specparam_tabular"
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


def test_source_localization_schema_accepts_runtime_options():
    config = _minimal_postprocessing_schema_config()
    source_localization = {
        "enabled": True,
        "method": "MNE",
        "lambda2": 0.125,
        "montage": "standard_1020",
        "resample_freq": 128.0,
        "max_memory_gb": 2.5,
    }
    config["postprocessing_analysis"] = {
        "enabled": True,
        "value": {"source_localization": source_localization},
    }

    validated = validate_task_module_config(config)

    assert (
        validated["postprocessing_analysis"]["value"]["source_localization"]
        == source_localization
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pick_ori", "normal"),
        ("n_jobs", 2),
        ("convert_to_eeg", True),
    ],
)
def test_source_localization_schema_rejects_ignored_options(field, value):
    config = _minimal_postprocessing_schema_config()
    config["postprocessing_analysis"] = {
        "enabled": True,
        "value": {
            "source_localization": {
                "enabled": True,
                field: value,
            }
        },
    }

    with pytest.raises(SchemaError):
        validate_task_module_config(config)


def test_source_localization_forwards_schema_options(task):
    task.settings = {
        "postprocessing_analysis": {
            "enabled": True,
            "value": {
                "source_localization": {
                    "enabled": True,
                    "input": "clean_epochs",
                    "method": "dSPM",
                    "lambda2": 0.25,
                    "montage": "standard_1020",
                    "resample_freq": 128.0,
                    "max_memory_gb": 3.0,
                }
            },
        }
    }

    with patch.object(task, "_update_metadata"):
        task.run_postprocessing_analysis()

    _, kwargs, enabled = task.calls[0]
    assert enabled is True
    assert kwargs == {
        "data": task.epochs,
        "method": "dSPM",
        "lambda2": 0.25,
        "montage": "standard_1020",
        "resample_freq": 128.0,
        "max_memory_gb": 3.0,
        "stage_name": "postprocessing_source_localization",
    }


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


def test_oscillation_blocks_schema_accepts_settings_and_rejects_bad_types():
    config = _minimal_postprocessing_schema_config()
    config["postprocessing_analysis"] = {
        "enabled": True,
        "value": {
            "power_coupling": {
                "enabled": True,
                "window_s": 1,
                "regions": {"cortex": "CTX_"},
            },
            "pac": {
                "enabled": True,
                "phase_bands": [[6, 9]],
                "amp_bands": [[70, 100]],
                "n_surrogates": 100,
                "seed": 1,
            },
            "hfo": {"enabled": True, "band": [120, 250], "min_cycles": 4},
            "connectivity": {
                "enabled": True,
                "bands": {"beta": [15, 25]},
                "methods": ["wpli"],
                "n_epochs": None,
            },
        },
    }
    validate_task_module_config(config)

    config["postprocessing_analysis"]["value"]["hfo"]["min_cycles"] = "four"
    with pytest.raises(SchemaError):
        validate_task_module_config(config)


def test_postprocessing_connectivity_records_epoch_multitaper_method(task):
    import mne

    raw = mne.io.RawArray(
        np.zeros((2, 100)), mne.create_info(["A", "B"], 100.0, "eeg"), verbose=False
    )
    frame = pd.DataFrame(
        {"channel_a": ["A"], "channel_b": ["B"], "band": ["theta"], "value": [0.1]}
    )
    common = {}
    with patch(
        "autoclean.calc.source_connectivity.compute_sensor_connectivity",
        return_value=frame,
    ):
        task._run_oscillation_block(
            "connectivity", {"bands": {"theta": [4, 8]}}, raw, common
        )

    assert common["method"] == "mne_connectivity_spectral_epochs_multitaper"


def test_postprocessing_tabular_fooof_drops_zero_hz_bin(task):
    freqs = np.arange(0.0, 45.0001, 0.25)
    psd = np.empty_like(freqs)
    psd[1:] = 10 ** (1.0 - 2.0 * np.log10(freqs[1:]))
    psd[0] = 1e3
    _run_fooof(
        task,
        pd.DataFrame({"frequency": freqs, "psd": psd}),
        freq_range=[0, 45],
        min_peak_height=50.0,
    )

    row = task.fooof_aperiodic_df.iloc[0]
    assert row["status"] == "SUCCESS"
    assert row["exponent"] == pytest.approx(2.0, abs=0.2)


def test_postprocessing_tabular_fooof_warns_when_every_spectrum_fails(task):
    freqs, psd = _synthetic_psd()
    psd = psd.copy()
    psd[10] = 0.0
    with patch("autoclean.core.task.message") as message:
        _run_fooof(task, pd.DataFrame({"frequency": freqs, "psd": psd}))

    warnings = [c for c in message.call_args_list if c.args[0] == "warning"]
    assert len(warnings) == 1
    assert "non-positive" in warnings[0].args[1]
