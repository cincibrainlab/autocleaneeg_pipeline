"""Tests for the lightweight ICA-rerun service (issue #275).

Covers the pure/testable pieces of ``autoclean.api.services.ica_rerun``:
the retained-data threshold check, structured ICLabel JSON shaping, and
recovery of the original run's ICA defaults from its metadata JSON. ICA
fitting itself is exercised via mocks (mirrors ``tests/functions/test_ica.py``)
since actually fitting/classifying is heavy and already covered by the
existing ICA function tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
import pytest
from mne.preprocessing import ICA

from autoclean.api.services import ica_rerun


def _make_epochs(
    n_epochs: int = 8, sfreq: float = 100.0, epoch_seconds: float = 1.0
) -> mne.EpochsArray:
    ch_names = ["Fz", "Cz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg", "eeg"])
    n_samples = int(epoch_seconds * sfreq)
    data = np.zeros((n_epochs, len(ch_names), n_samples), dtype=float)
    events = np.column_stack(
        [
            np.arange(n_epochs),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"standard": 1},
        tmin=0.0,
        baseline=None,
        verbose=False,
    )


class TestCheckRetainedEpochsThreshold:
    def test_passes_when_above_both_floors(self):
        epochs = _make_epochs(n_epochs=50, epoch_seconds=3.0)
        result = ica_rerun.check_retained_epochs_threshold(epochs)
        assert result["ok"] is True
        assert result["reason"] is None
        assert result["n_epochs"] == 50

    def test_fails_when_too_few_epochs(self):
        epochs = _make_epochs(n_epochs=5, epoch_seconds=3.0)
        result = ica_rerun.check_retained_epochs_threshold(epochs)
        assert result["ok"] is False
        assert "epochs retained" in result["reason"]

    def test_fails_when_duration_too_short(self):
        epochs = _make_epochs(n_epochs=50, epoch_seconds=0.5)
        result = ica_rerun.check_retained_epochs_threshold(epochs)
        assert result["ok"] is False
        assert "retained data" in result["reason"]

    def test_custom_thresholds_are_respected(self):
        epochs = _make_epochs(n_epochs=5, epoch_seconds=0.5)
        result = ica_rerun.check_retained_epochs_threshold(
            epochs, min_epochs=1, min_duration=0.1
        )
        assert result["ok"] is True


class TestBuildStructuredIclabelJson:
    def test_shapes_components_and_marks_rejected(self):
        ica = MagicMock(spec=ICA)
        ica.n_components_ = 3
        ica.exclude = [1]
        ica_flags = pd.DataFrame(
            {
                "ic_type": ["brain", "heart", "muscle"],
                "confidence": [0.9, 0.8, 0.7],
            },
            index=[0, 1, 2],
        )

        result = ica_rerun.build_structured_iclabel_json(
            ica, ica_flags, n_epochs=42, classification_method="iclabel"
        )

        assert result["components"] == [
            {"component": "IC0", "type": "brain", "confidence": 0.9, "rejected": False},
            {"component": "IC1", "type": "ecg", "confidence": 0.8, "rejected": True},
            {
                "component": "IC2",
                "type": "muscle",
                "confidence": 0.7,
                "rejected": False,
            },
        ]
        assert result["structure"] == {
            "n_components": 3,
            "method": "iclabel",
            "fitted_on": "epochs",
            "n_epochs": 42,
        }


class TestReadOriginalIcaDefaults:
    def test_missing_metadata_falls_back_to_defaults(self):
        defaults = ica_rerun.read_original_ica_defaults(None)
        assert defaults["classification_method"] == "iclabel"
        assert defaults["ica_kwargs"]["max_iter"] == "auto"
        assert defaults["ica_kwargs"]["random_state"] == 97
        assert defaults["parent_run_id"] is None

    def test_reads_kwargs_and_method_from_metadata_json(self, tmp_path: Path):
        metadata_path = tmp_path / "stem_autoclean_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "run_id": "run-123",
                    "metadata": {
                        "step_run_ica": {
                            "ica": {
                                "ica_kwargs": {
                                    "method": "infomax",
                                    "n_components": 15,
                                    "max_iter": 200,
                                    "random_state": 7,
                                    "temp_highpass_for_ica": 1.0,
                                }
                            }
                        },
                        "classify_ica_components": {
                            "ica": {"classification_method": "icvision"}
                        },
                    },
                }
            )
        )

        defaults = ica_rerun.read_original_ica_defaults(metadata_path)

        assert defaults["parent_run_id"] == "run-123"
        assert defaults["classification_method"] == "icvision"
        assert defaults["ica_kwargs"]["method"] == "infomax"
        assert defaults["ica_kwargs"]["n_components"] == 15
        assert defaults["ica_kwargs"]["max_iter"] == 200
        assert defaults["ica_kwargs"]["random_state"] == 7
        # temp_highpass_for_ica only applies when fitting on raw; must not
        # leak into the epochs-based rerun kwargs.
        assert "temp_highpass_for_ica" not in defaults["ica_kwargs"]

    def test_nonexistent_path_falls_back_to_defaults(self, tmp_path: Path):
        defaults = ica_rerun.read_original_ica_defaults(tmp_path / "missing.json")
        assert defaults["classification_method"] == "iclabel"
        assert defaults["parent_run_id"] is None


class TestLoadRetainedEpochs:
    def test_raises_when_no_postedit_export_exists(self, tmp_path: Path):
        task_root = tmp_path / "task"
        task_root.mkdir()
        with pytest.raises(ica_rerun.IcaRerunError):
            ica_rerun.load_retained_epochs(
                task_root, task_root / "exports" / "sub_epo.set"
            )


class TestFitAndClassify:
    def test_delegates_to_standalone_functions(self):
        epochs = _make_epochs(n_epochs=4)
        fake_ica = MagicMock(spec=ICA)
        fake_flags = pd.DataFrame({"ic_type": ["brain"], "confidence": [0.9]})

        with (
            patch.object(ica_rerun, "fit_ica", return_value=fake_ica) as mock_fit,
            patch.object(
                ica_rerun, "classify_ica_components", return_value=fake_flags
            ) as mock_classify,
        ):
            ica, flags = ica_rerun.fit_and_classify(
                epochs, {"n_components": 2}, "iclabel"
            )

        mock_fit.assert_called_once_with(epochs, n_components=2)
        mock_classify.assert_called_once_with(epochs, fake_ica, method="iclabel")
        assert ica is fake_ica
        assert flags is fake_flags
