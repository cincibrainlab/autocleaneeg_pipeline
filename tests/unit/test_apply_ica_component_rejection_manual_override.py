"""Tests for manual ICA component rejection overrides."""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from autoclean.mixins.base import BaseMixin
from autoclean.mixins.signal_processing import ica as ica_module
from autoclean.mixins.signal_processing.ica import IcaMixin


class DummyICA:
    """Simple stand-in for an MNE ICA object."""

    def __init__(self):
        self.exclude: List[int] = []
        self.applied_to: List[Any] = []

    def apply(self, data):
        self.applied_to.append(data)
        return data


class DummyICATask(IcaMixin, BaseMixin):
    """Minimal task harness exposing ICA component rejection."""

    def __init__(self, raw: mne.io.Raw):
        self.raw = raw
        self.final_ica = DummyICA()
        self.ica_flags = None
        self.flagged = False
        self.flagged_reasons: List[str] = []
        self.metadata_log: List[Tuple[str, Dict[str, Any]]] = []
        self.saved_results: List[Tuple[str, Any]] = []
        self.config = {
            "run_id": "unit-test-run",
            "task": "dummy",
            "tasks": {"dummy": {"settings": {}}},
        }

    def _update_metadata(self, operation: str, metadata_dict: Dict[str, Any]) -> None:
        self.metadata_log.append((operation, metadata_dict))

    def _auto_export_if_enabled(self, *args, **kwargs):
        # Disable export side effects during tests
        return None


@pytest.fixture
def dummy_raw() -> mne.io.Raw:
    ch_names = [f"E{i}" for i in range(1, 5)]
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    data = np.random.randn(len(ch_names), 256)
    return mne.io.RawArray(data, info)


def test_manual_ica_override_skips_auto(monkeypatch, dummy_raw):
    task = DummyICATask(dummy_raw)

    auto_rejection_mock = MagicMock()
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.ica.apply_ica_component_rejection",
        auto_rejection_mock,
    )
    monkeypatch.setattr(ica_module, "CACHE_AVAILABLE", False)

    task.apply_ica_component_rejection(
        manual_rejected_components=[2, 2, 0],
    )

    auto_rejection_mock.assert_not_called()
    assert task.final_ica.exclude == [0, 2]
    assert task.final_ica.applied_to == [task.raw]

    operation, metadata = task.metadata_log[-1]
    assert operation == "step_apply_ica_component_rejection"
    nested = metadata["ica"]
    assert nested["method"] == "ManualOverride"
    assert nested["final_excluded_indices"] == [0, 2]


def test_manual_ica_override_empty_list(monkeypatch, dummy_raw):
    task = DummyICATask(dummy_raw)

    auto_rejection_mock = MagicMock()
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.ica.apply_ica_component_rejection",
        auto_rejection_mock,
    )
    monkeypatch.setattr(ica_module, "CACHE_AVAILABLE", False)

    task.apply_ica_component_rejection(manual_rejected_components=[])

    auto_rejection_mock.assert_not_called()
    assert task.final_ica.exclude == []
    assert task.final_ica.applied_to == []

    operation, metadata = task.metadata_log[-1]
    assert operation == "step_apply_ica_component_rejection"
    nested = metadata["ica"]
    assert nested["method"] == "ManualOverride"
    assert nested["final_excluded_indices"] == []


def test_prerejection_snapshot_created_before_apply(monkeypatch, dummy_raw):
    task = DummyICATask(dummy_raw)

    auto_rejection_mock = MagicMock()
    monkeypatch.setattr(
        "autoclean.mixins.signal_processing.ica.apply_ica_component_rejection",
        auto_rejection_mock,
    )
    monkeypatch.setattr(ica_module, "CACHE_AVAILABLE", False)

    original_data = task.raw.get_data().copy()

    task.apply_ica_component_rejection(
        manual_rejected_components=[0],
    )

    # A pre-rejection snapshot should now exist
    assert hasattr(task, "raw_prerejection")
    assert task.raw_prerejection is not None

    # It must be a distinct object, not the same reference as self.raw
    assert task.raw_prerejection is not task.raw

    # Its data should match what self.raw contained before rejection ran
    np.testing.assert_array_equal(task.raw_prerejection.get_data(), original_data)

    # The actual rejection should still have been applied to self.raw itself
    assert task.final_ica.applied_to == [task.raw]
