"""Integration test: rejected ICA components retain real activity for reporting.

Regression test for the bug where rejected ICA component report pages showed
blank/flat diagnostic panels. Root cause: ICA.apply() mutates self.raw in
place, zeroing out rejected components' contribution to the signal, and the
report generator read from self.raw *after* that mutation. The fix snapshots
self.raw before rejection runs and points report generation at that snapshot.
"""

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from autoclean.core.task import Task
from tests.fixtures.synthetic_data import create_synthetic_raw


class _ICAIntegrationTask(Task):
    """Minimal real Task exposing ICA fit/rejection/reporting for integration testing."""

    def run(self):
        pass


@pytest.fixture
def ica_integration_task(tmp_path):
    settings: Dict[str, Any] = {
        "ICA": {
            "enabled": True,
            "value": {
                "n_components": 5,
                "method": "fastica",
                "max_iter": 500,
            },
        }
    }
    config = {
        "run_id": "ica-prerejection-integration-test",
        "unprocessed_file": tmp_path / "test.fif",
        "task": "ica-prerejection-integration-test",
        "tasks": {"ica-prerejection-integration-test": {"settings": settings}},
    }
    task = _ICAIntegrationTask(config)
    task.raw = create_synthetic_raw(
        montage="standard_1020", n_channels=16, duration=30.0, sfreq=250.0
    )
    return task


def test_rejected_component_retains_activity_after_manual_rejection(
    ica_integration_task,
):
    """After a real ICA fit + manual rejection, the rejected component's real
    activity should still be recoverable from the pre-rejection snapshot,
    even though it is zeroed out in self.raw itself.
    """
    task = ica_integration_task

    # Step 1 & 2: Real ICA fit + manual rejection, with database writes
    # patched out (this test cares about ICA data flow, not persistence).
    with patch("autoclean.utils.database.manage_database", return_value=None):
        task.run_ica()
        assert task.final_ica is not None
        assert task.final_ica.n_components_ == 5

        # Reject component 0 manually, skipping classification entirely
        task.apply_ica_component_rejection(manual_rejected_components=[0])

    # Step 3: Confirm the pre-rejection snapshot exists and is a distinct object
    assert hasattr(task, "raw_prerejection")
    assert task.raw_prerejection is not None
    assert task.raw_prerejection is not task.raw

    # Step 4: Extract component 0's source activity from both versions
    sources_prerejection = task.final_ica.get_sources(task.raw_prerejection)
    sources_postrejection = task.final_ica.get_sources(task.raw)

    prerejection_component_0 = sources_prerejection.get_data()[0]
    postrejection_component_0 = sources_postrejection.get_data()[0]

    # The actual bug assertion: pre-rejection data must retain real variance,
    # while self.raw (post-rejection) collapses to ~0 for the rejected IC.
    assert np.var(prerejection_component_0) > 1e-10
    assert np.var(postrejection_component_0) < 1e-10

    # Step 5: Confirm the reporting helper selects the correct data source
    report_data = task._get_ica_report_data()
    assert report_data is task.raw_prerejection
