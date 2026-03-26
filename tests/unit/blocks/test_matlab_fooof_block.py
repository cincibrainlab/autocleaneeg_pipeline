"""Tests for the bundled MATLAB FOOOF block."""

from __future__ import annotations

import json
from pathlib import Path

from autoclean.blocks.analysis.matlab_fooof.algorithm import (
    load_matlab_fooof_manifest,
    resolve_matlab_fooof_context,
)
from autoclean.blocks.analysis.matlab_fooof.mixin import MatlabFooofBlockMixin


def test_resolve_matlab_fooof_context_creates_subject_output_dir(tmp_path: Path) -> None:
    context = resolve_matlab_fooof_context(
        {
            "unprocessed_file": tmp_path / "0006_rest_comp_epo.set",
            "derivatives_dir": tmp_path / "derivatives",
        },
        {"artifacts_subdir": "matlab/fooof", "spect_freqs": [1, 55]},
        module_dir=Path("src/autoclean/blocks/analysis/matlab_fooof"),
    )

    assert context["subject_id"] == "0006_rest_comp_epo"
    assert context["block_root"].exists()
    assert context["freq_range"] == (1.0, 55.0)


def test_load_matlab_fooof_manifest_reads_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"subject_id": "0006", "summary_csv": "summary.csv"}),
        encoding="utf-8",
    )

    manifest = load_matlab_fooof_manifest(manifest_path)

    assert manifest["subject_id"] == "0006"


def test_apply_matlab_fooof_calls_runtime_and_updates_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "derivatives" / "matlab" / "fooof" / "0006_rest_comp_epo" / "0006_rest_comp_epo_fooof_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "subject_id": "0006_rest_comp_epo",
                "summary_csv": str(manifest_path.parent / "summary.csv"),
                "aperiodic_csv": str(manifest_path.parent / "aperiodic.csv"),
                "matlab_output_dir": str(manifest_path.parent / "eeg_htpCalcFooof"),
                "summary_row_count": 10,
                "aperiodic_row_count": 5,
                "n_channels": 128,
                "n_epochs": 60,
                "sampling_rate": 250.0,
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_call_matlab(function_name, *args, **kwargs):
        captured["function_name"] = function_name
        captured["args"] = args
        captured["kwargs"] = kwargs
        return str(manifest_path)

    monkeypatch.setattr("autoclean.blocks.analysis.matlab_fooof.mixin.call_matlab", _fake_call_matlab)

    class DummyTask(MatlabFooofBlockMixin):
        def __init__(self) -> None:
            self.config = {
                "run_id": "run-1",
                "unprocessed_file": tmp_path / "0006_rest_comp_epo.set",
                "derivatives_dir": tmp_path / "derivatives",
            }
            self.settings = {
                "apply_matlab_fooof": {
                    "enabled": True,
                    "value": {
                        "vhtp_path": str(tmp_path / "vhtp"),
                        "eeglab_path": str(tmp_path / "eeglab"),
                        "spect_freqs": [1, 55],
                        "save_fooof_img": False,
                        "parallel": False,
                    },
                }
            }
            self.metadata_updates: list[tuple[str, dict]] = []

        def _check_step_enabled(self, step_name):
            step = self.settings[step_name]
            return step["enabled"], step

        def _get_block_info(self, block_name: str):
            return {"source_commit": "abc123", "block_name": block_name}

        def _update_metadata(self, operation: str, metadata_dict: dict) -> None:
            self.metadata_updates.append((operation, metadata_dict))

    task = DummyTask()
    manifest, returned_manifest_path = task.apply_matlab_fooof()

    assert manifest is not None
    assert returned_manifest_path == manifest_path.resolve()
    assert captured["function_name"] == "autoclean_eeglab_fooof"
    assert task.metadata_updates
    assert task.metadata_updates[0][0] == "step_apply_matlab_fooof"
