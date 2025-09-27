"""Tests for deterministic LLM reporting helpers."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "autoclean"
    / "reporting"
    / "llm_reporting.py"
)
SPEC = spec_from_file_location(
    "autoclean.reporting.llm_reporting",
    MODULE_PATH,
)
assert SPEC and SPEC.loader  # narrow mypy / type-checker complaints
LLM_REPORTING = module_from_spec(SPEC)
sys.modules[SPEC.name] = LLM_REPORTING
SPEC.loader.exec_module(LLM_REPORTING)

EpochStats = LLM_REPORTING.EpochStats
FilterParams = LLM_REPORTING.FilterParams
ICAStats = LLM_REPORTING.ICAStats
RunContext = LLM_REPORTING.RunContext
render_methods = LLM_REPORTING.render_methods


def _base_context(**overrides):
    params = dict(
        run_id="run-1",
        dataset_name=None,
        input_file="subject01.raw",
        montage=None,
        resample_hz=None,
        reference=None,
        filter_params=FilterParams(
            l_freq=None,
            h_freq=None,
            notch_freqs=[],
            notch_widths=None,
        ),
        ica=None,
        epochs=None,
        durations_s=None,
        n_channels=None,
        bids_root=None,
        bids_subject_id=None,
        pipeline_version="2.3.0",
        mne_version=None,
        compliance_user=None,
        notes=[],
        figures={},
    )
    params.update(overrides)
    return RunContext(**params)


def test_render_methods_full_context(tmp_path: Path) -> None:
    context = _base_context(
        resample_hz=250.0,
        reference="average",
        montage="GSN-HydroCel-129",
        bids_root=str(tmp_path / "bids-root"),
        mne_version="1.10.1",
        filter_params=FilterParams(
            l_freq=1.0,
            h_freq=40.0,
            notch_freqs=[60.0, 120.0],
            notch_widths=1.0,
        ),
        ica=ICAStats(
            method="fastica",
            n_components=20,
            removed_indices=[0, 1],
            labels_histogram={"eye": 1},
            classifier="iclabel",
        ),
        epochs=EpochStats(
            tmin=-0.2,
            tmax=0.5,
            baseline=(None, 0.0),
            total_epochs=120,
            kept_epochs=100,
            rejected_epochs=20,
            rejection_rules={"mag": 5e-12},
        ),
    )

    rendered = render_methods(context)

    assert (
        rendered
        == "EEG preprocessing was performed using AutoCleanEEG v2.3.0 (MNE-Python 1.10.1). "
        "Data were converted to BIDS and organized under bids-root. Signals were resampled to 250 Hz. "
        "Data were filtered (high-pass at 1 Hz; low-pass at 40 Hz; notch at 60, 120 Hz). "
        "Signals were re-referenced to average. Independent Component Analysis was performed using fastica (20 components). "
        "Components were classified with iclabel and 2 components were removed. Data were epoched from -0.2s to 0.5s "
        "with baseline correction (None to 0.0 s). Epoch counts: total=120, kept=100, rejected=20. "
        "Automated epoch rejection thresholds: {'mag': 5e-12}. Electrodes were assigned to the GSN-HydroCel-129 montage."
    )


def test_render_methods_minimal_context() -> None:
    context = _base_context()

    rendered = render_methods(context)

    assert (
        rendered
        == "EEG preprocessing was performed using AutoCleanEEG v2.3.0 (MNE-Python n/a). "
        "Data were converted to BIDS and organized under a BIDS-compliant folder."
    )
