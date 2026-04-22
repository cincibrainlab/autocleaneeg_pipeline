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
create_reports = LLM_REPORTING.create_reports
render_methods = LLM_REPORTING.render_methods
resolve_llm_settings = LLM_REPORTING._resolve_llm_settings
build_llm_client = LLM_REPORTING._build_llm_client


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
        pipeline_version="3.0.0-alpha",
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
        == "EEG preprocessing was performed using AutoCleanEEG v3.0.0-alpha (MNE-Python 1.10.1). "
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
        == "EEG preprocessing was performed using AutoCleanEEG v3.0.0-alpha (MNE-Python n/a). "
        "Data were converted to BIDS and organized under a BIDS-compliant folder."
    )


def test_resolve_llm_settings_task_overrides_global(monkeypatch) -> None:
    monkeypatch.setenv("GLOBAL_LLM_KEY", "global-key")
    monkeypatch.setenv("TASK_LLM_KEY", "task-key")
    monkeypatch.setattr(
        LLM_REPORTING,
        "load_user_config",
        lambda: {
            "llm_reporting": {
                "enabled": True,
                "api_key": "$GLOBAL_LLM_KEY",
                "base_url": "https://global.example/v1/",
                "model": "global-model",
                "temperature": 0.0,
                "seed": 1,
            }
        },
    )

    resolved = resolve_llm_settings(
        {
            "api_key": "$TASK_LLM_KEY",
            "model": "task-model",
            "temperature": 0.25,
        }
    )

    assert resolved == {
        "enabled": True,
        "api_key": "task-key",
        "base_url": "https://global.example/v1",
        "model": "task-model",
        "temperature": 0.25,
        "seed": 1,
    }


def test_build_llm_client_uses_config_before_env(monkeypatch) -> None:
    captured = {}

    class FakeLLMClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("TASK_LLM_KEY", "task-key")
    monkeypatch.setattr(
        LLM_REPORTING,
        "load_user_config",
        lambda: {
            "llm_reporting": {
                "base_url": "https://global.example/v1",
                "model": "global-model",
            }
        },
    )
    monkeypatch.setattr(LLM_REPORTING, "LLMClient", FakeLLMClient)

    client = build_llm_client(
        {
            "api_key": "$TASK_LLM_KEY",
            "model": "task-model",
            "temperature": 0.5,
            "seed": 7,
        }
    )

    assert client is not None
    assert captured == {
        "api_key": "task-key",
        "base_url": "https://global.example/v1",
        "model": "task-model",
        "temperature": 0.5,
        "seed": 7,
    }


def test_create_reports_uses_llm_settings(tmp_path: Path, monkeypatch) -> None:
    created = {}

    class FakeLLMClient:
        def __init__(self, **kwargs):
            created.update(kwargs)
            self.model = kwargs.get("model", "fake-model")
            self.temperature = kwargs.get("temperature", 0.0)
            self.seed = kwargs.get("seed", 0)

        def generate_json(self, system: str, user: str, schema_hint: str):
            if "executive summary" in user:
                return {
                    "title": "Run summary",
                    "bullets": ["LLM-backed summary created"],
                    "notes": [],
                }
            return {
                "summary": "QC narrative created",
                "recommendations": ["Keep current defaults"],
            }

    monkeypatch.setattr(LLM_REPORTING, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(LLM_REPORTING, "load_user_config", lambda: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("REPORT_KEY", "report-key")

    out_dir = tmp_path / "reports"
    create_reports(
        _base_context(),
        out_dir,
        llm_settings={
            "enabled": True,
            "api_key": "$REPORT_KEY",
            "base_url": "https://local-llm.example/v1",
            "model": "custom-model",
            "temperature": 0.1,
            "seed": 3,
        },
    )

    assert created == {
        "api_key": "report-key",
        "base_url": "https://local-llm.example/v1",
        "model": "custom-model",
        "temperature": 0.1,
        "seed": 3,
    }
    assert (out_dir / "context.json").exists()
    assert (out_dir / "methods.md").exists()
    assert (out_dir / "executive_summary.md").exists()
    assert (out_dir / "qc_narrative.md").exists()
