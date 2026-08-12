from pathlib import Path

from autoclean.data.builtins.tasks.auditory.ASSR_40Hz import config as assr_config
from autoclean.data.builtins.tasks.auditory.MMN_Standard import config as mmn_config
from autoclean.data.builtins.tasks.resting.RestingEyesClosed import (
    config as resting_closed_config,
)
from autoclean.data.builtins.tasks.resting.RestingEyesOpen import (
    config as resting_open_config,
)
from autoclean.tasks.RestingState_Basic import config as resting_basic_config
from autoclean.tasks.RestingState_Basic_128 import config as resting_basic_128_config
from autoclean.tasks.RestingState_BasicWavelet import (
    config as resting_basic_wavelet_config,
)
from autoclean.templates.custom_task_template import config as custom_template_config
from autoclean.utils.template_renderer import render_template


def _reprocess_context(**overrides):
    context = {
        "timestamp": "2026-08-12T00:00:00",
        "original_file": "sub-01.raw",
        "fix_type": "ica",
        "class_name": "ReprocessTask",
        "bad_channels": [],
        "rejected_ica": [],
    }
    context.update(overrides)
    return context


def test_resting_task_configs_default_component_classifier_to_iclabel() -> None:
    configs = [
        resting_basic_config,
        resting_basic_128_config,
        resting_basic_wavelet_config,
        resting_open_config,
        resting_closed_config,
    ]

    assert {
        config["component_rejection"]["method"] for config in configs
    } == {"iclabel"}


def test_auditory_task_configs_default_component_classifier_to_iclabel() -> None:
    configs = [assr_config, mmn_config]

    assert {
        config["component_rejection"]["method"] for config in configs
    } == {"iclabel"}


def test_custom_task_template_defaults_component_classifier_to_iclabel() -> None:
    assert custom_template_config["component_rejection"]["method"] == "iclabel"


def test_reprocess_template_defaults_component_classifier_to_iclabel() -> None:
    template = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "autoclean"
        / "templates"
        / "reprocess_with_overrides.jinja"
    )

    rendered = render_template(template, _reprocess_context())

    assert '"method": "iclabel"' in rendered
    assert "classify_ica_components(method=\"iclabel\"" in rendered


def test_reprocess_template_preserves_explicit_icvision_override() -> None:
    template = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "autoclean"
        / "templates"
        / "reprocess_with_overrides.jinja"
    )

    rendered = render_template(
        template,
        _reprocess_context(ica_classification_method="icvision"),
    )

    assert '"method": "icvision"' in rendered
    assert "classify_ica_components(method=\"icvision\"" in rendered


def test_ica_fitting_method_defaults_are_unchanged() -> None:
    assert resting_basic_config["ICA"]["value"]["method"] == "infomax"
    assert custom_template_config["ICA"]["value"]["method"] == "fastica"
