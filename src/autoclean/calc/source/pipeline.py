"""Composable pipeline utilities for source-space analyses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import mne
from mne import SourceEstimate

from .connectivity import (
    calculate_aec_connectivity,
    calculate_source_connectivity,
)
from .conversion import convert_stc_to_eeg
from .fooof import (
    calculate_fooof_aperiodic,
    calculate_fooof_periodic,
    calculate_vertex_peak_frequencies,
)
from .pac import calculate_source_pac
from .psd import calculate_source_psd
from .vertex import (
    calculate_vertex_level_spectral_power,
    calculate_vertex_psd_for_fooof,
)

Runner = Callable[..., Any]


@dataclass
class PipelineContext:
    """Shared context passed to each pipeline step."""

    subject_id: str
    output_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineData:
    """Container for artefacts shared between pipeline steps."""

    raw: Optional[mne.io.Raw] = None
    epochs: Optional[mne.Epochs] = None
    stc: Optional[SourceEstimate] = None
    stc_list: Optional[List[SourceEstimate]] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

    def get_stcs(self) -> List[SourceEstimate]:
        """Return a list of source estimates available to the pipeline."""
        if self.stc_list:
            return list(self.stc_list)
        if self.stc is not None:
            return [self.stc]
        raise ValueError("Pipeline data does not include source estimates")


@dataclass
class PipelineStep:
    """Definition of a pipeline step."""

    name: str
    runner: Runner
    default_kwargs: Dict[str, Any] = field(default_factory=dict)

    def execute(
        self,
        data: PipelineData,
        context: PipelineContext,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Run the step with merged kwargs and return its result."""
        kwargs: Dict[str, Any] = dict(self.default_kwargs)
        if overrides:
            kwargs.update(dict(overrides))
        return self.runner(data, context, **kwargs)


class SourceAnalysisPipeline:
    """Pipeline orchestrating independent analysis steps."""

    def __init__(self, loader: Callable[[Path], PipelineData], steps: Iterable[PipelineStep]):
        self.loader = loader
        self.steps = list(steps)

    def run(
        self,
        data_path: Path | str,
        subject_id: str,
        output_dir: Path | str,
        step_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineData:
        """Run the configured pipeline and return populated data."""
        data = self.loader(Path(data_path))
        context = PipelineContext(
            subject_id=subject_id,
            output_dir=Path(output_dir),
            metadata=metadata or {},
        )
        context.output_dir.mkdir(parents=True, exist_ok=True)

        for step in self.steps:
            overrides = step_overrides.get(step.name) if step_overrides else None
            result = step.execute(data, context, overrides)
            data.results[step.name] = result

        return data


def load_source_estimates_from_directory(
    directory: Path | str,
    pattern: Optional[str] = None,
    *,
    allow_single: bool = True,
) -> PipelineData:
    """Load source estimates saved in a directory."""
    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    if pattern is not None:
        candidates = sorted(path.glob(pattern))
    else:
        candidates = sorted(path.glob("*.stc.h5"))
        if not candidates:
            candidates = sorted(path.glob("*psd-stc.h5"))
        if not candidates:
            stc_files = list(path.glob("*.stc"))
            base_names = sorted(
                {
                    f.name.replace("-lh.stc", "").replace("-rh.stc", "")
                    for f in stc_files
                }
            )
            candidates = [path / name for name in base_names]

    if not candidates:
        raise FileNotFoundError(
            f"No source estimate files found in {path}. Provide a pattern if files use a custom suffix."
        )

    stcs = [mne.read_source_estimate(str(candidate), verbose=False) for candidate in candidates]
    data = PipelineData(stc_list=stcs)
    if allow_single and len(stcs) == 1:
        data.stc = stcs[0]
    return data


def _prepare_step_kwargs(
    context: PipelineContext,
    step_name: str,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(kwargs)
    output_dir = Path(merged.pop("output_dir", context.output_dir / step_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    merged["output_dir"] = str(output_dir)
    merged.setdefault("subject_id", context.subject_id)
    return merged


def psd_step(name: str = "psd", **default_kwargs: Any) -> PipelineStep:
    """Create a pipeline step that computes ROI power spectra."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_source_psd(stcs, **merged)
        data.extras[f"{name}_psd"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def connectivity_step(name: str = "connectivity", **default_kwargs: Any) -> PipelineStep:
    """Create a pipeline step computing connectivity metrics."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_source_connectivity(stcs, **merged)
        data.extras[f"{name}_connectivity"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def aec_step(name: str = "aec", **default_kwargs: Any) -> PipelineStep:
    """Create a step that calculates amplitude envelope correlation."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_aec_connectivity(stcs, **merged)
        data.extras[f"{name}_connectivity"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def pac_step(name: str = "pac", **default_kwargs: Any) -> PipelineStep:
    """Create a step that runs phase-amplitude coupling analysis."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_source_pac(stcs, **merged)
        data.extras[f"{name}_results"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def vertex_power_step(name: str = "vertex_power", **default_kwargs: Any) -> PipelineStep:
    """Create a step that computes vertex-level spectral power."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_vertex_level_spectral_power(stcs, **merged)
        data.extras[f"{name}_results"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def vertex_psd_step(name: str = "vertex_psd", **default_kwargs: Any) -> PipelineStep:
    """Create a step that converts source data into PSD source estimates."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        stc_psd, file_path = calculate_vertex_psd_for_fooof(stcs, **merged)
        data.extras["stc_psd"] = stc_psd
        data.extras[f"{name}_file"] = file_path
        return stc_psd, file_path

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def fooof_aperiodic_step(name: str = "fooof_aperiodic", **default_kwargs: Any) -> PipelineStep:
    """Create a step that runs aperiodic FOOOF fits."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stc_psd: Optional[SourceEstimate] = kwargs.pop(
            "stc_psd", data.extras.get("stc_psd")
        )
        if stc_psd is None:
            raise ValueError(
                "FOOOF analysis requires PSD source estimates. Run vertex_psd_step first or pass stc_psd explicitly."
            )
        merged = _prepare_step_kwargs(context, name, kwargs)
        subject = merged.pop("subject_id", context.subject_id)
        output_dir = merged.pop("output_dir")
        result = calculate_fooof_aperiodic(
            stc_psd,
            subject_id=subject,
            output_dir=output_dir,
            **merged,
        )
        data.extras[f"{name}_results"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def fooof_periodic_step(name: str = "fooof_periodic", **default_kwargs: Any) -> PipelineStep:
    """Create a step that runs periodic FOOOF analysis."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stc_psd: Optional[SourceEstimate] = kwargs.pop(
            "stc_psd", data.extras.get("stc_psd")
        )
        if stc_psd is None:
            raise ValueError("Periodic FOOOF analysis requires PSD source estimates.")
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_fooof_periodic(stc_psd, **merged)
        data.extras[f"{name}_results"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def fooof_peak_step(name: str = "fooof_peaks", **default_kwargs: Any) -> PipelineStep:
    """Create a step that extracts spectral peaks from PSD source estimates."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stc_psd: Optional[SourceEstimate] = kwargs.pop(
            "stc_psd", data.extras.get("stc_psd")
        )
        if stc_psd is None:
            raise ValueError("Peak extraction requires PSD source estimates.")
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = calculate_vertex_peak_frequencies(stc_psd, **merged)
        data.extras[f"{name}_results"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


def conversion_step(name: str = "conversion", **default_kwargs: Any) -> PipelineStep:
    """Create a step that exports source data to EEG formats."""

    def runner(data: PipelineData, context: PipelineContext, **kwargs: Any):
        stcs = data.get_stcs()
        merged = _prepare_step_kwargs(context, name, kwargs)
        result = convert_stc_to_eeg(stcs, **merged)
        data.extras[f"{name}_results"] = result
        return result

    return PipelineStep(name=name, runner=runner, default_kwargs=default_kwargs)


__all__ = [
    "PipelineContext",
    "PipelineData",
    "PipelineStep",
    "SourceAnalysisPipeline",
    "load_source_estimates_from_directory",
    "psd_step",
    "connectivity_step",
    "aec_step",
    "pac_step",
    "vertex_power_step",
    "vertex_psd_step",
    "fooof_aperiodic_step",
    "fooof_periodic_step",
    "fooof_peak_step",
    "conversion_step",
]
