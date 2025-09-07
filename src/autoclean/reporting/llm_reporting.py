"""LLM-backed textual reporting utilities."""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from autoclean.utils.auth import require_authentication
from autoclean.utils.logging import message

# ---------- Data models ----------


@dataclass
class ICAStats:
    """Information about ICA processing."""

    method: str
    n_components: Optional[int]
    removed_indices: List[int]
    labels_histogram: Dict[str, int]
    classifier: Optional[str]


@dataclass
class EpochStats:
    """Statistics about epoching."""

    tmin: Optional[float]
    tmax: Optional[float]
    baseline: Optional[Tuple[Optional[float], Optional[float]]]
    total_epochs: Optional[int]
    kept_epochs: Optional[int]
    rejected_epochs: Optional[int]
    rejection_rules: Dict[str, Any]


@dataclass
class FilterParams:
    """Filtering parameters."""

    l_freq: Optional[float]
    h_freq: Optional[float]
    notch_freqs: List[float]
    notch_widths: Optional[float]


@dataclass
class RunContext:
    """Context information for a pipeline run."""

    run_id: str
    dataset_name: Optional[str]
    input_file: str
    montage: Optional[str]
    resample_hz: Optional[float]
    reference: Optional[str]
    filter_params: FilterParams
    ica: Optional[ICAStats]
    epochs: Optional[EpochStats]
    durations_s: Optional[float]
    n_channels: Optional[int]
    bids_root: Optional[str]
    bids_subject_id: Optional[str]
    pipeline_version: str
    mne_version: Optional[str]
    compliance_user: Optional[Dict[str, Any]]
    notes: List[str]
    figures: Dict[str, str]


# ---------- Deterministic “methods” templater ----------


def render_methods(context: RunContext) -> str:
    """Render a deterministic methods paragraph."""

    fp = context.filter_params
    e = context.epochs
    ica = context.ica
    parts = []
    parts.append(
        f"EEG preprocessing was performed using AutoCleanEEG v{context.pipeline_version} "
        f"(MNE-Python {context.mne_version or 'n/a'}). Data were converted to BIDS and organized under "
        f"{Path(context.bids_root).name if context.bids_root else 'a BIDS-compliant folder'}."
    )
    if context.resample_hz:
        parts.append(f"Signals were resampled to {context.resample_hz:.0f} Hz.")
    if fp.l_freq or fp.h_freq or fp.notch_freqs:
        band = []
        if fp.l_freq is not None:
            band.append(f"high-pass at {fp.l_freq:g} Hz")
        if fp.h_freq is not None:
            band.append(f"low-pass at {fp.h_freq:g} Hz")
        if fp.notch_freqs:
            band.append(
                f"notch at {', '.join(map(lambda x: str(int(x)), fp.notch_freqs))} Hz"
            )
        parts.append("Data were filtered (" + "; ".join(band) + ").")
    if context.reference:
        parts.append(f"Signals were re-referenced to {context.reference}.")
    if ica:
        parts.append(
            f"Independent Component Analysis was performed using {ica.method} "
            f"({ica.n_components if ica.n_components is not None else 'n'} components). "
            f"Components were classified with {ica.classifier or 'unspecified'} and "
            f"{len(ica.removed_indices)} components were removed."
        )
    if e and (e.tmin is not None or e.tmax is not None):
        base = ""
        if e.baseline and any(x is not None for x in e.baseline):
            base = f" with baseline correction ({e.baseline[0]} to {e.baseline[1]} s)"
        parts.append(f"Data were epoched from {e.tmin}s to {e.tmax}s{base}.")
        if e.total_epochs is not None:
            parts.append(
                f"Epoch counts: total={e.total_epochs}, kept={e.kept_epochs}, rejected={e.rejected_epochs}."
            )
        if e.rejection_rules:
            parts.append(f"Automated epoch rejection thresholds: {e.rejection_rules}.")
    if context.montage:
        parts.append(f"Electrodes were assigned to the {context.montage} montage.")
    return " ".join(parts)


# ---------- LLM provider (OpenAI-compatible; easy to swap) ----------


class LLMClient:
    """Minimal client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        seed: Optional[int] = 0,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def generate_json(self, system: str, user: str, schema_hint: str) -> Dict[str, Any]:
        """Generate a JSON object from the LLM using a strict schema."""

        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {
                    "role": "system",
                    "content": f"Output must be a JSON object matching this schema:\n{schema_hint}",
                },
            ],
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content)


# ---------- Prompts ----------


def _exec_summary_prompt(context: RunContext) -> Tuple[str, str, str]:
    system = (
        "You are an EEG preprocessing reporting assistant. "
        "You strictly summarize ONLY the facts contained in the provided JSON context. "
        "If a detail is not present, write 'not specified'. No speculation."
    )
    user = (
        "Produce a concise 6-10 sentence executive summary of the run, covering: "
        "dataset/run identifiers; preprocessing steps (resample, filters, reference, montage); "
        "ICA method and removal counts with label tallies; epoch yield and main rejection causes; "
        "QC highlights (e.g., residual line noise, flat/bad channels); any flags/notes. "
        "Refer to figures by filename when available. Here is the context JSON:\n\n"
        + json.dumps(asdict(context), indent=2)
    )
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "bullets": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "bullets"],
        },
        indent=2,
    )
    return system, user, schema


def _qc_narrative_prompt(context: RunContext) -> Tuple[str, str, str]:
    system = (
        "You write a short QC narrative for EEG preprocessing using only provided context. "
        "Prefer precise thresholds and counts when present. Include 2-3 recommendations."
    )
    user = "Write a 1-2 paragraph QC narrative. Context:\n\n" + json.dumps(
        asdict(context), indent=2
    )
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "recommendations": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary"],
        },
        indent=2,
    )
    return system, user, schema


# ---------- Helpers ----------


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    message("success", f"Wrote {path}")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    message("success", f"Wrote {path}")


# ---------- Public API ----------


@require_authentication
def create_reports(
    context: RunContext,
    out_dir: Path,
    llm: Optional[LLMClient] = None,
    emit_methods: bool = True,
    emit_exec: bool = True,
    emit_qc: bool = True,
) -> None:
    """Create reports for a pipeline run."""

    _write_json(out_dir / "context.json", asdict(context))

    if emit_methods:
        methods_md = render_methods(context)
        _write_text(out_dir / "methods.md", methods_md)

    if llm is None:
        llm_enabled = bool(os.getenv("OPENAI_API_KEY"))
        if not llm_enabled:
            message("warning", "OPENAI_API_KEY not set; skipping LLM summaries.")
            emit_exec = emit_qc = False
        else:
            llm = LLMClient()

    if emit_exec and llm:
        system, user, schema = _exec_summary_prompt(context)
        js = llm.generate_json(system, user, schema)
        title = js.get("title", "Run summary")
        bullets = js.get("bullets", [])
        notes = js.get("notes", [])
        md = f"# {title}\n\n" + "\n".join(f"- {b}" for b in bullets)
        if notes:
            md += "\n\n**Notes**\n" + "\n".join(f"- {n}" for n in notes)
        _write_text(out_dir / "executive_summary.md", md)
        _append_trace(out_dir, "exec_summary", system, user, js, llm)

    if emit_qc and llm:
        system, user, schema = _qc_narrative_prompt(context)
        js = llm.generate_json(system, user, schema)
        md = js["summary"]
        if js.get("recommendations"):
            md += "\n\n**Recommendations**\n" + "\n".join(
                f"- {r}" for r in js["recommendations"]
            )
        _write_text(out_dir / "qc_narrative.md", md)
        _append_trace(out_dir, "qc_narrative", system, user, js, llm)


def _append_trace(
    out_dir: Path,
    kind: str,
    system: str,
    user: str,
    result: Dict[str, Any],
    llm: LLMClient,
) -> None:
    rec = {
        "ts": time.time(),
        "kind": kind,
        "model": llm.model,
        "temperature": llm.temperature,
        "seed": llm.seed,
        "system": system,
        "user": _hash_prompt(user),
        "result": result,
    }
    path = out_dir / "llm_trace.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _hash_prompt(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
def run_context_from_dict(data: Dict[str, Any]) -> RunContext:
    """Coerce a plain dict (e.g., loaded from JSON) into a RunContext.

    Ensures nested fields are proper dataclass instances and fills sensible
    defaults for optional collections.
    """

    def _filter_params(fp: Any) -> FilterParams:
        if isinstance(fp, FilterParams):
            return fp
        fp = fp or {}
        return FilterParams(
            l_freq=fp.get("l_freq"),
            h_freq=fp.get("h_freq"),
            notch_freqs=list(fp.get("notch_freqs") or []),
            notch_widths=fp.get("notch_widths"),
        )

    def _ica(ica: Any) -> Optional[ICAStats]:
        if ica is None:
            return None
        if isinstance(ica, ICAStats):
            return ica
        return ICAStats(
            method=str(ica.get("method", "unspecified")),
            n_components=ica.get("n_components"),
            removed_indices=list(ica.get("removed_indices") or []),
            labels_histogram=dict(ica.get("labels_histogram") or {}),
            classifier=ica.get("classifier"),
        )

    def _epochs(e: Any) -> Optional[EpochStats]:
        if e is None:
            return None
        if isinstance(e, EpochStats):
            return e
        return EpochStats(
            tmin=e.get("tmin"),
            tmax=e.get("tmax"),
            baseline=tuple(e.get("baseline")) if e.get("baseline") is not None else None,
            total_epochs=e.get("total_epochs"),
            kept_epochs=e.get("kept_epochs"),
            rejected_epochs=e.get("rejected_epochs"),
            rejection_rules=dict(e.get("rejection_rules") or {}),
        )

    return RunContext(
        run_id=str(data.get("run_id", "")),
        dataset_name=data.get("dataset_name"),
        input_file=str(data.get("input_file", "")),
        montage=data.get("montage"),
        resample_hz=data.get("resample_hz"),
        reference=data.get("reference"),
        filter_params=_filter_params(data.get("filter_params")),
        ica=_ica(data.get("ica")),
        epochs=_epochs(data.get("epochs")),
        durations_s=data.get("durations_s"),
        n_channels=data.get("n_channels"),
        bids_root=data.get("bids_root"),
        bids_subject_id=data.get("bids_subject_id"),
        pipeline_version=str(data.get("pipeline_version", "")),
        mne_version=data.get("mne_version"),
        compliance_user=data.get("compliance_user"),
        notes=list(data.get("notes") or []),
        figures=dict(data.get("figures") or {}),
    )
