"""Prior preprocessing detection and reporting helpers.

The detector keeps documented metadata separate from signal-based inference. It
is intentionally conservative: findings are warnings and confidence labels, not
hard failures, unless callers opt into strict validation handling.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CONFIDENCE_DOCUMENTED = "documented"
CONFIDENCE_LIKELY = "likely"
CONFIDENCE_POSSIBLE = "possible"
CONFIDENCE_UNKNOWN = "unknown"
UNKNOWN = "unknown"
UNAVAILABLE = "unavailable"
SCHEMA_VERSION = "1.0"
POWERLINE_FREQS = (50.0, 60.0, 100.0, 120.0)


def detect_prior_preprocessing(
    eeg_data: Any | None = None,
    *,
    import_metadata: Mapping[str, Any] | None = None,
    provenance_summary: Mapping[str, Any] | None = None,
    task_config: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Return a conservative prior-preprocessing status summary.

    Parameters are deliberately plain mappings/objects so tests and importers can
    pass MNE Raw/Epochs, light stubs, or #202 EEGLAB provenance summaries.
    """

    import_metadata = import_metadata or {}
    task_config = task_config or {}
    documented = _extract_documented_metadata(
        eeg_data=eeg_data,
        import_metadata=import_metadata,
        provenance_summary=provenance_summary,
    )
    signal = _infer_from_signal(eeg_data)
    findings = _merge_findings(documented, signal)
    warnings = build_prior_preprocessing_warnings(findings, task_config)
    strict_violations = warnings if strict else []
    summary_row = _build_summary_row(findings, documented, signal, warnings)

    return {
        "schema_version": SCHEMA_VERSION,
        "documented_metadata": documented,
        "signal_inference": signal,
        "findings": findings,
        "warnings": warnings,
        "strict_validation": bool(strict),
        "strict_violations": strict_violations,
        "summary_row": summary_row,
        "artifact_paths": {},
    }


def build_prior_preprocessing_dataset_summary(
    summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-file prior-preprocessing summaries."""

    rows = [dict(summary.get("summary_row", {})) for summary in summaries]
    warning_counts = Counter(
        warning for summary in summaries for warning in summary.get("warnings", [])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "warning_counts": dict(sorted(warning_counts.items())),
        "warnings": _dataset_consistency_warnings(rows),
    }


def write_prior_preprocessing_artifacts(
    summary: dict[str, Any], output_dir: Path, stem: str
) -> dict[str, str]:
    """Write machine-readable, Markdown, and aggregate summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_prior_preprocessing.json"
    report_path = output_dir / f"{stem}_prior_preprocessing.md"
    dataset_path = output_dir / "prior_preprocessing_dataset_summary.json"
    artifact_paths = {
        "json": str(json_path),
        "report": str(report_path),
        "dataset_summary": str(dataset_path),
    }
    summary["artifact_paths"] = artifact_paths
    json_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    report_path.write_text(render_prior_preprocessing_report(summary), encoding="utf-8")
    dataset_summary = _append_prior_preprocessing_dataset_summary(dataset_path, summary)
    dataset_path.write_text(
        json.dumps(_json_safe(dataset_summary), indent=2), encoding="utf-8"
    )
    return artifact_paths


def _append_prior_preprocessing_dataset_summary(
    dataset_path: Path, summary: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    warning_counts: Counter[str] = Counter()
    if dataset_path.exists():
        try:
            existing = json.loads(dataset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        rows = [dict(row) for row in existing.get("rows", [])]
        warning_counts.update(existing.get("warning_counts", {}))

    row = dict(summary.get("summary_row", {}))
    source_file = row.get("source_file")
    if source_file:
        rows = [
            existing for existing in rows if existing.get("source_file") != source_file
        ]
    rows.append(row)
    warning_counts.update(summary.get("warnings", []))

    return {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "warning_counts": dict(sorted(warning_counts.items())),
        "warnings": _dataset_consistency_warnings(rows),
    }


def render_prior_preprocessing_report(summary: Mapping[str, Any]) -> str:
    """Render a concise Markdown report for human review."""

    row = summary["summary_row"]
    lines = [
        f"# Prior Preprocessing Status: {row.get('source_file', UNAVAILABLE)}",
        "",
        "## Documented Metadata",
    ]
    documented = summary.get("documented_metadata", {})
    for key in (
        "file_type",
        "importer",
        "data_type",
        "sampling_rate",
        "channel_count",
        "epoch_count",
        "epoch_window",
        "reference",
        "event_labels",
        "event_codes",
        "event_counts",
    ):
        lines.append(f"- {key}: {_compact(documented.get(key, UNAVAILABLE))}")

    lines.extend(["", "## Signal-Based Inference"])
    for key, finding in summary.get("findings", {}).items():
        lines.append(
            f"- {key}: {finding.get('confidence', CONFIDENCE_UNKNOWN)}"
            f" ({finding.get('evidence', UNAVAILABLE)})"
        )

    lines.extend(["", "## Warnings"])
    warnings = summary.get("warnings", [])
    (
        lines.extend(f"- {warning}" for warning in warnings)
        if warnings
        else lines.append("- none")
    )
    lines.append("")
    return "\n".join(lines)


def resolve_prior_preprocessing_dir(autoclean_dict: Mapping[str, Any]) -> Path:
    """Resolve artifact directory using existing run directory conventions."""

    if autoclean_dict.get("reports_dir"):
        return (
            Path(autoclean_dict["reports_dir"]) / "run_reports" / "prior_preprocessing"
        )
    if autoclean_dict.get("metadata_dir"):
        return Path(autoclean_dict["metadata_dir"]) / "prior_preprocessing"
    return Path(autoclean_dict["unprocessed_file"]).parent / "prior_preprocessing"


def resolve_prior_preprocessing_provenance(
    import_metadata: Mapping[str, Any], autoclean_dict: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    """Return the richest available #202-style provenance summary.

    Prefer a caller-provided full summary. Fall back to compact import metadata
    (`metadata["import_eeg"]["eeglab_provenance"]`) when #202 has already run but
    only persisted summary fields are available.
    """

    direct = autoclean_dict.get("eeglab_provenance")
    if isinstance(direct, Mapping):
        return direct

    from_import = import_metadata.get("eeglab_provenance")
    if isinstance(from_import, Mapping):
        return from_import

    return None


def build_prior_preprocessing_warnings(
    findings: Mapping[str, Mapping[str, Any]], task_config: Mapping[str, Any]
) -> list[str]:
    """Warn when task config may repeat likely/documented prior steps."""

    warnings: list[str] = []
    if _step_enabled(task_config, "filtering"):
        if _is_known(findings.get("highpass_filter")) or _is_known(
            findings.get("lowpass_filter")
        ):
            warnings.append("Task filtering may repeat prior high/low-pass filtering")
        notch_freqs = _configured_notches(task_config)
        repeated = [
            freq
            for freq in notch_freqs
            if _is_known(findings.get(f"notch_filter_{int(freq)}hz"))
        ]
        if repeated:
            joined = ", ".join(str(int(freq)) for freq in repeated)
            warnings.append(
                f"Task notch filtering may repeat prior notch at {joined} Hz"
            )

    baseline_step = _nested(task_config, "epoch_settings", "remove_baseline")
    if isinstance(baseline_step, Mapping) and baseline_step.get("enabled", False):
        if _is_known(findings.get("baseline_applied")):
            warnings.append(
                "Task baseline correction may repeat prior baseline correction"
            )

    if _step_enabled(task_config, "ICA") and _is_known(findings.get("ica_present")):
        warnings.append("Task ICA may repeat prior ICA decomposition")

    if _step_enabled(task_config, "reference_step") and _is_known(
        findings.get("reference")
    ):
        warnings.append("Task rereferencing may repeat or replace documented reference")

    if _step_enabled(task_config, "epoch_settings") and _is_known(
        findings.get("epoching")
    ):
        warnings.append("Task epoching may repeat prior epoching")

    expected_events = _nested(task_config, "epoch_settings", "event_id")
    documented_events = findings.get("event_codes", {})
    if isinstance(expected_events, Mapping) and documented_events.get("value") not in (
        None,
        UNKNOWN,
        UNAVAILABLE,
    ):
        observed = {str(item) for item in _as_list(documented_events.get("value"))}
        expected = {str(item) for item in expected_events.values()}
        if expected and observed and expected.isdisjoint(observed):
            warnings.append("Task event codes do not match documented event codes")

    return warnings


def _extract_documented_metadata(
    *,
    eeg_data: Any | None,
    import_metadata: Mapping[str, Any],
    provenance_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    documented = dict((provenance_summary or {}).get("documented_provenance", {}))
    summary_row = dict((provenance_summary or {}).get("summary_row", {}))

    info = getattr(eeg_data, "info", {}) or {}
    ch_names = list(getattr(eeg_data, "ch_names", []) or [])
    is_epochs = _is_epochs_like(eeg_data, import_metadata)
    events = getattr(eeg_data, "events", None)

    channel_types = _channel_types(eeg_data, ch_names)
    source_file = import_metadata.get("unprocessedFile") or summary_row.get(
        "source_file"
    )
    sampling_rate = _first_known(
        summary_row.get("srate"),
        documented.get("srate"),
        import_metadata.get("sampleRate"),
        info.get("sfreq"),
    )
    channel_count = _first_known(
        summary_row.get("nbchan"),
        documented.get("nbchan"),
        import_metadata.get("channelCount"),
        len(ch_names) or UNAVAILABLE,
    )
    epoch_count = _first_known(
        summary_row.get("trials"),
        documented.get("trials"),
        import_metadata.get("n_epochs"),
        _safe_len(eeg_data) if is_epochs else UNAVAILABLE,
    )
    epoch_window = _first_known(
        summary_row.get("epoch_window"),
        documented.get("epoch_window"),
        _epoch_window(eeg_data, import_metadata),
    )
    events_doc = (
        documented.get("events", {})
        if isinstance(documented.get("events"), Mapping)
        else {}
    )
    channels_doc = (
        documented.get("channels", {})
        if isinstance(documented.get("channels"), Mapping)
        else {}
    )

    return {
        "source_file": source_file or UNAVAILABLE,
        "file_type": import_metadata.get("file_format", UNAVAILABLE),
        "importer": import_metadata.get("plugin_used", UNAVAILABLE),
        "data_type": "epochs" if is_epochs else "raw",
        "continuous_or_epoched": "epoched" if is_epochs else "continuous",
        "sampling_rate": sampling_rate,
        "channel_count": channel_count,
        "epoch_count": epoch_count,
        "epoch_window": epoch_window,
        "reference": _first_known(
            summary_row.get("reference"), documented.get("reference"), _reference(info)
        ),
        "ica": documented.get("ica", UNAVAILABLE),
        "iclabel": documented.get("iclabel", UNAVAILABLE),
        "bad_or_interpolated_channels": _first_known(
            documented.get("interpolation"), info.get("bads"), UNAVAILABLE
        ),
        "channel_labels": _first_known(
            channels_doc.get("labels"), ch_names or UNAVAILABLE
        ),
        "channel_types": _first_known(
            channels_doc.get("types"), channel_types or UNAVAILABLE
        ),
        "event_labels": _first_known(
            events_doc.get("labels"), import_metadata.get("event_dict"), UNAVAILABLE
        ),
        "event_codes": _first_known(
            events_doc.get("codes"),
            _event_codes(events),
            import_metadata.get("unique_event_types"),
            UNAVAILABLE,
        ),
        "event_counts": _first_known(
            events_doc.get("counts"), _event_counts(events), UNAVAILABLE
        ),
        "history": documented.get("history", UNAVAILABLE),
        "comments": documented.get("comments", UNAVAILABLE),
        "etc_keys": documented.get("etc_keys", UNAVAILABLE),
        "provenance_integration": _provenance_integration_status(provenance_summary),
    }


def _infer_from_signal(eeg_data: Any | None) -> dict[str, Any]:
    if eeg_data is None:
        return _unknown_signal_inference("no signal object provided")

    info = getattr(eeg_data, "info", {}) or {}
    ch_names = list(getattr(eeg_data, "ch_names", []) or [])
    data = _get_data(eeg_data)
    times = np.asarray(getattr(eeg_data, "times", []), dtype=float)
    sfreq = _to_float(info.get("sfreq"))
    channel_types = _channel_types(eeg_data, ch_names)
    inference: dict[str, Any] = {
        "data_shape": list(data.shape) if data is not None else UNAVAILABLE,
        "eog_ecg_misc_channel_presence": _infer_aux_channels(channel_types, ch_names),
        "channel_count_suggests_dropping": _infer_channel_count_drop(len(ch_names)),
        "baseline_applied": _infer_baseline(data, times),
        "highpass_filter": _infer_highpass(data, sfreq),
        "lowpass_filter": _infer_lowpass(data, sfreq),
        "ica_pruned": _possible("ICA pruning cannot be confirmed from signal alone"),
        "ica_capable": _finding(
            CONFIDENCE_POSSIBLE if len(ch_names) >= 2 else CONFIDENCE_UNKNOWN,
            len(ch_names) >= 2,
            (
                f"{len(ch_names)} channels available for ICA"
                if ch_names
                else "channel names unavailable"
            ),
        ),
    }
    inference.update(_infer_notches(data, sfreq))
    return inference


def _merge_findings(
    documented: Mapping[str, Any], signal: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    findings = {
        "epoching": _documented_or_unknown(
            documented.get("continuous_or_epoched") == "epoched",
            documented.get("continuous_or_epoched"),
            "import metadata data_type",
        ),
        "sampling_rate": _documented_value(documented.get("sampling_rate")),
        "epoch_window": _documented_value(documented.get("epoch_window")),
        "reference": _documented_value(documented.get("reference")),
        "event_codes": _documented_value(documented.get("event_codes")),
        "event_counts": _documented_value(documented.get("event_counts")),
        "ica_present": _ica_present(documented.get("ica")),
        "iclabel_present": _iclabel_present(documented.get("iclabel")),
        "interpolated_channels": _documented_value(
            documented.get("bad_or_interpolated_channels")
        ),
        "channel_types": _documented_value(documented.get("channel_types")),
    }

    text = " ".join(
        str(documented.get(key, "")) for key in ("history", "comments", "etc_keys")
    ).lower()
    for name, needles in {
        "highpass_filter": ("highpass", "high-pass", "pop_eegfilt", "eegfilt"),
        "lowpass_filter": ("lowpass", "low-pass", "pop_eegfilt", "eegfilt"),
        "baseline_applied": ("baseline", "pop_rmbase", "rmbase"),
        "ica_pruned": ("subcomp", "reject component", "ica prune"),
    }.items():
        documented_finding = _documented_or_unknown(
            any(needle in text for needle in needles), True, "EEG history/comments"
        )
        findings[name] = _prefer_documented(documented_finding, signal.get(name))

    for freq in POWERLINE_FREQS:
        key = f"notch_filter_{int(freq)}hz"
        documented_finding = _documented_or_unknown(
            any(
                token in text
                for token in (f"notch {int(freq)}", f"{int(freq)}hz", f"{int(freq)} hz")
            ),
            True,
            "EEG history/comments",
        )
        findings[key] = _prefer_documented(documented_finding, signal.get(key))

    findings["eog_ecg_misc_channel_presence"] = signal.get(
        "eog_ecg_misc_channel_presence", _unknown("channel type inference unavailable")
    )
    findings["channel_count_suggests_dropping"] = signal.get(
        "channel_count_suggests_dropping", _unknown("channel count unavailable")
    )
    return findings


def _infer_notches(
    data: np.ndarray | None, sfreq: float | None
) -> dict[str, dict[str, Any]]:
    result = {}
    freqs, spectrum = _mean_spectrum(data, sfreq)
    for freq in POWERLINE_FREQS:
        key = f"notch_filter_{int(freq)}hz"
        if freqs is None or spectrum is None or sfreq is None or freq >= sfreq / 2:
            result[key] = _unknown("frequency outside available Nyquist range")
            continue
        ratio = _band_power_ratio(
            freqs, spectrum, freq, inner_width=1.0, outer_width=4.0
        )
        if ratio is None:
            result[key] = _unknown("insufficient spectral bins")
        elif ratio < 0.45:
            result[key] = _finding(
                CONFIDENCE_LIKELY,
                True,
                f"attenuation ratio {ratio:.2f} near {freq:g} Hz",
            )
        elif ratio < 0.75:
            result[key] = _finding(
                CONFIDENCE_POSSIBLE,
                True,
                f"attenuation ratio {ratio:.2f} near {freq:g} Hz",
            )
        else:
            result[key] = _unknown(f"no clear attenuation near {freq:g} Hz")
    return result


def _infer_highpass(data: np.ndarray | None, sfreq: float | None) -> dict[str, Any]:
    freqs, spectrum = _mean_spectrum(data, sfreq)
    if freqs is None or spectrum is None:
        return _unknown("spectrum unavailable")
    low = _mean_power(freqs, spectrum, 0.1, 0.8)
    mid = _mean_power(freqs, spectrum, 2.0, 8.0)
    if low is None or mid is None or mid <= 0:
        return _unknown("insufficient low-frequency bins")
    ratio = low / mid
    if ratio < 0.25:
        return _finding(CONFIDENCE_POSSIBLE, True, f"sub-1 Hz power ratio {ratio:.2f}")
    return _unknown(f"sub-1 Hz power ratio {ratio:.2f}")


def _infer_lowpass(data: np.ndarray | None, sfreq: float | None) -> dict[str, Any]:
    freqs, spectrum = _mean_spectrum(data, sfreq)
    if freqs is None or spectrum is None or sfreq is None or sfreq < 80:
        return _unknown("spectrum unavailable or Nyquist too low")
    high_start = min(70.0, sfreq / 2 * 0.7)
    high = _mean_power(freqs, spectrum, high_start, sfreq / 2 * 0.95)
    mid = _mean_power(freqs, spectrum, 10.0, min(30.0, sfreq / 2 * 0.6))
    if high is None or mid is None or mid <= 0:
        return _unknown("insufficient high-frequency bins")
    ratio = high / mid
    if ratio < 0.2:
        return _finding(
            CONFIDENCE_POSSIBLE, True, f"high-frequency power ratio {ratio:.2f}"
        )
    return _unknown(f"high-frequency power ratio {ratio:.2f}")


def _infer_baseline(data: np.ndarray | None, times: np.ndarray) -> dict[str, Any]:
    if data is None or times.size == 0 or data.ndim < 2:
        return _unknown("data or times unavailable")
    baseline_mask = (times >= -0.25) & (times <= 0.0)
    if not np.any(baseline_mask):
        return _unknown("no pre-stimulus baseline window")
    time_axis = data.ndim - 1
    baseline = np.take(data, np.where(baseline_mask)[0], axis=time_axis)
    all_scale = float(np.nanstd(data))
    baseline_mean = float(abs(np.nanmean(baseline)))
    if all_scale <= 0:
        return _unknown("flat or unavailable signal scale")
    ratio = baseline_mean / all_scale
    if ratio < 0.05:
        return _finding(
            CONFIDENCE_POSSIBLE, True, f"baseline mean/std ratio {ratio:.3f}"
        )
    return _unknown(f"baseline mean/std ratio {ratio:.3f}")


def _infer_aux_channels(
    channel_types: Mapping[str, int], ch_names: Sequence[str]
) -> dict[str, Any]:
    present = {
        kind: count
        for kind, count in channel_types.items()
        if kind in {"eog", "ecg", "misc"}
    }
    if present:
        return _finding(
            CONFIDENCE_LIKELY, present, f"typed auxiliary channels {present}"
        )
    name_hits = [
        name
        for name in ch_names
        if any(token in name.lower() for token in ("eog", "ecg", "ekg", "emg", "misc"))
    ]
    if name_hits:
        return _finding(
            CONFIDENCE_POSSIBLE,
            name_hits,
            f"auxiliary-looking channel names {name_hits}",
        )
    return _unknown("no auxiliary channel types or names detected")


def _infer_channel_count_drop(channel_count: int) -> dict[str, Any]:
    common_counts = (32, 64, 128, 129, 256)
    if channel_count <= 0:
        return _unknown("channel count unavailable")
    if any(
        0 < expected - channel_count <= max(2, expected * 0.1)
        for expected in common_counts
    ):
        return _finding(
            CONFIDENCE_POSSIBLE,
            True,
            f"{channel_count} channels is just below a common montage count",
        )
    return _unknown(f"{channel_count} channels does not suggest dropping by itself")


def _mean_spectrum(
    data: np.ndarray | None, sfreq: float | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if data is None or sfreq is None or sfreq <= 0:
        return None, None
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 3:
        arr = arr.reshape((-1, arr.shape[-1]))
    elif arr.ndim == 2:
        pass
    elif arr.ndim == 1:
        arr = arr.reshape((1, -1))
    else:
        return None, None
    if arr.shape[-1] < 4:
        return None, None
    arr = arr - np.nanmean(arr, axis=-1, keepdims=True)
    spectrum = np.nanmean(np.abs(np.fft.rfft(arr, axis=-1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(arr.shape[-1], d=1.0 / sfreq)
    return freqs, spectrum


def _band_power_ratio(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    center: float,
    *,
    inner_width: float,
    outer_width: float,
) -> float | None:
    inner = (freqs >= center - inner_width) & (freqs <= center + inner_width)
    outer = (freqs >= center - outer_width) & (freqs <= center + outer_width) & ~inner
    inner_power = _safe_mean(spectrum[inner])
    outer_power = _safe_mean(spectrum[outer])
    if inner_power is None or outer_power is None or outer_power <= 0:
        return None
    return inner_power / outer_power


def _mean_power(
    freqs: np.ndarray, spectrum: np.ndarray, low: float, high: float
) -> float | None:
    mask = (freqs >= low) & (freqs <= high)
    return _safe_mean(spectrum[mask])


def _safe_mean(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    mean = float(np.nanmean(values))
    return mean if np.isfinite(mean) else None


def _get_data(eeg_data: Any) -> np.ndarray | None:
    if eeg_data is None or not hasattr(eeg_data, "get_data"):
        return None
    try:
        data = eeg_data.get_data()
    except TypeError:
        data = eeg_data.get_data(copy=False)
    except Exception:  # pragma: no cover - defensive for third-party objects
        return None
    return np.asarray(data, dtype=float)


def _is_epochs_like(eeg_data: Any | None, import_metadata: Mapping[str, Any]) -> bool:
    if import_metadata.get("data_type") == "epochs":
        return True
    return (
        hasattr(eeg_data, "events")
        and hasattr(eeg_data, "tmin")
        and hasattr(eeg_data, "tmax")
    )


def _channel_types(eeg_data: Any | None, ch_names: Sequence[str]) -> dict[str, int]:
    if eeg_data is not None and hasattr(eeg_data, "get_channel_types"):
        try:
            return dict(
                Counter(str(kind).lower() for kind in eeg_data.get_channel_types())
            )
        except Exception:  # pragma: no cover - defensive
            pass
    return dict(Counter(_type_from_name(name) for name in ch_names))


def _type_from_name(name: str) -> str:
    lowered = name.lower()
    if "eog" in lowered:
        return "eog"
    if "ecg" in lowered or "ekg" in lowered:
        return "ecg"
    if "misc" in lowered or "stim" in lowered or "status" in lowered:
        return "misc"
    return "eeg"


def _event_codes(events: Any) -> list[int] | str:
    if events is None:
        return UNAVAILABLE
    arr = np.asarray(events)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return UNAVAILABLE
    return sorted(int(value) for value in set(arr[:, 2].tolist()))


def _event_counts(events: Any) -> dict[str, int] | str:
    codes = _event_codes(events)
    if codes == UNAVAILABLE:
        return UNAVAILABLE
    arr = np.asarray(events)
    return {str(code): int(np.sum(arr[:, 2] == code)) for code in codes}


def _epoch_window(
    eeg_data: Any | None, import_metadata: Mapping[str, Any]
) -> dict[str, Any] | str:
    tmin = _first_known(
        import_metadata.get("tmin"), getattr(eeg_data, "tmin", UNAVAILABLE)
    )
    tmax = _first_known(
        import_metadata.get("tmax"), getattr(eeg_data, "tmax", UNAVAILABLE)
    )
    if tmin == UNAVAILABLE and tmax == UNAVAILABLE:
        return UNAVAILABLE
    return {"tmin": tmin, "tmax": tmax}


def _reference(info: Mapping[str, Any]) -> Any:
    custom_ref = info.get("custom_ref_applied", UNAVAILABLE)
    if custom_ref not in (None, False, UNAVAILABLE):
        return custom_ref
    return info.get("description", UNAVAILABLE)


def _ica_present(ica: Any) -> dict[str, Any]:
    if isinstance(ica, Mapping):
        present = any(
            isinstance(value, Mapping) and value.get("present")
            for value in ica.values()
        )
        return _documented_or_unknown(present, present, "documented ICA fields")
    return _unknown("ICA metadata unavailable")


def _iclabel_present(iclabel: Any) -> dict[str, Any]:
    if isinstance(iclabel, Mapping):
        present = bool(iclabel.get("present"))
        return _documented_or_unknown(present, present, "documented ICLabel fields")
    return _unknown("ICLabel metadata unavailable")


def _documented_value(value: Any) -> dict[str, Any]:
    if value in (None, UNAVAILABLE, UNKNOWN, [], {}):
        return _unknown("documented value unavailable")
    return _finding(CONFIDENCE_DOCUMENTED, value, "documented metadata")


def _documented_or_unknown(
    condition: bool, value: Any, evidence: str
) -> dict[str, Any]:
    if condition:
        return _finding(CONFIDENCE_DOCUMENTED, value, evidence)
    return _unknown(evidence)


def _prefer_documented(
    documented: Mapping[str, Any], signal: Mapping[str, Any] | None
) -> dict[str, Any]:
    if documented.get("confidence") == CONFIDENCE_DOCUMENTED:
        return dict(documented)
    return dict(signal or documented)


def _finding(confidence: str, value: Any, evidence: str) -> dict[str, Any]:
    return {"confidence": confidence, "value": _json_safe(value), "evidence": evidence}


def _unknown(evidence: str) -> dict[str, Any]:
    return _finding(CONFIDENCE_UNKNOWN, UNKNOWN, evidence)


def _possible(evidence: str) -> dict[str, Any]:
    return _finding(CONFIDENCE_POSSIBLE, UNKNOWN, evidence)


def _unknown_signal_inference(evidence: str) -> dict[str, Any]:
    result = {
        "data_shape": UNAVAILABLE,
        "eog_ecg_misc_channel_presence": _unknown(evidence),
        "channel_count_suggests_dropping": _unknown(evidence),
        "baseline_applied": _unknown(evidence),
        "highpass_filter": _unknown(evidence),
        "lowpass_filter": _unknown(evidence),
        "ica_pruned": _unknown(evidence),
        "ica_capable": _unknown(evidence),
    }
    result.update(
        {f"notch_filter_{int(freq)}hz": _unknown(evidence) for freq in POWERLINE_FREQS}
    )
    return result


def _provenance_integration_status(
    provenance_summary: Mapping[str, Any] | None,
) -> dict[str, str]:
    if not provenance_summary:
        return {
            "status": "unavailable",
            "detail": "No #202 provenance summary was provided.",
        }
    if provenance_summary.get("documented_provenance"):
        return {
            "status": "full_summary",
            "detail": "Full #202 documented_provenance fields are available.",
        }
    if provenance_summary.get("summary_row"):
        return {
            "status": "summary_row_only",
            "detail": "Only compact #202 summary_row metadata is available; full documented fields require #202 integration.",
        }
    return {
        "status": "unknown_shape",
        "detail": "Provided provenance summary did not match the #202 contract.",
    }


def _build_summary_row(
    findings: Mapping[str, Mapping[str, Any]],
    documented: Mapping[str, Any],
    signal: Mapping[str, Any],
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "source_file": documented.get("source_file", UNAVAILABLE),
        "data_type": documented.get("data_type", UNAVAILABLE),
        "sampling_rate": _finding_value(findings.get("sampling_rate")),
        "channel_count": documented.get("channel_count", UNAVAILABLE),
        "epoching": _finding_confidence(findings.get("epoching")),
        "epoch_window": _finding_value(findings.get("epoch_window")),
        "reference": _finding_confidence(findings.get("reference")),
        "ica_present": _finding_confidence(findings.get("ica_present")),
        "ica_pruned": _finding_confidence(findings.get("ica_pruned")),
        "baseline_applied": _finding_confidence(findings.get("baseline_applied")),
        "notch_filter_50hz": _finding_confidence(findings.get("notch_filter_50hz")),
        "notch_filter_60hz": _finding_confidence(findings.get("notch_filter_60hz")),
        "notch_filter_100hz": _finding_confidence(findings.get("notch_filter_100hz")),
        "notch_filter_120hz": _finding_confidence(findings.get("notch_filter_120hz")),
        "interpolated_or_dropped_channels": _finding_confidence(
            findings.get("interpolated_channels")
        ),
        "event_codes": _finding_confidence(findings.get("event_codes")),
        "event_code_values": _compact(_finding_value(findings.get("event_codes"))),
        "highpass_filter": _finding_confidence(findings.get("highpass_filter")),
        "lowpass_filter": _finding_confidence(findings.get("lowpass_filter")),
        "aux_channels": _finding_confidence(
            findings.get("eog_ecg_misc_channel_presence")
        ),
        "signal_shape": _compact(signal.get("data_shape", UNAVAILABLE)),
        "warning_count": len(warnings),
    }


def _dataset_consistency_warnings(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    warnings = []
    for key in ("sampling_rate", "epoch_window", "reference"):
        values = {
            str(row.get(key))
            for row in rows
            if row.get(key) not in (None, UNKNOWN, UNAVAILABLE)
        }
        if len(values) > 1:
            warnings.append(
                f"Inconsistent prior preprocessing {key}: {', '.join(sorted(values))}"
            )
    return warnings


def _configured_notches(task_config: Mapping[str, Any]) -> list[float]:
    notch = _nested(task_config, "filtering", "value", "notch_freqs")
    return [
        _to_float(value) for value in _as_list(notch) if _to_float(value) is not None
    ]


def _step_enabled(task_config: Mapping[str, Any], key: str) -> bool:
    step = task_config.get(key)
    return isinstance(step, Mapping) and bool(step.get("enabled", False))


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _is_known(finding: Mapping[str, Any] | None) -> bool:
    return bool(finding) and finding.get("confidence") in {
        CONFIDENCE_DOCUMENTED,
        CONFIDENCE_LIKELY,
        CONFIDENCE_POSSIBLE,
    }


def _finding_confidence(finding: Mapping[str, Any] | None) -> str:
    return str((finding or {}).get("confidence", CONFIDENCE_UNKNOWN))


def _finding_value(finding: Mapping[str, Any] | None) -> Any:
    return (finding or {}).get("value", UNKNOWN)


def _first_known(*values: Any) -> Any:
    for value in values:
        if value not in (None, UNAVAILABLE, UNKNOWN, [], {}):
            return value
    return UNAVAILABLE


def _safe_len(value: Any) -> int | str:
    try:
        return len(value)
    except Exception:
        return UNAVAILABLE


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _compact(value: Any) -> str:
    if value in (None, UNKNOWN, UNAVAILABLE):
        return UNAVAILABLE if value is None else str(value)
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), sort_keys=True)
