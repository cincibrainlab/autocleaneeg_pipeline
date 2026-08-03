"""Lightweight service for refitting ICA on post-review retained epochs.

This intentionally bypasses the Task/Pipeline machinery (see issue #275):
it operates directly on the epochs already exported to disk after a
reviewer rejects epochs during QA (``exclude.py``'s ``_postedit_path``),
reusing the same standalone ICA/ICLabel functions the pipeline mixins call.
Results are written to a distinct ``post_epoch_rejection`` pass subtree so
the original ICA outputs are never touched until a reviewer explicitly
promotes this pass (see ``exclude.py``'s promote/discard endpoints).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from mne.preprocessing import ICA

from autoclean.api.routes._exclude_paths import _load_epochs, _postedit_path
from autoclean.functions.ica.ica_processing import (
    classify_ica_components,
    fit_ica,
    normalize_ic_type,
)
from autoclean.functions.visualization.icvision_layouts import (
    plot_component_for_classification,
    plot_ica_topographies_overview,
)

PASS_NAME = "post_epoch_rejection"

# Conservative floors below which a fresh ICA decomposition is unreliable.
# Both must pass. See the issue #275 implementation plan for rationale --
# no channel-count-aware formula is used because that would require extra
# plumbing (montage/original n_components lookup) this lightweight service
# doesn't otherwise need.
MIN_RETAINED_EPOCHS = 40
MIN_RETAINED_DURATION_SECONDS = 120.0


class IcaRerunError(Exception):
    """Raised for user-facing failures in the ICA rerun flow."""


def check_retained_epochs_threshold(
    epochs: mne.BaseEpochs,
    min_epochs: int = MIN_RETAINED_EPOCHS,
    min_duration: float = MIN_RETAINED_DURATION_SECONDS,
) -> dict[str, Any]:
    """Check whether enough data remains to fit a reliable ICA decomposition."""
    n_epochs = len(epochs)
    epoch_duration = float(epochs.tmax - epochs.tmin)
    duration_seconds = n_epochs * epoch_duration

    reasons: list[str] = []
    if n_epochs < min_epochs:
        reasons.append(f"only {n_epochs} epochs retained (minimum {min_epochs})")
    if duration_seconds < min_duration:
        reasons.append(
            f"only {duration_seconds:.1f}s of retained data "
            f"(minimum {min_duration:.0f}s)"
        )

    return {
        "ok": not reasons,
        "n_epochs": n_epochs,
        "duration_seconds": duration_seconds,
        "reason": "; ".join(reasons) if reasons else None,
    }


def load_retained_epochs(task_root: Path, file_path: Path) -> mne.BaseEpochs:
    """Load the reviewer's retained (postedit) epochs for a file.

    Raises IcaRerunError if the reviewer hasn't rejected any epochs yet --
    there is nothing distinct from the original run to refit ICA on.
    """
    postedit_path = _postedit_path(task_root, file_path)
    if not postedit_path.exists():
        raise IcaRerunError(
            "No epoch rejections recorded for this file -- nothing to rerun "
            "ICA on. Reject at least one epoch during review first."
        )
    return _load_epochs(postedit_path)


def read_original_ica_defaults(metadata_path: Optional[Path]) -> dict[str, Any]:
    """Read the original run's ICA fit kwargs / classification method.

    Falls back to sensible defaults (mirrors ``fit_ica``/
    ``classify_ica_components`` defaults) when metadata is missing or the
    expected keys aren't present, so a rerun is always possible even for
    older runs.
    """
    ica_kwargs: dict[str, Any] = {}
    classification_method = "iclabel"
    parent_run_id: Optional[str] = None

    if metadata_path is not None and metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text())
        except Exception:
            data = {}
        parent_run_id = data.get("run_id")
        metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
        run_ica_kwargs = (
            metadata.get("step_run_ica", {}).get("ica", {}).get("ica_kwargs")
        )
        if isinstance(run_ica_kwargs, dict):
            ica_kwargs = dict(run_ica_kwargs)
        method = (
            metadata.get("classify_ica_components", {})
            .get("ica", {})
            .get("classification_method")
        )
        if isinstance(method, str) and method:
            classification_method = method

    # These are only meaningful when fitting on the same data shape/type the
    # original run used; fitting on epochs here regardless of what the
    # original was fit on is the whole point of this feature.
    ica_kwargs.pop("temp_highpass_for_ica", None)
    ica_kwargs.setdefault("max_iter", "auto")
    ica_kwargs.setdefault("random_state", 97)

    return {
        "ica_kwargs": ica_kwargs,
        "classification_method": classification_method,
        "parent_run_id": parent_run_id,
    }


def fit_and_classify(
    epochs: mne.BaseEpochs,
    ica_kwargs: dict[str, Any],
    classification_method: str = "iclabel",
) -> tuple[ICA, pd.DataFrame]:
    """Fit ICA and classify components, reusing the standalone pipeline functions."""
    ica = fit_ica(epochs, **ica_kwargs)
    ica_flags = classify_ica_components(epochs, ica, method=classification_method)
    return ica, ica_flags


def build_structured_iclabel_json(
    ica: ICA,
    ica_flags: pd.DataFrame,
    *,
    n_epochs: int,
    classification_method: str,
) -> dict[str, Any]:
    """Shape ICLabel results to match the frontend's ExcludeIcaSummaryResponse.

    This is the structured-JSON counterpart to the PDF-scraped
    ``extract_ica_full`` result used for the original ICA pass, so the
    review UI needs no new parsing logic for this pass -- only a new data
    source.
    """
    excluded = {int(v) for v in ica.exclude}
    components: list[dict[str, Any]] = []
    for idx, row in ica_flags.iterrows():
        component_idx = int(idx)
        components.append(
            {
                "component": f"IC{component_idx}",
                "type": normalize_ic_type(str(row.get("ic_type", ""))) or "other",
                "confidence": float(row.get("confidence", 0.0) or 0.0),
                "rejected": component_idx in excluded,
            }
        )
    return {
        "components": components,
        "structure": {
            "n_components": int(ica.n_components_ or 0),
            "method": classification_method,
            "fitted_on": "epochs",
            "n_epochs": int(n_epochs),
        },
    }


def _build_synthetic_raw(epochs: mne.BaseEpochs) -> mne.io.RawArray:
    """Concatenate epoch data into a continuous Raw purely for plotting.

    The existing ICA report pipeline is raw-only (see
    ``ICAReportingMixin._get_ica_report_data``, which explicitly returns
    None when ICA was fit on epochs), so this synthesizes a raw-shaped
    object just so the genuinely-standalone plotting primitives in
    ``icvision_layouts.py`` can be reused. Epoch-boundary discontinuities
    are a cosmetic artifact only -- ``ica.get_sources()`` on this object is
    still numerically correct per-sample.
    """
    data = np.concatenate(epochs.get_data(copy=False), axis=-1)
    return mne.io.RawArray(data, epochs.info.copy(), verbose="ERROR")


def generate_rerun_report(
    ica: ICA,
    epochs: mne.BaseEpochs,
    ica_flags: pd.DataFrame,
    *,
    classification_method: str,
    stem: str,
    output_pdf_path: Path,
    psd_fmax: Optional[float] = None,
) -> Path:
    """Render a component-review PDF for the rerun pass.

    Re-implements only the summary-table assembly from the mixin's
    ``_plot_ica_components`` (the rest of that method is entangled with
    ``self.config``/``self._resolve_report_path``/``self._update_metadata``,
    none of which exist in this request context); the per-component and
    topography-overview plots are reused verbatim from
    ``icvision_layouts.py``.
    """
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    raw_synth = _build_synthetic_raw(epochs)
    component_indices = list(range(ica.n_components_ or 0))
    excluded_set = set(int(v) for v in ica.exclude)

    color_map = {
        "brain": "#a9dfbf",
        "eog": "#f9e79f",
        "muscle": "#f5b7b1",
        "ecg": "#d7bde2",
        "ch_noise": "#ffd700",
        "line_noise": "#add8e6",
        "other": "#f0f0f0",
    }

    with PdfPages(output_pdf_path) as pdf:
        # --- Summary table page(s) ---------------------------------------
        components_per_page = 20
        num_pages = max(1, int(np.ceil(len(component_indices) / components_per_page)))
        for page in range(num_pages):
            start_idx = page * components_per_page
            end_idx = min((page + 1) * components_per_page, len(component_indices))
            page_components = component_indices[start_idx:end_idx]

            fig_table, ax_table = plt.subplots(figsize=(11, 8.5))
            ax_table.axis("off")

            table_data = []
            colors = []
            for idx in page_components:
                comp_info = ica_flags.loc[idx] if idx in ica_flags.index else None
                if comp_info is not None:
                    ic_type = normalize_ic_type(str(comp_info.get("ic_type", "")))
                    table_data.append(
                        [
                            f"IC{idx}",
                            ic_type or "other",
                            f"{float(comp_info.get('confidence', 0.0) or 0.0):.2f}",
                            "Yes" if idx in excluded_set else "No",
                        ]
                    )
                    colors.append([color_map.get(ic_type, "white")] * 4)
                else:
                    table_data.append(
                        [
                            f"IC{idx}",
                            "N/A",
                            "N/A",
                            "Yes" if idx in excluded_set else "No",
                        ]
                    )
                    colors.append(["white"] * 4)

            if table_data:
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=["Component", "Type", "Confidence", "Rejected"],
                    loc="center",
                    cellLoc="center",
                    cellColours=colors,
                    colWidths=[0.2, 0.4, 0.2, 0.2],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig_table.suptitle(
                f"ICA Components Summary [Post-Epoch-Rejection] - {stem}\n"
                f"(Page {page + 1} of {num_pages})\n"
                f"Generated: {timestamp}",
                fontsize=12,
                y=0.95,
            )
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
                for color in color_map.values()
            ]
            ax_table.legend(
                legend_elements,
                color_map.keys(),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(color_map),
                title="Component Types",
            )
            plt.subplots_adjust(top=0.85, bottom=0.25)
            pdf.savefig(fig_table)
            plt.close(fig_table)

        # --- Topography overview ------------------------------------------
        for topo_fig in plot_ica_topographies_overview(ica, component_indices):
            pdf.savefig(topo_fig)
            plt.close(topo_fig)

        # --- Per-component classification pages ----------------------------
        for idx in component_indices:
            comp_info = ica_flags.loc[idx] if idx in ica_flags.index else None
            classification_label = (
                comp_info.get("ic_type") if comp_info is not None else None
            )
            classification_confidence = (
                comp_info.get("confidence") if comp_info is not None else None
            )
            fig = plot_component_for_classification(
                ica,
                raw_synth,
                idx,
                output_dir=output_pdf_path.parent,
                return_fig_object=True,
                classification_label=classification_label,
                classification_confidence=classification_confidence,
                classification_method=classification_method,
                raw_full=raw_synth,
                source_filename=stem,
                psd_fmax=psd_fmax,
            )
            if isinstance(fig, plt.Figure):
                pdf.savefig(fig)
                plt.close(fig)
            else:
                plt.close("all")

    return output_pdf_path


def pass_dir(task_root: Path) -> Path:
    return task_root / "ica" / PASS_NAME


def pass_report_dir(task_root: Path) -> Path:
    return task_root / "reports" / "ica_components" / PASS_NAME


def pass_metadata_path(task_root: Path, stem: str) -> Path:
    return pass_dir(task_root) / f"{stem}-pass-metadata.json"


def pass_iclabel_path(task_root: Path, stem: str) -> Path:
    return pass_dir(task_root) / f"{stem}-iclabel.json"


def pass_ica_fif_path(task_root: Path, stem: str) -> Path:
    return pass_dir(task_root) / f"{stem}-ica.fif"


def pass_report_pdf_path(task_root: Path, stem: str) -> Path:
    return pass_report_dir(task_root) / f"{stem}_ica_components_all.pdf"


def save_rerun_artifacts(
    *,
    task_root: Path,
    stem: str,
    ica: ICA,
    ica_flags: pd.DataFrame,
    epochs: mne.BaseEpochs,
    classification_method: str,
    ica_kwargs: dict[str, Any],
    epochs_before: int,
    parent_run_id: Optional[str],
    psd_fmax: Optional[float] = None,
) -> dict[str, Any]:
    """Write all artifacts for a rerun pass under a pass-specific subtree.

    Never touches the original ``ica/``/``reports/ica_components/`` paths --
    this pass lives entirely under ``.../post_epoch_rejection/`` until an
    explicit promote.
    """
    epochs_after = len(epochs)
    duration_seconds = epochs_after * float(epochs.tmax - epochs.tmin)

    pass_dir(task_root).mkdir(parents=True, exist_ok=True)
    ica_fif_path = pass_ica_fif_path(task_root, stem)
    ica.save(ica_fif_path, overwrite=True)

    iclabel_json = build_structured_iclabel_json(
        ica,
        ica_flags,
        n_epochs=epochs_after,
        classification_method=classification_method,
    )
    iclabel_path = pass_iclabel_path(task_root, stem)
    iclabel_path.write_text(json.dumps(iclabel_json, indent=2))

    report_pdf_path = pass_report_pdf_path(task_root, stem)
    generate_rerun_report(
        ica,
        epochs,
        ica_flags,
        classification_method=classification_method,
        stem=stem,
        output_pdf_path=report_pdf_path,
        psd_fmax=psd_fmax,
    )

    pass_metadata = {
        "pass_name": PASS_NAME,
        "created_at": datetime.now().isoformat(),
        "parent_run_id": parent_run_id,
        "epochs_before": int(epochs_before),
        "epochs_after": int(epochs_after),
        "retained_duration_seconds": duration_seconds,
        "ica_kwargs": {k: v for k, v in ica_kwargs.items() if k != "picks"},
        "classification_method": classification_method,
        "status": "pending_review",
    }
    pass_metadata_path(task_root, stem).write_text(
        json.dumps(pass_metadata, indent=2, default=str)
    )

    return {
        "ica_fif": str(ica_fif_path),
        "iclabel_json": str(iclabel_path),
        "report_pdf": str(report_pdf_path),
        "pass_metadata": str(pass_metadata_path(task_root, stem)),
        "epochs_before": epochs_before,
        "epochs_after": epochs_after,
        "duration_seconds": duration_seconds,
    }


def load_pass_metadata(task_root: Path, stem: str) -> Optional[dict[str, Any]]:
    path = pass_metadata_path(task_root, stem)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_pass_iclabel(task_root: Path, stem: str) -> Optional[dict[str, Any]]:
    path = pass_iclabel_path(task_root, stem)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def discard_pass(task_root: Path, stem: str) -> None:
    """Delete a rerun pass's artifacts entirely. Never touches active outputs."""
    for path in [
        pass_ica_fif_path(task_root, stem),
        pass_iclabel_path(task_root, stem),
        pass_metadata_path(task_root, stem),
        pass_report_pdf_path(task_root, stem),
    ]:
        path.unlink(missing_ok=True)


def apply_pass_rejection(
    task_root: Path, stem: str, rejected_components: list[int]
) -> tuple[ICA, mne.BaseEpochs]:
    """Load the pass's ICA + retained epochs and apply the reviewer's exclusions.

    Rejection is always explicit/manual for this pass (never the
    config-driven automatic thresholds the mixin's
    ``apply_ica_component_rejection`` uses) -- those thresholds live in the
    live Task's assembled config, which isn't cheaply recoverable outside a
    running Task, and the reviewer explicitly reviews this pass before
    promoting it anyway.
    """
    ica_fif_path = pass_ica_fif_path(task_root, stem)
    if not ica_fif_path.exists():
        raise IcaRerunError("Rerun pass ICA fit not found -- run the rerun first")
    ica = mne.preprocessing.read_ica(ica_fif_path, verbose="ERROR")
    ica.exclude = sorted({int(v) for v in rejected_components})

    postedit_dir = task_root / "postedit"
    postedit_matches = list(postedit_dir.glob(f"{stem}_postedit.set"))
    if not postedit_matches:
        raise IcaRerunError("Retained epochs for this pass are no longer available")
    epochs = _load_epochs(postedit_matches[0])
    ica.apply(epochs)
    return ica, epochs
