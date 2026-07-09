"""Condition-wise ERP epoch rejection utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mne
import numpy as np
import pandas as pd

from autoclean.utils.logging import message


class ConditionwiseEpochRejectionMixin:
    """Mixin for condition-wise ERP epoch rejection with audit outputs."""

    DEFAULT_CONDITIONWISE_METRICS = (
        "rms",
        "std",
        "max_abs",
        "ptp",
        "mean_gradient",
    )

    def apply_conditionwise_epoch_rejection(
        self,
        epochs: Optional[mne.BaseEpochs] = None,
        robust_z_threshold: float = 4.0,
        minimum_metric_flags: int = 2,
        absolute_amplitude_uv: Optional[float] = None,
        max_reject_fraction: float = 0.10,
        minimum_epochs: int = 20,
        exclude_channels_matching: Optional[Iterable[str]] = None,
        exclude_channel_types: Optional[Iterable[str]] = None,
        mode: str = "apply",
        group_by: str = "event_id",
        stage_name: str = "conditionwise_epoch_rejection",
    ) -> tuple[mne.BaseEpochs, pd.DataFrame, pd.DataFrame]:
        """Score and optionally reject ERP epochs separately within each condition."""

        if hasattr(self, "_check_step_enabled"):
            is_enabled, config_value = self._check_step_enabled(
                "conditionwise_epoch_rejection"
            )
            if not is_enabled:
                message("info", "Condition-wise epoch rejection step is disabled")
                current_epochs = (
                    epochs if epochs is not None else getattr(self, "epochs", None)
                )
                return current_epochs, pd.DataFrame(), pd.DataFrame()

            if config_value and isinstance(config_value, dict):
                params = config_value.get("value", config_value)
                robust_z_threshold = params.get(
                    "robust_z_threshold", robust_z_threshold
                )
                minimum_metric_flags = params.get(
                    "minimum_metric_flags", minimum_metric_flags
                )
                absolute_amplitude_uv = params.get(
                    "absolute_amplitude_uv", absolute_amplitude_uv
                )
                max_reject_fraction = params.get(
                    "max_reject_fraction", max_reject_fraction
                )
                minimum_epochs = params.get("minimum_epochs", minimum_epochs)
                exclude_channels_matching = params.get(
                    "exclude_channels_matching", exclude_channels_matching
                )
                exclude_channel_types = params.get(
                    "exclude_channel_types", exclude_channel_types
                )
                mode = params.get("mode", mode)
                group_by = params.get("group_by", group_by)

        if epochs is None:
            epochs = getattr(self, "epochs", None)
        if epochs is None:
            raise ValueError("No epochs available for condition-wise epoch rejection.")
        if not isinstance(epochs, mne.BaseEpochs):
            raise TypeError(f"epochs must be an MNE Epochs object, got {type(epochs)}")
        if group_by != "event_id":
            raise ValueError(
                "condition-wise epoch rejection currently supports group_by='event_id'"
            )
        if mode not in {"apply", "report_only"}:
            raise ValueError("mode must be 'apply' or 'report_only'")
        if not 0 <= float(max_reject_fraction) <= 1:
            raise ValueError("max_reject_fraction must be between 0 and 1")
        if int(minimum_epochs) < 0:
            raise ValueError("minimum_epochs must be non-negative")

        picks = self._conditionwise_rejection_picks(
            epochs,
            include_channel_types=["eeg"],
            exclude_channel_types=exclude_channel_types or ["eog", "ecg", "misc"],
            exclude_channels_matching=exclude_channels_matching or [],
        )
        if not picks:
            raise ValueError(
                "No channels available for condition-wise rejection scoring."
            )

        data = epochs.get_data(picks=picks)
        metrics = self._conditionwise_epoch_metrics(data)
        labels = self._conditionwise_event_labels(epochs)
        audit_df = self._conditionwise_epoch_audit(
            metrics=metrics,
            labels=labels,
            robust_z_threshold=float(robust_z_threshold),
            minimum_metric_flags=int(minimum_metric_flags),
            absolute_amplitude_uv=absolute_amplitude_uv,
        )
        audit_df, summary_df = self._conditionwise_rejection_decisions(
            audit_df,
            max_reject_fraction=float(max_reject_fraction),
            minimum_epochs=int(minimum_epochs),
            mode=mode,
        )

        clean_epochs = epochs
        rejected_indices = audit_df.loc[audit_df["rejected"], "epoch_index"].tolist()
        if mode == "apply" and rejected_indices:
            keep_mask = np.ones(len(epochs), dtype=bool)
            keep_mask[rejected_indices] = False
            clean_epochs = epochs[np.where(keep_mask)[0]]
            if getattr(self, "epochs", None) is epochs:
                self.epochs = clean_epochs

        artifact_paths = self._save_conditionwise_rejection_outputs(
            audit_df=audit_df,
            summary_df=summary_df,
            stage_name=stage_name,
        )
        metadata = {
            "stage_name": stage_name,
            "mode": mode,
            "group_by": group_by,
            "robust_z_threshold": float(robust_z_threshold),
            "minimum_metric_flags": int(minimum_metric_flags),
            "absolute_amplitude_uv": absolute_amplitude_uv,
            "max_reject_fraction": float(max_reject_fraction),
            "minimum_epochs": int(minimum_epochs),
            "scored_channels": [epochs.ch_names[idx] for idx in picks],
            "original_epochs": int(len(epochs)),
            "rejected_epochs": int(len(rejected_indices)),
            "retained_epochs": int(len(clean_epochs)),
            "rejected_epoch_indices": [int(index) for index in rejected_indices],
            "artifact_reports": artifact_paths,
            "warnings": summary_df.loc[summary_df["warning"] != "", "warning"].tolist(),
        }
        if hasattr(self, "_update_metadata"):
            try:
                self._update_metadata("step_conditionwise_epoch_rejection", metadata)
            except Exception as exc:  # Unavailable in unit test contexts.
                message(
                    "warning",
                    f"Condition-wise epoch rejection metadata was not persisted: {exc}",
                )

        self.conditionwise_epoch_rejection_audit = audit_df
        self.conditionwise_epoch_rejection_summary = summary_df
        return clean_epochs, audit_df, summary_df

    @staticmethod
    def _conditionwise_rejection_picks(
        epochs: mne.BaseEpochs,
        include_channel_types: Optional[Iterable[str]],
        exclude_channel_types: Iterable[str],
        exclude_channels_matching: Iterable[str],
    ) -> list[int]:
        included_types = (
            {str(ch_type).lower() for ch_type in include_channel_types}
            if include_channel_types is not None
            else None
        )
        excluded_types = {str(ch_type).lower() for ch_type in exclude_channel_types}
        excluded_patterns = [
            str(pattern).lower() for pattern in exclude_channels_matching
        ]
        channel_types = epochs.get_channel_types()
        picks = []
        for idx, (name, ch_type) in enumerate(zip(epochs.ch_names, channel_types)):
            lowered_type = ch_type.lower()
            if included_types is not None and lowered_type not in included_types:
                continue
            if lowered_type in excluded_types:
                continue
            lowered = name.lower()
            if any(pattern in lowered for pattern in excluded_patterns):
                continue
            picks.append(idx)
        return picks

    @staticmethod
    def _conditionwise_epoch_metrics(data: np.ndarray) -> Dict[str, np.ndarray]:
        gradient = np.diff(data, axis=2)
        return {
            "rms": np.sqrt(np.mean(np.square(data), axis=(1, 2))),
            "std": np.std(data, axis=(1, 2)),
            "max_abs": np.max(np.abs(data), axis=(1, 2)),
            "ptp": np.ptp(data, axis=(1, 2)),
            "mean_gradient": np.mean(np.abs(gradient), axis=(1, 2)),
        }

    @staticmethod
    def _conditionwise_event_labels(epochs: mne.BaseEpochs) -> list[dict[str, Any]]:
        inverse_event_id = {int(code): name for name, code in epochs.event_id.items()}
        labels = []
        for idx, code in enumerate(epochs.events[:, 2].astype(int)):
            labels.append(
                {
                    "epoch_index": idx,
                    "event_code": int(code),
                    "condition": inverse_event_id.get(int(code), str(code)),
                }
            )
        return labels

    @staticmethod
    def _robust_z(values: np.ndarray) -> np.ndarray:
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        deviations = np.abs(values - median)
        if mad == 0:
            scores = np.zeros_like(values, dtype=float)
            scores[deviations > 0] = np.inf
            return scores
        return np.abs(0.6745 * (values - median) / mad)

    def _conditionwise_epoch_audit(
        self,
        metrics: Dict[str, np.ndarray],
        labels: list[dict[str, Any]],
        robust_z_threshold: float,
        minimum_metric_flags: int,
        absolute_amplitude_uv: Optional[float],
    ) -> pd.DataFrame:
        audit = pd.DataFrame(labels)
        for metric_name, values in metrics.items():
            audit[metric_name] = values
            audit[f"{metric_name}_robust_z"] = 0.0
            audit[f"{metric_name}_flagged"] = False

        for condition, group_index in audit.groupby("condition").groups.items():
            idx = list(group_index)
            for metric_name, values in metrics.items():
                scores = self._robust_z(values[idx])
                audit.loc[idx, f"{metric_name}_robust_z"] = scores
                audit.loc[idx, f"{metric_name}_flagged"] = scores >= robust_z_threshold

        metric_flag_columns = [f"{name}_flagged" for name in metrics]
        audit["metric_flag_count"] = audit[metric_flag_columns].sum(axis=1).astype(int)
        audit["absolute_amplitude_flagged"] = False
        if absolute_amplitude_uv is not None:
            threshold_volts = float(absolute_amplitude_uv) * 1e-6
            audit["absolute_amplitude_flagged"] = audit["max_abs"] > threshold_volts

        audit["candidate_reject"] = (
            audit["metric_flag_count"] >= int(minimum_metric_flags)
        ) | audit["absolute_amplitude_flagged"]
        audit["rejected"] = False
        audit["rejection_reason"] = ""
        return audit

    @staticmethod
    def _conditionwise_rejection_decisions(
        audit_df: pd.DataFrame,
        max_reject_fraction: float,
        minimum_epochs: int,
        mode: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        audit_df = audit_df.copy()
        summaries = []
        robust_cols = [col for col in audit_df.columns if col.endswith("_robust_z")]
        audit_df["max_robust_z"] = audit_df[robust_cols].max(axis=1)

        for condition, group in audit_df.groupby("condition", sort=False):
            original = int(len(group))
            candidates = group[group["candidate_reject"]].copy()
            max_by_fraction = int(np.floor(original * max_reject_fraction))
            max_by_minimum = max(0, original - int(minimum_epochs))
            allowed_rejections = max(0, min(max_by_fraction, max_by_minimum))
            warning_parts = []
            if original and (len(candidates) / original) > max_reject_fraction:
                warning_parts.append("candidate_fraction_exceeds_max_reject_fraction")
            if original - len(candidates) < minimum_epochs:
                warning_parts.append("candidate_retention_below_minimum_epochs")

            rejected_indices = []
            if mode == "apply" and allowed_rejections > 0 and not candidates.empty:
                ordered = candidates.sort_values(
                    ["metric_flag_count", "max_robust_z"], ascending=[False, False]
                )
                rejected_indices = ordered.head(allowed_rejections)[
                    "epoch_index"
                ].tolist()
                audit_df.loc[
                    audit_df["epoch_index"].isin(rejected_indices), "rejected"
                ] = True
                audit_df.loc[
                    audit_df["epoch_index"].isin(rejected_indices), "rejection_reason"
                ] = "conditionwise_metric_threshold"

            rejected = int(len(rejected_indices))
            retained = original - rejected
            event_codes = sorted(group["event_code"].unique().tolist())
            summaries.append(
                {
                    "condition": condition,
                    "event_codes": ";".join(str(code) for code in event_codes),
                    "original_epochs": original,
                    "candidate_rejected_epochs": int(len(candidates)),
                    "rejected_epochs": rejected,
                    "retained_epochs": retained,
                    "rejection_percentage": (
                        (rejected / original * 100.0) if original else 0.0
                    ),
                    "candidate_rejection_percentage": (
                        len(candidates) / original * 100.0 if original else 0.0
                    ),
                    "max_reject_fraction": float(max_reject_fraction),
                    "minimum_epochs": int(minimum_epochs),
                    "mode": mode,
                    "warning": ";".join(warning_parts),
                }
            )

        total_original = int(len(audit_df))
        total_candidates = int(audit_df["candidate_reject"].sum())
        total_rejected = int(audit_df["rejected"].sum())
        total_retained = total_original - total_rejected
        summaries.append(
            {
                "condition": "__recording_total__",
                "event_codes": "all",
                "original_epochs": total_original,
                "candidate_rejected_epochs": total_candidates,
                "rejected_epochs": total_rejected,
                "retained_epochs": total_retained,
                "rejection_percentage": (
                    total_rejected / total_original * 100.0 if total_original else 0.0
                ),
                "candidate_rejection_percentage": (
                    total_candidates / total_original * 100.0 if total_original else 0.0
                ),
                "max_reject_fraction": float(max_reject_fraction),
                "minimum_epochs": int(minimum_epochs),
                "mode": mode,
                "warning": "",
            }
        )

        return audit_df, pd.DataFrame(summaries)

    def _save_conditionwise_rejection_outputs(
        self, audit_df: pd.DataFrame, summary_df: pd.DataFrame, stage_name: str
    ) -> dict[str, str]:
        output_dir = self._conditionwise_output_dir()
        subject_id = "unknown_subject"
        if hasattr(self, "config") and self.config.get("unprocessed_file"):
            subject_id = Path(self.config["unprocessed_file"]).stem
        audit_path = output_dir / f"{subject_id}_{stage_name}_audit.csv"
        summary_path = output_dir / f"{subject_id}_{stage_name}_summary.csv"
        audit_df.to_csv(audit_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        message("info", f"Saved condition-wise epoch rejection audit to {audit_path}")
        message(
            "info", f"Saved condition-wise epoch rejection summary to {summary_path}"
        )
        return {
            "conditionwise_epoch_rejection_audit": str(
                self._conditionwise_report_relative_path(audit_path)
            ),
            "conditionwise_epoch_rejection_summary": str(
                self._conditionwise_report_relative_path(summary_path)
            ),
        }

    def _conditionwise_output_dir(self) -> Path:
        if hasattr(self, "_resolve_report_path"):
            return self._resolve_report_path("conditionwise_epoch_rejection")
        config = getattr(self, "config", {}) or {}
        reports_dir = (
            config.get("reports_dir") or config.get("metadata_dir") or Path.cwd()
        )
        output_dir = Path(reports_dir) / "conditionwise_epoch_rejection"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _conditionwise_report_relative_path(self, path: Path) -> Path:
        if hasattr(self, "_report_relative_path"):
            return self._report_relative_path(path)
        return path
