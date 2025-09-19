"""ICA reporting mixin for autoclean tasks.

This module provides specialized ICA visualization and reporting functionality for
the AutoClean pipeline. It defines methods for generating comprehensive visualizations
and reports of Independent Component Analysis (ICA) results, including:

- Full-duration component activations
- Component properties and classifications
- Rejected components with their properties
- Interactive and static reports

These reports help users understand the ICA decomposition and validate component rejection
decisions to ensure appropriate artifact removal.

"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mne.preprocessing import ICA

from autoclean.functions.visualization.icvision_layouts import (
    plot_component_for_classification,
    plot_ica_topographies_overview,
)
from autoclean.mixins.viz._ica_sources_cache import (
    get_cached_ica_sources,
    invalidate_ica_cache,
    get_ica_cache_stats,
    cache_aware_ica_method,
)
from autoclean.utils.logging import message

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")


class ICAReportingMixin:
    """Mixin providing ICA reporting functionality for EEG data.

    This mixin extends the base ReportingMixin with specialized methods for
    generating visualizations and reports of ICA results. It provides tools for
    assessing component properties, visualizing component activations, and
    documenting component rejection decisions.

    All reporting methods respect configuration toggles from `autoclean_config.yaml`,
    checking if their corresponding step is enabled before execution. Each method
    can be individually enabled or disabled via configuration.

    Available ICA reporting methods include:

    - `plot_ica_full`: Plot all ICA components over the full time series
    - `generate_ica_reports`: Create a comprehensive report of ICA decomposition results
    - `verify_topography_plot`: Use a basicica topograph to verify MEA channel placement.
    """

    def plot_ica_full(self) -> plt.Figure:
        """Plot ICA components over the full time series with their labels and probabilities.

        This method creates a figure showing each ICA component's time course over the full
        time series. Components are color-coded by their classification/rejection status,
        and probability scores are indicated for each component.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure with ICA components.

        Raises
        ------
        ValueError
            If no ICA object is found in the pipeline.

        Examples
        --------
        >>> # After performing ICA
        >>> fig = task.plot_ica_full()
        >>> plt.show()

        Notes:
            - Components classified as artifacts are highlighted in red
            - Classification probabilities are shown for each component
            - The method respects configuration settings via the `ica_full_plot_step` config
        """
        # Get raw and ICA from pipeline
        raw = self.raw.copy()
        ica = self.final_ica
        ic_labels = self.ica_flags

        # Get ICA activations using cache for better performance
        ica_sources = get_cached_ica_sources(ica, raw)
        ica_data = ica_sources.get_data()
        times = raw.times
        n_components, _ = ica_data.shape

        # Normalize each component individually for better visibility
        for idx in range(n_components):
            component = ica_data[idx]
            # Scale to have a consistent peak-to-peak amplitude
            ptp = np.ptp(component)
            if ptp == 0:
                scaling_factor = 2.5  # Avoid division by zero
            else:
                scaling_factor = 2.5 / ptp
            ica_data[idx] = component * scaling_factor

        # Determine appropriate spacing
        spacing = 2  # Fixed spacing between components

        # Calculate figure size proportional to duration
        total_duration = times[-1] - times[0]
        width_per_second = 0.1  # Increased from 0.02 to 0.1 for wider view
        fig_width = total_duration * width_per_second
        max_fig_width = 200  # Doubled from 100 to allow wider figures
        fig_width = min(fig_width, max_fig_width)
        fig_height = max(6, n_components * 0.5)  # Ensure a minimum height

        # Create plot with wider figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create a colormap for the components
        cmap = plt.cm.get_cmap("tab20", n_components)
        line_colors = [cmap(i) for i in range(n_components)]

        # Plot components in original order
        for idx in range(n_components):
            offset = idx * spacing
            ax.plot(
                times, ica_data[idx] + offset, color=line_colors[idx], linewidth=0.5
            )

        # Set y-ticks and labels
        yticks = [idx * spacing for idx in range(n_components)]
        yticklabels = []
        for idx in range(n_components):
            annotator = (
                ic_labels["annotator"][idx]
                if hasattr(ic_labels, "columns") and "annotator" in ic_labels.columns
                else "ic_label"
            )
            source_tag = (
                " [Vision]" if str(annotator).lower() in {"ic_vision", "vision"} else ""
            )
            label_text = (
                f"IC{idx + 1}: {ic_labels['ic_type'][idx]} "
                f"({ic_labels['confidence'][idx]:.2f}){source_tag}"
            )
            yticklabels.append(label_text)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=8)

        # Customize axes
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_title("ICA Component Activations (Full Duration)", fontsize=14)
        ax.set_xlim(times[0], times[-1])

        # Adjust y-axis limits
        ax.set_ylim(-spacing, (n_components - 1) * spacing + spacing)

        # Remove y-axis label as we have custom labels
        ax.set_ylabel("")

        # Invert y-axis to have the first component at the top
        ax.invert_yaxis()

        # Color the labels red or black based on component type
        artifact_types = ["eog", "muscle", "ecg", "other"]
        for ticklabel, idx in zip(ax.get_yticklabels(), range(n_components)):
            ic_type = ic_labels["ic_type"][idx]
            if ic_type in artifact_types:
                ticklabel.set_color("red")
            else:
                ticklabel.set_color("black")

        # Adjust layout
        plt.tight_layout()

        basename = self.config["bids_path"].basename
        basename = basename.replace("_eeg", "_ica_components_full_duration")
        target_figure = self._resolve_report_path("ica_components", basename)

        # Save figure with higher DPI for better resolution of wider plot
        fig.savefig(target_figure, dpi=300, bbox_inches="tight")

        artifact_relpath = self._report_relative_path(Path(target_figure))
        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "ica_components_full_duration": str(artifact_relpath),
            }
        }

        self._update_metadata("plot_ica_full", metadata)

        return fig

    def generate_ica_reports(
        self,
        duration: int = 10,
    ) -> None:
        """Generate comprehensive ICA reports using the _plot_ica_components method.

        Parameters
        ----------
        duration : Optional[int]
            Duration in seconds for plotting time series data
        """
        # Generate report for all components
        report_filename = self._plot_ica_components(
            duration=duration,
            components="all",
        )

        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "ica_all_components": report_filename,
            }
        }

        self._update_metadata("generate_ica_reports", metadata)

        # Generate report for rejected components
        report_filename = self._plot_ica_components(
            duration=duration,
            components="rejected",
        )

        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "ica_rejected_components": report_filename,
            }
        }

        self._update_metadata("generate_ica_reports", metadata)

    def _plot_ica_components(
        self,
        duration: int = 10,
        components: str = "all",
    ):
        """
        Plots ICA components with labels and saves reports.

        Parameters:
        -----------
        duration : int
            Duration in seconds to plot.
        components : str
            'all' to plot all components, 'rejected' to plot only rejected components.
        """
        if self.raw is None or self.final_ica is None:
            message("warning", "ICA plotting skipped because raw or ICA data is missing.")
            return None

        raw = self.raw
        ica = self.final_ica
        ic_labels = getattr(self, "ica_flags", None)

        classification_method = getattr(self, "ica_classification_method", None)
        if not classification_method and ic_labels is not None:
            annot_col = getattr(ic_labels, "columns", [])
            if "annotator" in annot_col:
                try:
                    annotators = {
                        str(value).lower()
                        for value in ic_labels["annotator"].dropna().unique()
                    }
                except Exception:
                    annotators = set()
                if {"ic_label", "ic_vision"}.issubset(annotators):
                    classification_method = "hybrid"
                elif "ic_vision" in annotators:
                    classification_method = "icvision"
                elif "ic_label" in annotators:
                    classification_method = "iclabel"

        def _format_method(value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            mapping = {"iclabel": "ICLabel", "icvision": "ICVision", "hybrid": "Hybrid"}
            return mapping.get(value.lower(), value)

        formatted_method = _format_method(classification_method)

        if components == "all":
            total_components = ica.n_components_ or 0
            component_indices = list(range(total_components))
            report_name = "ica_components_all"
        elif components == "rejected":
            component_indices = list(ica.exclude)
            report_name = "ica_components_rejected"
            if not component_indices:
                message("info", "No rejected components. Skipping rejected components report.")
                return None
        else:
            raise ValueError("components parameter must be 'all' or 'rejected'.")

        if not component_indices:
            message("info", "No ICA components available for plotting.")
            return None

        sfreq = raw.info["sfreq"]
        n_samples = int(min(duration * sfreq, raw.n_times))
        tmax = raw.times[n_samples - 1] if n_samples > 0 else 0.0
        raw_fast = raw.copy().crop(tmin=0, tmax=tmax)

        basename = self.config["bids_path"].basename
        basename = basename.replace("_eeg", report_name)
        pdf_path = self._resolve_report_path("ica_components", basename).with_suffix(".pdf")

        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        color_map = {
            "brain": "#d4edda",
            "eog": "#f9e79f",
            "muscle": "#f5b7b1",
            "ecg": "#d7bde2",
            "ch_noise": "#ffd700",
            "line_noise": "#add8e6",
            "other": "#f0f0f0",
        }
        excluded_set = set(ica.exclude)

        with PdfPages(pdf_path) as pdf:
            components_per_page = 20
            component_count = len(component_indices)
            num_pages = max(1, int(np.ceil(component_count / components_per_page)))

            for page in range(num_pages):
                start_idx = page * components_per_page
                end_idx = min((page + 1) * components_per_page, component_count)
                page_components = component_indices[start_idx:end_idx]

                fig_table, ax_table = plt.subplots(figsize=(11, 8.5))
                ax_table.axis("off")

                table_data = []
                colors = []

                for idx in page_components:
                    if ic_labels is not None and idx < len(ic_labels):
                        comp_info = ic_labels.iloc[idx]
                        annot = str(comp_info.get("annotator", "ic_label")).lower()
                        src_suffix = " [Vision]" if annot in {"ic_vision", "vision"} else ""
                        table_data.append(
                            [
                                f"IC{idx + 1}",
                                f"{comp_info['ic_type']}{src_suffix}",
                                f"{comp_info['confidence']:.2f}",
                                "Yes" if idx in excluded_set else "No",
                            ]
                        )
                        colors.append([
                            color_map.get(comp_info["ic_type"].lower(), "white")
                        ] * 4)
                    else:
                        table_data.append(
                            [
                                f"IC{idx + 1}",
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
                method_tag = f" [{formatted_method}]" if formatted_method else ""
                summary_title = f"ICA Components Summary{method_tag}"
                fig_table.suptitle(
                    f"{summary_title} - {self.config['bids_path'].basename}\n"
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
                    loc="upper right",
                    title="Component Types",
                )

                plt.subplots_adjust(top=0.85, bottom=0.15)

                pdf.savefig(fig_table)
                plt.close(fig_table)

            for topo_fig in plot_ica_topographies_overview(ica, component_indices):
                pdf.savefig(topo_fig)
                plt.close(topo_fig)

            if components == "rejected":
                end_time = min(float(duration), raw_fast.times[-1])
                raw_copy = raw_fast.copy()
                ica_ch_names = self.final_ica.ch_names
                if len(ica_ch_names) != len(raw_copy.ch_names):
                    message(
                        "warning",
                        "Channel count mismatch between ICA and raw. Using ICA channels only for overlay.",
                    )
                    raw_copy.pick_channels(ica_ch_names)

                fig_overlay = self.final_ica.plot_overlay(
                    raw_copy,
                    start=0,
                    stop=end_time,
                    exclude=component_indices,
                    show=False,
                )
                fig_overlay.set_size_inches(15, 10)
                pdf.savefig(fig_overlay)
                plt.close(fig_overlay)

            source_name = self.config["bids_path"].basename
            for idx in component_indices:
                classification_label = None
                classification_confidence = None
                classification_reason = None

                if ic_labels is not None and idx < len(ic_labels):
                    comp_info = ic_labels.iloc[idx]
                    classification_label = comp_info.get("ic_type")
                    classification_confidence = comp_info.get("confidence")
                    classification_reason = (
                        comp_info.get("vision_reason")
                        or comp_info.get("reason")
                        or comp_info.get("notes")
                    )

                fig = plot_component_for_classification(
                    ica,
                    raw_fast,
                    idx,
                    output_dir=pdf_path.parent,
                    return_fig_object=True,
                    classification_label=classification_label,
                    classification_confidence=classification_confidence,
                    classification_reason=classification_reason,
                    classification_method=classification_method,
                    raw_full=raw,
                    source_filename=source_name,
                )

                if isinstance(fig, plt.Figure):
                    pdf.savefig(fig)
                    plt.close(fig)
                else:
                    plt.close("all")

        message("success", f"ICA report saved to {pdf_path}")
        return str(self._report_relative_path(Path(pdf_path)))

    def get_cache_info(self) -> dict:
        """Get ICA sources cache statistics for monitoring.
        
        Returns
        -------
        dict
            Cache statistics including size, entries, and utilization
        """
        return get_ica_cache_stats()

    def clear_ica_sources_cache(self):
        """Clear all cached ICA sources to free memory."""
        invalidate_ica_cache(self.final_ica)
        message("info", "ICA sources cache cleared")

    def log_cache_performance(self):
        """Log current cache performance statistics."""
        stats = get_ica_cache_stats()
        if stats['entries'] > 0:
            message("info", 
                f"ICA Cache: {stats['entries']} entries, "
                f"{stats['total_size_mb']:.1f}MB used "
                f"({stats['utilization_percent']:.1f}% of limit)")
        else:
            message("info", "ICA Cache: empty")

    def verify_topography_plot(self) -> bool:
        """Use ica topograph to verify MEA channel placement.
        This function simply runs fast ICA then plots the topography.
        It is used on mouse files to verify channel placement.

        """
        ica = ICA(  # pylint: disable=not-callable
            n_components=len(self.raw.ch_names) - len(self.raw.info["bads"]),
            method="fastica",
            random_state=42,
        )
        ica.fit(self.raw)

        fig = ica.plot_components(
            picks=range(len(self.raw.ch_names) - len(self.raw.info["bads"])), show=False
        )

        target_path = self._resolve_report_path("ica_components", "ica_topography.png")
        fig.savefig(target_path)

    def compare_vision_iclabel_classifications(self):
        """Compare ICLabel and Vision API classifications for ICA components.

        This method creates a comparison report between ICLabel and OpenAI Vision
        classifications of ICA components, highlighting agreements and disagreements.
        It requires both classify_ica_components_vision and run_ICLabel to have been run.

        Returns
        -------
        matplotlib.figure.Figure
            Figure showing the comparison of classifications.
        """
        # Check if both ICLabel and Vision classifications exist
        if not hasattr(self, "ica_flags") or self.ica_flags is None:
            message("error", "ICLabel results not found. Please run run_ICLabel first.")
            return None

        if not hasattr(self, "ica_vision_flags") or self.ica_vision_flags is None:
            message(
                "error",
                "Vision classification results not found. Please run classify_ica_components_vision first.",
            )
            return None

        # Get the classification results
        iclabel_results = self.ica_flags
        vision_results = self.ica_vision_flags

        # Prepare data for comparison
        n_components = len(iclabel_results)

        # Create mapping for ICLabel categories to binary brain/artifact
        iclabel_mapping = {
            "brain": "brain",
            "eog": "artifact",
            "muscle": "artifact",
            "ecg": "artifact",
            "ch_noise": "artifact",
            "line_noise": "artifact",
            "other": "artifact",
        }

        # Create a figure for the comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # First subplot: Bar chart comparison
        indices = np.arange(n_components)
        bar_width = 0.4

        # Create binary coding (1 for brain, 0 for artifact)
        iclabel_binary = np.array(
            [
                (
                    1
                    if iclabel_mapping.get(
                        iclabel_results.iloc[i]["ic_type"].lower(), "artifact"
                    )
                    == "brain"
                    else 0
                )
                for i in range(n_components)
            ]
        )
        vision_binary = np.array(
            [
                1 if vision_results.iloc[i]["label"] == "brain" else 0
                for i in range(n_components)
            ]
        )

        # Plot bars
        ax1.bar(
            indices - bar_width / 2,
            iclabel_binary,
            bar_width,
            label="ICLabel",
            color="blue",
            alpha=0.6,
        )
        ax1.bar(
            indices + bar_width / 2,
            vision_binary,
            bar_width,
            label="Vision API",
            color="orange",
            alpha=0.6,
        )

        # Highlight disagreements
        disagreements = np.where(iclabel_binary != vision_binary)[0]
        if len(disagreements) > 0:
            for idx in disagreements:
                ax1.annotate(
                    "*",
                    xy=(idx, 1.1),
                    xytext=(idx, 1.1),
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="red",
                )

        # Customize plot
        ax1.set_title("Classification Comparison: ICLabel vs. Vision API", fontsize=14)
        ax1.set_xlabel("Component Number", fontsize=12)
        ax1.set_xticks(indices)
        ax1.set_xticklabels([f"IC{i + 1}" for i in range(n_components)])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(["Artifact", "Brain"])
        ax1.legend()

        # Second subplot: Agreement table
        ax2.axis("tight")
        ax2.axis("off")

        # Prepare table data
        table_data = []
        cell_colors = []
        agreement_count = 0

        for i in range(n_components):
            iclabel_category = iclabel_results.iloc[i]["ic_type"]
            iclabel_type = iclabel_mapping.get(iclabel_category.lower(), "artifact")
            iclabel_conf = iclabel_results.iloc[i]["confidence"]

            vision_type = vision_results.iloc[i]["label"]
            vision_conf = vision_results.iloc[i]["confidence"]

            agreement = "✓" if iclabel_type == vision_type else "✗"
            if iclabel_type == vision_type:
                agreement_count += 1
                bg_color = "#d4edda"  # Light green
            else:
                bg_color = "#f8d7da"  # Light red

            table_data.append(
                [
                    f"IC{i + 1}",
                    iclabel_category,
                    f"{iclabel_conf:.2f}",
                    vision_type.title(),
                    f"{vision_conf:.2f}",
                    agreement,
                ]
            )

            cell_colors.append([bg_color] * 6)

        # Add agreement percentage to the end
        agreement_pct = (agreement_count / n_components) * 100

        # Create and customize table
        table = ax2.table(
            cellText=table_data,
            colLabels=[
                "Component",
                "ICLabel Category",
                "ICLabel Conf.",
                "Vision Type",
                "Vision Conf.",
                "Agreement",
            ],
            loc="center",
            cellLoc="center",
            cellColours=cell_colors,
        )

        # Customize table appearance
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Add agreement percentage as text
        ax2.text(
            0.5,
            -0.1,
            f"Overall Agreement: {agreement_pct:.1f}% ({agreement_count}/{n_components} components)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
            fontweight="bold",
        )

        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(hspace=0.3)

        basename = self.config["bids_path"].basename
        basename = basename.replace("_eeg", "_ica_classification_comparison")
        target_figure = self._resolve_report_path("ica_components", basename)

        # Save figure with higher DPI
        fig.savefig(target_figure, dpi=300, bbox_inches="tight")

        artifact_relpath = self._report_relative_path(Path(target_figure))
        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "ica_classification_comparison": str(artifact_relpath),
            }
        }

        self._update_metadata("compare_vision_iclabel_classifications", metadata)

        return fig
