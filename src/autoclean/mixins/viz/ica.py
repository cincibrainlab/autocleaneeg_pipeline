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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

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

        # Get ICA activations and create time vector
        ica_sources = ica.get_sources(raw)
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
            label_text = (
                f"IC{idx + 1}: {ic_labels['ic_type'][idx]} "
                f"({ic_labels['confidence'][idx]:.2f})"
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

        derivatives_dir = Path(self.config["derivatives_dir"])
        basename = self.config["bids_path"].basename
        basename = basename.replace("_eeg", "_ica_components_full_duration")
        target_figure = derivatives_dir / basename

        # Save figure with higher DPI for better resolution of wider plot
        fig.savefig(target_figure, dpi=300, bbox_inches="tight")

        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "ica_components_full_duration": Path(target_figure).name,
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

        # Get raw and ICA from pipeline
        raw = self.raw
        ica = self.final_ica
        ic_labels = self.ica_flags

        # Determine components to plot
        if components == "all":
            component_indices = range(ica.n_components_)
            report_name = "ica_components_all"
        elif components == "rejected":
            component_indices = ica.exclude
            report_name = "ica_components_rejected"
            if not component_indices:
                print(
                    "No components were rejected. Skipping rejected components report."
                )
                return
        else:
            raise ValueError("components parameter must be 'all' or 'rejected'.")

        # Get ICA activations
        ica_sources = ica.get_sources(raw)
        ica_data = ica_sources.get_data()

        # Limit data to specified duration
        sfreq = raw.info["sfreq"]
        n_samples = int(duration * sfreq)
        times = raw.times[:n_samples]

        # Create output path for the PDF report
        derivatives_dir = Path(self.config["derivatives_dir"])
        basename = self.config["bids_path"].basename
        basename = basename.replace("_eeg", report_name)
        pdf_path = derivatives_dir / basename
        pdf_path = pdf_path.with_suffix(".pdf")

        # Remove existing file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        with PdfPages(pdf_path) as pdf:
            # Calculate how many components to show per page
            components_per_page = 20
            num_pages = int(np.ceil(len(component_indices) / components_per_page))

            # Create summary tables split across pages
            for page in range(num_pages):
                start_idx = page * components_per_page
                end_idx = min((page + 1) * components_per_page, len(component_indices))
                page_components = component_indices[start_idx:end_idx]

                fig_table = plt.figure(figsize=(11, 8.5))
                ax_table = fig_table.add_subplot(111)
                ax_table.axis("off")

                # Prepare table data for this page
                table_data = []
                colors = []
                for idx in page_components:
                    comp_info = ic_labels.iloc[idx]
                    table_data.append(
                        [
                            f"IC{idx + 1}",
                            comp_info["ic_type"],
                            f"{comp_info['confidence']:.2f}",
                            "Yes" if idx in ica.exclude else "No",
                        ]
                    )

                    # Define colors for different IC types
                    color_map = {
                        "brain": "#d4edda",  # Light green
                        "eog": "#f9e79f",  # Light yellow
                        "muscle": "#f5b7b1",  # Light red
                        "ecg": "#d7bde2",  # Light purple,
                        "ch_noise": "#ffd700",  # Light orange
                        "line_noise": "#add8e6",  # Light blue
                        "other": "#f0f0f0",  # Light grey
                    }
                    colors.append(
                        [color_map.get(comp_info["ic_type"].lower(), "white")] * 4
                    )

                # Create and customize table
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=["Component", "Type", "Confidence", "Rejected"],
                    loc="center",
                    cellLoc="center",
                    cellColours=colors,
                    colWidths=[0.2, 0.3, 0.25, 0.25],
                )

                # Customize table appearance
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)  # Reduced vertical scaling

                # Add title with page information, filename and timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fig_table.suptitle(
                    f"ICA Components Summary - {self.config['bids_path'].basename}\n"
                    f"(Page {page + 1} of {num_pages})\n"
                    f"Generated: {timestamp}",
                    fontsize=12,
                    y=0.95,
                )
                # Add legend for colors
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

                # Add margins
                plt.subplots_adjust(top=0.85, bottom=0.15)

                pdf.savefig(fig_table)
                plt.close(fig_table)

            # First page: Component topographies overview
            fig_topo = ica.plot_components(picks=component_indices, show=False)
            if isinstance(fig_topo, list):
                for f in fig_topo:
                    pdf.savefig(f)
                    plt.close(f)
            else:
                pdf.savefig(fig_topo)
                plt.close(fig_topo)

            # If rejected components, add overlay plot
            if components == "rejected":
                fig_overlay = plt.figure()
                end_time = min(30.0, self.raw.times[-1])

                # Create a copy of raw data with only the channels used in ICA training
                # to avoid shape mismatch during pre-whitening
                raw_copy = self.raw.copy()

                # Get the channel names that were used for ICA training
                ica_ch_names = self.final_ica.ch_names

                # Pick only those channels from the raw data
                if len(ica_ch_names) != len(raw_copy.ch_names):
                    message(
                        "warning",
                        f"Channel count mismatch: ICA has {len(ica_ch_names)} channels, "
                        f"raw has {len(raw_copy.ch_names)}. Using only ICA channels for plotting.",
                    )
                    # Keep only the channels that were used in ICA
                    raw_copy.pick_channels(ica_ch_names)

                fig_overlay = self.final_ica.plot_overlay(
                    raw_copy,
                    start=0,
                    stop=end_time,
                    exclude=component_indices,
                    show=False,
                )
                fig_overlay.set_size_inches(15, 10)  # Set size after creating figure

                pdf.savefig(fig_overlay)
                plt.close(fig_overlay)

            # For each component, create detailed plots
            for idx in component_indices:
                fig = plt.figure(constrained_layout=True, figsize=(12, 8))
                gs = GridSpec(nrows=3, ncols=3, figure=fig)

                # Axes for ica.plot_properties
                ax1 = fig.add_subplot(gs[0, 0])  # Data
                ax2 = fig.add_subplot(gs[0, 1])  # Epochs image
                ax3 = fig.add_subplot(gs[0, 2])  # ERP/ERF
                ax4 = fig.add_subplot(gs[1, 0])  # Spectrum
                ax5 = fig.add_subplot(gs[1, 1])  # Topomap
                ax_props = [ax1, ax2, ax3, ax4, ax5]

                # Plot properties
                ica.plot_properties(
                    raw,
                    picks=[idx],
                    axes=ax_props,
                    dB=True,
                    plot_std=True,
                    log_scale=False,
                    reject="auto",
                    show=False,
                )

                # Add time series plot
                ax_timeseries = fig.add_subplot(gs[2, :])  # Last row, all columns
                ax_timeseries.plot(times, ica_data[idx, :n_samples], linewidth=0.5)
                ax_timeseries.set_xlabel("Time (seconds)")
                ax_timeseries.set_ylabel("Amplitude")
                ax_timeseries.set_title(
                    f"Component {idx + 1} Time Course ({duration}s)"
                )

                # Add labels
                comp_info = ic_labels.iloc[idx]
                label_text = (
                    f"Component {comp_info['component']}\n"
                    f"Type: {comp_info['ic_type']}\n"
                    f"Confidence: {comp_info['confidence']:.2f}"
                )

                fig.suptitle(
                    label_text,
                    fontsize=14,
                    fontweight="bold",
                    color=(
                        "red"
                        if comp_info["ic_type"]
                        in ["eog", "muscle", "ch_noise", "line_noise", "ecg"]
                        else "black"
                    ),
                )

                # Save the figure
                pdf.savefig(fig)
                plt.close(fig)

            print(f"Report saved to {pdf_path}")
            return Path(pdf_path).name

    def verify_topography_plot(self) -> bool:
        """Use ica topograph to verify MEA channel placement.
        This function simply runs fast ICA then plots the topography.
        It is used on mouse files to verify channel placement.

        """
        # pylint: disable=import-outside-toplevel
        from mne.preprocessing import ICA

        derivatives_dir = Path(self.config["derivatives_dir"])

        ica = ICA(  # pylint: disable=not-callable
            n_components=len(self.raw.ch_names) - len(self.raw.info["bads"]),
            method="fastica",
            random_state=42,
        )
        ica.fit(self.raw)

        fig = ica.plot_components(
            picks=range(len(self.raw.ch_names) - len(self.raw.info["bads"])), show=False
        )

        fig.savefig(derivatives_dir / "ica_topography.png")

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
        ax1.set_xticklabels([f"IC{i+1}" for i in range(n_components)])
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
                    f"IC{i+1}",
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

        # Save the figure
        derivatives_dir = Path(self.config["derivatives_dir"])
        basename = self.config["bids_path"].basename
        basename = basename.replace("_eeg", "_ica_classification_comparison")
        target_figure = derivatives_dir / basename

        # Save figure with higher DPI
        fig.savefig(target_figure, dpi=300, bbox_inches="tight")

        metadata = {
            "artifact_reports": {
                "creationDateTime": datetime.now().isoformat(),
                "ica_classification_comparison": Path(target_figure).name,
            }
        }

        self._update_metadata("compare_vision_iclabel_classifications", metadata)

        return fig
