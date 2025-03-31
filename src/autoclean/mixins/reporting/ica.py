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

Example:
    ```python
    from autoclean.core.task import Task
    from mne.preprocessing import ICA
    
    class MyICATask(Task):
        def process(self, raw, pipeline, autoclean_dict):
            # Perform ICA decomposition
            ica = ICA(n_components=20)
            ica.fit(raw)
            
            # Create visualizations and reports
            self.plot_ica_components(ica, raw, autoclean_dict, pipeline)
            self.generate_ica_report(ica, raw, autoclean_dict, pipeline)
    ```
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.gridspec import GridSpec

from autoclean.utils.logging import message
from autoclean.mixins.reporting.base import BaseVizMixin

# Force matplotlib to use non-interactive backend for async operations
matplotlib.use("Agg")


class ICAReportingMixin(BaseVizMixin):
    """Mixin providing ICA reporting functionality for EEG data.
    
    This mixin extends the base ReportingMixin with specialized methods for
    generating visualizations and reports of ICA results. It provides tools for
    assessing component properties, visualizing component activations, and
    documenting component rejection decisions.
    
    All reporting methods respect configuration toggles from `autoclean_config.yaml`,
    checking if their corresponding step is enabled before execution. Each method
    can be individually enabled or disabled via configuration.
    
    Available ICA reporting methods include:
    
    - `plot_ica_full`: Plot all ICA components over the full recording duration
    - `plot_ica_components`: Generate plots of ICA components with properties
    - `plot_ica_sources`: Plot ICA sources with EOG channel overlay
    - `generate_ica_report`: Create a comprehensive report of ICA decomposition results
    """
    
    def plot_ica_full(self, pipeline: Any, autoclean_dict: Dict[str, Any]) -> plt.Figure:
        """Plot ICA components over the full duration with their labels and probabilities.
        
        This method creates a figure showing each ICA component's time course over the full
        recording duration. Components are color-coded by their classification/rejection status,
        and probability scores are indicated for each component.
        
        Args:
            pipeline: Pipeline object containing raw data and fitted ICA object.
            autoclean_dict: Dictionary containing metadata about the processing run.
            
        Returns:
            matplotlib.figure.Figure: The generated figure with ICA components.
            
        Raises:
            ValueError: If no ICA object is found in the pipeline.
            
        Example:
            ```python
            # After performing ICA
            fig = task.plot_ica_full(pipeline, autoclean_dict)
            plt.show()
            ```
            
        Notes:
            - Components classified as artifacts are highlighted in red
            - Classification probabilities are shown for each component
            - The method respects configuration settings via the `ica_full_plot_step` config
        """
        # Get raw and ICA from pipeline
        raw = pipeline.raw
        ica = pipeline.ica2
        ic_labels = pipeline.flags["ic"]

        # Get ICA activations and create time vector
        ica_sources = ica.get_sources(raw)
        ica_data = ica_sources.get_data()
        times = raw.times
        n_components, n_samples = ica_data.shape

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
            ax.plot(times, ica_data[idx] + offset, color=line_colors[idx], linewidth=0.5)

        # Set y-ticks and labels
        yticks = [idx * spacing for idx in range(n_components)]
        yticklabels = []
        for idx in range(n_components):
            label_text = f"IC{idx + 1}: {ic_labels['ic_type'][idx]} ({ic_labels['confidence'][idx]:.2f})"
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

        # Get output path for ICA components figure
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
        target_figure = str(
            derivatives_path.copy().update(
                suffix="ica_components_full_duration", extension=".png", datatype="eeg"
            )
        )

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
        pipeline: Any,
        autoclean_dict: Dict[str, Any],
        duration: int = 60,
    ) -> None:
        """Generate comprehensive ICA reports.
        
        Generates two reports:
        1. All ICA components with properties and activations
        2. Only the rejected ICA components with their properties
        
        Parameters:
        -----------
        pipeline : pylossless.Pipeline
            The pipeline object containing the ICA and raw data
        cleaned_raw : mne.io.Raw
            Cleaned raw data after processing
        autoclean_dict : dict
            Dictionary containing configuration and paths
        duration : int
            Duration in seconds for plotting time series data
        """
        # Generate report for all components
        report_filename = self._plot_ica_components(
            pipeline = pipeline,
            autoclean_dict = autoclean_dict,
            duration = duration,
            components = "all"
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
            pipeline = pipeline,
            autoclean_dict = autoclean_dict,
            duration = duration,
            components = "rejected"
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
        pipeline: Any,
        autoclean_dict: Dict[str, Any],
        duration: int = 60,
        components: str = "all",
    ):
        """
        Plots ICA components with labels and saves reports.

        Parameters:
        -----------
        pipeline : pylossless.Pipeline
            Pipeline object containing raw data and ICA.
        autoclean_dict : dict
            Autoclean dictionary containing metadata.
        duration : int
            Duration in seconds to plot.
        components : str
            'all' to plot all components, 'rejected' to plot only rejected components.
        """
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.gridspec import GridSpec

        # Get raw and ICA from pipeline
        raw = pipeline.raw
        ica = pipeline.ica2
        ic_labels = pipeline.flags["ic"]

        # Determine components to plot
        if components == "all":
            component_indices = range(ica.n_components_)
            report_name = "ica_components_all"
        elif components == "rejected":
            component_indices = ica.exclude
            report_name = "ica_components_rejected"
            if not component_indices:
                print("No components were rejected. Skipping rejected components report.")
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
        derivatives_path = pipeline.get_derivative_path(autoclean_dict["bids_path"])
        pdf_path = str(derivatives_path.copy().update(suffix=report_name, extension=".pdf"))

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
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fig_table.suptitle(
                    f"ICA Components Summary - {autoclean_dict['bids_path'].basename}\n"
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
                end_time = min(30.0, pipeline.raw.times[-1])
                
                # Create a copy of raw data with only the channels used in ICA training
                # to avoid shape mismatch during pre-whitening
                raw_copy = pipeline.raw.copy()
                
                # Get the channel names that were used for ICA training
                ica_ch_names = pipeline.ica2.ch_names
                
                # Pick only those channels from the raw data
                if len(ica_ch_names) != len(raw_copy.ch_names):
                    message('warning',f"Channel count mismatch: ICA has {len(ica_ch_names)} channels, raw has {len(raw_copy.ch_names)}. Using only ICA channels for plotting.")
                    # Keep only the channels that were used in ICA
                    raw_copy.pick_channels(ica_ch_names)
                
                fig_overlay = pipeline.ica2.plot_overlay(
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
                ax_timeseries.set_title(f"Component {idx + 1} Time Course ({duration}s)")

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