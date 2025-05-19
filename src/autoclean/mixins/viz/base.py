"""Base reporting mixin for autoclean tasks.

This module provides the foundation for all reporting mixins in the AutoClean
pipeline. It defines the base class that all specialized reporting mixins
inherit from, providing common utility methods and a consistent interface for
generating visualizations and reports from EEG data.

The ReportingMixin class is designed to be used as a mixin with Task classes,
providing them with reporting capabilities while maintaining a clean separation
of concerns. This modular approach allows for flexible composition of reporting
functionality across different task types.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from mne_bids import BIDSPath

from autoclean.utils.database import manage_database
from autoclean.utils.logging import message


class BaseVizMixin:
    """Base mixin class providing reporting functionality for EEG data.

    This mixin serves as the foundation for all reporting operations in the
    AutoClean pipeline. It provides utility methods for generating visualizations,
    reports, and summaries from EEG data processing results.

    The ReportingMixin is designed to be used with Task classes through multiple
    inheritance, allowing tasks to gain reporting capabilities while maintaining
    a clean separation of concerns. Specialized reporting mixins inherit from
    this base class and extend it with specific functionality.

    Attributes:
        config (Dict[str, Any]): Task configuration dictionary (provided by the parent class)
        raw (mne.io.Raw): MNE Raw object containing the EEG data (if available)
        epochs (mne.Epochs): MNE Epochs object containing epoched data (if available)

    Note:
        This class expects to be mixed in with a class that provides access to
        configuration settings via the `config` attribute and data objects via
        the `raw` and/or `epochs` attributes.
    """

    def _get_derivatives_path(self, bids_path: Optional[BIDSPath] = None) -> Path:
        """Get the derivatives path for saving reports and visualizations.

        Args:
            bids_path: Optional BIDSPath object. If None, attempts to get from pipeline or config

        Returns:
            Path object pointing to the derivatives directory

        Raises:
            ValueError: If derivatives path cannot be determined
        """
        # If bids_path is provided, use it directly
        if self.config:
            if "derivatives_dir" in self.config:
                return Path(self.config["derivatives_dir"])

        raise ValueError("Could not determine derivatives path")

    def _update_metadata(self, operation: str, metadata_dict: Dict[str, Any]) -> None:
        """Update the database with metadata about a reporting operation.

        Args:
            operation: Name of the operation
            metadata_dict: Dictionary of metadata to store
        """
        if not hasattr(self, "config") or not self.config.get("run_id"):
            return

        # Add creation timestamp if not present
        if "creationDateTime" not in metadata_dict:
            metadata_dict["creationDateTime"] = datetime.now().isoformat()

        metadata = {operation: metadata_dict}

        run_id = self.config.get("run_id")
        manage_database(
            operation="update", update_record={"run_id": run_id, "metadata": metadata}
        )

    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """Save a matplotlib figure to the derivatives directory.

        Args:
            fig: Matplotlib figure to save
            filename: Base filename (without path or extension)
            dpi: Resolution for saving the figure

        Returns:
            Full path to the saved figure
        """
        try:
            derivatives_path = self._get_derivatives_path()
            figure_path = derivatives_path / f"{filename}.png"

            # Save figure
            fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            message("info", f"Figure saved to {figure_path}")
            return str(figure_path)

        except Exception as e:  # pylint: disable=broad-exception-caught
            message("error", f"Error saving figure: {str(e)}")
            return ""
