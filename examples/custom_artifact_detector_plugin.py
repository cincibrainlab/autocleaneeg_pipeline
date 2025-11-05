#!/usr/bin/env python3
"""Example: Custom ML-Based Artifact Detection Plugin

This example demonstrates the CORRECT use of the plugin system:
- Extends pipeline functionality (doesn't duplicate existing mixins)
- Imports helper functions from pipeline (zero code duplication)
- Provides new capability not available in core pipeline
- Easy to share within research group or community

This is a TEMPLATE - not a working implementation.
"""

from typing import Optional
import mne
import numpy as np

# Import from pipeline - this ensures zero code duplication
from autoclean.utils.logging import message

__block_metadata__ = {
    "name": "ml_artifact_detector",
    "version": "1.0.0",
    "description": "Machine learning based artifact detection using pre-trained models",
    "author": "Research Group X",
    "maintainer": "researcher@university.edu",
    "license": "MIT",
    "category": "signal_processing",
    "tags": ["artifact-detection", "machine-learning", "custom"],
    "dependencies": {
        "python": ">=3.10",
        "packages": {
            "tensorflow": ">=2.10.0",  # Example ML dependency
            "scikit-learn": ">=1.2.0"
        },
        "autocleaneeg-pipeline": ">=3.0.0-alpha"
    }
}


class MLArtifactDetectorMixin:
    """Mixin for ML-based artifact detection.

    This extends the pipeline with custom functionality that doesn't
    exist in the core implementation. It uses helper functions from
    the pipeline but adds new capabilities.
    """

    def apply_ml_artifact_detection(
        self,
        data: Optional[mne.io.BaseRaw] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.8,
    ) -> Optional[mne.io.BaseRaw]:
        """Apply ML-based artifact detection.

        Parameters
        ----------
        data
            Optional raw object to analyze. If not provided, uses self.raw.
        model_path
            Path to pre-trained model. If None, uses default model.
        confidence_threshold
            Minimum confidence for artifact classification (0.0-1.0).

        Returns
        -------
        Optional[mne.io.BaseRaw]
            The cleaned raw instance with artifacts marked/removed.

        Notes
        -----
        This is a TEMPLATE demonstrating plugin architecture.
        Replace with actual ML implementation.
        """
        inst = data if data is not None else getattr(self, "raw", None)
        if inst is None:
            message("warning", "ML artifact detection skipped: no raw data available")
            return inst

        # Check if step is enabled in configuration
        is_enabled, settings = self._check_step_enabled("ml_artifact_detection")
        if not is_enabled:
            message("info", "ML artifact detection disabled in configuration")
            return inst

        # Extract parameters from config
        params = (settings or {}).get("value", {})
        model_path = params.get("model_path", model_path)
        confidence_threshold = params.get("confidence_threshold", confidence_threshold)

        message("info", f"Applying ML artifact detection (threshold={confidence_threshold})")

        try:
            # PLACEHOLDER: Replace with actual ML implementation
            # This would typically:
            # 1. Load pre-trained model
            # 2. Extract features from EEG data
            # 3. Run inference
            # 4. Mark/remove detected artifacts

            # Example structure (not functional):
            # model = load_model(model_path)
            # features = extract_features(inst.get_data())
            # predictions = model.predict(features)
            # clean_data = remove_artifacts(inst, predictions, confidence_threshold)

            message("success", "ML artifact detection complete")
            message("info", "Note: This is a template - implement actual ML logic here")

            # For now, return unchanged data
            return inst

        except Exception as e:
            message("error", f"ML artifact detection failed: {e}")
            return inst

    def export_artifact_report(
        self,
        output_path: Optional[str] = None,
    ) -> None:
        """Export artifact detection report with ML confidence scores.

        Parameters
        ----------
        output_path
            Path to save report. If None, saves to derivatives directory.

        Notes
        -----
        This demonstrates how plugins can add reporting capabilities.
        """
        if not hasattr(self, "raw") or self.raw is None:
            message("warning", "Cannot export report: no data available")
            return

        message("info", "Generating ML artifact detection report")

        # PLACEHOLDER: Implement actual reporting
        # This would typically generate:
        # - Visualization of detected artifacts
        # - Confidence scores per channel/time
        # - Summary statistics
        # - Comparison with traditional methods

        message("success", "Report exported (placeholder)")


# Optional: Helper function that doesn't need to be in the mixin
def _load_ml_model(model_path: str):
    """Load pre-trained artifact detection model.

    This is an internal helper function for the plugin.
    """
    # Placeholder for actual model loading
    pass


def ml_artifact_descriptor():
    """Schema descriptor for ML artifact detection configuration.

    This defines the configuration structure for the plugin.
    Follows the same pattern as internal mixins.

    Returns
    -------
    dict
        Configuration schema for the ML artifact detection step.
    """
    return {
        "enabled": True,
        "value": {
            "model_path": None,  # Path to pre-trained model
            "confidence_threshold": 0.8,  # Confidence threshold (0.0-1.0)
            "export_report": True,  # Generate detailed report
            "features": ["amplitude", "frequency", "connectivity"],  # Features to extract
        }
    }