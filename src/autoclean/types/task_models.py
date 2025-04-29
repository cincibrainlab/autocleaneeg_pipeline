# task_models.py
"""Pydantic models for defining and validating task metadata in the autoclean pipeline.

This module provides structured data models for capturing, validating, and storing
metadata at various stages of EEG processing. The models ensure consistency in
data representation and enable type checking during development.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BaseMetadata(BaseModel):
    """Common fields for all metadata entries."""

    creationDateTime: str = Field(default_factory=lambda: datetime.now().isoformat())
    step_name: str  # Name of the processing step (e.g., "import_eeg")


class ImportMetadata(BaseMetadata):
    """Metadata for the import step."""

    step_name: str = "import_eeg"
    import_function: str = "import_eeg"
    unprocessedFile: str
    eegSystem: str
    sampleRate: float
    channelCount: int
    durationSec: float
    numberSamples: int
    hasEvents: bool

    class Config:
        """Configuration for the ImportMetadata class."""
        arbitrary_types_allowed = True  # Allow Path or mne.io.Raw if passed directly


# Main metadata class with optional steps
class ProcessingMetadata(BaseModel):
    """Container for all processing stage metadata.
    
    This class serves as the top-level model for organizing metadata from
    different processing stages. Each field represents metadata from a specific
    processing step, which may or may not be present depending on the pipeline's
    execution.
    """
    import_eeg: Optional[ImportMetadata] = None
