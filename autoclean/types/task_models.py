# task_models.py
from pydantic import BaseModel, Field, StrictFloat
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import mne

class BaseMetadata(BaseModel):
    """Common fields for all metadata entries."""
    creationDateTime: str = Field(default_factory=lambda: datetime.now().isoformat())
    step_name: str  # Name of the processing step (e.g., "step_import")

class ImportMetadata(BaseMetadata):
    """Metadata for the import step."""
    step_name: str = "step_import"
    import_function: str = "step_import"
    unprocessedFile: str
    eegSystem: str
    sampleRate: float
    channelCount: int
    durationSec: float
    numberSamples: int
    hasEvents: bool

    class Config:
        arbitrary_types_allowed = True  # Allow Path or mne.io.Raw if passed directly

# Main metadata class with optional steps
class ProcessingMetadata(BaseModel):
    step_import: Optional[ImportMetadata] = None