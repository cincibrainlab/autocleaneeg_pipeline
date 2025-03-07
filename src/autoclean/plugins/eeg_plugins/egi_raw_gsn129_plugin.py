# src/autoclean/plugins/eeg_plugins/egi_raw_gsn129_plugin.py
"""EGI .raw file plugin with GSN-HydroCel-129 montage configuration.

This plugin handles the complete import and montage configuration
for EGI .raw files with GSN-HydroCel-129 electrode system.
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path

from autoclean.step_functions.io import BaseEEGPlugin
from autoclean.utils.logging import message


class EGIRawGSN129Plugin(BaseEEGPlugin):
    """Plugin for EGI .raw files with GSN-HydroCel-129 montage.
    
    This plugin handles the specific combination of EGI .raw files
    with the GSN-HydroCel-129 electrode system, which requires special
    handling for the 129th electrode.
    """
    
    VERSION = "1.0.0"
    
    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check if this plugin supports the given format and montage combination."""
        return format_id == "EGI_RAW" and montage_name == "GSN-HydroCel-129"
    
    def import_and_configure(self, file_path: Path, autoclean_dict: dict, preload: bool = True):
        """Import EGI .raw file and configure GSN-HydroCel-129 montage."""
        message("info", f"Loading EGI .raw file with GSN-HydroCel-129 montage: {file_path}")
        
        try:
            # Step 1: Import the .raw file
            raw = mne.io.read_raw_egi(
                input_fname=file_path,
                preload=preload,
                events_as_annotations=True,
                verbose=True
            )
            message("success", "Successfully loaded .raw file")
            
            # Step 2: Configure the GSN-HydroCel-129 montage
            message("info", "Configuring GSN-HydroCel-129 montage")
            
            # Create montage and set the special 129th electrode name
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
            montage.ch_names[128] = "E129"  # Special handling for 129th electrode
            
            # Apply the montage
            raw.set_montage(montage, match_case=False)
            
            # Pick only EEG channels
            raw.pick("eeg")
            
            message("success", "Successfully configured GSN-HydroCel-129 montage")
            
            # Step 3: Process events
            # EGI .raw files have events stored as TTL pulses in the recording
            # These are converted to annotations by mne when loading the file
            
            # Step 4: Apply task-specific processing if needed
            task = autoclean_dict.get("task", None)
            if task:
                if task == "p300_grael4k":
                    message("info", "Processing P300 task-specific annotations")
                    mapping = {"13": "Standard", "14": "Target"}
                    raw.annotations.rename(mapping)
                elif task == "hbcd_mmn":
                    message("info", "Processing HBCD MMN task-specific annotations")
                    # This task may require special handling that was not present in the
                    # original code for EGI .raw files
                    message("warning", "HBCD MMN task not fully implemented for EGI .raw files")
                
                # Check if task-specific event handling is needed
                if "tasks" in autoclean_dict and task in autoclean_dict["tasks"]:
                    # Get epoch settings
                    settings = autoclean_dict["tasks"][task]["settings"]
                    epoch_settings = settings.get("epoch_settings", {})
                    event_id = epoch_settings.get("event_id")
                    
                    # If event_id is not None, process the events
                    if event_id is not None:
                        try:
                            target_event_id = event_id
                            events, event_id_map = mne.events_from_annotations(raw)
                            rev_target_event_id = dict(map(reversed, target_event_id.items()))
                            raw.set_annotations(None)
                            raw.set_annotations(
                                mne.annotations_from_events(
                                    events, raw.info["sfreq"], event_desc=rev_target_event_id
                                )
                            )
                            message("success", "Successfully processed task-specific events")
                        except Exception as e:
                            message("warning", f"Failed to process task-specific events: {str(e)}")
            
            return raw
            
        except Exception as e:
            raise RuntimeError(f"Failed to process EGI .raw file with GSN-HydroCel-129 montage: {str(e)}")
    
    def process_events(self, raw: mne.io.Raw, autoclean_dict: dict) -> tuple:
        """Process events and annotations in the EEG data."""
        message("info", "Processing events from EGI .raw file")
        try:
            # Get events from annotations
            events, event_id = mne.events_from_annotations(raw)
            
            # Create a more detailed events DataFrame
            if events is not None and len(events) > 0:
                events_df = pd.DataFrame({
                    'time': events[:, 0] / raw.info['sfreq'],
                    'sample': events[:, 0],
                    'id': events[:, 2],
                    'type': [event_id.get(str(id), f"Unknown-{id}") for id in events[:, 2]]
                })
                return events, event_id, events_df
            else:
                return None, None, None
                
        except Exception as e:
            message("warning", f"Failed to process events: {str(e)}")
            return None, None, None
    
    def get_metadata(self) -> dict:
        """Get additional metadata about this plugin."""
        return {
            "plugin_name": self.__class__.__name__,
            "plugin_version": self.VERSION,
            "montage_details": {
                "type": "GSN-HydroCel-129",
                "channel_count": 129,
                "manufacturer": "Electrical Geodesics, Inc. (EGI)",
                "reference": "Common Reference",
                "layout": "Geodesic",
                "file_format": "EGI .raw binary format"
            }
        }