# AutoClean Unified Plugin System: User Guide

This guide explains how to use and extend the AutoClean plugin system to add support for new file formats and EEG montages without modifying core code.

## Overview

The AutoClean plugin system uses a unified approach where each plugin handles:
1. File format import
2. Montage configuration 
3. Event processing

All in a single integrated unit, making it easier to handle specific format+montage combinations.

## Directory Structure

```
src/autoclean/
├── step_functions/
│   └── io.py                     # Core interfaces (don't modify)
├── plugins/
│   ├── formats/                  # Format registrations
│   │   ├── __init__.py
│   │   └── additional_formats.py # Register new formats here
│   ├── eeg_plugins/              # Unified plugins for format+montage combinations
│   │   ├── __init__.py
│   │   └── eeglab_mea30_plugin.py # Example plugin
```

## Basic Usage

### Using Plugins to Import EEG Data

```python
from autoclean.step_functions.io import import_eeg

# Configure the import
autoclean_dict = {
    "run_id": "my_analysis_001",
    "unprocessed_file": "/path/to/my/data.set",  # Can be any supported format
    "eeg_system": "MEA30",                       # Montage/system name
    "task": "oddball",                           # Optional task type
    # ...other configuration options...
}

# Import the data - plugins are automatically discovered and used
eeg_data = import_eeg(autoclean_dict)

# eeg_data can be either Raw or Epochs depending on the plugin
```

## Creating a Custom Plugin

### Step 1: Register Your File Format (if needed)

If you're adding support for a new file format, register it first:

```python
# src/autoclean/plugins/formats/my_formats.py
from autoclean.step_functions.io import register_format

# Register your format - extension without dot, format ID in UPPERCASE
register_format('xyz', 'XYZ_FORMAT')
```

### Step 2: Create Your Plugin

Create a file in `src/autoclean/plugins/eeg_plugins/` named descriptively to indicate the format+montage combination:

```python
# src/autoclean/plugins/eeg_plugins/xyz_custom64_plugin.py
from autoclean.step_functions.io import BaseEEGPlugin
from autoclean.utils.logging import message

class XYZCustom64Plugin(BaseEEGPlugin):
    """Plugin for XYZ files with Custom-64 montage."""
    
    VERSION = "1.0.0"  # Track your plugin version
    
    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        """Check which format+montage combinations this plugin supports."""
        return format_id == 'XYZ_FORMAT' and montage_name == 'CustomCap-64'
    
    def import_and_configure(self, file_path, autoclean_dict, preload=True):
        """Import data and configure montage in one step."""
        message("info", f"Processing {file_path} with {autoclean_dict['eeg_system']}")
        
        try:
            # 1. Import the data
            # Your import code here...
            
            # 2. Configure the montage
            # Your montage configuration code here...
            
            # 3. Process events if applicable
            # Your event processing code here...
            
            # Return either Raw or Epochs object
            return raw  # or return epochs
            
        except Exception as e:
            raise RuntimeError(f"Failed to process data: {str(e)}")
    
    def process_events(self, raw, autoclean_dict):
        """Process events after import (for Raw data)."""
        # This is called automatically by import_eeg for Raw data
        # Events code here...
        return events, event_id, events_df
    
    def get_metadata(self):
        """Return additional metadata about this plugin."""
        return {
            "plugin_version": self.VERSION,
            # Any other metadata you want to include
        }
```

### Step 3: Use Your Plugin

Your plugin will be automatically discovered and used when the appropriate format and montage are specified:

```python
autoclean_dict = {
    "unprocessed_file": "my_data.xyz",  # Your custom format
    "eeg_system": "CustomCap-64",       # Your custom montage
    # ...other options...
}

# Your plugin will be automatically selected
eeg_data = import_eeg(autoclean_dict)
```

## Complete Example

Here's a complete example that adds support for XDF files with a BioSemi-64 montage:

```python
# src/autoclean/plugins/formats/bids_formats.py
from autoclean.step_functions.io import register_format
register_format('xdf', 'XDF_FORMAT')

# src/autoclean/plugins/eeg_plugins/xdf_biosemi64_plugin.py
import mne
import pyxdf
from autoclean.step_functions.io import BaseEEGPlugin
from autoclean.utils.logging import message

class XDFBioSemi64Plugin(BaseEEGPlugin):
    """Plugin for XDF files with BioSemi-64 montage."""
    
    VERSION = "1.0.0"
    
    @classmethod
    def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
        return format_id == 'XDF_FORMAT' and montage_name == 'biosemi64'
    
    def import_and_configure(self, file_path, autoclean_dict, preload=True):
        message("info", f"Processing XDF file with BioSemi-64 montage: {file_path}")
        
        try:
            # Load XDF file using pyxdf
            streams, header = pyxdf.load_xdf(file_path)
            
            # Find EEG stream and extract data
            eeg_stream = next((s for s in streams if s['info']['type'][0] == 'EEG'), None)
            if not eeg_stream:
                raise RuntimeError("No EEG stream found in XDF file")
                
            # Extract data, create MNE Raw object
            data = eeg_stream['time_series'].T
            sfreq = float(eeg_stream['info']['nominal_srate'][0])
            ch_names = [f"EEG{i+1}" for i in range(data.shape[0])]
            
            # Create Raw object
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data, info)
            
            # Apply BioSemi-64 montage
            montage = mne.channels.make_standard_montage('biosemi64')
            raw.set_montage(montage)
            
            # Process events from marker stream if present
            marker_stream = next((s for s in streams if s['info']['type'][0] == 'Markers'), None)
            if marker_stream:
                onsets = marker_stream['time_stamps'] - eeg_stream['time_stamps'][0]
                descriptions = [str(m[0]) for m in marker_stream['time_series']]
                raw.set_annotations(mne.Annotations(onsets, np.zeros_like(onsets), descriptions))
            
            return raw
            
        except Exception as e:
            raise RuntimeError(f"Failed to process XDF file: {str(e)}")
```

## Plugin Naming Conventions

Use clear, descriptive names that indicate the format and montage combination:

- `eeglab_mea30_plugin.py` - EEGLAB files with MEA30 montage
- `egi_raw_gsn129_plugin.py` - EGI Raw files with GSN-HydroCel-129 montage
- `brainvision_standard1020_plugin.py` - BrainVision files with standard 10-20 montage

## Returning Raw vs. Epochs

Your plugin can return either a `mne.io.Raw` or `mne.Epochs` object:

```python
def import_and_configure(self, file_path, autoclean_dict, preload=True):
    # ... processing code ...
    
    if autoclean_dict.get("return_epochs", False):
        # Create and return epochs
        events, _ = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, preload=True)
        return epochs
    else:
        # Return continuous data
        return raw
```

## Advanced Plugin Features

### Task-Specific Processing

Handle different experimental paradigms:

```python
def import_and_configure(self, file_path, autoclean_dict, preload=True):
    # ... basic import code ...
    
    # Apply task-specific processing
    task = autoclean_dict.get("task", None)
    if task == "oddball":
        # Apply oddball-specific processing
        mapping = {"1": "Standard", "2": "Target"}
        raw.annotations.rename(mapping)
    elif task == "mmn":
        # Apply MMN-specific processing
        # ...
        
    return raw
```

### Multiple Format/Montage Support

A single plugin can support multiple combinations:

```python
@classmethod
def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
    # Support multiple combinations
    supported = [
        ('EEGLAB_SET', 'GSN-HydroCel-64'),
        ('EEGLAB_SET', 'GSN-HydroCel-128'),
        ('EEGLAB_SET', 'GSN-HydroCel-256'),
    ]
    return (format_id, montage_name) in supported
```

### Custom Configuration Options

Pass additional configuration via `autoclean_dict`:

```python
def import_and_configure(self, file_path, autoclean_dict, preload=True):
    # Get custom options
    reference = autoclean_dict.get("reference", "average")
    filter_settings = autoclean_dict.get("filter", {"l_freq": 0.1, "h_freq": 40})
    
    # ... import code ...
    
    # Apply reference
    if reference == "average":
        raw.set_eeg_reference("average")
    elif reference == "mastoids":
        raw.set_eeg_reference(["M1", "M2"])
    
    # Apply filters
    raw.filter(l_freq=filter_settings["l_freq"], h_freq=filter_settings["h_freq"])
    
    return raw
```

## Troubleshooting

If your plugin isn't working:

1. **Plugin Discovery**:
   - Ensure your plugin is in the correct directory (`plugins/eeg_plugins/`)
   - Check that `__init__.py` files exist in all plugin directories

2. **Format Registration**:
   - Verify your format is registered correctly in a file in `plugins/formats/`
   - Check for typos in format IDs

3. **Debugging Tips**:
   - Add `message("info", "Debug message")` calls in your plugin
   - Check if `supports_format_montage()` is returning `True` as expected
   - Test your plugin with a simple test script before integrating

4. **Common Issues**:
   - Mismatch between registered format ID and what's checked in `supports_format_montage()`
   - Forgetting to handle exceptions properly
   - Using incompatible MNE versions
