# PSD Frequency Limit (psd_fmax) Implementation

## Overview

Your senior developer has successfully integrated the `psd_fmax` parameter from the `icvision` package into the AutoClean pipeline. This allows users to limit the frequency range of Power Spectral Density (PSD) plots generated during ICA component classification.

## What Was Changed

### 1. **Core ICA Processing Function** (`src/autoclean/functions/ica/ica_processing.py`)
- Added `**kwargs` parameter to `classify_ica_components()` function
- Updated docstring to document the new parameter support
- Modified icvision call to pass through kwargs: `label_components(raw, ica, **kwargs)`

### 2. **ICA Mixin** (`src/autoclean/mixins/signal_processing/ica.py`)
- Added `psd_fmax: float | None = None` parameter to `classify_ica_components()` method
- Updated docstring with parameter description and examples
- Implemented conditional forwarding of psd_fmax to the underlying function
- Added code examples showing different usage patterns

## How to Use

### Option 1: Configuration File (Recommended)
Add `psd_fmax` to your task configuration YAML under the ICLabel section:

```yaml
ICLabel:
  enabled: true
  psd_fmax: 40.0  # Limit PSD plots to 1-40 Hz
  ic_flags_to_reject: ["eog", "muscle"]
  ic_rejection_threshold: 0.80
```

### Option 2: In Task Code (Automatic)
The method now automatically reads from config, so you just need:

```python
def run(self):
    # ... other processing steps ...
    
    # psd_fmax is automatically read from config!
    self.classify_ica_components(
        method="icvision",
        reject=True
    )
```

If you want to override the config value:

```python
# Override with explicit value
self.classify_ica_components(
    method="icvision",
    reject=True,
    psd_fmax=30.0  # Override config value
)
```

### Option 3: Hardcoded Value
For specific use cases, you can hardcode the value:

```python
self.classify_ica_components(
    method="icvision",
    reject=True,
    psd_fmax=30.0  # Always limit to 30 Hz
)
```

## Benefits

1. **Focused Analysis**: Limit PSD plots to relevant frequency ranges (e.g., 1-40 Hz for standard EEG)
2. **Smaller File Sizes**: Reduced plot file sizes by excluding irrelevant high frequencies
3. **Backward Compatible**: Existing code works without modification (psd_fmax defaults to None)
4. **Flexible**: Can be set per-task or per-run as needed

## Requirements

- `autoclean-icvision >= 0.5.0` (for psd_fmax support)
- No changes needed for tasks using ICLabel (parameter is ignored)

## Migration Guide

For existing tasks, no changes are required. To opt-in to the feature:

1. Add `psd_fmax: 40.0` to your config file under ICLabel section
2. Update your task to pass the config value as shown above

## Technical Details

- The parameter is only used when `method="icvision"`
- ICLabel ignores this parameter (no error)
- Default behavior (when psd_fmax=None) uses 80 Hz or Nyquist frequency
- The implementation uses conditional kwargs to maintain compatibility