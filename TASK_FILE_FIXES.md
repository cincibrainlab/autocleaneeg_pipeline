# Task File Structure Fixes

## Issues Found in Your Task File

### 1. **Missing Import Statement**
```python
# ❌ Missing
from autoclean.core.task import Task

# ✅ Fixed - Added at the top
from typing import Any, Dict
from autoclean.core.task import Task
```

### 2. **Missing `__init__` Method**
The Task base class requires proper initialization:

```python
# ❌ Missing __init__ method
class p300_grael4k(Task):
    def run(self) -> None:
        # ...

# ✅ Fixed - Added proper __init__
class p300_grael4k(Task):
    def __init__(self, config: Dict[str, Any]):
        self.raw = None
        self.original_raw = None
        self.epochs = None
        super().__init__(config)  # IMPORTANT!
```

### 3. **Using psd_fmax with Nested Config**
Your config has nested structure with 'value' dictionaries:

```python
# Your config structure:
'ICLabel': {
    'enabled': True,
    'value': {  # Note: nested inside 'value'
        'ic_flags_to_reject': [...],
        'ic_rejection_threshold': 0.3,
        'psd_fmax': 40.0
    }
}

# ✅ Correct way to access:
psd_fmax = self.settings.get('ICLabel', {}).get('value', {}).get('psd_fmax')
```

### 4. **Passing psd_fmax to classify_ica_components**
```python
# ❌ Original - not passing psd_fmax
self.classify_ica_components(method='icvision')

# ✅ Fixed - passing psd_fmax from config
psd_fmax = self.settings.get('ICLabel', {}).get('value', {}).get('psd_fmax')
self.classify_ica_components(
    method='icvision',
    reject=True,
    psd_fmax=psd_fmax
)
```

## Key Points

1. **Task Base Class**: Always call `super().__init__(config)` in your __init__ method
2. **Config Access**: Use `self.settings` to access your module-level config
3. **Nested Config**: Navigate through nested dictionaries carefully
4. **Parameter Passing**: Extract psd_fmax and pass it explicitly

## Testing Your Fix

1. Copy the corrected version to your tasks directory
2. Run with: `autoclean-eeg process p300_grael4k your_data.raw`
3. Check that PSD plots are limited to 40 Hz in the output

## Alternative Config Structure

If you prefer flatter config structure:

```python
# Alternative flatter structure
'ICLabel': {
    'enabled': True,
    'ic_flags_to_reject': ['muscle', 'heart', 'eog'],
    'ic_rejection_threshold': 0.3,
    'psd_fmax': 40.0  # At same level as other params
}

# Then access simply:
psd_fmax = self.settings.get('ICLabel', {}).get('psd_fmax')
```