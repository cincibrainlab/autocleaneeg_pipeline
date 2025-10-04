# MNE pick_types() to inst.pick() Migration Guide

## Overview
This document outlines the migration from the legacy `mne.pick_types()` and `mne.pick_channels()` functions to the modern `inst.pick()` method in the autocleaneeg_pipeline codebase.

## Background
As of MNE-Python 1.6+, both `pick_types()` and `pick_channels()` are marked as LEGACY functions. The recommended replacement is the `inst.pick()` method, which provides a more flexible and intuitive API for channel selection.

## Migration Patterns

### Pattern 1: pick_types() on instance (eeg=True)
**Old:**
```python
inst.pick_types(eeg=True, exclude=[])
```

**New:**
```python
inst.pick('eeg', exclude=[])
```

### Pattern 2: mne.pick_types() for getting indices
**Old:**
```python
picks = mne.pick_types(inst.info, eeg=True, meg=False, ref_meg=False, exclude=[])
```

**New:**
```python
# Option 1: Use inst.pick() if modifying the instance
inst.pick('eeg', exclude=[])

# Option 2: Use pick_info if just getting indices (internal use)
from mne.io.pick import pick_info
picks = pick_info(inst.info, 'eeg', exclude=[])

# Option 3: Use channel_indices_by_type
from mne.io import pick_types as _pick_types
picks = _pick_types(inst.info, eeg=True, meg=False, ref_meg=False, exclude=[])
```

**Note:** For getting channel indices without modifying the instance, we need to research the recommended MNE approach. The `pick()` method modifies the instance in-place and returns the instance, not indices.

### Pattern 3: pick_types() for EOG channels
**Old:**
```python
eog_picks = mne.pick_types(data.info, eog=True)
```

**New:**
```python
# Need to research: How to get indices for specific channel type without modifying instance
# Possible approach: use pick_info or similar internal function
```

### Pattern 4: pick_channels() for channel name lists
**Old:**
```python
picks = mne.pick_channels(inst.ch_names, channel_list)
```

**New:**
```python
# Option 1: If modifying the instance
inst.pick(channel_list)

# Option 2: If getting indices
picks = [inst.ch_names.index(ch) for ch in channel_list if ch in inst.ch_names]
```

### Pattern 5: pick_channels() on instance
**Old:**
```python
raw_copy.pick_channels(ica_ch_names)
```

**New:**
```python
raw_copy.pick(ica_ch_names)
```

## Files Requiring Updates

### pick_types() Usage (5 occurrences in 4 files, 2 removed with combined_image.py)

1. **src/autoclean/plugins/eeg_plugins/eeglab_mea30_plugin.py:61**
   - Pattern: `raw.pick_types(eeg=True, exclude=[])`
   - Migration: `raw.pick('eeg', exclude=[])`

2. **src/autoclean/functions/preprocessing/wavelet_thresholding.py:109**
   - Pattern: `mne.pick_types(inst.info, eeg=True, meg=False, ref_meg=False, exclude=[])`
   - Migration: **REQUIRES RESEARCH** - Need indices, not in-place modification
   - Context: Returns channel indices for processing

3. **src/autoclean/io/export.py:628**
   - Pattern: `eeg_epochs.pick_types(eeg=True, exclude=[])`
   - Migration: `eeg_epochs.pick('eeg', exclude=[])`

4. **src/autoclean/io/export.py:638**
   - Pattern: `raw.pick_types(eeg=True, exclude=[])`
   - Migration: `raw.pick('eeg', exclude=[])`

5. **src/autoclean/mixins/signal_processing/channels.py:97**
   - Pattern: `mne.pick_types(data.info, eog=True)`
   - Migration: **REQUIRES RESEARCH** - Need indices for EOG channels
   - Context: Gets EOG channel indices to convert them to EEG type

6. **src/autoclean/mixins/signal_processing/channels.py:475**
   - Pattern: `mne.pick_types(data.info, eog=True)`
   - Migration: **REQUIRES RESEARCH** - Need indices for EOG channels
   - Context: Detects EOG channels

7. **src/autoclean/mixins/reporting/combined_image.py:992** *(FILE REMOVED)*
   - Pattern: `raw_obj.pick_types(eeg=True, exclude=[])`
   - Migration: N/A - File removed in fastplot removal (remove-fastplot-qa-integration branch)

8. **src/autoclean/mixins/reporting/combined_image.py:1018** *(FILE REMOVED)*
   - Pattern: `eeg_epochs.pick_types(eeg=True, exclude=[])`
   - Migration: N/A - File removed in fastplot removal (remove-fastplot-qa-integration branch)

### pick_channels() Usage (6 occurrences in 4 files)

1. **src/autoclean/step_functions/continuous.py:155-156** (COMMENTED OUT)
   - Pattern: `mne.pick_channels(raw.ch_names, bad_channels)`
   - Migration: Not needed - code is commented out

2. **src/autoclean/functions/analysis/statistical_learning.py:396**
   - Pattern: `mne.pick_channels(itc.ch_names, picks)`
   - Migration: **REQUIRES RESEARCH** - Need indices from channel names
   - Context: Gets indices for specific channels

3. **src/autoclean/functions/analysis/statistical_learning.py:652**
   - Pattern: `mne.pick_channels(itc.ch_names, picks)`
   - Migration: **REQUIRES RESEARCH** - Need indices from channel names

4. **src/autoclean/functions/analysis/statistical_learning.py:764**
   - Pattern: `mne.pick_channels(itc.ch_names, picks)`
   - Migration: **REQUIRES RESEARCH** - Need indices from channel names

5. **src/autoclean/mixins/viz/ica.py:456**
   - Pattern: `raw_copy.pick_channels(ica_ch_names)`
   - Migration: `raw_copy.pick(ica_ch_names)`

6. **src/autoclean/mixins/viz/visualization.py:316-317**
   - Pattern: `mne.pick_channels(raw.ch_names, bad_channels)`
   - Migration: **REQUIRES RESEARCH** - Need indices from channel names
   - Context: Gets indices for bad channels

## Research Needed

### Critical Question: How to get channel indices without modifying instance?

Many uses of `pick_types()` and `pick_channels()` are to GET channel indices without modifying the instance. The new `inst.pick()` method modifies the instance in-place.

**Research tasks:**
1. What is the recommended MNE-Python approach for getting channel indices by type?
2. Is there a public API replacement for getting indices?
3. Should we use internal functions like `_pick_info`?
4. What about using list comprehensions vs MNE utilities?

**Possible approaches:**
- Use `mne.channel_indices_by_type()`
- Use internal `mne.io.pick._picks_to_idx()`
- Use list comprehensions: `[inst.ch_names.index(ch) for ch in channel_list]`
- Keep using `pick_types()` for index-only operations (if it's not truly deprecated)

## Implementation Strategy

### Phase 1: Simple replacements (4 occurrences after combined_image.py removal)
Replace straightforward `inst.pick_types()` and `inst.pick_channels()` calls that modify the instance.
Note: 2 occurrences in combined_image.py were removed when that file was deleted in the fastplot removal.

### Phase 2: Research index-based operations
Investigate the proper MNE-Python approach for getting channel indices without modifying instances.

### Phase 3: Update index-based operations (7 occurrences)
Implement the researched approach for operations that need indices.

### Phase 4: Testing
Test all changes to ensure behavior is preserved.

## References
- [MNE Forum: pick_channels() is a legacy function](https://mne.discourse.group/t/pick-channels-is-a-legacy-function/8445)
- [MNE Documentation: mne.io.Raw.pick()](https://mne.tools/stable/generated/mne.io.Raw.html)
- [MNE Documentation: mne.pick_types()](https://mne.tools/stable/generated/mne.pick_types.html)
