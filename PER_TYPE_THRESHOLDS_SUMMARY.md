# Per-IC-Type Confidence Thresholds - Implementation Complete

## 🎯 **What Was Implemented**

A flexible system allowing users to set different confidence thresholds for different IC artifact types in their task configuration files.

## 📋 **Changes Made**

### 1. **Core Function Enhancement** (`ica_processing.py`)
- Added `ic_rejection_overrides` parameter to `apply_ica_component_rejection()`
- Updated rejection logic to use per-type thresholds when available
- Enhanced documentation and type hints

### 2. **Mixin Integration** (`ica.py`)
- Automatic extraction of `ic_rejection_overrides` from config
- Enhanced logging to show per-type thresholds
- Warning for unused overrides
- Metadata tracking of threshold configuration

### 3. **Backward Compatibility**
- Existing configs work unchanged
- `ic_rejection_overrides` is completely optional
- Default behavior identical when overrides not specified

## 🔧 **How to Use**

### Basic Configuration (Unchanged)
```python
'ICLabel': {
    'enabled': True,
    'value': {
        'ic_flags_to_reject': ['muscle', 'heart', 'eog'],
        'ic_rejection_threshold': 0.5  # Global threshold
    }
}
```

### Advanced Configuration (New)
```python
'ICLabel': {
    'enabled': True,
    'value': {
        'ic_flags_to_reject': ['muscle', 'heart', 'eog', 'ch_noise'],
        'ic_rejection_threshold': 0.3,     # Global default
        'ic_rejection_overrides': {        # Optional per-type overrides
            'muscle': 0.99,                # Very conservative (only 99% confidence)
            'heart': 0.80,                 # Moderate threshold
            'eog': 0.60                    # More aggressive
            # ch_noise uses global 0.3
        }
    }
}
```

## 📊 **Logging Output Examples**

### Without Overrides
```
Will reject ICs of types: ['muscle', 'heart', 'eog'] with confidence > 0.5
```

### With Overrides
```
Will reject ICs with per-type thresholds: {'muscle': 0.99, 'heart': 0.80, 'eog': 0.60}
```

### With Warnings
```
WARNING: Threshold overrides specified for types not in rejection list: {'ecg'}
```

## 🧪 **Testing**

The implementation includes:
- Backward compatibility verification
- Per-type threshold functionality
- Edge case handling (empty overrides, unused overrides)
- Integration with existing AutoClean workflow

## 🏃‍♂️ **Use Cases**

### Movement Studies
```python
'ic_rejection_overrides': {
    'muscle': 0.99,  # Be very conservative with muscle artifacts
    'eog': 0.5       # More aggressive with eye movements
}
```

### Sleep Studies
```python
'ic_rejection_overrides': {
    'eog': 0.3,      # Aggressive EOG removal for sleep spindles
    'muscle': 0.8,   # Moderate muscle artifact removal
    'heart': 0.9     # Conservative heart artifact removal
}
```

### Resting State
```python
'ic_rejection_overrides': {
    'heart': 0.95,   # Very conservative with heart artifacts
    'line_noise': 0.6  # More aggressive line noise removal
}
```

## 🔍 **Technical Details**

### Implementation Strategy
- **Minimal Changes**: Only 3 core locations modified
- **Reuse Existing Patterns**: Leveraged existing config parsing
- **Fail Gracefully**: Invalid configs warn but don't crash
- **Performance**: Negligible overhead (single dict lookup per component)

### Code Locations
- `src/autoclean/functions/ica/ica_processing.py`: Core rejection logic
- `src/autoclean/mixins/signal_processing/ica.py`: Config extraction and validation
- Task files: User configuration

### Validation Features
- Warns about overrides for non-rejected types
- Logs complete threshold configuration
- Tracks configuration in metadata
- Maintains audit trail

## ✅ **Verification**

To verify the implementation works:

1. **Run test script**:
   ```bash
   python test_per_type_thresholds.py
   ```

2. **Use example task**:
   ```bash
   autoclean-eeg process p300_grael4k_with_overrides data.raw
   ```

3. **Check logs** for per-type threshold messages

## 🚀 **Next Steps**

The implementation is complete and ready for use. Consider:
- Adding to configuration wizard
- Including in task templates
- Adding to documentation examples
- Performance validation with large datasets

## 📈 **Benefits Achieved**

- ✅ **Flexibility**: Different thresholds for different artifact types
- ✅ **Backward Compatibility**: No breaking changes
- ✅ **User-Friendly**: Clear configuration and logging
- ✅ **Robust**: Proper validation and error handling
- ✅ **Maintainable**: Minimal, focused changes