# EOG Channel Reference Guide

Quick reference for configuring EOG (electrooculogram) channels across different EEG montage systems used in AutoClean EEG pipeline.

## Overview

EOG channels capture eye movement and blink artifacts. After ICA processing removes eye-related components, these channels can be dropped from the dataset using the `eog_drop: True` parameter.

## Montage-Specific Channel Mappings

### 1. EGI GSN-HydroCel-129 (High-Density Net)

**System**: 129-channel Geodesic Sensor Net
**Montage ID**: `GSN-HydroCel-129`
**Manufacturer**: Electrical Geodesics Inc. (EGI)

**EOG Channel Configuration**:
```python
"eog_step": {
    "enabled": True,
    "value": {
        "eog_indices": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
        "eog_drop": True,
    },
}
```

**Channel Descriptions**:
- **E1** - Outer right eye (vertical EOG)
- **E32** - Outer left eye (vertical EOG)
- **E8, E14, E17, E21, E25** - Periocular channels around eyes
- **E125, E126, E127, E128** - Lower face/neck channels (capture blink artifacts)

**Total**: 11 EOG channels

---

### 2. Standard 10-20 System

**System**: International 10-20 electrode placement system
**Montage ID**: `standard_1020`
**Common Use**: Clinical EEG, qEEG, Grael systems, portable EEG devices

**EOG Channel Configuration**:
```python
"eog_step": {
    "enabled": True,
    "value": {
        "eog_indices": [31, 32],  # Fp1, Fp2
        "eog_drop": True,
    },
}
```

**Channel Descriptions**:
- **Channel 31 (Fp1)** - Left frontal pole
- **Channel 32 (Fp2)** - Right frontal pole

**Note**: In standard 10-20 systems, Fp1 and Fp2 serve dual purposes:
- Primary function: Record frontal brain activity
- Secondary function: Capture vertical eye movements and blinks due to proximity to eyes

**Total**: 2 EOG channels

---

### 3. Grael 4K System (Standard 10-20 Implementation)

**System**: Compumedics Grael 4K
**Montage ID**: `standard_1020`
**Common Configurations**: 19-channel, 21-channel, or custom layouts

**EOG Channel Configuration**:
```python
"eog_step": {
    "enabled": True,
    "value": {
        "eog_indices": [31, 32],  # Fp1, Fp2 (or adjust based on your channel order)
        "eog_drop": True,
    },
}
```

**Important Notes for Grael Systems**:
1. **Verify channel order**: Grael systems may use different channel numbering
2. **Check your montage file**: Confirm Fp1/Fp2 positions in your specific setup
3. **Dedicated EOG electrodes**: If your Grael setup includes dedicated EOG electrodes (often placed laterally for horizontal EOG), add those indices to the list
4. **Example with dedicated EOG**:
   ```python
   "eog_indices": [31, 32, 33, 34],  # Fp1, Fp2, HEOG-L, HEOG-R
   ```

**Total**: 2-4 EOG channels (depending on configuration)

---

### 4. EGI GSN-HydroCel-124

**System**: 124-channel Geodesic Sensor Net
**Montage ID**: `GSN-HydroCel-124`

**EOG Channel Configuration**:
```python
"eog_step": {
    "enabled": True,
    "value": {
        "eog_indices": [1, 32, 8, 14, 17, 21, 25, 121, 122, 123, 124],
        "eog_drop": True,
    },
}
```

**Note**: Similar periocular coverage as 129-channel net, adjusted for 124-channel layout.

---

### 5. MEA30 (Multi-Electrode Array - Mouse/Rodent Systems)

**System**: 30-channel mouse EEG array
**Montage ID**: `MEA30`
**Use Case**: Preclinical rodent studies

**EOG Configuration**:
- MEA30 systems typically do not have dedicated EOG channels
- Rodent eye movements are less prominent in scalp recordings
- If needed, consult your specific MEA30 layout documentation

---

## Configuration Best Practices

### Full Dict Format (Recommended)
```python
"eog_step": {
    "enabled": True,
    "value": {
        "eog_indices": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
        "eog_drop": True,  # Drop channels after ICA removes EOG components
    },
}
```

### Backward-Compatible Bare List Format
```python
"eog_step": {
    "enabled": True,
    "value": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],  # eog_drop defaults to False
}
```

### Keep EOG Channels (No Removal)
```python
"eog_step": {
    "enabled": True,
    "value": {
        "eog_indices": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
        "eog_drop": False,  # Keep EOG channels in final dataset
    },
}
```

---

## Workflow: How EOG Channels Are Processed

1. **Assignment** (`assign_eog_channels`): Channels marked as EOG type
2. **ICA Processing**: ICA components capturing EOG artifacts are identified and removed
3. **Channel Removal** (`drop_eog_channels`): If `eog_drop: True`, EOG channels are removed from dataset
4. **Final Export**: Clean EEG data without EOG artifacts

---

## Troubleshooting

### Issue: "Specified EOG channels not found in data"
**Solution**: Verify channel indices match your actual data:
```python
# Check channel names in your raw data
print(raw.ch_names)
print(f"Total channels: {len(raw.ch_names)}")
```

### Issue: "EOG artifacts still present after ICA"
**Solutions**:
1. Ensure EOG channels are properly assigned before ICA
2. Increase ICA components: `"n_components": None` (auto)
3. Check ICA classification threshold: `"ic_rejection_threshold": 0.3`
4. Verify `eog` is in `ic_flags_to_reject` list

### Issue: "How do I find the right channel indices?"
**Solutions**:
1. Check your electrode cap/net layout diagram
2. Use MNE channel visualization: `raw.plot_sensors()`
3. Consult manufacturer documentation (EGI, Compumedics, etc.)
4. For 10-20 systems, Fp1/Fp2 are typically channels 31-32, but **always verify** with your specific setup

---

## Example Task Configurations

### High-Density EGI Task
```python
config = {
    "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
    "eog_step": {
        "enabled": True,
        "value": {
            "eog_indices": [1, 32, 8, 14, 17, 21, 25, 125, 126, 127, 128],
            "eog_drop": True,
        },
    },
}
```

### Clinical 10-20 Task
```python
config = {
    "montage": {"enabled": True, "value": "standard_1020"},
    "eog_step": {
        "enabled": True,
        "value": {
            "eog_indices": [31, 32],  # Fp1, Fp2
            "eog_drop": True,
        },
    },
}
```

### Grael 4K with Dedicated HEOG
```python
config = {
    "montage": {"enabled": True, "value": "standard_1020"},
    "eog_step": {
        "enabled": True,
        "value": {
            "eog_indices": [31, 32, 33, 34],  # Fp1, Fp2, HEOG-L, HEOG-R
            "eog_drop": True,
        },
    },
}
```

---

## References

- **EGI Documentation**: https://www.egi.com/
- **10-20 System**: Jasper, H.H. (1958). The ten-twenty electrode system of the International Federation
- **MNE-Python Montages**: https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html

---

**Document Version**: 1.0.0
**Last Updated**: 2025-09-29
**Maintainer**: AutoClean EEG Development Team