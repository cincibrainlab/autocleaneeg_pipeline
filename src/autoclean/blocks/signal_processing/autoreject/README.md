# AutoReject Processing Block

**Category**: Signal Processing
**Version**: 1.0.0
**Status**: Stable

## Overview

**AutoReject** is a machine learning-based method for automated artifact rejection and channel interpolation in EEG epochs. It uses cross-validation to learn optimal, data-driven rejection thresholds for each channel individually, eliminating the need for manual inspection and arbitrary threshold setting.

This implementation is based on the autoreject Python package by Jas et al. (2017), which has become a standard tool for objective quality control in EEG research.

## Scientific Background

### How AutoReject Works

AutoReject employs a sophisticated cross-validation approach to determine optimal rejection parameters:

```
1. Create parameter grid:
   - n_interpolate: [1, 4, 8] channels to interpolate
   - consensus: [0.1, 0.25, 0.5, 0.75, 0.9] thresholds

2. For each parameter combination:
   - Split data into K folds (default: 4-fold CV)
   - Train on K-1 folds, validate on held-out fold
   - Calculate reconstruction error

3. Select optimal parameters:
   - Choose combination with lowest CV error
   - Apply to entire dataset

4. Clean epochs:
   - Interpolate bad channels within epochs
   - Reject epochs that still exceed thresholds
```

### Key Advantages

✅ **Data-driven**: Thresholds learned from your specific data characteristics
✅ **Per-channel optimization**: Different thresholds for different channels
✅ **Objective**: Reproducible decisions without manual inspection
✅ **Channel interpolation**: Repairs bad channels instead of always rejecting
✅ **Cross-validated**: Prevents overfitting to specific artifact patterns

### Comparison with Other Methods

| Method | Threshold | Inspection | Interpolation | Speed |
|--------|-----------|------------|---------------|-------|
| **AutoReject** | Data-driven, per-channel | Automated | Yes | Slow (1-10 min) |
| **Manual rejection** | Subjective | Manual | Manual | Very slow (hours) |
| **Fixed threshold** | Arbitrary, global | Automated | No | Fast (seconds) |
| **Wavelet threshold** | Universal, continuous data | Automated | N/A | Fast (seconds) |

**When to use AutoReject over alternatives:**
- Need objective, reproducible quality control
- Have sufficient epochs (100+) for robust CV
- Want to repair channels rather than just reject
- Processing event-related data where epochs are precious
- Batch processing multiple datasets

**When to use alternatives:**
- Need real-time processing (use simple thresholding)
- Have very few epochs (<20) (use manual inspection)
- Processing continuous data (use wavelet threshold)
- Speed is critical (use fixed thresholds)

## Configuration

### Basic Configuration

```python
config = {
    "apply_autoreject": {
        "enabled": True,
        "value": {
            "n_interpolate": [1, 4, 8],
            "consensus": [0.1, 0.25, 0.5, 0.75, 0.9],
            "n_jobs": 4,
            "cv": 4,
            "random_state": 42
        }
    }
}
```

### Parameter Guide

#### `n_interpolate` (list of int)
**Number of channels to interpolate during parameter search**

**Default**: `[1, 4, 8]`

**Guidelines by montage**:
- **High-density (128+ channels)**: `[1, 4, 8, 16]`
- **Standard (32-64 channels)**: `[1, 4, 8]` (recommended)
- **Low-density (<32 channels)**: `[1, 2, 4]`

**Trade-off**: Higher values = more repair attempts, but risk over-interpolation

#### `consensus` (list of float)
**Consensus percentages for determining bad epochs (0.0-1.0)**

**Default**: `[0.1, 0.25, 0.5, 0.75, 0.9]`

**Interpretation**:
- `0.1`: Liberal (more aggressive rejection)
- `0.5`: Moderate (balanced approach)
- `0.9`: Conservative (preserve more epochs)

**Typical ranges**:
- **High-quality data**: `[0.5, 0.75, 0.9]`
- **Standard data**: `[0.1, 0.25, 0.5, 0.75, 0.9]` (recommended)
- **Noisy data**: `[0.1, 0.25, 0.5]`

#### `n_jobs` (int)
**Number of parallel jobs for cross-validation**

**Default**: `1`
**Recommended**: `4` to `-1` (all cores)

**Performance impact**:
- `1`: Single-threaded (slowest, lowest memory)
- `4`: 4-core parallel (good balance)
- `-1`: All available cores (fastest, highest memory)

#### `cv` (int)
**Number of cross-validation folds**

**Default**: `4`
**Range**: `2-10`

**Guidelines**:
- Minimum 2 folds required
- 4-5 folds typical (good balance)
- More folds = more robust but slower
- Requires at least `cv` epochs in dataset

#### `random_state` (int or null)
**Random seed for reproducible CV splits**

**Default**: `null` (non-reproducible)
**Recommended**: Set to integer (e.g., `42`) for reproducibility

#### `picks` (list of str or null)
**Channel names to include**

**Default**: `null` (all EEG channels)

**When to use**:
- Focus quality control on specific regions
- Exclude non-EEG channels
- Process subset for speed

#### `thresh_method` (str)
**Threshold optimization method**

**Options**:
- `"bayesian_optimization"` (default, recommended)
- `"random_search"` (faster, less optimal)

## Usage Examples

### Example 1: Basic Epoch Cleaning

```python
from autoclean.core.task import Task

config = {
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {"enabled": True, "value": {"l_freq": 0.1, "h_freq": 40}},

    # Create epochs first
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -0.2, "tmax": 0.8}
    },

    # Apply AutoReject to epochs
    "apply_autoreject": {
        "enabled": True,
        "value": {
            "n_interpolate": [1, 4, 8],
            "consensus": [0.1, 0.25, 0.5, 0.75, 0.9],
            "n_jobs": 4
        }
    }
}

class ERP_With_AutoReject(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()

        # Create epochs
        self.create_eventid_epochs()

        # Apply AutoReject
        self.apply_autoreject()

        # Continue processing with cleaned epochs
        self.generate_reports()
```

### Example 2: Conservative Cleaning (High-Quality Data)

```python
config = {
    "apply_autoreject": {
        "enabled": True,
        "value": {
            "n_interpolate": [1, 2, 4],      # Less interpolation
            "consensus": [0.5, 0.75, 0.9],   # More conservative
            "n_jobs": -1,
            "random_state": 42
        }
    }
}
```

### Example 3: Aggressive Cleaning (Noisy Data)

```python
config = {
    "apply_autoreject": {
        "enabled": True,
        "value": {
            "n_interpolate": [1, 4, 8, 16],  # More interpolation options
            "consensus": [0.1, 0.25, 0.5],   # More aggressive
            "n_jobs": -1,
            "random_state": 42
        }
    }
}
```

### Example 4: Fast Processing (Limited CV)

```python
config = {
    "apply_autoreject": {
        "enabled": True,
        "value": {
            "n_interpolate": [1, 4],          # Fewer options
            "consensus": [0.25, 0.5, 0.75],   # Fewer options
            "cv": 2,                          # Faster CV
            "n_jobs": -1,
            "thresh_method": "random_search"  # Faster method
        }
    }
}
```

## Output and Quality Control

### Generated Output

**Cleaned epochs**: Saved to `derivatives/apply_autoreject/`
**File**: `*_autoreject_epo.set`

### Metadata Logged

```python
{
    "initial_epochs": 150,
    "final_epochs": 142,
    "rejected_epochs": 8,
    "rejection_percent": 5.33,
    "interpolated_channels": ["Fp1", "Fp2", "F7"],
    "n_interpolate": [1, 4, 8],
    "consensus": [0.1, 0.25, 0.5, 0.75, 0.9],
    "cv_folds": 4,
    "cv_scores": [...],
    "epoch_duration": 1.0,
    "total_duration_sec": 142.0
}
```

### Interpreting Results

**Rejection percentage guidelines**:
- `<10%`: Good quality, typical for clean data
- `10-25%`: Moderate artifacts, acceptable
- `25-50%`: High artifacts, consider preprocessing or parameters
- `>50%`: Very noisy, review data quality or use alternative methods

**Interpolated channels**: Common channels (Fp1/Fp2, T7/T8) indicate typical eye/muscle artifacts

## When to Use AutoReject

### ✅ Ideal For:

- **Event-related potentials (ERPs)**: Preserve as many epochs as possible while maintaining quality
- **Oddball paradigms**: Especially important for rare events
- **Multi-site studies**: Objective, reproducible criteria across labs
- **Large-scale analysis**: Automated QC for hundreds of datasets
- **Publication-quality data**: Defendable, data-driven rejection criteria

### ⚠️ Limitations:

- **Requires sufficient epochs**: Minimum 20-30, optimal 100+
- **Computationally intensive**: 1-10 minutes per dataset
- **Memory requirements**: Scales with (n_epochs × n_channels × n_times × cv)
- **Not for continuous data**: Use wavelet threshold or filtering instead
- **Doesn't fix all artifacts**: Some epochs may still be noisy after cleaning

## Troubleshooting

### Not Enough Epochs for CV

**Error**: `ValueError: Need at least 4 epochs for 4-fold cross-validation`

**Solution**: Reduce `cv` parameter or create more epochs:
```python
"cv": 2  # Use 2-fold CV for limited data
```

### High Rejection Rate (>50%)

**Causes**:
1. Poor data quality (check raw data)
2. Too aggressive consensus thresholds
3. Inadequate preprocessing (filtering, bad channels)

**Solutions**:
```python
# More conservative thresholds
"consensus": [0.5, 0.75, 0.9]

# Less interpolation
"n_interpolate": [1, 2, 4]

# Review preprocessing steps
# - Apply proper filtering first
# - Remove obvious bad channels before AutoReject
# - Check for line noise, DC drifts
```

### Slow Processing

**Solutions**:
```python
# Use more CPU cores
"n_jobs": -1

# Reduce parameter grid
"n_interpolate": [1, 4],
"consensus": [0.25, 0.5, 0.75],

# Reduce CV folds
"cv": 2,

# Faster optimization method
"thresh_method": "random_search"
```

### Memory Errors

**Solutions**:
- Process fewer epochs at a time (chunk data)
- Reduce `cv` folds
- Use fewer `n_jobs`
- Downsample data if appropriate

## Integration with Other Processing Steps

### Recommended Preprocessing Order

```python
def run(self):
    # 1. Basic preprocessing
    self.import_raw()
    self.resample_data()
    self.filter_data()              # IMPORTANT: Filter before epoching

    # 2. Bad channel detection (optional but recommended)
    self.clean_bad_channels()

    # 3. Create epochs
    self.create_eventid_epochs()

    # 4. Apply AutoReject
    self.apply_autoreject()         # ← AutoReject here

    # 5. Continue with clean data
    self.average_epochs()
    self.generate_reports()
```

### Combining with ICA

**Option A: ICA before AutoReject** (recommended for eye artifacts)
```python
def run(self):
    self.filter_data()
    self.run_ica()                  # Remove eye movements, heartbeat
    self.create_epochs()
    self.apply_autoreject()         # Clean remaining artifacts
```

**Option B: AutoReject before ICA** (alternative)
```python
def run(self):
    self.filter_data()
    self.create_epochs()
    self.apply_autoreject()         # Remove bad epochs first
    # Convert back to continuous for ICA
    self.run_ica()
```

### Combining with Wavelet Threshold

```python
def run(self):
    self.filter_data()
    self.apply_wavelet_threshold()  # Remove transient artifacts in continuous data
    self.create_epochs()
    self.apply_autoreject()         # Clean epochs
```

## References

### Primary Papers

**Jas, M., Engemann, D. A., Bekhti, Y., Raimondo, F., & Gramfort, A. (2017)**.
Autoreject: Automated artifact rejection for MEG and EEG data.
*NeuroImage*, 159, 417-429.
[DOI: 10.1016/j.neuroimage.2017.06.030](https://doi.org/10.1016/j.neuroimage.2017.06.030)

**Jas, M., Engemann, D. A., Raimondo, F., Bekhti, Y., & Gramfort, A. (2016)**.
Automated rejection and repair of bad trials in MEG/EEG.
*2016 International Workshop on Pattern Recognition in Neuroimaging (PRNI)*, 1-4.
[DOI: 10.1109/PRNI.2016.7552336](https://doi.org/10.1109/PRNI.2016.7552336)

### Software

**AutoReject Package**
Official documentation: [https://autoreject.github.io/](https://autoreject.github.io/)
GitHub repository: [https://github.com/autoreject/autoreject](https://github.com/autoreject/autoreject)

## See Also

- [Wavelet Threshold Block](../wavelet_threshold/) - For continuous data artifact removal
- [Task Library](https://docs.autocleaneeg.org/tasks/task-library) - Pre-built tasks using AutoReject
- [Processing Blocks Introduction](https://docs.autocleaneeg.org/processing-blocks/introduction)

## Version History

- **1.0.0** (2025-09-29): Initial release
  - Core AutoReject implementation
  - Cross-validation based parameter optimization
  - Channel interpolation and epoch rejection
  - Comprehensive metadata logging

## License

MIT License - See repository root for details

## Contributors

- AutoCleanEEG Team
- Maintainer: Ernest Pedapati (ernest.pedapati@cchmc.org)