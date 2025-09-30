# Wavelet Thresholding Processing Block

**Category**: Signal Processing
**Version**: 1.0.0
**Status**: Stable

## Overview

Wavelet thresholding is a powerful signal processing technique for removing transient artifacts from EEG data using discrete wavelet transform (DWT) with universal thresholding. This method is particularly effective at removing high-amplitude transient artifacts (eye movements, muscle activity, electrode pops) while preserving the underlying neural signals.

This implementation follows the HAPPE pipeline approach and provides both standard and ERP-preserving modes.

## Scientific Background

### Discrete Wavelet Transform (DWT)

The DWT decomposes a signal into approximation and detail coefficients at multiple scales:

```
Signal → DWT → [Approximation, Detail₁, Detail₂, ..., Detailₙ]
                ↓
          Threshold details
                ↓
          Inverse DWT → Denoised Signal
```

### Universal Thresholding

The universal threshold (Donoho & Johnstone, 1994) is calculated as:

```
T = σ × √(2 × log(N))
```

Where:
- `σ` = median absolute deviation (MAD) of finest detail coefficients / 0.6745
- `N` = signal length

**Key advantages:**
- Data-driven threshold selection
- Adapts to signal characteristics
- Proven theoretical optimality

### Soft vs Hard Thresholding

**Soft thresholding** (default, recommended):
- Shrinks coefficients toward zero
- Smoother reconstructed signal
- Better for EEG with gradual transitions

**Hard thresholding** (aggressive):
- Binary keep/discard decision
- Can create discontinuities
- Use for very high-amplitude artifacts

## Configuration

### Basic Configuration

```python
config = {
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",              # Wavelet family
            "level": 5,                      # Decomposition level
            "threshold_mode": "soft",        # "soft" or "hard"
            "is_erp": False,                 # ERP-preserving mode
            "threshold_scale": 1.0,          # Threshold multiplier
            "bandpass": (1.0, 30.0),        # For ERP mode
            "psd_fmax": None,                # PSD analysis ceiling
            "picks": None,                   # Channel selection
            "filter_kwargs": None            # Additional filter args
        }
    }
}
```

### Parameter Guide

#### `wavelet` (string)
**Options**: Any PyWavelets family (e.g., `"sym4"`, `"db4"`, `"coif1"`)
**Default**: `"sym4"` (symlet 4)
**Recommendation**: `"sym4"` works well for most EEG applications

Popular choices:
- `"sym4"`: Balanced, good for general EEG
- `"db4"`: Daubechies, similar to sym4
- `"coif1"`: Coiflets, more regular
- `"haar"`: Simplest, use for testing only

#### `level` (integer or "auto")
**Range**: 0 to max_level (automatically clamped)
**Default**: `5`
**Recommendation**:
- 5-6 for typical EEG (250-1000 Hz sampling)
- "auto" to use maximum possible level

Higher levels = more aggressive denoising but risk over-smoothing.

#### `threshold_mode` (string)
**Options**: `"soft"` or `"hard"`
**Default**: `"soft"`
**Recommendation**: Use `"soft"` for most cases

- **Soft**: Gradual shrinkage, smoother results
- **Hard**: HAPPE's high-artifact mode, more aggressive

#### `is_erp` (boolean)
**Default**: `False`
**Recommendation**: `True` for event-related potentials, `False` for continuous EEG

**ERP-preserving mode** (when `True`):
1. Filters signal to ERP band (bandpass)
2. Applies wavelet denoising to filtered signal
3. Estimates artifacts from filtered space
4. Subtracts artifacts from original unfiltered signal
5. Applies final bandpass filter

This preserves ERP morphology better than applying wavelet directly.

#### `threshold_scale` (float)
**Range**: > 0.0
**Default**: `1.0`
**Recommendation**:
- `1.0`: Standard universal threshold
- `< 1.0`: Less aggressive, preserves more signal
- `> 1.0`: More aggressive, removes more artifacts

Adjust based on artifact severity:
- High-quality data: 0.8-1.0
- Moderate artifacts: 1.0-1.5
- Heavy artifacts: 1.5-2.0

#### `bandpass` (tuple or None)
**Format**: `(low_freq, high_freq)`
**Default**: `(1.0, 30.0)`
**Used only when**: `is_erp=True`

Typical ERP bands:
- P300: (1.0, 30.0)
- MMN: (1.0, 20.0)
- ASSR: (1.0, 100.0)

#### `psd_fmax` (float or None)
**Default**: `None` (uses 45 Hz)
**Recommendation**: Set to Nyquist - 0.5 Hz for full spectrum

Controls PSD analysis ceiling in reports.

#### `picks` (string, list, or None)
**Options**:
- `None`: All channels
- `"eeg"`: EEG channels only
- `["Fz", "Cz", "Pz"]`: Specific channels
- `[0, 1, 2]`: Channel indices

**Recommendation**: `None` or `"eeg"` for most cases

#### `filter_kwargs` (dict or None)
**Default**: `None`
**Example**: `{"n_jobs": 4, "method": "fir"}`

Additional arguments passed to MNE's filter method (ERP mode only).

## Usage Examples

### Example 1: Basic Continuous EEG

```python
from autoclean.core.task import Task

config = {
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",
            "level": 5,
            "threshold_mode": "soft",
            "is_erp": False,
            "threshold_scale": 1.0
        }
    }
}

class RestingWithWavelet(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.assign_eog_channels()

        # Apply wavelet thresholding
        self.apply_wavelet_threshold()

        self.create_regular_epochs(export=True)
        self.generate_reports()
```

### Example 2: Event-Related Potentials (ERP Mode)

```python
config = {
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",
            "level": 5,
            "threshold_mode": "soft",
            "is_erp": True,                # ERP-preserving mode
            "bandpass": (1.0, 30.0),       # ERP frequency band
            "threshold_scale": 1.2         # Slightly more aggressive
        }
    }
}

class MMN_WithWavelet(Task):
    def run(self):
        self.import_raw()
        self.resample_data()

        # Apply before epoching for ERPs
        self.apply_wavelet_threshold()

        self.create_eventid_epochs(export=True)
        self.generate_reports()
```

### Example 3: Heavy Artifacts (Aggressive Cleaning)

```python
config = {
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",
            "level": 6,                    # Higher level
            "threshold_mode": "hard",      # HAPPE high-artifact mode
            "is_erp": False,
            "threshold_scale": 1.5,        # More aggressive threshold
            "picks": "eeg"                 # EEG channels only
        }
    }
}
```

### Example 4: Wavelet-Only Pipeline (No ICA)

```python
config = {
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 1, "h_freq": 100}
    },
    "wavelet_threshold": {
        "enabled": True,
        "value": {
            "wavelet": "sym4",
            "level": 5,
            "threshold_mode": "soft",
            "threshold_scale": 1.0
        }
    },
    "ICA": {"enabled": False},  # Skip ICA
    "epoch_settings": {
        "enabled": True,
        "value": {"tmin": -1, "tmax": 1}
    }
}

class RestingWaveletOnly(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()

        # Wavelet replaces ICA
        self.apply_wavelet_threshold()

        self.create_regular_epochs(export=True)
        self.generate_reports()
```

## When to Use Wavelet Thresholding

### ✅ Good For:
- **Transient high-amplitude artifacts**: Eye blinks, muscle twitches, electrode pops
- **Pediatric/developmental data**: High artifact rates (HAPPE pipeline standard)
- **Quick preprocessing**: Faster than ICA, no component inspection needed
- **Complementary to ICA**: Use before or after ICA for optimal cleaning
- **Event-related data**: ERP-preserving mode maintains signal morphology

### ⚠️ Limitations:
- **Stationary artifacts**: Less effective for continuous EMG, line noise
- **Low-frequency drifts**: Use high-pass filtering first
- **Systematic artifacts**: ICA may be better for stereotyped patterns
- **Over-cleaning risk**: Too aggressive settings can distort signal

## Comparison with Other Methods

### Wavelet vs ICA

| Aspect | Wavelet Thresholding | ICA |
|--------|---------------------|-----|
| **Speed** | Fast (seconds) | Slow (minutes) |
| **Artifacts** | Transient, non-stationary | Stationary, stereotyped |
| **User input** | Minimal (parameter tuning) | Moderate (component inspection) |
| **Reproducibility** | High (deterministic) | Moderate (random initialization) |
| **ERP preservation** | Excellent (ERP mode) | Good |
| **Best for** | High-amplitude transients | Eye movements, heartbeat, line noise |

### Wavelet vs AutoReject

| Aspect | Wavelet Thresholding | AutoReject |
|--------|---------------------|------------|
| **When applied** | Continuous data | Epoched data |
| **Strategy** | Signal reconstruction | Epoch/channel rejection |
| **Data loss** | Minimal | Can be substantial |
| **Computation** | Fast | Slow (cross-validation) |
| **Best for** | Transient artifacts in continuous data | Epoch-level quality control |

## Output and Quality Control

### Generated Report

The block automatically generates a PDF report (`*_wavelet_threshold.pdf`) containing:

1. **Summary Table**
   - Channels analyzed
   - Sampling rate and duration
   - Effective decomposition level
   - Threshold parameters
   - Mean artifact reduction metrics

2. **Signal Comparison**
   - Before/after waveform comparison
   - Peak-to-peak reduction per channel
   - Top 10 channels by artifact reduction

3. **Power Spectral Density**
   - Mean PSD before/after
   - Band power changes (delta, theta, alpha, beta, gamma)

4. **Top Channels Table**
   - Detailed metrics for most affected channels
   - P2P and STD reduction percentages

### Metadata Logged

Processing metadata automatically recorded:
- Wavelet family and level used
- Threshold mode and scale
- Mean absolute difference (μV)
- Mean peak-to-peak reduction (%)
- Number of channels processed
- Report path (for later review)

## Parameter Tuning Guide

### Step 1: Start with Defaults
```python
"wavelet": "sym4", "level": 5, "threshold_scale": 1.0, "threshold_mode": "soft"
```

### Step 2: Visual Inspection
- Check the PDF report
- Look for over-cleaning (excessive smoothing)
- Look for under-cleaning (artifacts remain)

### Step 3: Adjust Threshold Scale
**If artifacts remain**: Increase `threshold_scale` (1.2, 1.5, 2.0)
**If signal looks over-smoothed**: Decrease `threshold_scale` (0.8, 0.9)

### Step 4: Consider Mode for Severe Artifacts
**If very high-amplitude artifacts**: Try `threshold_mode="hard"`

### Step 5: Level Adjustment (Rare)
Usually not needed, but:
- **Lower level** (3-4): Preserve fine temporal structure
- **Higher level** (6-7): More aggressive smoothing

## Integration with Other Steps

### Recommended Preprocessing Order

**Option A: Wavelet + ICA** (comprehensive)
```python
def run(self):
    self.import_raw()
    self.resample_data()
    self.filter_data()
    self.apply_wavelet_threshold()  # Remove transients first
    self.run_ica()                   # Then ICA for stationary artifacts
    self.create_epochs()
```

**Option B: Wavelet Only** (fast, good for developmental data)
```python
def run(self):
    self.import_raw()
    self.resample_data()
    self.filter_data()
    self.apply_wavelet_threshold()
    self.create_epochs()
```

**Option C: ICA + Wavelet** (alternative ordering)
```python
def run(self):
    self.import_raw()
    self.resample_data()
    self.filter_data()
    self.run_ica()                   # ICA first
    self.apply_wavelet_threshold()   # Clean up remaining transients
    self.create_epochs()
```

## Troubleshooting

### Issue: Signal Looks Over-Smoothed
**Solution**: Decrease `threshold_scale` (try 0.8)

### Issue: Artifacts Still Visible
**Solution**:
1. Increase `threshold_scale` (try 1.5)
2. Try `threshold_mode="hard"`
3. Ensure filtering is applied first

### Issue: ERP Morphology Changed
**Solution**: Use `is_erp=True` with appropriate `bandpass`

### Issue: Processing Too Slow
**Solution**:
- Decrease `level` (try 4 instead of 5)
- Process subset of channels with `picks`

### Issue: Report Generation Fails
**Solution**: Check that reportlab and matplotlib are installed

## References

1. **Donoho DL & Johnstone IM (1994)**. Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455. [DOI: 10.1093/biomet/81.3.425](https://doi.org/10.1093/biomet/81.3.425)

2. **Donoho DL (1995)**. De-noising by soft-thresholding. *IEEE Transactions on Information Theory*, 41(3), 613-627. [DOI: 10.1109/18.382009](https://doi.org/10.1109/18.382009)

3. **Gabard-Durnam LJ, et al. (2018)**. The Harvard Automated Processing Pipeline for Electroencephalography (HAPPE): Standardized Processing Software for Developmental and High-Artifact Data. *Frontiers in Neuroscience*, 12, 97. [HAPPE on GitHub](https://github.com/PINE-Lab/HAPPE)

4. **Mallat S (1989)**. A theory for multiresolution signal decomposition: the wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674-693.

## See Also

- [Task Library](https://docs.autocleaneeg.org/tasks/task-library) - Pre-built tasks using wavelet thresholding
- [AutoReject Block](../autoreject/) - Complementary epoch-level cleaning
- [ICA Block](../advanced_ica/) - For stationary artifact removal
- [HAPPE Pipeline](https://github.com/PINE-Lab/HAPPE) - MATLAB implementation reference

## Version History

- **1.0.0** (2025-09-29): Initial release
  - Core wavelet thresholding implementation
  - ERP-preserving mode
  - Comprehensive PDF reporting
  - Flexible parameter configuration

## License

MIT License - See repository root for details

## Contributors

- AutoCleanEEG Team
- Maintainer: Ernest Pedapati (ernest.pedapati@cchmc.org)