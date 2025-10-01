# Zapline DSS-Based Line Noise Removal Block

**Category**: Signal Processing
**Version**: 1.0.0
**Status**: Stable

## Overview

Zapline is a sophisticated line noise removal method that uses Denoising Source Separation (DSS) to remove power line artifacts (50 or 60 Hz) and their harmonics from EEG/MEG data. Unlike traditional notch filters that remove specific frequency bands, Zapline exploits the spatial structure of line noise across channels to achieve better signal preservation while effectively removing artifacts.

This implementation is based on the MEEGkit Python toolbox and provides both single-pass and iterative removal modes.

## Scientific Background

### Power Line Artifacts

Power line artifacts are among the most common and problematic noise sources in EEG/MEG recordings:

- **Source**: Electrical power systems (50 Hz in Europe/Asia, 60 Hz in Americas)
- **Characteristics**: Narrow-band oscillations with harmonics (120 Hz, 180 Hz, etc.)
- **Impact**: Can obscure neural signals, especially in frequency domain analyses
- **Spatial pattern**: Typically consistent across channels (environmental contamination)

### Denoising Source Separation (DSS)

DSS is a powerful method that combines spectral and spatial information:

```
Multi-channel EEG → DSS → Noise Components + Clean Components
                     ↓
            Remove noise components
                     ↓
              Reconstruct signal
```

**Key principle**: Line noise has:
1. **Spectral structure**: Concentrated at line frequency
2. **Spatial consistency**: Similar pattern across channels

DSS finds spatial filters that maximize power at the line frequency, then removes those components.

### Why Zapline is Better than Notch Filtering

**Traditional Notch Filter**:
- Removes frequency band regardless of whether it contains signal or noise
- Cannot distinguish neural oscillations from line noise
- May distort signal in adjacent frequencies

**Zapline DSS Approach**:
- Removes only spatially-consistent noise at line frequency
- Preserves neural activity at the same frequency
- Better signal-to-noise ratio improvement
- Adaptive to noise characteristics

### DSS Algorithm Steps

1. **Fourier transform**: Decompose data into frequency components
2. **Narrow-band extraction**: Extract signal around line frequency (e.g., 58-62 Hz)
3. **DSS**: Find spatial components that maximize narrow-band power
4. **Component removal**: Remove top N noise components (typically 1)
5. **Reconstruction**: Project cleaned data back to sensor space

### Iterative Mode

For severe line noise contamination, iterative mode (`use_iter=True`) repeats the process:
- Automatically detects when noise is sufficiently reduced
- Typically converges in 1-3 iterations
- More thorough but slightly slower

## Configuration

### Basic Configuration

```python
config = {
    "apply_zapline": {
        "enabled": True,
        "value": {
            "fline": 60,           # Line frequency (Hz)
            "nkeep": 1,            # Number of components to remove
            "use_iter": False,     # Iterative refinement
            "max_iter": 10         # Max iterations (if use_iter=True)
        }
    }
}
```

### Parameter Guide

#### `fline` (float)
**Options**: `50` or `60` (Hz)
**Default**: `60`
**Recommendation**:
- **60 Hz**: United States, Canada, Central/South America, parts of Asia
- **50 Hz**: Europe, Africa, most of Asia, Australia

```python
# US/Americas
"fline": 60

# Europe/Asia
"fline": 50
```

**Important**: Must be below Nyquist frequency (sampling_rate / 2)

#### `nkeep` (integer)
**Range**: 1-5
**Default**: `1`
**Recommendation**:
- **1**: Remove strongest line noise component (typical)
- **2**: Remove main component + first harmonic
- **3+**: Severe contamination with multiple harmonics

```python
# Standard (remove primary 60 Hz)
"nkeep": 1

# Also remove harmonics (60 Hz, 120 Hz)
"nkeep": 2
```

**Warning**: Higher values may remove more signal. Start with 1 and increase only if needed.

#### `use_iter` (boolean)
**Default**: `False`
**Recommendation**:
- **False**: Single-pass removal, faster, usually sufficient
- **True**: Iterative refinement, more thorough, use for severe contamination

```python
# Standard (fast, recommended)
"use_iter": False

# Severe contamination
"use_iter": True
```

**Iterative mode**:
- Automatically determines convergence
- Typically completes in 1-3 iterations
- Stops when noise reduction plateaus

#### `max_iter` (integer)
**Range**: 1-50
**Default**: `10`
**Only used when**: `use_iter=True`

```python
# Standard
"max_iter": 10

# Conservative (stop early)
"max_iter": 5

# Aggressive (allow more iterations)
"max_iter": 20
```

## Usage Examples

### Example 1: Basic 60 Hz Removal (US)

```python
from autoclean.core.task import Task

config = {
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 1, "h_freq": 100}
    },
    "apply_zapline": {
        "enabled": True,
        "value": {
            "fline": 60,
            "nkeep": 1,
            "use_iter": False
        }
    }
}

class RestingWithZapline(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()

        # Remove line noise before ICA
        self.apply_zapline()

        self.run_ica()
        self.create_regular_epochs(export=True)
        self.generate_reports()
```

### Example 2: 50 Hz Removal (Europe)

```python
config = {
    "apply_zapline": {
        "enabled": True,
        "value": {
            "fline": 50,       # Europe/Asia
            "nkeep": 1,
            "use_iter": False
        }
    }
}

class EuropeanData(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.apply_zapline()
        self.run_ica()
        self.create_epochs()
```

### Example 3: Severe Contamination (Iterative Mode)

```python
config = {
    "apply_zapline": {
        "enabled": True,
        "value": {
            "fline": 60,
            "nkeep": 2,           # Remove main + first harmonic
            "use_iter": True,     # Iterative refinement
            "max_iter": 10
        }
    }
}

class SevereLineNoise(Task):
    def run(self):
        self.import_raw()
        self.resample_data()

        # Apply aggressive Zapline before filtering
        self.apply_zapline()

        self.filter_data()
        self.run_ica()
        self.create_epochs()
```

### Example 4: Zapline-Only Pipeline (No Notch)

```python
config = {
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1,
            "h_freq": 100,
            "notch_freqs": None  # No notch filter - using Zapline
        }
    },
    "apply_zapline": {
        "enabled": True,
        "value": {"fline": 60, "nkeep": 1}
    },
    "ICA": {"enabled": True}
}

class ZaplineNoNotch(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()      # No notch
        self.apply_zapline()     # DSS-based line removal
        self.run_ica()
        self.create_epochs()
```

### Example 5: Minimal Preprocessing (Zapline + Basic Filter)

```python
config = {
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {"l_freq": 1, "h_freq": 45}  # Low-pass to avoid aliasing
    },
    "apply_zapline": {
        "enabled": True,
        "value": {"fline": 60, "nkeep": 1}
    },
    "ICA": {"enabled": False}  # Skip ICA
}

class MinimalZapline(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.apply_zapline()
        self.create_regular_epochs(export=True)
```

## When to Use Zapline

### ✅ Excellent For:
- **Multi-channel data**: Works best with >32 channels (exploits spatial structure)
- **High line noise**: Alternative to aggressive notch filtering
- **Frequency domain analysis**: Spectral analysis, connectivity, FOOOF
- **Preserved signal**: When neural oscillations overlap with line frequency
- **Research settings**: Controlled environments with consistent line noise
- **Alternative to notch**: Better signal preservation than traditional filters

### ⚠️ Limitations:
- **Few channels**: Requires ≥2 channels, works best with 32+
- **Non-stationary noise**: Assumes spatial consistency of line noise
- **Variable contamination**: May struggle if noise pattern changes over time
- **Computational cost**: Slower than notch filtering (but still fast)

### ⚙️ Technical Requirements:
- **Minimum**: 2 channels
- **Recommended**: 32+ channels
- **Sampling rate**: Line frequency must be < Nyquist (sfreq/2)
- **Dependencies**: meegkit>=0.1.9

## Comparison with Other Methods

### Zapline vs Notch Filter

| Aspect | Zapline DSS | Notch Filter |
|--------|------------|--------------|
| **Method** | Spatial filtering | Frequency filtering |
| **Signal preservation** | Excellent (removes only noise) | Moderate (removes all activity at freq) |
| **Speed** | Moderate (seconds) | Fast (instant) |
| **Channel requirement** | Multi-channel (32+ best) | Single channel OK |
| **Harmonics** | Removes automatically (nkeep>1) | Requires separate notch per harmonic |
| **Neural oscillations** | Preserves if not spatially consistent | Removes regardless of source |
| **Best for** | Research, spectral analysis | Quick preprocessing, single channel |

### Zapline vs ICA

| Aspect | Zapline DSS | ICA |
|--------|------------|-----|
| **Target** | Line noise only | Multiple artifact types |
| **Speed** | Fast (seconds) | Slow (minutes) |
| **User input** | Minimal (just line freq) | Moderate (component inspection) |
| **Reproducibility** | High (deterministic) | Moderate (random init) |
| **Data requirement** | Multi-channel | Multi-channel |
| **Best use** | Line noise removal | General artifact cleaning |

**Recommendation**: Use Zapline for line noise, then ICA for other artifacts (eye movements, muscle, heartbeat).

## Pipeline Integration

### Recommended Processing Order

**Option A: Zapline → ICA** (recommended)
```python
def run(self):
    self.import_raw()
    self.resample_data()
    self.filter_data()         # Basic bandpass (no notch)
    self.apply_zapline()        # Remove line noise first
    self.run_ica()              # Then other artifacts
    self.create_epochs()
```

**Why this order?**
- Line noise can interfere with ICA decomposition
- Removing line noise first improves ICA component separation
- Cleaner ICA components are easier to identify

**Option B: Filter → Zapline → Epochs** (no ICA)
```python
def run(self):
    self.import_raw()
    self.resample_data()
    self.filter_data()
    self.apply_zapline()
    self.create_epochs()        # Skip ICA
```

**When to use?**
- High-quality data with minimal artifacts
- Fast preprocessing pipeline
- Resting-state with good data quality

**Option C: Zapline + Wavelet** (comprehensive)
```python
def run(self):
    self.import_raw()
    self.resample_data()
    self.filter_data()
    self.apply_zapline()           # Stationary line noise
    self.apply_wavelet_threshold() # Transient artifacts
    self.create_epochs()
```

**When to use?**
- Pediatric/developmental data
- High artifact rates
- Alternative to ICA

## Output and Quality Metrics

### Console Output

The block provides detailed logging:

```
[INFO] Pre-Zapline: 65.3 dB at 60 Hz (SNR: 3.42)
[HEADER] Applying Zapline (fline=60 Hz, nkeep=1, iter=False)...
[INFO] Post-Zapline: 45.1 dB at 60 Hz (reduction: 20.2 dB, SNR: 0.98)
[SUCCESS] Zapline complete (1 iterations)
```

### Quality Metrics Logged

Automatically tracked metadata:
- **Power before/after**: dB at line frequency
- **Reduction**: dB decrease at line frequency
- **SNR**: Line freq power / background power
- **Method**: `dss_line` or `dss_line_iter`
- **Iterations**: Number of iterations completed
- **Parameters**: fline, nkeep, use_iter, max_iter

### Expected Results

**Good line noise removal**:
- ≥10 dB reduction at line frequency
- SNR reduced from >2 to <1
- Visual inspection: Line noise peak reduced in PSD

**Insufficient removal** (<10 dB):
- Try increasing `nkeep` (2 or 3)
- Enable iterative mode (`use_iter=True`)
- Check data quality (is noise spatially consistent?)

## Parameter Tuning Guide

### Step 1: Start with Defaults
```python
"fline": 60,  # or 50 for Europe
"nkeep": 1,
"use_iter": False
```

### Step 2: Check Reduction Metric
Look at console output:
- **≥20 dB**: Excellent removal
- **10-20 dB**: Good removal
- **<10 dB**: Needs adjustment

### Step 3: If Reduction is Low (<10 dB)

**Try iterative mode**:
```python
"use_iter": True
```

**Or increase components**:
```python
"nkeep": 2  # Remove main + harmonic
```

### Step 4: Visual Inspection

Plot power spectrum before/after:
```python
import matplotlib.pyplot as plt
from scipy import signal

# Before Zapline
freqs, psd_before = signal.welch(raw_before.get_data(), fs=sfreq)

# After Zapline
freqs, psd_after = signal.welch(raw_after.get_data(), fs=sfreq)

plt.semilogy(freqs, psd_before.mean(axis=0), label='Before')
plt.semilogy(freqs, psd_after.mean(axis=0), label='After')
plt.axvline(60, color='r', linestyle='--', label='Line freq')
plt.legend()
plt.show()
```

Look for:
- Sharp peak at line frequency reduced
- No distortion in adjacent frequencies
- Overall spectrum shape preserved

## Channel Count Considerations

### Optimal Performance

| Channels | Expected Performance | Recommendation |
|----------|---------------------|----------------|
| **1 channel** | Won't work | Use notch filter instead |
| **2-8 channels** | Poor | Use notch filter instead |
| **8-32 channels** | Moderate | Works, but notch may be better |
| **32-64 channels** | Good | Zapline recommended |
| **64-128+ channels** | Excellent | Zapline strongly recommended |

**Why channel count matters**:
- DSS exploits spatial consistency across channels
- More channels → better noise subspace estimation
- Few channels → insufficient spatial information

## Troubleshooting

### Issue: "Zapline requires at least 2 channels"
**Cause**: Single-channel data
**Solution**: Use notch filter instead of Zapline

### Issue: Low Noise Reduction (<10 dB)
**Possible causes**:
1. Few channels (<32) → Insufficient spatial information
2. Non-stationary noise → DSS assumes consistency
3. Single component insufficient → Try `nkeep=2`

**Solutions**:
```python
# Try iterative mode
"use_iter": True

# Or increase components
"nkeep": 2

# Or use notch filter instead
```

### Issue: "Line frequency must be below Nyquist"
**Cause**: Line frequency ≥ sampling_rate / 2
**Solution**:
```python
# Check sampling rate
print(f"Nyquist: {raw.info['sfreq'] / 2} Hz")

# If sfreq=100, Nyquist=50, cannot use fline=60
# Resample to higher rate first
self.resample_data()  # e.g., to 250 Hz
```

### Issue: Warning "works best with >32 channels"
**Cause**: 2-32 channels
**Not an error**: Zapline will run, but performance may be limited
**Options**:
1. Continue with Zapline (may still help)
2. Use notch filter instead
3. Combine: Light Zapline + notch filter

### Issue: Signal Looks Distorted
**Rare, but possible**:
- `nkeep` too high (removing signal components)
- Data issues (very non-stationary noise)

**Solution**:
```python
# Reduce to single component
"nkeep": 1

# Disable iterative mode
"use_iter": False

# Or revert to notch filter
```

### Issue: Processing Very Slow
**Causes**:
- Iterative mode with many iterations
- Very high sampling rate
- Many channels

**Solutions**:
```python
# Disable iterative mode
"use_iter": False

# Reduce max iterations
"max_iter": 5

# Resample to moderate rate
"resample_step": {"enabled": True, "value": 250}
```

## Advanced Usage

### Combining with Notch Filter

For very severe line noise, use both:

```python
config = {
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1,
            "h_freq": 100,
            "notch_freqs": [60, 120]  # Light notch
        }
    },
    "apply_zapline": {
        "enabled": True,
        "value": {
            "fline": 60,
            "nkeep": 1,
            "use_iter": True  # Aggressive Zapline
        }
    }
}
```

### Custom Line Frequency

If your line frequency is unusual (not 50 or 60):
```python
"fline": 62.5  # Custom frequency
```

**Warning**: Will trigger warning message but will work.

### Removing Multiple Harmonics

```python
# Remove 60 Hz and 120 Hz
"fline": 60,
"nkeep": 2  # Removes 2 components (usually captures harmonics)
```

**Note**: You don't specify harmonic frequencies explicitly. DSS finds the strongest components in the narrow-band around `fline`.

## Performance Benchmarks

**Typical processing time** (64-channel EEG, 10 minutes recording, 250 Hz):
- **Single-pass mode**: 2-5 seconds
- **Iterative mode** (3 iterations): 6-15 seconds

**Memory usage**: 1-2 GB for typical datasets

**Scales with**:
- Number of channels (linear)
- Recording duration (linear)
- Iterations (linear)

## References

1. **de Cheveigné, A. (2020)**. ZapLine: A simple and effective method to remove power line artifacts. *NeuroImage*, 207, 116356.
   [DOI: 10.1016/j.neuroimage.2019.116356](https://doi.org/10.1016/j.neuroimage.2019.116356)
   - Original ZapLine paper describing the method
   - Comparison with notch filtering
   - Validation on real data

2. **de Cheveigné, A., & Simon, J. Z. (2008)**. Denoising based on spatial filtering. *Journal of Neuroscience Methods*, 171(2), 331-339.
   [DOI: 10.1016/j.jneumeth.2008.03.015](https://doi.org/10.1016/j.jneumeth.2008.03.015)
   - Original DSS method paper
   - Theoretical foundation
   - Applications to M/EEG

3. **MEEGkit Python Toolbox**
   [GitHub: python-meegkit](https://github.com/nbara/python-meegkit)
   - Python implementation of DSS and ZapLine
   - Used by this block

## See Also

- [Wavelet Threshold Block](../wavelet_threshold/) - For transient artifacts
- [AutoReject Block](../autoreject/) - For epoch-level quality control
- [ICA Blocks](../../analysis/) - For other artifact types
- [Task Library](https://docs.autocleaneeg.org/tasks/) - Pre-built tasks with Zapline

## Version History

- **1.0.0** (2025-10-01): Initial release
  - Core Zapline DSS implementation
  - Single-pass and iterative modes
  - Automatic quality metrics
  - Comprehensive parameter validation

## Installation

Zapline requires the `meegkit` package:

```bash
pip install meegkit>=0.1.9
```

Or with full AutoCleanEEG installation:
```bash
pip install autocleaneeg-pipeline
```

## License

MIT License - See repository root for details

## Contributors

- AutoCleanEEG Team
- Maintainer: Ernest Pedapati (ernest.pedapati@cchmc.org)

## Citation

If you use Zapline in your research, please cite:

```bibtex
@article{deCheveigné2020zapline,
  title={ZapLine: A simple and effective method to remove power line artifacts},
  author={de Cheveigné, Alain},
  journal={NeuroImage},
  volume={207},
  pages={116356},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.neuroimage.2019.116356}
}
```
