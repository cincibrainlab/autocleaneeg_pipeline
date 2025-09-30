# FOOOF Periodic Analysis Block

Vertex-level extraction of oscillatory peak parameters from source-localized EEG data using the FOOOF algorithm.

## Overview

This block implements FOOOF (Fitting Oscillations & One Over F) analysis to extract periodic (oscillatory) components from neural power spectra. It identifies dominant oscillatory peaks in specified frequency bands across 20,484 cortical surface vertices.

## Scientific Background

**Reference**: Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655-1665.

The periodic component reflects synchronized neural oscillations. For each frequency band, FOOOF identifies:

- **Center frequency**: Peak location in Hz (e.g., 10 Hz for alpha)
- **Power**: Peak amplitude (after removing aperiodic component)
- **Bandwidth**: Peak width in Hz (relates to rhythmicity)
  - Narrow bandwidth (<2 Hz) → more rhythmic oscillation
  - Wide bandwidth (>4 Hz) → less rhythmic, may be noise

## Algorithm

### Input
- Uses pre-computed vertex-level PSD from `apply_fooof_aperiodic()`
- OR accepts standalone PSD SourceEstimate

### Processing
1. Fits FOOOF model to each vertex spectrum
2. Identifies dominant peak in each frequency band
3. Extracts peak parameters (frequency, power, bandwidth)
4. Returns NaN for bands with no detected peaks

### Peak Selection
- Uses `get_band_peak_fm()` from FOOOF
- Selects highest peak if multiple peaks in band
- Validates peak meets threshold criteria

## Parameters

```python
"apply_fooof_periodic": {
    "enabled": True,
    "value": {
        "freq_bands": {              # Optional, defaults to standard bands
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45)
        },
        "n_jobs": 10,                # Parallel jobs
        "aperiodic_mode": "knee"     # 'fixed' or 'knee'
    }
}
```

### Default Frequency Bands
- **Delta**: 1-4 Hz (slow wave sleep)
- **Theta**: 4-8 Hz (memory, navigation)
- **Alpha**: 8-13 Hz (relaxed wakefulness)
- **Beta**: 13-30 Hz (active thinking, motor)
- **Gamma**: 30-45 Hz (sensory processing, binding)

### Custom Bands
```python
"freq_bands": {
    "slow_alpha": (8, 10),
    "fast_alpha": (10, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "low_gamma": (30, 45),
    "high_gamma": (45, 80)
}
```

## Inputs

- **Preferred**: Uses `self.stc_psd` from `apply_fooof_aperiodic()`
- **Alternative**: Accepts standalone PSD SourceEstimate
- **Data requirement**: Vertex-level PSD with frequencies as timepoints

## Outputs

### Files Created
```
derivatives/fooof/
  ├── {subject}_fooof_periodic.parquet        # Periodic parameters
  └── {subject}_fooof_periodic.csv            # Same data in CSV
```

### DataFrame Columns
- `subject`: Subject identifier
- `vertex`: Vertex index (0-20483)
- `band`: Frequency band name
- `center_frequency`: Peak location in Hz (NaN if no peak)
- `power`: Peak amplitude (NaN if no peak)
- `bandwidth`: Peak width in Hz (NaN if no peak)

## Usage Recommendations

### When to Use

✅ **Good Use Cases**:
- Identifying individual alpha frequency (IAF)
- Comparing oscillatory power across regions
- Peak frequency shifts with development/aging
- Task-related changes in oscillatory activity

### Interpretation Guidelines

**Center Frequency**:
- Within-band variation is meaningful (e.g., 9 Hz vs 11 Hz alpha)
- Slower frequencies often in frontal regions
- Faster frequencies often in posterior regions

**Power**:
- Reflects oscillation strength *after* removing 1/f background
- More interpretable than raw power
- Can compare across bands fairly

**Bandwidth**:
- <2 Hz: Strong, rhythmic oscillation
- 2-4 Hz: Moderate rhythmicity
- >4 Hz: Weak rhythmicity, may be noise
- NaN: No peak detected in band

### Quality Control

Check for:
- **Reasonable frequencies**: Peaks should be near band centers
- **Narrow bandwidths**: <4 Hz is typical for real oscillations
- **Spatial consistency**: Adjacent vertices should have similar peaks
- **Missing peaks**: NaN is OK, means no oscillation in that band

## Pipeline Integration

### After Aperiodic Analysis (Recommended)
```python
class MyTask(Task):
    def run(self):
        # Preprocessing...
        self.apply_source_localization()
        self.apply_fooof_aperiodic()   # Creates self.stc_psd
        self.apply_fooof_periodic()    # Uses self.stc_psd
```

### Standalone (Advanced)
```python
class MyTask(Task):
    def run(self):
        # Preprocessing...
        self.apply_source_localization()

        # Manual PSD calculation
        from autoclean.calc.fooof_analysis import calculate_vertex_psd_for_fooof
        stc_psd, _ = calculate_vertex_psd_for_fooof(
            self.stc, fmin=1.0, fmax=45.0, n_jobs=10
        )

        # Periodic analysis
        self.apply_fooof_periodic(stc_psd=stc_psd)
```

## Example Analysis

### Find Individual Alpha Frequency
```python
import pandas as pd

# Load periodic results
df = pd.read_parquet('subject_fooof_periodic.parquet')

# Get alpha peaks
alpha_df = df[df['band'] == 'alpha']

# Calculate mean IAF across all vertices
mean_iaf = alpha_df['center_frequency'].mean()
print(f"Individual Alpha Frequency: {mean_iaf:.2f} Hz")

# Find occipital vertices with strongest alpha
occipital_vertices = range(5000, 7000)  # Example range
occipital_alpha = alpha_df[alpha_df['vertex'].isin(occipital_vertices)]
strongest_alpha = occipital_alpha.nlargest(10, 'power')
```

## Dependencies

- `fooof` or `specparam`: Spectral parameterization library
- MNE-Python >= 1.10.1
- numpy, scipy, pandas

**Installation**:
```bash
pip install fooof
# OR (newer version)
pip install specparam
```

## Performance

- **Computation time**: ~4-6 minutes for 20,484 vertices × 5 bands
- **Memory usage**: ~2-4 GB RAM peak
- **Parallelization**: Scales well with n_jobs (4-10 recommended)

## Validation

Typical values for good data:
- **Peak detection**: 60-80% of vertices should have peaks in primary bands
- **Alpha frequency**: 8-13 Hz, typically ~10 Hz
- **Beta frequency**: 15-25 Hz typical
- **Gamma frequency**: 30-45 Hz if detected

## References

1. Donoghue T, et al. (2020). Parameterizing neural power spectra. *Nature Neuroscience*, 23(12), 1655-1665.
2. Klimesch W (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. *Brain Res Rev*, 29(2-3), 169-195.
3. Haegens S, et al. (2014). Inter- and intra-individual variability in alpha peak frequency. *NeuroImage*, 92, 46-55.