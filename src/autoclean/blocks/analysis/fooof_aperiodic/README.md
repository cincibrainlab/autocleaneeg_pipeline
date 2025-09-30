# FOOOF Aperiodic Analysis Block

Vertex-level spectral parameterization for extracting aperiodic (1/f background) parameters from source-localized EEG data using the FOOOF algorithm.

## Overview

This block implements FOOOF (Fitting Oscillations & One Over F) analysis to decompose neural power spectra into aperiodic components. It operates at the vertex level (20,484 cortical surface vertices) to provide high spatial resolution characterization of the 1/f background.

## Scientific Background

**Reference**: Donoghue T, et al. (2020). Parameterizing neural power spectra into periodic and aperiodic components. *Nature Neuroscience*, 23(12), 1655-1665.

The aperiodic component reflects the 1/f background of neural power spectra, which is thought to index the balance of excitation and inhibition in neural populations:

- **Offset**: Overall power level (intercept)
- **Exponent**: Slope of 1/f decay (spectral tilt)
  - Lower exponent → more excitation
  - Higher exponent → more inhibition
- **Knee** (optional): Bend point in spectrum (reflects temporal integration timescale)

## Algorithm

### Step 1: Vertex-Level PSD Calculation
- Uses Welch's method with 4-second windows and 50% overlap
- Batch processing (4000 vertices/batch) for memory efficiency
- Configurable frequency range (default: 1-45 Hz)

### Step 2: FOOOF Model Fitting
- Fits FOOOF model to each vertex independently
- Two modes: 'fixed' (linear 1/f) or 'knee' (bent 1/f)
- Robust error handling with automatic fallback parameters
- Parallel batch processing (2000 vertices/batch)
- Success rate tracking and validation

## Parameters

```python
"apply_fooof_aperiodic": {
    "enabled": True,
    "value": {
        "fmin": 1.0,              # Minimum frequency (Hz)
        "fmax": 45.0,             # Maximum frequency (Hz)
        "n_jobs": 10,             # Parallel jobs
        "aperiodic_mode": "knee"  # 'fixed' or 'knee'
    }
}
```

## Inputs

- **Required**: Source estimates from source localization (self.stc)
- **Data type**: Continuous Raw data (not epochs)
- **Minimum duration**: 60 seconds recommended for stable estimates

## Outputs

### Files Created
```
derivatives/fooof/
  ├── {subject}_psd-stc.h5                    # Vertex-level PSD
  ├── {subject}_fooof_aperiodic.parquet       # Aperiodic parameters
  └── {subject}_fooof_aperiodic.csv           # Same data in CSV
```

### DataFrame Columns
- `subject`: Subject identifier
- `vertex`: Vertex index (0-20483)
- `offset`: Overall power level
- `knee`: Bend point (NaN if mode='fixed')
- `exponent`: Slope of 1/f decay
- `r_squared`: Model fit quality (aim for >0.9)
- `error`: Fitting error
- `status`: SUCCESS, FITTING_FAILED, NAN_PARAMS, INVALID_PARAMS, INVALID_EXPONENT

## Usage Recommendations

### When to Use

✅ **Good Use Cases**:
- Resting-state EEG with broadband spectrum (1-45 Hz)
- Studying developmental changes in E/I balance
- Comparing aperiodic vs periodic contributions
- Longitudinal studies (medications, aging, disease)

❌ **Poor Use Cases**:
- Narrow-band filtered data
- Very short recordings (<30 seconds)
- Data with extreme artifacts/noise
- Task-based EEG with rapid power changes

### Mode Selection

**'knee' mode** (default):
- Use for: Broadband recordings (0.5-45 Hz or wider)
- Captures: Bend in 1/f curve
- Best for: Resting state, eyes open/closed, long epochs
- Requires: 60+ seconds of data

**'fixed' mode**:
- Use for: Narrow frequency ranges (<1 octave)
- Captures: Simple linear 1/f slope
- Best for: Targeted band analysis, noisy data, shorter recordings
- More stable: Higher success rates

## Validation

Check these metrics in output:
- **Success rate**: Aim for >80% successful fits
- **R² distribution**: Should be >0.9 for good data
- **Exponent range**: 0.5-2.5 is typical for EEG
- **Knee range** (if knee mode): 0-20 Hz is typical

## Example

```python
from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
    # ... preprocessing steps ...
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,
            "pick_ori": "normal",
            "n_jobs": 10
        }
    },
    "apply_fooof_aperiodic": {
        "enabled": True,
        "value": {
            "fmin": 1.0,
            "fmax": 45.0,
            "n_jobs": 4,
            "aperiodic_mode": "knee"
        }
    }
}

class MyTask(Task):
    def run(self):
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.crop_duration()
        self.rereference_data()
        self.apply_source_localization()
        self.apply_fooof_aperiodic()
```

## Dependencies

- `fooof` or `specparam`: Spectral parameterization library
- MNE-Python >= 1.10.1
- numpy, scipy, pandas, h5py

**Installation**:
```bash
pip install fooof
# OR (newer version)
pip install specparam
```

## Performance

- **Computation time**: ~3-5 minutes for 20,484 vertices (60s data)
- **Memory usage**: ~2-4 GB RAM peak
- **Parallelization**: Scales well with n_jobs (4-10 recommended)

## References

1. Donoghue T, et al. (2020). Parameterizing neural power spectra. *Nature Neuroscience*, 23(12), 1655-1665.
2. https://fooof-tools.github.io/fooof/
3. Voytek B, Knight RT (2015). Dynamic network communication as a unifying neural basis for cognition, development, aging, and disease. *Biol Psychiatry*, 77(12), 1089-1097.