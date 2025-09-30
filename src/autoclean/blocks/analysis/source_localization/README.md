# Source Localization Block

**Version:** 1.0.0
**Category:** Analysis
**Status:** Stable

## Overview

The Source Localization block estimates cortical sources from sensor-space EEG data using Minimum Norm Estimation (MNE). This fundamental analysis step projects scalp-recorded electrical activity to the cortical surface, enabling region-of-interest (ROI) analyses, connectivity studies, and spatially localized power analyses.

## What It Does

Source localization solves the EEG inverse problem by:
1. Creating a forward model linking cortical sources to scalp sensors
2. Computing an inverse operator with regularization
3. Applying the inverse to sensor data to estimate source activations
4. Producing source estimates (STCs) with 10,242 cortical vertices

The block uses the **fsaverage** template brain with ico-5 source space and an identity noise covariance matrix, making it suitable for group studies and resting-state analyses without requiring individual anatomical MRI scans.

## When to Use

**Use source localization when you need:**
- Region-of-interest (ROI) power analyses
- Functional connectivity between brain regions
- Spatially localized spectral analyses
- Network analyses of brain dynamics
- Source-level event-related potentials (ERPs)

**Requirements:**
- Minimum 19 EEG channels (64+ recommended for best spatial resolution)
- Standard 10-20 montage or compatible electrode layout
- Data after artifact rejection (ICA, AutoReject)
- Adequate SNR (signal-to-noise ratio)

## How It Works

### Algorithm

```
Sensor Data (n_channels × n_times)
         ↓
Forward Solution (fsaverage template)
         ↓
Inverse Operator (identity noise covariance)
         ↓
Source Estimates (n_vertices × n_times)
    10,242 cortical vertices
```

### Technical Details

- **Method:** Minimum Norm Estimation (MNE)
- **Source Space:** fsaverage ico-5 (10,242 vertices)
- **BEM:** 3-layer (skin, skull, brain)
- **Regularization:** λ² = 1/9 (default)
- **Orientation:** Normal to cortical surface
- **Noise Covariance:** Identity matrix

## Configuration

### Basic Configuration

```python
config = {
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,  # 1/9
            "pick_ori": "normal",
            "n_jobs": 10
        }
    }
}
```

### Advanced Configuration

```python
# For higher spatial resolution (dSPM)
config = {
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "dSPM",           # Depth-weighted
            "lambda2": 0.111,
            "pick_ori": "normal",
            "n_jobs": 10
        }
    }
}

# For connectivity analyses (unconstrained orientation)
config = {
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,
            "pick_ori": None,            # Free orientation
            "n_jobs": 10
        }
    }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"MNE"` | Inverse method: `"MNE"`, `"dSPM"`, `"sLORETA"` |
| `lambda2` | float | `0.111` | Regularization parameter (1/SNR²) |
| `pick_ori` | str | `"normal"` | Source orientation: `"normal"` (constrained), `None` (free) |
| `n_jobs` | int | `10` | Number of parallel jobs for forward solution |
| `convert_to_eeg` | bool | `False` | Convert STCs to 68-channel EEG (Desikan-Killiany ROIs) |

### EEG Conversion Feature

Optionally convert source estimates (10,242 vertices) to standardized 68-channel EEG format using Desikan-Killiany atlas ROIs. This enables:
- Analysis with standard EEG software (EEGLAB, FieldTrip, etc.)
- BIDS-compatible derivatives
- Simplified ROI-based analyses
- Compatibility with existing EEG pipelines

```python
config = {
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,
            "convert_to_eeg": True  # Enable conversion
        }
    }
}
```

**Outputs when `convert_to_eeg=True`:**
- `{subject}_dk_regions.set` - EEGLAB file with 68 ROI time courses
- `{subject}_dk_montage.fif` - MNE montage with ROI centroid positions
- `{subject}_region_info.csv` - ROI metadata (names, hemispheres, coordinates)

## Usage in Tasks

### Resting-State Example

```python
from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,
            "n_jobs": 10
        }
    }
}

class RestingStateSource(Task):
    def run(self):
        # Preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.rereference_data()
        self.run_ica()

        # Create epochs for source analysis
        self.create_regular_epochs()

        # Apply source localization
        stc_list = self.apply_source_localization()

        # stc_list is now available for downstream analyses
        # Each STC: (10242 vertices × time_points)
```

### Event-Related Example

```python
class ERPSource(Task):
    def run(self):
        # Preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.rereference_data()

        # Epoch around events
        self.epoch_data()

        # Source localization on epochs
        stc_list = self.apply_source_localization()

        # Average across epochs for ERP
        if stc_list:
            stc_avg = sum(stc_list) / len(stc_list)
```

## Outputs

### Source Estimate Objects (STCs)

For **Raw** input:
- `self.stc`: Single SourceEstimate object
- Shape: (n_vertices, n_times)
- Saved to: `derivatives/source_localization/`

For **Epochs** input:
- `self.stc_list`: List of SourceEstimate objects
- Length: n_epochs
- Each STC shape: (n_vertices, n_times_per_epoch)
- First 3 epochs saved as examples

### STC Properties

```python
stc.data              # (10242, n_times) array of source activations
stc.vertices          # [lh_vertices, rh_vertices] vertex indices
stc.times             # Time vector
stc.sfreq             # Sampling frequency
stc.tmin              # Start time
```

### Metadata

Stored in run database:
- `method`: Inverse method used
- `lambda2`: Regularization parameter
- `n_vertices`: Number of source vertices
- `sfreq`: Sampling frequency
- `duration_sec` or `n_epochs`: Data duration/count

## Downstream Analyses

Source estimates enable:

1. **ROI Power Analysis** (Source PSD block)
   ```python
   self.apply_source_localization()
   self.apply_source_psd()
   ```

2. **Functional Connectivity** (Source Connectivity block)
   ```python
   self.apply_source_localization()
   self.apply_source_connectivity()
   ```

3. **Custom ROI Extraction**
   ```python
   stc_list = self.apply_source_localization()
   labels = mne.read_labels_from_annot('fsaverage', 'aparc')
   roi_tc = stc_list[0].extract_label_time_course(labels[0], mode='mean')
   ```

## Performance

**Computational Cost:**
- Forward solution: ~30-60 seconds (first time)
- Inverse operator: ~10-20 seconds
- Apply inverse (Raw): ~30-90 seconds per minute of data
- Apply inverse (Epochs): ~1-3 seconds per epoch

**Memory Requirements:**
- Forward solution: ~500 MB
- Source estimates (Raw, 5 min): ~2 GB
- Source estimates (Epochs, 100): ~500 MB

**Optimization Tips:**
- Increase `n_jobs` for parallel processing (default: 10)
- Process shorter segments for memory-constrained systems
- Cache fsaverage data locally (auto-downloaded first run)

## Limitations

1. **Spatial Resolution**
   - Limited by number of electrodes (64+ recommended)
   - Deep sources less accurately localized than surface sources
   - Cannot distinguish nearby sources (point-spread function)

2. **Inverse Problem**
   - Non-unique solution (regularization required)
   - Assumptions: sources normal to cortex, distributed
   - Identity noise covariance may not be optimal for all cases

3. **Template Brain**
   - fsaverage is an average brain (may not fit individual anatomy)
   - No individual head modeling or MRI co-registration
   - Best for group studies and relative comparisons

4. **Data Requirements**
   - Requires adequate SNR after preprocessing
   - Montage information must be accurate
   - Reference must be consistent

## Troubleshooting

### "fsaverage data not found"
**Solution:** MNE will automatically download fsaverage (~200 MB) on first run. Ensure internet connection.

### "Forward solution failed"
**Solution:** Check montage is set correctly. Use `self.set_montage()` before source localization.

### High memory usage
**Solution:** Reduce data length, increase swap space, or process in segments.

### Unrealistic source patterns
**Solution:**
- Verify preprocessing quality (ICA, artifact rejection)
- Check electrode locations are accurate
- Consider adjusting lambda2 (lower = less smoothing, higher = more)

## Scientific References

1. **Minimum Norm Estimation**
   - Hämäläinen MS & Ilmoniemi RJ (1994). Interpreting magnetic fields of the brain: minimum norm estimates. *Medical & Biological Engineering & Computing*, 32(1), 35-42.

2. **MNE-Python Software**
   - Gramfort A, et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267.

3. **fsaverage Template**
   - Fischl B, et al. (1999). High-resolution intersubject averaging and a coordinate system for the cortical surface. *Human Brain Mapping*, 8(4), 272-284.

4. **Inverse Methods Comparison**
   - Grech R, et al. (2008). Review on solving the inverse problem in EEG source analysis. *Journal of NeuroEngineering and Rehabilitation*, 5(1), 25.

5. **Desikan-Killiany Atlas**
   - Desikan RS, et al. (2006). An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest. *NeuroImage*, 31(3), 968-980.

## Version History

- **1.0.0** (2025-09-29): Initial release
  - MNE source localization for Raw and Epochs
  - fsaverage template brain
  - Identity noise covariance
  - Automatic saving and metadata tracking