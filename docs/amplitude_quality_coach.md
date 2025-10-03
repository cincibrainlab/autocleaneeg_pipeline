# Amplitude Quality Coach

## Overview

The Amplitude Quality Coach is an interpretable diagnostic system that transforms raw numeric threshold flags into meaningful, actionable feedback about EEG data quality. Instead of silently flagging epochs, it provides human-readable analysis with concrete suggestions for improvement.

## Problem Solved

Traditional amplitude-based rejection is opaque:
- Epochs are flagged/dropped with minimal explanation
- Users don't know if thresholds are appropriate
- No guidance on whether to adjust thresholds or fix channels
- Difficult to distinguish systematic channel issues from transient artifacts
- No empirical basis for choosing threshold values

## Solution: `summarize_amplitude_quality()` Method

An automatic coaching system that:
- Computes peak-to-peak amplitudes per channel per epoch
- Compares amplitudes to configured thresholds
- Identifies problematic patterns (strict thresholds, bad channels, etc.)
- Provides specific, actionable recommendations
- Displays results in clear, structured format

### Key Features

#### 1. **Always-On Analysis with Preview-Then-Validate**
Runs automatically in two stages during epoch creation:
- **Preview Mode (Before Rejection)**: Shows impact of thresholds on all epochs with actionable coaching
- **Validation Mode (After Rejection)**: Confirms final data quality with descriptive statistics

#### 2. **Per-Channel-Type Statistics**
```
EEG Channels:
--------------------------------------------------------------------------------
  Average mean amplitude: 0.000085 V
  Maximum mean amplitude: 0.000320 V
  Configured threshold:   0.000200 V
```

#### 3. **Problem Detection**
Identifies channels exceeding thresholds in >20% of epochs:
```
  ⚠️  Channels exceeding threshold in >20% of epochs:
    • E46: 28.3% of epochs (mean: 0.000245 V)
    • E68: 24.7% of epochs (mean: 0.000230 V)
```

#### 4. **Actionable Suggestions**
Provides context-specific recommendations:

**Many channels flagged (>30%):**
```
  💡 Suggestions:
    • >30% of channels flagged - threshold may be too strict
    • Consider increasing threshold to ~0.000213 V
```

**Few channels flagged (≤3):**
```
  💡 Suggestions:
    • Few channels flagged - likely bad electrode contact
    • Consider interpolating: E46, E68
```

**No channels flagged but low amplitudes:**
```
  💡 Threshold may be too loose for this data
    • Consider tightening to ~0.000095 V for better artifact detection
```

#### 5. **Overall Quality Metrics**
```
Quality Summary Statistics:
--------------------------------------------------------------------------------
Overall mean amplitude (across all channels): 0.000085 V
Overall max amplitude (across all channels):  0.000450 V
Overall min amplitude (across all channels):  0.000012 V
Overall rejection rate: 12.3% (2460/20000 channel-epoch pairs)
```

## Example Output

### Stage 1: Preview Mode (Before Rejection)

```
================================================================================
Amplitude Quality Preview (Before Rejection)
================================================================================
Analyzing impact of configured thresholds on your data...

Computing peak-to-peak amplitudes across epochs...
================================================================================
Analyzed 100 epochs across 64 channels
================================================================================

EEG Channels:
--------------------------------------------------------------------------------
  Average mean amplitude: 0.000085 V
  Maximum mean amplitude: 0.000320 V
  Number of channels:     64
  Configured threshold:   0.000200 V

  ⚠️  Channels exceeding threshold in >20% of epochs:
    • E68: 67.8% of epochs (mean: 0.000450 V)
    • E71: 42.3% of epochs (mean: 0.000380 V)

  💡 Suggestions:
    • Few channels flagged - likely bad electrode contact
    • Consider interpolating: E68, E71

EOG Channels:
--------------------------------------------------------------------------------
  Average mean amplitude: 0.000180 V
  Maximum mean amplitude: 0.000520 V
  Number of channels:     2
  ✓ All channels within acceptable limits

================================================================================
Quality Summary Statistics:
--------------------------------------------------------------------------------
Overall mean amplitude (across all channels): 0.000092 V
Overall max amplitude (across all channels):  0.000520 V
Overall min amplitude (across all channels):  0.000012 V
Overall rejection rate: 8.5% (544/6400 channel-epoch pairs)
================================================================================

📊 Impact Preview:
  • With current thresholds, 8.5% of channel-epoch pairs will be flagged
  • Proceeding with rejection of flagged epochs...
```

### Stage 2: Validation Mode (After Rejection)

```
Epoch Drop Log Summary:
  Total epochs: 100
  Good epochs: 92
  Epochs with EEG: 8

================================================================================
Final Data Quality Validation
================================================================================
Computing peak-to-peak amplitudes across epochs...
================================================================================
Analyzed 92 epochs across 64 channels
================================================================================

EEG Channels:
--------------------------------------------------------------------------------
  Average mean amplitude: 0.000068 V
  Maximum mean amplitude: 0.000185 V
  Number of channels:     64

EOG Channels:
--------------------------------------------------------------------------------
  Average mean amplitude: 0.000165 V
  Maximum mean amplitude: 0.000480 V
  Number of channels:     2

================================================================================
Quality Summary Statistics:
--------------------------------------------------------------------------------
Overall mean amplitude (across all channels): 0.000072 V
Overall max amplitude (across all channels):  0.000480 V
Overall min amplitude (across all channels):  0.000008 V
================================================================================

✓ Final Dataset Quality:
  • Mean amplitude: 0.000072 V
  • Data retained: 92/100 epochs (92.0%)
```

## Usage

### Automatic Integration

The coach runs automatically during `create_eventid_epochs()`:

```python
# Just create epochs normally - coaching happens automatically
epochs = processor.create_eventid_epochs(
    event_id={"target": 4},
    volt_threshold={"eeg": 0.0002},
)

# Output includes full amplitude quality analysis with suggestions
```

### Standalone Analysis

Run on any epochs object for diagnosis:

```python
# Analyze existing epochs with different thresholds
quality_df = processor.summarize_amplitude_quality(
    epochs=epochs,
    volt_threshold={"eeg": 0.00015, "eog": 0.0005}
)

# Returns DataFrame with per-channel statistics
print(quality_df[["channel", "mean_amp", "flagged_pct"]])
```

### Returned DataFrame

The method returns a detailed DataFrame with columns:
- `channel`: Channel name
- `ch_type`: Channel type (eeg, eog, etc.)
- `mean_amp`: Mean peak-to-peak amplitude across epochs
- `max_amp`: Maximum peak-to-peak amplitude
- `min_amp`: Minimum peak-to-peak amplitude
- `flagged_count`: Number of epochs exceeding threshold
- `flagged_pct`: Percentage of epochs exceeding threshold
- `threshold`: Configured threshold for this channel type

## Workflow Examples

### 1. First-Time Threshold Selection

```python
# Step 1: Analyze without rejection
epochs_all = processor.create_eventid_epochs(
    event_id={"stimulus": 4},
    volt_threshold=None,  # No rejection
    keep_all_epochs=True,
)

# Coach shows amplitude distribution without thresholds
# Output: "Average mean amplitude: 0.000085 V"

# Step 2: Apply coach-suggested threshold
quality_df = processor.summarize_amplitude_quality(epochs_all, None)
suggested_threshold = float(quality_df["mean_amp"].mean() * 2.5)

epochs_clean = processor.create_eventid_epochs(
    event_id={"stimulus": 4},
    volt_threshold={"eeg": suggested_threshold},
)

# Coach validates threshold effectiveness
```

### 2. Identifying Bad Channels

```python
epochs = processor.create_eventid_epochs(
    event_id={"target": 4},
    volt_threshold={"eeg": 0.0002},
)

# Coach output identifies problem channels
# Look for: "Few channels flagged - likely bad electrode contact"
# Suggestion: "Consider interpolating: E68, E71"

# Apply interpolation
raw.info['bads'] = ['E68', 'E71']
raw.interpolate_bads()

# Rerun epochs with interpolated channels
epochs_fixed = processor.create_eventid_epochs(
    event_id={"target": 4},
    volt_threshold={"eeg": 0.0002},
)
```

### 3. Threshold Tuning

```python
# Coach says: ">30% of channels flagged - threshold may be too strict"
# Suggestion: "Consider increasing threshold to ~0.000213 V"

# Apply suggestion
epochs = processor.create_eventid_epochs(
    event_id={"target": 4},
    volt_threshold={"eeg": 0.000213},  # Use suggested value
)

# Coach confirms: "✓ All channels within acceptable limits"
```

## Implementation Details

### Method Signature

```python
def summarize_amplitude_quality(
    self,
    epochs: mne.Epochs,
    volt_threshold: Optional[Dict[str, float]] = None,
) -> Optional[pd.DataFrame]
```

### Algorithm

1. **Extract epoch data**: `epochs.get_data()` → (n_epochs, n_channels, n_times)
2. **Compute peak-to-peak per channel**: `data[:, ch_idx, :].ptp(axis=1)`
3. **Calculate statistics**: mean, max, min amplitudes
4. **Compare to thresholds**: Count/percentage of epochs exceeding limits
5. **Aggregate by channel type**: Group EEG, EOG, etc.
6. **Analyze patterns**:
   - Many flagged → strict threshold
   - Few flagged → bad channels
   - None flagged + low amp → loose threshold
7. **Generate suggestions**: Context-specific recommendations
8. **Return DataFrame**: Detailed per-channel statistics

### Performance

- **Complexity**: O(n_epochs × n_channels × n_timepoints)
- **Acceptable** because data is already preloaded for epoching
- **Fast**: Typical 100 epochs × 64 channels × 1000 samples = ~0.5s

### Integration Points

**In `create_eventid_epochs()`:**

```python
# After epochs_clean is finalized, before metadata update
quality_df = self.summarize_amplitude_quality(
    epochs=epochs_clean,
    volt_threshold=volt_threshold
)

# Store summary in metadata
if quality_df is not None:
    metadata["amplitude_quality"] = {
        "overall_mean_amplitude": float(quality_df["mean_amp"].mean()),
        "overall_max_amplitude": float(quality_df["max_amp"].max()),
        "channels_analyzed": len(quality_df),
        "threshold_config": volt_threshold,
        "total_flagged_pairs": int(quality_df["flagged_count"].sum()),
    }
```

## Decision Tree

The coach uses this logic to generate suggestions:

```
volt_threshold provided?
├─ Yes → Compare amplitudes to thresholds
│   ├─ Problem channels (>20% flagged)?
│   │   ├─ Many (>30% of channels) → "Threshold too strict, increase to X"
│   │   ├─ Few (≤3 channels) → "Bad electrode contact, interpolate: X, Y, Z"
│   │   └─ Moderate → "Review electrode placement"
│   └─ No problem channels
│       └─ Low amplitudes (<30% of threshold)?
│           └─ Yes → "Threshold too loose, tighten to X"
└─ No → Only show descriptive statistics
```

## Metadata Export

Quality statistics are automatically stored in processing metadata:

```json
{
  "step_create_eventid_epochs": {
    "amplitude_quality": {
      "overall_mean_amplitude": 0.000085,
      "overall_max_amplitude": 0.000450,
      "channels_analyzed": 64,
      "threshold_config": {"eeg": 0.0002},
      "total_flagged_pairs": 544
    }
  }
}
```

This enables:
- Reproducibility (know exactly what thresholds were used)
- Quality tracking across processing stages
- Automated quality control in batch pipelines

## Benefits

### For Users

- ✅ **Understand** why epochs are rejected
- ✅ **Tune** thresholds empirically, not by guesswork
- ✅ **Identify** bad channels before ICA
- ✅ **Validate** that data quality is acceptable
- ✅ **Learn** best practices through suggestions
- ✅ **Confidence** in preprocessing decisions

### For Analysis

- ✅ **Transparent QC**: All decisions documented
- ✅ **Reproducible**: Threshold values tracked in metadata
- ✅ **Auditable**: Clear rationale for rejections
- ✅ **Comparable**: Consistent quality metrics across datasets

### For Support

- ✅ **Reduced questions**: Users get immediate guidance
- ✅ **Easier debugging**: Quality summaries in logs
- ✅ **Self-service**: Suggestions handle common issues
- ✅ **Better reports**: Users can share quality DataFrames

## Design Principles

### 1. **Interpretable Over Precise**
Prefer clear explanations over exact technical details. Users need to understand *why*, not just *what*.

### 2. **Proactive Over Reactive**
Show quality metrics even when no problems detected. Prevention beats correction.

### 3. **Actionable Over Descriptive**
Every insight includes a concrete next step. Numbers mean nothing without context.

### 4. **Educational Over Automated**
Teach users to fish rather than catching fish for them. Build understanding, not dependency.

## Related

- **Event Discovery Coach**: Helps configure event_id
- **Amplitude Quality Coach**: Helps tune volt_threshold (this document)
- Future: Channel interpolation coach, ICA component coach, etc.

## References

- MNE Epochs Quality: https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html#rejecting-bad-epochs
- Threshold-based rejection: https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs
- AutoCleanEEG Processing: `docs/tutorials/first_time_processing.rst`

---

The Amplitude Quality Coach transforms opaque rejection into transparent, guided decision-making. It's coaching, not just analysis.

