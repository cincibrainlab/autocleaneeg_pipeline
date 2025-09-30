# Source Connectivity Block

**Version:** 1.0.0
**Category:** Analysis
**Status:** Stable

## Overview

The Source Connectivity block calculates functional connectivity between brain regions from source-localized EEG data and computes graph theory metrics to characterize brain network organization. This analysis reveals how different brain regions communicate across frequency bands and provides quantitative measures of network architecture.

## What It Does

Source-level connectivity analysis performs:
1. Functional connectivity calculation using 5 methods (WPLI, PLV, coherence, PLI, AEC)
2. ROI time course extraction from source estimates
3. Epoch-based connectivity averaging (4s epochs, 40 epochs default)
4. Frequency-band-specific connectivity (delta through gamma)
5. Graph theory metrics (clustering, efficiency, modularity, path length, small-worldness)
6. Comprehensive outputs (connectivity matrices, pairwise values, graph metrics)

The block processes continuous source estimates, extracts ROI time courses, and computes connectivity matrices for multiple methods and frequency bands, producing both edge-level and network-level characterizations.

## When to Use

**Use source connectivity when you need:**
- Functional brain network analysis
- Connectivity biomarkers for clinical populations
- Network dynamics across frequency bands
- Graph theory characterization of brain organization
- ROI-to-ROI coupling quantification

**Requirements:**
- Prior source localization (self.stc must exist)
- Minimum 160 seconds of clean data (4s × 40 epochs)
- Adequate SNR for reliable connectivity estimation
- fsaverage brain data and Desikan-Killiany atlas

## How It Works

### Algorithm

```
Source Estimate (continuous STC)
         ↓
ROI Time Course Extraction
    (Desikan-Killiany atlas, 8 sensorimotor ROIs)
         ↓
Random Epoch Selection
    (40 epochs of 4s each from available data)
         ↓
Connectivity Calculation
    5 methods × 5 frequency bands = 25 matrices
    Methods: WPLI, PLV, Coh, PLI, AEC
    Bands: Delta, Theta, Alpha, Beta, Gamma
         ↓
Graph Theory Metrics
    Clustering, Efficiency, Modularity, etc.
         ↓
Output: Connectivity Matrices + Graph Metrics
```

### Technical Details

- **Methods:** WPLI (primary), PLV, coherence, PLI, AEC
- **Frequency Bands:** Delta (1-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-45 Hz)
- **Atlas:** Desikan-Killiany (aparc annotation)
- **Default ROIs:** 8 sensorimotor regions (precentral, postcentral, paracentral, caudalmiddlefrontal - bilateral)
- **Epoching:** Random selection from available data
- **Connectivity Estimation:** Spectral_connectivity_time (MNE-Connectivity)
- **Graph Metrics:** NetworkX + BCTpy

### Connectivity Methods

1. **WPLI (Weighted Phase Lag Index)** - Robust to volume conduction, recommended
2. **PLV (Phase Locking Value)** - Phase synchronization measure
3. **Coherence** - Magnitude-squared coherence
4. **PLI (Phase Lag Index)** - Volume conduction robust
5. **AEC (Amplitude Envelope Correlation)** - Amplitude coupling

## Configuration

### Basic Configuration

```python
config = {
    "apply_source_connectivity": {
        "enabled": True,
        "value": {
            "epoch_length": 4.0,  # seconds
            "n_epochs": 40,
            "n_jobs": 4
        }
    }
}
```

### Advanced Configuration

```python
# Short epochs, many averages
config = {
    "apply_source_connectivity": {
        "enabled": True,
        "value": {
            "epoch_length": 2.0,
            "n_epochs": 80,
            "n_jobs": 8
        }
    }
}

# Long epochs, fewer averages
config = {
    "apply_source_connectivity": {
        "enabled": True,
        "value": {
            "epoch_length": 8.0,
            "n_epochs": 20,
            "n_jobs": 4
        }
    }
}
```

## Usage in Tasks

### Resting-State Example

```python
from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
    "apply_source_localization": {
        "enabled": True,
        "value": {"method": "MNE", "lambda2": 0.111}
    },
    "apply_source_connectivity": {
        "enabled": True,
        "value": {"epoch_length": 4.0, "n_epochs": 40}
    }
}

class RestingConnectivity(Task):
    def run(self):
        # Preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.rereference_data()
        self.run_ica()

        # Create epochs
        self.create_regular_epochs()

        # Source analysis pipeline
        self.apply_source_localization()  # Creates self.stc
        conn_df, summary_path = self.apply_source_connectivity()

        # Access specific connectivity
        alpha_wpli = conn_df[
            (conn_df['method'] == 'wpli') &
            (conn_df['band'] == 'alpha')
        ]
        print(f"Alpha WPLI mean: {alpha_wpli['connectivity'].mean():.3f}")
```

## Outputs

### Connectivity Summary (CSV)

**File:** `{subject}_connectivity_summary.csv`

**Columns:**
- `subject`: Subject identifier
- `method`: Connectivity method (wpli, plv, coh, pli, aec)
- `band`: Frequency band (delta, theta, alpha, beta, gamma)
- `roi1`: First ROI name
- `roi2`: Second ROI name
- `connectivity`: Connectivity value

**Example usage:**
```python
import pandas as pd
conn = pd.read_csv("subject_connectivity_summary.csv")

# Filter for alpha WPLI
alpha_wpli = conn[(conn['method'] == 'wpli') & (conn['band'] == 'alpha')]

# Get specific connection
precentral_conn = alpha_wpli[
    alpha_wpli['roi1'].str.contains('precentral') &
    alpha_wpli['roi2'].str.contains('postcentral')
]
```

### Connectivity Matrices (CSV)

**Files:** `{subject}_{method}_{band}_matrix.csv`

Full ROI × ROI matrices for each method and band combination (25 files).

### Graph Metrics (CSV)

**File:** `{subject}_graph_metrics.csv`

**Columns:**
- `subject`: Subject identifier
- `method`: Connectivity method
- `band`: Frequency band
- `clustering`: Mean clustering coefficient
- `global_efficiency`: Network efficiency
- `char_path_length`: Characteristic path length
- `modularity`: Community structure strength
- `strength`: Mean node strength
- `assortativity`: Degree assortativity
- `small_worldness`: Small-worldness index

## Performance

**Computational Cost:**
- ROI extraction: ~5-10 seconds
- Connectivity per method/band: ~30-60 seconds
- Total for 5 methods × 5 bands: ~10-15 minutes
- Graph metrics: ~1-2 minutes
- **Total:** ~10-20 minutes per subject

**Memory Requirements:**
- Source estimates: ~500 MB - 2 GB
- ROI time courses: ~50 MB
- Connectivity matrices: ~10 MB
- Peak memory: ~16 GB (parallel processing)

**Optimization Tips:**
- Reduce `n_epochs` for faster processing (minimum 20 recommended)
- Adjust `epoch_length` based on lowest frequency of interest (4s = 0.25 Hz resolution)
- Increase `n_jobs` for parallel connectivity calculation (4-8 recommended)
- Process on high-memory systems for best performance

## Downstream Analyses

Source connectivity enables:

1. **Statistical Testing**
   ```python
   # Compare alpha connectivity between groups
   from scipy import stats
   group_a_wpli = [load_connectivity(f, 'wpli', 'alpha') for f in files_a]
   group_b_wpli = [load_connectivity(f, 'wpli', 'alpha') for f in files_b]
   t_stat, p_val = stats.ttest_ind(group_a_wpli, group_b_wpli)
   ```

2. **Network Visualization**
   ```python
   import networkx as nx
   import matplotlib.pyplot as plt

   # Create network from connectivity matrix
   G = nx.from_pandas_edgelist(
       alpha_wpli, 'roi1', 'roi2', 'connectivity'
   )
   nx.draw(G, with_labels=True, node_size=1000)
   ```

3. **Graph Theory Correlations**
   ```python
   # Correlate modularity with clinical scores
   metrics = pd.read_csv("subject_graph_metrics.csv")
   alpha_metrics = metrics[metrics['band'] == 'alpha']
   correlation = alpha_metrics['modularity'].corr(clinical_scores)
   ```

## Limitations

1. **ROI Selection**
   - Default: 8 sensorimotor ROIs (customizable via labels parameter)
   - Limited spatial coverage (not whole-brain by default)
   - Parcellation-dependent results

2. **Data Requirements**
   - Requires 160+ seconds of clean data (40 epochs × 4s)
   - Low SNR → unreliable connectivity estimates
   - Assumes stationarity within epochs

3. **Computational Demands**
   - Memory-intensive (16GB recommended)
   - Time-consuming (10-20 minutes per subject)
   - Parallel processing helps but still substantial

4. **Volume Conduction**
   - Some methods (PLV, coherence) sensitive to volume conduction
   - WPLI and PLI more robust but not perfect
   - Source localization quality crucial

## Troubleshooting

### "No source estimates found"
**Solution:** Apply source localization first (`self.apply_source_localization()`)

### "Insufficient epochs available"
**Solution:**
- Reduce `n_epochs` (minimum 20)
- Reduce `epoch_length` (minimum 2s)
- Ensure adequate clean data after preprocessing

### High memory usage
**Solution:**
- Reduce `n_jobs`
- Process fewer methods/bands
- Increase system swap space
- Use high-memory machine

### "Network analysis libraries not available"
**Solution:** Install dependencies:
```bash
pip install networkx bctpy
```

### Unrealistic connectivity patterns
**Solution:**
- Check source localization quality
- Verify preprocessing (ICA, artifact rejection)
- Check for line noise contamination
- Ensure sufficient data length

## Scientific References

1. **WPLI**
   - Vinck M, et al. (2011). The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization. *NeuroImage*, 51(1), 112-122.

2. **Graph Theory in Neuroscience**
   - Rubinov M & Sporns O (2010). Complex network measures of brain connectivity: Uses and interpretations. *NeuroImage*, 52(3), 1059-1069.

3. **MNE-Connectivity**
   - Larson-Prior LJ, et al. (2013). Adding dynamics to the Human Connectome Project with MEG. *NeuroImage*, 80, 190-201.

4. **Phase Lag Index**
   - Stam CJ, et al. (2007). Phase lag index: assessment of functional connectivity from multi channel EEG and MEG with diminished bias from common sources. *Human Brain Mapping*, 28(11), 1178-1193.

5. **Small-World Networks**
   - Watts DJ & Strogatz SH (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.

## Version History

- **1.0.0** (2025-09-29): Initial release
  - 5 connectivity methods (WPLI, PLV, coherence, PLI, AEC)
  - 5 frequency bands (delta through gamma)
  - 7 graph theory metrics
  - Detailed logging and error handling
  - CSV outputs for connectivity and graph metrics