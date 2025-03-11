# ASSR Analysis and Group-Level Processing

This guide explains how to use the ASSR (Auditory Steady-State Response) analysis tools in the autoclean pipeline, with a focus on exporting heat maps for group-level analysis.

## Overview

The ASSR analysis module provides tools for:

1. Computing time-frequency representations of EEG data
2. Visualizing Inter-Trial Coherence (ITC), Event-Related Spectral Perturbation (ERSP), and Single Trial Power (STP)
3. Exporting heat maps for group-level analysis
4. Performing group-level analysis across multiple subjects

These tools are particularly useful for analyzing 40 Hz ASSR paradigms, which are commonly used to study neural synchronization in various populations.

## Individual Subject Analysis

### Basic Usage

To analyze a single subject's ASSR data:

```python
from autoclean.calc.assr_analysis import analyze_assr
from autoclean.calc.assr_viz import plot_all_figures

# Run the analysis
analysis_results = analyze_assr('path/to/eeglab_file.set', output_dir='output_directory', save_results=True)

# Generate visualizations
figures = plot_all_figures(
    analysis_results['tf_data'], 
    analysis_results['epochs'], 
    output_dir='output_directory', 
    save_figures=True
)
```

Alternatively, you can use the command-line interface:

```bash
python -m autoclean.calc.assr_viz path/to/eeglab_file.set --output_dir output_directory
```

## Visualization Guide

The ASSR visualization module provides several functions for creating different types of visualizations. Here's a detailed guide on how to use each visualization function.

### Available Visualization Functions

The main visualization functions include:

1. `plot_itc_channels`: Plots Inter-Trial Coherence for each channel
2. `plot_global_mean_itc`: Plots global mean ITC across all channels
3. `plot_topomap`: Creates a topographic map of ITC at a specific time point
4. `plot_ersp_channels`: Plots Event-Related Spectral Perturbation for each channel
5. `plot_global_mean_ersp`: Plots global mean ERSP across all channels
6. `plot_stp_channels`: Plots Single Trial Power for each channel
7. `plot_global_mean_stp`: Plots global mean STP across all channels
8. `plot_all_figures`: Generates all of the above visualizations

### Visualizing Inter-Trial Coherence (ITC)

ITC measures the consistency of phase across trials at each time-frequency point. Values range from 0 (random phase) to 1 (perfect phase alignment).

#### Channel-Level ITC Visualization

```python
from autoclean.calc.assr_viz import plot_itc_channels

# Plot ITC for each channel
fig_itc = plot_itc_channels(
    analysis_results['tf_data'],
    analysis_results['epochs'],
    output_dir='output_directory',
    save_figures=True,
    file_basename='subject_01'
)
```

This creates a grid of plots, one for each channel, showing the ITC values across time and frequency. The 40 Hz line is highlighted in red to help identify the ASSR response.

![Example ITC Channels Plot](../../animations/itc_channels_example.png)

#### Global Mean ITC Visualization

```python
from autoclean.calc.assr_viz import plot_global_mean_itc

# Plot global mean ITC
fig_global_itc = plot_global_mean_itc(
    analysis_results['tf_data'],
    output_dir='output_directory',
    save_figures=True,
    epochs=analysis_results['epochs'],
    file_basename='subject_01'
)
```

This creates a single plot showing the average ITC across all channels, which is useful for getting an overall view of the ASSR response.

![Example Global Mean ITC Plot](../../animations/global_itc_example.png)

#### ITC Topographic Map

```python
from autoclean.calc.assr_viz import plot_topomap

# Plot topographic map at 0.3 seconds
fig_topo = plot_topomap(
    analysis_results['tf_data'],
    analysis_results['epochs'],
    time_point=0.3,  # Time point in seconds
    output_dir='output_directory',
    save_figures=True,
    file_basename='subject_01'
)
```

This creates a topographic map showing the spatial distribution of ITC at 40 Hz at the specified time point.

![Example Topographic Map](../../animations/topomap_example.png)

### Visualizing Event-Related Spectral Perturbation (ERSP)

ERSP measures changes in spectral power relative to a baseline period. Positive values indicate power increases, while negative values indicate power decreases.

#### Channel-Level ERSP Visualization

```python
from autoclean.calc.assr_viz import plot_ersp_channels

# Plot ERSP for each channel
fig_ersp = plot_ersp_channels(
    analysis_results['tf_data'],
    analysis_results['epochs'],
    output_dir='output_directory',
    save_figures=True,
    file_basename='subject_01'
)
```

This creates a grid of plots, one for each channel, showing the ERSP values across time and frequency.

![Example ERSP Channels Plot](../../animations/ersp_channels_example.png)

#### Global Mean ERSP Visualization

```python
from autoclean.calc.assr_viz import plot_global_mean_ersp

# Plot global mean ERSP
fig_global_ersp = plot_global_mean_ersp(
    analysis_results['tf_data'],
    output_dir='output_directory',
    save_figures=True,
    epochs=analysis_results['epochs'],
    file_basename='subject_01'
)
```

This creates a single plot showing the average ERSP across all channels.

![Example Global Mean ERSP Plot](../../animations/global_ersp_example.png)

### Visualizing Single Trial Power (STP)

STP shows the raw spectral power without baseline correction, which can be useful for examining the absolute power levels.

#### Channel-Level STP Visualization

```python
from autoclean.calc.assr_viz import plot_stp_channels

# Plot STP for each channel
fig_stp = plot_stp_channels(
    analysis_results['tf_data'],
    analysis_results['epochs'],
    output_dir='output_directory',
    save_figures=True,
    file_basename='subject_01'
)
```

This creates a grid of plots, one for each channel, showing the STP values across time and frequency.

![Example STP Channels Plot](../../animations/stp_channels_example.png)

#### Global Mean STP Visualization

```python
from autoclean.calc.assr_viz import plot_global_mean_stp

# Plot global mean STP
fig_global_stp = plot_global_mean_stp(
    analysis_results['tf_data'],
    analysis_results['epochs'],
    output_dir='output_directory',
    save_figures=True,
    file_basename='subject_01'
)
```

This creates a single plot showing the average STP across all channels.

![Example Global Mean STP Plot](../../animations/global_stp_example.png)

### Generating All Visualizations

Instead of calling each visualization function separately, you can use the `plot_all_figures` function to generate all visualizations at once:

```python
from autoclean.calc.assr_viz import plot_all_figures

# Generate all visualizations
figures = plot_all_figures(
    analysis_results['tf_data'],
    analysis_results['epochs'],
    output_dir='output_directory',
    save_figures=True,
    file_basename='subject_01'
)
```

This returns a dictionary containing all the figure objects, which you can further customize if needed:

```python
# Access and customize a specific figure
import matplotlib.pyplot as plt

# Get the global mean ITC figure
fig_global_itc = figures['global_mean_itc']

# Customize the figure
plt.figure(fig_global_itc.number)
plt.title('Custom Title for Global Mean ITC')
plt.savefig('custom_global_itc.png', dpi=300)
```

### Customizing Visualizations

You can customize the visualizations by modifying the parameters of the plotting functions:

```python
# Customize ITC visualization
from autoclean.calc.assr_viz import plot_global_mean_itc
import matplotlib.pyplot as plt

# Create the figure
fig_global_itc = plot_global_mean_itc(
    analysis_results['tf_data'],
    output_dir=None,  # Don't save automatically
    save_figures=False,
    epochs=analysis_results['epochs']
)

# Customize the figure
plt.figure(fig_global_itc.number)
plt.title('Custom Title for Global Mean ITC')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.colorbar().set_label('ITC Value')
plt.tight_layout()

# Save the customized figure
plt.savefig('custom_global_itc.png', dpi=300)
```

### Exporting Heat Maps for Group Analysis

To export heat maps for later group-level analysis, use the `export_data` parameter:

```python
# Generate visualizations and export data
figures = plot_all_figures(
    analysis_results['tf_data'], 
    analysis_results['epochs'], 
    output_dir='output_directory', 
    save_figures=True,
    export_data=True
)
```

From the command line:

```bash
python -m autoclean.calc.assr_viz path/to/eeglab_file.set --output_dir output_directory --export_data
```

This will create a directory structure like:

```
output_directory/
├── figures/
│   ├── subject_fig_itc_channels.png
│   ├── subject_summary_global_itc.png
│   └── ...
└── exported_data/
    ├── subject_itc_data.npy
    ├── subject_ersp_data.npy
    ├── subject_stp_data.npy
    └── subject_export_metadata.npy
```

### Exporting to CSV Format

For compatibility with other analysis tools, you can also export the data in CSV format:

```python
# Generate visualizations and export data as CSV
figures = plot_all_figures(
    analysis_results['tf_data'], 
    analysis_results['epochs'], 
    output_dir='output_directory', 
    save_figures=True,
    export_data=True,
    export_csv=True
)
```

From the command line:

```bash
python -m autoclean.calc.assr_viz path/to/eeglab_file.set --output_dir output_directory --export_data --export_csv
```

This will create additional CSV files in the `exported_data` directory:

```
exported_data/
├── subject_itc_data_info.csv
├── subject_itc_channel_Fz.csv
├── subject_itc_channel_Cz.csv
├── subject_itc_global_mean.csv
└── ...
```

## Group-Level Analysis

### Running Group Analysis

After processing multiple subjects and exporting their data, you can perform group-level analysis:

```python
from autoclean.calc.assr_viz import group_analysis_heatmaps

# Perform group analysis on ITC data
group_data = group_analysis_heatmaps(
    data_dir='path/to/exported_data',
    data_type='itc',  # Options: 'itc', 'ersp', or 'stp'
    output_dir='group_results',
    save_figures=True,
    group_name='my_study'
)
```

From the command line:

```bash
python -m autoclean.calc.assr_viz --group_analysis --data_dir path/to/exported_data --data_type itc --output_dir group_results --group_name my_study
```

This will:

1. Find all ITC data files in the specified directory
2. Load and average the data across subjects
3. Calculate standard error of the mean
4. Create and save group-level figures
5. Save the group-level data for further analysis

### Group Visualization Examples

#### Group-Level Global Mean Visualization

When you run the `group_analysis_heatmaps` function, it automatically generates a global mean visualization for the group data:

![Example Group Global Mean ITC](../../animations/group_itc_example.png)

This shows the average response across all subjects, which is useful for identifying consistent patterns in the data.

#### Group-Level Topographic Maps

You can create topographic maps at specific frequencies and time points:

```python
from autoclean.calc.assr_viz import plot_group_topomap

# Load group data
import numpy as np
group_data = np.load('group_results/my_study_itc_data.npy', allow_pickle=True).item()

# Create topographic map at 40 Hz and 0.3 seconds
fig = plot_group_topomap(
    group_data,
    freq=40.0,
    time_point=0.3,
    output_dir='group_results/topomaps',
    save_figures=True,
    group_name='my_study'
)
```

From the command line:

```bash
python -m autoclean.calc.assr_viz --plot_topomap_only --data_dir group_results --data_type itc --freq 40.0 --time 0.3 --output_dir group_results/topomaps --group_name my_study
```

![Example Group Topographic Map](../../animations/group_topomap_example.png)

This shows the spatial distribution of the group-average response at the specified frequency and time point.

#### Comparing Multiple Groups

You can compare multiple groups by running the group analysis separately for each group and then comparing the results:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load group data for two groups
group1_data = np.load('group_results/group1_itc_data.npy', allow_pickle=True).item()
group2_data = np.load('group_results/group2_itc_data.npy', allow_pickle=True).item()

# Calculate global mean for each group
group1_global = np.mean(group1_data['data'], axis=0)
group2_global = np.mean(group2_data['data'], axis=0)

# Find the index of 40 Hz in the frequency array
freq_idx = np.argmin(np.abs(group1_data['freqs'] - 40))

# Plot the 40 Hz response over time for both groups
plt.figure(figsize=(10, 6))
plt.plot(group1_data['times'], group1_global[freq_idx], label='Group 1')
plt.plot(group2_data['times'], group2_global[freq_idx], label='Group 2')
plt.axvline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('ITC Value')
plt.title('40 Hz ASSR Response Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('group_comparison.png', dpi=300)
```

![Example Group Comparison](../../animations/group_comparison_example.png)

### Exporting Group Data to CSV

You can export the group-level data to CSV format for use in other analysis tools:

```bash
python -m autoclean.calc.assr_viz --group_analysis --data_dir path/to/exported_data --data_type itc --output_dir group_results --group_name my_study --export_csv
```

## Data Structure

### Individual Subject Data

The exported data for each subject is stored in NumPy files with the following structure:

```python
{
    'data': array,        # Shape: (channels, freqs, times)
    'times': array,       # Time points in seconds
    'freqs': array,       # Frequencies in Hz
    'ch_names': list,     # Channel names
    'subject_id': str     # Subject identifier
}
```

### Group-Level Data

The group-level data is stored with the following structure:

```python
{
    'data': array,        # Shape: (channels, freqs, times) - group average
    'sem': array,         # Standard error of the mean
    'times': array,       # Time points in seconds
    'freqs': array,       # Frequencies in Hz
    'ch_names': list,     # Channel names
    'n_subjects': int,    # Number of subjects
    'subject_ids': list,  # List of subject identifiers
    'data_type': str      # Type of data ('itc', 'ersp', or 'stp')
}
```

## Advanced Usage

### Customizing Analysis Parameters

You can customize the analysis parameters by modifying the `analyze_assr` function call:

```python
analysis_results = analyze_assr(
    'path/to/eeglab_file.set',
    output_dir='output_directory',
    save_results=True,
    fmin=1,              # Minimum frequency
    fmax=100,            # Maximum frequency
    n_freqs=50,          # Number of frequency points
    n_cycles=7,          # Number of cycles for wavelet transform
    use_fft=False,       # Whether to use FFT instead of wavelets
    baseline=(-0.5, 0),  # Baseline period for normalization
    baseline_mode='logratio'  # Baseline normalization method
)
```

### Working with the Exported Data

You can load and work with the exported data directly:

```python
import numpy as np
import matplotlib.pyplot as plt

# Load subject data
subject_data = np.load('output_directory/exported_data/subject_itc_data.npy', allow_pickle=True).item()

# Extract data for a specific channel
channel_idx = subject_data['ch_names'].index('Cz')
channel_data = subject_data['data'][channel_idx]

# Plot data for a specific frequency
freq_idx = np.argmin(np.abs(subject_data['freqs'] - 40))
plt.figure(figsize=(10, 6))
plt.plot(subject_data['times'], channel_data[freq_idx])
plt.title(f"40 Hz ITC at channel Cz")
plt.xlabel("Time (s)")
plt.ylabel("ITC Value")
plt.show()
```

### Custom Group Analysis

You can perform custom group analysis by loading and manipulating the exported data:

```python
import numpy as np
import glob
from pathlib import Path

# Find all ITC data files
data_files = glob.glob('path/to/exported_data/*_itc_data.npy')

# Load data from all files
all_data = []
subject_ids = []

for file_path in data_files:
    data_dict = np.load(file_path, allow_pickle=True).item()
    all_data.append(data_dict['data'])
    subject_ids.append(data_dict['subject_id'])

# Convert to numpy array
all_data = np.array(all_data)  # Shape: (subjects, channels, freqs, times)

# Calculate group average
group_avg = np.mean(all_data, axis=0)

# Calculate standard error of the mean
group_sem = np.std(all_data, axis=0) / np.sqrt(len(all_data))

# Perform statistical tests
# ...
```

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure that the data directory contains the expected files with the correct naming pattern.

2. **Channel mismatch**: When performing group analysis, ensure that all subjects have the same channel configuration.

3. **Memory errors**: If you encounter memory errors when processing large datasets, try reducing the frequency or time resolution.

### Getting Help

If you encounter issues not covered in this guide, please:

1. Check the [troubleshooting](troubleshooting.md) section for general pipeline issues
2. Review the API documentation for detailed function descriptions
3. Submit an issue on the project's GitHub repository with a detailed description of the problem 