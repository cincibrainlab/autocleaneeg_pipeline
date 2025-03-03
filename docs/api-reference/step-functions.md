# Step Functions API Reference

Step functions are modular processing units that perform specific operations on EEG data within the AutoClean Pipeline. This page documents the core step functions available in the pipeline.

## Module Structure

Step functions are organized in modules based on their functionality:

```
autoclean.step_functions
├── io.py             # File I/O operations
├── preprocessing.py  # Basic signal processing
├── artifacts.py      # Artifact detection and rejection
├── ica.py            # ICA-related operations
├── epochs.py         # Epoching functions
├── frequency.py      # Spectral analysis
├── connectivity.py   # Connectivity measures
├── source.py         # Source localization
└── reports.py        # Visualization and reporting
```

## Common Step Function Pattern

All step functions follow a consistent pattern:

```python
def step_function_name(pipeline, param1=default1, param2=default2, ...):
    """Description of what the step function does.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance containing the EEG data
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    
    Returns
    -------
    None
        The pipeline is modified in-place
    """
    # Implementation
    pass
```

## I/O Functions

### load_raw

```python
from autoclean.step_functions.io import load_raw

def load_raw(pipeline, file_path=None, preload=True, verbose=None):
    """Load raw EEG data into the pipeline.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    file_path : str or Path, optional
        Path to the EEG file. If None, uses pipeline.unprocessed_file
    preload : bool, default=True
        Whether to preload the data into memory
    verbose : bool or None, optional
        Whether to print verbose output
    
    Returns
    -------
    None
        Sets pipeline.raw to the loaded Raw object
    """
```

### save_raw

```python
from autoclean.step_functions.io import save_raw

def save_raw(pipeline, file_path=None, overwrite=True, fmt='auto'):
    """Save the processed raw EEG data to disk.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    file_path : str or Path, optional
        Path where to save the file. If None, uses pipeline's default output path
    overwrite : bool, default=True
        Whether to overwrite existing files
    fmt : str, default='auto'
        File format to use ('auto', 'single', 'double', or 'int')
    
    Returns
    -------
    None
        Saves the data to disk
    """
```

### export_to_eeglab

```python
from autoclean.step_functions.io import export_to_eeglab

def export_to_eeglab(pipeline, file_path=None):
    """Export the EEG data to EEGLAB format (.set/.fdt).
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    file_path : str or Path, optional
        Path where to save the files. If None, uses pipeline's default output path
    
    Returns
    -------
    None
        Exports the data to EEGLAB format
    """
```

## Preprocessing Functions

### resample

```python
from autoclean.step_functions.preprocessing import resample

def resample(pipeline, sfreq=None, npad='auto'):
    """Resample the raw EEG data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    sfreq : float, optional
        New sampling rate in Hz. If None, uses pipeline.config['sampling_rate']
    npad : int or 'auto', default='auto'
        Number of samples to use for padding
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place
    """
```

### apply_bandpass_filter

```python
from autoclean.step_functions.preprocessing import apply_bandpass_filter

def apply_bandpass_filter(pipeline, l_freq=1.0, h_freq=40.0, method='fir', 
                          iir_params=None, fir_design='firwin', 
                          skip_by_annotation=('edge', 'bad_acq_skip')):
    """Apply a bandpass filter to the raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    l_freq : float, default=1.0
        Lower frequency bound in Hz
    h_freq : float, default=40.0
        Upper frequency bound in Hz
    method : str, default='fir'
        Filter method ('fir' or 'iir')
    iir_params : dict | None, default=None
        Parameters to use for IIR filtering
    fir_design : str, default='firwin'
        FIR design method
    skip_by_annotation : list, default=('edge', 'bad_acq_skip')
        Annotations to skip during filtering
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place
    """
```

### apply_notch_filter

```python
from autoclean.step_functions.preprocessing import apply_notch_filter

def apply_notch_filter(pipeline, freqs=None, notch_widths=None, trans_bandwidth=1,
                      method='fir', iir_params=None, mt_bandwidth=None,
                      p_value=0.05, picks=None, filter_length='auto', 
                      phase='zero', fir_design='firwin', fir_window='hamming'):
    """Apply a notch filter to remove power line noise.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    freqs : float or array of float, optional
        Frequencies to filter out. If None, uses pipeline.config['line_noise']
    notch_widths : float or array of float, optional
        Width of the notch (Hz)
    trans_bandwidth : float, default=1
        Width of the transition band (Hz)
    method : str, default='fir'
        'fir' or 'iir'
    
    ... (other parameters omitted for brevity)
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place
    """
```

### apply_reference

```python
from autoclean.step_functions.preprocessing import apply_reference

def apply_reference(pipeline, ref_channels='average', projection=False):
    """Apply a new reference to the raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    ref_channels : list of str | str, default='average'
        The channel(s) to use as reference. If 'average', compute average reference
    projection : bool, default=False
        If True, use projection matrix to re-reference. Else, directly modify data
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place
    """
```

## Artifact Detection

### detect_bad_channels

```python
from autoclean.step_functions.artifacts import detect_bad_channels

def detect_bad_channels(pipeline, method='correlation', threshold=0.7, 
                        min_correlation=0.4, fraction_bad=0.1):
    """Detect bad channels in the raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    method : str, default='correlation'
        Method to use ('correlation', 'pyprep', 'ransac', 'autoreject')
    threshold : float, default=0.7
        Threshold for detection (meaning depends on method)
    min_correlation : float, default=0.4
        Minimum correlation to consider (for 'correlation' method)
    fraction_bad : float, default=0.1
        Maximum fraction of channels allowed to be bad
    
    Returns
    -------
    None
        Adds bad channels to pipeline.raw.info['bads']
    """
```

### detect_muscle_artifacts

```python
from autoclean.step_functions.artifacts import detect_muscle_artifacts

def detect_muscle_artifacts(pipeline, threshold=5, min_duration=0.1, 
                           max_duration=0.5):
    """Detect muscle (EMG) artifacts in the raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    threshold : float, default=5
        Threshold in z-scores for detection
    min_duration : float, default=0.1
        Minimum duration of muscle artifact (seconds)
    max_duration : float, default=0.5
        Maximum duration to mark as muscle artifact (seconds)
    
    Returns
    -------
    None
        Adds annotations to pipeline.raw
    """
```

## ICA Functions

### fit_ica

```python
from autoclean.step_functions.ica import fit_ica

def fit_ica(pipeline, n_components=0.999, method='fastica', random_state=42,
           max_iter=200, fit_params=None):
    """Fit ICA to the raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    n_components : int | float, default=0.999
        Number of components to extract. If float between 0-1, it's treated as explained variance ratio
    method : str, default='fastica'
        ICA method ('fastica', 'infomax', 'extended-infomax', 'picard')
    random_state : int, default=42
        Random seed for reproducibility
    max_iter : int, default=200
        Maximum number of iterations
    fit_params : dict | None, default=None
        Additional parameters to pass to the ICA algorithm
    
    Returns
    -------
    None
        Sets pipeline.ica to the fitted ICA object
    """
```

### detect_artifact_components

```python
from autoclean.step_functions.ica import detect_artifact_components

def detect_artifact_components(pipeline, method='correlation', threshold=0.3,
                             eog_channels=None, ecg_channels=None):
    """Detect ICA components related to artifacts.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    method : str, default='correlation'
        Method for detection ('correlation', 'auto', 'EOGcorr', 'ECGcorr')
    threshold : float, default=0.3
        Correlation threshold for detection
    eog_channels : list of str | None, default=None
        Channels to use for EOG detection. If None, uses pipeline's channel types
    ecg_channels : list of str | None, default=None
        Channels to use for ECG detection. If None, uses pipeline's channel types
    
    Returns
    -------
    None
        Sets pipeline.ica.exclude with the indices of artifact components
    """
```

### apply_ica

```python
from autoclean.step_functions.ica import apply_ica

def apply_ica(pipeline, exclude=None):
    """Apply ICA to remove artifacts from raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    exclude : list of int | None, default=None
        Indices of components to exclude. If None, uses pipeline.ica.exclude
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place with artifact components removed
    """
```

## Epoching Functions

### create_epochs

```python
from autoclean.step_functions.epochs import create_epochs

def create_epochs(pipeline, events=None, event_id=None, tmin=-0.2, tmax=0.5,
                baseline=(None, 0), preload=True, reject=None, flat=None,
                reject_by_annotation=True):
    """Create epochs from raw data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    events : array | None, default=None
        Events array. If None, events are detected from stim channel
    event_id : dict | None, default=None
        Mapping from event names to event codes
    tmin : float, default=-0.2
        Start time of epoch in seconds relative to event
    tmax : float, default=0.5
        End time of epoch in seconds relative to event
    baseline : tuple, default=(None, 0)
        Baseline period (start, end) in seconds
    preload : bool, default=True
        Whether to preload the data
    reject : dict | None, default=None
        Rejection parameters based on peak-to-peak amplitude
    flat : dict | None, default=None
        Rejection parameters based on flatness of signal
    reject_by_annotation : bool, default=True
        Whether to reject based on annotations
    
    Returns
    -------
    None
        Sets pipeline.epochs to the created Epochs object
    """
```

### create_evoked

```python
from autoclean.step_functions.epochs import create_evoked

def create_evoked(pipeline, condition=None):
    """Create evoked object from epochs.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    condition : str | list | None, default=None
        Condition(s) to use for averaging. If None, average all epochs
    
    Returns
    -------
    None
        Sets pipeline.evoked to the created Evoked object
    """
```

## Frequency Analysis

### compute_psd

```python
from autoclean.step_functions.frequency import compute_psd

def compute_psd(pipeline, fmin=0, fmax=100, tmin=None, tmax=None, method='welch',
               n_fft=2048, n_overlap=0, n_per_seg=None, picks=None,
               window='hamming'):
    """Compute power spectral density (PSD) of the data.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    fmin : float, default=0
        Minimum frequency of interest
    fmax : float, default=100
        Maximum frequency of interest
    tmin : float | None, default=None
        Start time of the data to use
    tmax : float | None, default=None
        End time of the data to use
    method : str, default='welch'
        Method to use ('welch', 'multitaper', 'fft')
    n_fft : int, default=2048
        Length of FFT
    n_overlap : int, default=0
        Number of points to overlap
    n_per_seg : int | None, default=None
        Length of each segment
    picks : list | None, default=None
        Channels to include
    window : str, default='hamming'
        Window function to use
    
    Returns
    -------
    None
        Sets pipeline.psd to the computed PSD object
    """
```

### extract_frequency_bands

```python
from autoclean.step_functions.frequency import extract_frequency_bands

def extract_frequency_bands(pipeline, bands=None):
    """Extract power in frequency bands from PSD.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    bands : dict | None, default=None
        Dictionary of frequency bands. If None, uses default bands:
        {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
         'beta': (13, 30), 'gamma': (30, 45)}
    
    Returns
    -------
    None
        Adds frequency band data to pipeline.metadata
    """
```

## Report Generation

### generate_report

```python
from autoclean.step_functions.reports import generate_report

def generate_report(pipeline, report_type='html', file_path=None, 
                   include_psd=True, include_ica=True, include_evoked=True):
    """Generate processing report.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    report_type : str, default='html'
        Type of report ('html', 'pdf')
    file_path : str | None, default=None
        Path to save the report. If None, uses pipeline's default report path
    include_psd : bool, default=True
        Whether to include PSD plots
    include_ica : bool, default=True
        Whether to include ICA component plots
    include_evoked : bool, default=True
        Whether to include evoked response plots
    
    Returns
    -------
    None
        Generates and saves the report
    """
```

## Custom Step Functions

You can create custom step functions to extend the pipeline:

```python
def my_custom_step(pipeline, param1=default1, param2=default2):
    """Description of what this step function does.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    """
    # Implementation
    # Get config values or use defaults
    param1 = pipeline.config.get('my_section', {}).get('param1', param1)
    
    # Log what you're doing
    pipeline.logger.info(f"Running my_custom_step with param1={param1}")
    
    # Perform processing
    # ...
    
    # Update pipeline state
    # pipeline.raw = ...
    
    # Update metadata
    pipeline.metadata['processing_steps'].append({
        'step': 'my_custom_step',
        'parameters': {
            'param1': param1,
            'param2': param2
        }
    })
```

## Using Step Functions

Step functions are typically used within tasks:

```python
from autoclean.core.task import Task
from autoclean.step_functions import preprocessing, artifacts, ica

class MyTask(Task):
    def run(self, pipeline):
        # Sequential processing steps
        preprocessing.resample(pipeline)
        preprocessing.apply_bandpass_filter(pipeline)
        artifacts.detect_bad_channels(pipeline)
        ica.fit_ica(pipeline)
        ica.detect_artifact_components(pipeline)
        ica.apply_ica(pipeline)
```

Or can be called directly:

```python
from autoclean.core.pipeline import Pipeline
from autoclean.step_functions import preprocessing

pipeline = Pipeline()
pipeline.process_file(file_path="my_file.raw", task="rest_eyesopen")

# Apply additional processing
preprocessing.apply_notch_filter(pipeline, freqs=[50, 100])
```
