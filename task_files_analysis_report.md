# AutoClean EEG Task Files Analysis and Configuration Guide

## Executive Summary

This report analyzes the built-in task files in the AutoClean EEG pipeline and provides an exhaustive guide for creating and configuring task files. The analysis reveals both strengths in the modular design and several inconsistencies that could be addressed to improve maintainability and user experience.

## Current State Analysis

### Built-in Task Files Inventory

The following built-in task files were analyzed:

1. **RawToSet.py** - Simple RAW to SET file conversion task
2. **assr_default.py** - ASSR (Auditory Steady-State Response) preprocessing
3. **chirp_default.py** - Chirp stimulus preprocessing
4. **bb_long.py** (actual file: resting_eyes_open.py) - Resting state preprocessing
5. **hbcd_VEP.py** - HBCD Visual Evoked Potential preprocessing
6. **hbcd_mmn.py** - HBCD Mismatch Negativity preprocessing
7. **mouse_xdat_assr.py** - Mouse XDAT ASSR preprocessing
8. **mouse_xdat_chirp.py** - Mouse XDAT Chirp preprocessing (appears to be copy of resting code)
9. **mouse_xdat_resting.py** - Mouse XDAT Resting state preprocessing

### Critical Issues Identified

#### 1. File Naming Inconsistencies
- **bb_long.py** contains a class named `BB_Long` but the docstring and content suggest it's for "resting state EEG preprocessing"
- **mouse_xdat_chirp.py** file header comment says "Mouse XDAT Chirp Task" but the class is named `MouseXdatChirp` and the actual file path shown in line 1 is "mouse_xdat_resting.py"

#### 2. Class Naming Inconsistencies
- Some classes follow PascalCase: `AssrDefault`, `ChirpDefault`, `RawToSet`
- Others use ALL_CAPS with underscores: `BB_Long`, `HBCD_VEP`, `HBCD_MMN`
- Mouse classes use mixed patterns: `MouseXdatAssr`, `MouseXdatChirp`, `MouseXdatResting`

#### 3. Method Naming Inconsistencies
- Most tasks use `self.import_raw()` for data import
- Mouse tasks use `self.import_data()`
- Some use `self.run_basic_steps()`, others use `self.basic_steps()`

#### 4. Missing or Inconsistent Configuration Validation
- Most tasks have identical `_validate_task_config()` methods with minimal validation
- No consistent pattern for required configuration fields
- Comments suggest configuration is auto-generated but validation is still hardcoded

#### 5. Duplicated Code Patterns
- Similar epoching blocks across multiple files
- Repeated report generation methods with identical implementation
- Nearly identical bad channel cleaning methods in mouse tasks

#### 6. Documentation Inconsistencies
- Some classes have comprehensive docstrings (HBCD tasks)
- Others have minimal documentation (mouse tasks)
- Inconsistent parameter documentation styles

## Epoching Methods Analysis

The analysis reveals three main epoching approaches available in the system:

### 1. Regular Epochs (`create_regular_epochs`)
- **Purpose**: Creates fixed-length epochs at regular intervals
- **Use Case**: Resting-state data without specific event markers
- **Used in**: bb_long.py (resting state), mouse_xdat_resting.py
- **Key Parameters**:
  - `tmin`, `tmax`: Time bounds for epochs
  - `baseline`: Baseline correction window
  - `reject_by_annotation`: Whether to reject epochs with bad annotations

### 2. Event-based Epochs (`create_eventid_epochs`)
- **Purpose**: Creates epochs centered around specific event markers
- **Use Case**: Task-based paradigms with stimulus/response events
- **Used in**: assr_default.py, chirp_default.py, hbcd_VEP.py, hbcd_mmn.py, mouse_xdat_assr.py, mouse_xdat_chirp.py
- **Key Parameters**:
  - `event_id`: Dictionary mapping event names to event IDs
  - `tmin`, `tmax`: Time bounds relative to events
  - `baseline`: Baseline correction window
  - `keep_all_epochs`: Whether to keep all epochs or apply rejection

### 3. Statistical Learning Epochs (`create_sl_epochs`)
- **Purpose**: Creates syllable-based epochs for statistical learning paradigms
- **Use Case**: Statistical learning experiments with syllable sequences
- **Used in**: Available via mixin but not used in built-in tasks
- **Key Parameters**:
  - `num_syllables`: Number of syllables per epoch
  - `tmin`: Start time relative to syllable onset
  - `baseline`: Whether to apply baseline correction

## Comprehensive Configuration Guide

### Task File Structure (v2.0.0)

AutoClean v2.0.0 introduced Python task files with embedded configuration, eliminating the need for separate YAML files:

```python
# Task file structure
from autoclean.core.task import Task

# Embedded configuration dictionary
config = {
    # Configuration sections here
}

class YourTaskName(Task):
    def __init__(self, config: Dict[str, Any]):
        # Initialize instance variables
        self.raw = None
        self.epochs = None
        # Call parent initialization
        super().__init__(config)
    
    def run(self) -> None:
        # Processing pipeline implementation
        pass
    
    def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Task-specific validation
        return config
```

### Core Configuration Sections

#### 1. Dataset and Input Configuration
```python
config = {
    "dataset_name": "MyExperiment_2024",        # Optional: Organize output directories
    "input_path": "/path/to/data/",             # Optional: Default input path
    "move_flagged_files": False,                # Keep flagged files in output
}
```

#### 2. Signal Processing Parameters

##### Resampling
```python
"resample_step": {
    "enabled": True,
    "value": 250                    # Target sampling frequency (Hz)
}
```

##### Filtering
```python
"filtering": {
    "enabled": True,
    "value": {
        "l_freq": 1,                # High-pass filter (Hz)
        "h_freq": 100,              # Low-pass filter (Hz) 
        "notch_freqs": [60, 120],   # Notch filter frequencies
        "notch_widths": 5           # Notch filter width
    }
}
```

##### Channel Management
```python
"drop_outerlayer": {
    "enabled": True,
    "value": [1, 32, 125, 126, 127, 128]  # Channel indices to drop
},
"eog_step": {
    "enabled": True,
    "value": [1, 32, 8, 14, 17, 21, 25]   # EOG channel indices
}
```

##### Temporal Trimming
```python
"trim_step": {
    "enabled": True, 
    "value": 4                      # Seconds to trim from start/end
},
"crop_step": {
    "enabled": True,
    "value": {
        "start": 30,                # Start time (seconds)
        "end": 300                  # End time (seconds)
    }
}
```

##### Referencing and Montage
```python
"reference_step": {
    "enabled": True,
    "value": "average"              # Options: 'average', channel list, or None
},
"montage": {
    "enabled": True,
    "value": "GSN-HydroCel-129"     # EEG montage specification
}
```

#### 3. Artifact Removal and ICA

##### ICA Configuration
```python
"ICA": {
    "enabled": True,
    "value": {
        "method": "fastica",        # Options: 'fastica', 'infomax', 'picard'
        "n_components": None,       # Number of components (None = auto)
        "fit_params": {}           # Additional ICA parameters
    }
}
```

##### Component Rejection
```python
"component_rejection": {
    "enabled": True,
    "method": "iclabel",           # Options: 'iclabel', 'icvision'
    "value": {
        "ic_flags_to_reject": [    # Component types to reject
            "muscle", "heart", "eog", "ch_noise", "line_noise"
        ],
        "ic_rejection_threshold": 0.3  # Confidence threshold
    }
}
```

#### 4. Epoching Configuration

##### Event-based Epoching
```python
"epoch_settings": {
    "enabled": True,
    "value": {
        "tmin": -0.5,               # Epoch start (seconds)
        "tmax": 2.0,                # Epoch end (seconds)
    },
    "event_id": {                   # Event ID mapping (None = auto-detect)
        "target": 1,
        "standard": 2,
        "deviant": 3
    },
    "remove_baseline": {
        "enabled": True,
        "window": [None, 0]         # Baseline correction window
    },
    "threshold_rejection": {
        "enabled": True,
        "volt_threshold": {
            "eeg": 0.000125         # Voltage threshold (V)
        }
    }
}
```

##### Regular Epoching (Resting State)
```python
"epoch_settings": {
    "enabled": True,
    "type": "regular",             # Specify regular epoching
    "value": {
        "tmin": 0,                 # Epoch start
        "tmax": 4,                 # Epoch duration (4 seconds)
        "overlap": 0               # Overlap between epochs
    },
    "reject_by_annotation": True   # Reject epochs with bad annotations
}
```

##### Statistical Learning Epoching
```python
"epoch_settings": {
    "enabled": True,
    "type": "statistical_learning",
    "value": {
        "num_syllables": 30,       # Syllables per epoch
        "tmin": 0,                 # Start relative to first syllable
        "baseline": True           # Apply baseline correction
    }
}
```

#### 5. Analysis Configuration

##### Inter-Trial Coherence Analysis
```python
"itc_analysis": {
    "enabled": True,
    "value": {
        "n_cycles": 7.0,           # Wavelet cycles
        "baseline": [-0.5, -0.1],  # Baseline period
        "analyze_bands": True,     # Frequency band analysis
        "time_window": [1.0, 8.0], # Analysis time window
        "calculate_wli": True      # Word Learning Index
    }
}
```

#### 6. Output and Export Settings
```python
"output": {
    "save_stages": [               # Stages to export
        "raw",                     # Post-import raw data
        "epochs",                  # Final epochs
        "ica_analysis",            # ICA results
        "itc_analysis"             # ITC analysis results
    ],
    "export_format": "set",        # Options: 'set', 'fif', 'edf'
    "debug_plots": True            # Generate debug visualizations
}
```

### EEG System Configuration

```python
"eeg_system": {
    "montage": "GSN-HydroCel-129", # Standard montage options:
                                   # - "standard_1020"
                                   # - "GSN-HydroCel-129" 
                                   # - "GSN-HydroCel-65"
                                   # - "biosemi64"
                                   # - "biosemi128"
    "reference": "average"         # Reference options:
                                   # - "average"
                                   # - ["Cz"] (specific channels)
                                   # - None (no re-referencing)
}
```

### Task-Specific Settings Structure

The `tasks` section allows task-specific parameter overrides:

```python
"tasks": {
    "YourTaskName": {
        "settings": {
            # Task-specific overrides here
            "filtering": {
                "value": {
                    "l_freq": 0.5,     # Override global filter settings
                    "h_freq": 80
                }
            },
            "epoch_settings": {
                "value": {
                    "tmin": -1.0,      # Task-specific epoch timing
                    "tmax": 3.0
                }
            }
        }
    }
}
```

## Processing Pipeline Methods

### Available Mixin Methods

Tasks inherit from multiple mixins that provide processing methods:

#### Data Import and Basic Processing
- `import_raw()` - Import EEG data
- `resample_data()` - Resample to target frequency
- `filter_data()` - Apply filtering
- `run_basic_steps()` - Combined basic preprocessing

#### Channel Management
- `drop_outer_layer()` - Remove edge channels
- `assign_eog_channels()` - Set EOG channel types  
- `clean_bad_channels()` - Detect and interpolate bad channels
- `rereference_data()` - Apply re-referencing

#### Artifact Detection and Removal
- `annotate_noisy_epochs()` - Mark noisy time segments
- `annotate_uncorrelated_epochs()` - Mark uncorrelated segments
- `detect_dense_oscillatory_artifacts()` - Detect oscillatory artifacts
- `run_ica()` - Perform ICA decomposition
- `classify_ica_components()` - Classify and reject ICA components

#### Epoching Methods
- `create_regular_epochs()` - Fixed-length epochs
- `create_eventid_epochs()` - Event-based epochs
- `create_sl_epochs()` - Statistical learning epochs

#### Quality Control and Outlier Detection
- `detect_outlier_epochs()` - Identify outlier epochs
- `gfp_clean_epochs()` - Clean epochs using Global Field Power

#### Visualization and Reporting
- `generate_reports()` - Create quality control reports
- `plot_raw_vs_cleaned_overlay()` - Compare raw vs cleaned data
- `step_psd_topo_figure()` - Power spectral density topography
- `plot_ica_full()` - ICA component plots

## Recommended Best Practices

### 1. Naming Conventions
- **Class Names**: Use PascalCase (e.g., `MyTaskName`)
- **File Names**: Use snake_case matching the main class purpose
- **Method Names**: Use snake_case consistently

### 2. Code Structure
```python
class MyTask(Task):
    def __init__(self, config: Dict[str, Any]):
        # Initialize instance variables first
        self.raw = None
        self.epochs = None
        self.original_raw = None
        # Call parent initialization
        super().__init__(config)
    
    def run(self) -> None:
        # Data import
        self.import_raw()
        
        # Basic preprocessing  
        self.run_basic_steps()
        
        # Store original for comparison
        self.original_raw = self.raw.copy()
        
        # Create BIDS paths
        self.create_bids_path()
        
        # Advanced preprocessing
        # ... processing steps ...
        
        # Epoching
        self.create_eventid_epochs()  # or appropriate epoching method
        
        # Quality control
        self.detect_outlier_epochs()
        self.gfp_clean_epochs()
        
        # Generate reports
        self.generate_reports()
```

### 3. Configuration Validation
Implement robust validation in `_validate_task_config()`:

```python
def _validate_task_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    required_fields = {
        "task": str,
        "eeg_system": dict,  # More specific than str
        "epoch_settings": dict
    }
    
    for field, field_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(config[field], field_type):
            raise TypeError(f"Field {field} must be {field_type}")
    
    # Validate EEG system configuration
    if "montage" not in config["eeg_system"]:
        raise ValueError("EEG system must specify montage")
    
    return config
```

### 4. Error Handling
- Always check for `self.raw is not None` before processing
- Use appropriate exception types (`ValueError`, `FileNotFoundError`, etc.)
- Provide informative error messages

### 5. Documentation Standards
- Include comprehensive class docstrings
- Document all parameters in method docstrings
- Provide usage examples where appropriate

## Recommendations for Improvement

### Short-term Fixes
1. **Fix naming inconsistencies** - Standardize class and file names
2. **Consolidate duplicate code** - Create shared utility functions
3. **Standardize method signatures** - Use consistent parameter names
4. **Improve validation** - Add meaningful configuration validation

### Medium-term Enhancements
1. **Create task templates** - Provide standardized templates for different paradigms
2. **Configuration schema** - Implement JSON schema validation for configurations
3. **Automated testing** - Add unit tests for each built-in task
4. **Documentation generation** - Auto-generate configuration documentation

### Long-term Architecture Improvements
1. **Plugin system enhancement** - Better plugin discovery and validation
2. **Configuration inheritance** - Allow tasks to inherit from base configurations
3. **Runtime validation** - Validate configurations at runtime with helpful error messages
4. **Integration testing** - End-to-end testing of complete pipelines

## Conclusion

The AutoClean EEG pipeline demonstrates a solid modular architecture with powerful mixin-based functionality. However, the built-in task files show several consistency issues that could benefit from standardization. The embedded configuration approach in v2.0.0 provides good flexibility, but would benefit from better validation and documentation standards.

Addressing the identified inconsistencies and implementing the recommended improvements would significantly enhance the user experience and maintainability of the codebase while preserving the powerful and flexible architecture that makes AutoClean effective for EEG processing research.