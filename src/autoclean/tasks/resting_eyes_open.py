from typing import Any, Dict
from autoclean.core.task import Task

# =============================================================================
#                     RESTING STATE EEG PREPROCESSING CONFIGURATION
# =============================================================================
# This configuration controls how your resting state EEG data will be 
# automatically cleaned and processed. Each section handles a different aspect
# of the preprocessing pipeline.
#
# 🟢 enabled: True  = Apply this processing step
# 🔴 enabled: False = Skip this processing step
#
# 💡 TIP: A web-based configuration wizard is available to generate this
#         automatically - you shouldn't need to edit this manually!
# =============================================================================

config = {
    
    # -------------------------------------------------------------------------
    # 📊 DATA RESAMPLING
    # -------------------------------------------------------------------------
    # Reduces file size and speeds up processing by lowering the sampling rate.
    # Most EEG analysis doesn't need sampling rates above 250-500 Hz.
    'resample_step': {
        'enabled': True,
        'value': 250  # Target sampling rate in Hz
    },
    
    # -------------------------------------------------------------------------  
    # 🔧 FREQUENCY FILTERING
    # -------------------------------------------------------------------------
    # Removes unwanted frequencies: slow drifts, muscle noise, electrical interference
    'filtering': {
        'enabled': True,
        'value': {
            'l_freq': 1,                # High-pass filter: Remove below 1 Hz 
            'h_freq': 100,              # Low-pass filter: Remove above 100 Hz 
            'notch_freqs': [60, 120],   # Remove electrical line noise 
            'notch_widths': 5           # Width of notch filter in Hz
        }
        # 📋 Typical ranges: l_freq (0.5-2 Hz), h_freq (40-150 Hz)
        # 🌍 Line noise: 60 Hz (US/Canada), 50 Hz (Europe/Asia)
    },
    
    # -------------------------------------------------------------------------
    # 🗑️ CHANNEL MANAGEMENT  
    # -------------------------------------------------------------------------
    # Options for removing problematic electrode channels
    'drop_outerlayer': {
        'enabled': False,           # Manually remove specific channels
        'value': []                 # List of channel names to remove, e.g., ['E17', 'E38']
    },
    
    'eog_step': {
        'enabled': False,           # Remove EOG (eye) channels  
        'value': []                 # EOG channel names, e.g., ['EOG1', 'EOG2']
    },
    
    # -------------------------------------------------------------------------
    # ✂️ TIME TRIMMING & CROPPING
    # -------------------------------------------------------------------------
    # Remove data from beginning/end or extract specific time windows
    'trim_step': {
        'enabled': True,
        'value': 4                  # Remove first/last 4 seconds (artifacts from start/stop)
    },
    
    'crop_step': {
        'enabled': False,           # Extract specific time window
        'value': {
            'start': 0,             # Start time in seconds
            'end': 60               # End time in seconds (False = use all data)
        }
    },
    
    # -------------------------------------------------------------------------
    # 📍 REFERENCE ELECTRODE
    # -------------------------------------------------------------------------
    # Changes the electrical reference for all channels
    'reference_step': {
        'enabled': True,
        'value': "average"          # Reference type: "average", "REST", or channel name
    },
    # 📋 Options: "average" (most common), "REST" (advanced), "Cz" (specific channel)
    
    # -------------------------------------------------------------------------
    # 🎯 EEG CAP CONFIGURATION  
    # -------------------------------------------------------------------------
    # Tells the system the layout of electrodes on your EEG cap
    'montage': {
        'enabled': True,
        'value': "GSN-HydroCel-129" # Your EEG cap model
    },
    
    # -------------------------------------------------------------------------
    # 🧠 INDEPENDENT COMPONENT ANALYSIS (ICA)
    # -------------------------------------------------------------------------
    # Identifies and separates brain signals from artifacts (eye blinks, muscle, etc.)
    'ICA': {
        'enabled': True,
        'value': {
            'method': 'picard',         # ICA algorithm (we have found picard to be the most reliable)
            'n_components': None,       # Number of components (None = automatic)
            'fit_params': {
                'ortho': False,         # 🔬 Advanced: Orthogonal constraint
                'extended': True        # 🔬 Advanced: Extended ICA for sub/super-Gaussian sources
            }
        }
    },
    # 📋 Methods: 'picard' (recommended), 'fastica', 'infomax'
    
    # -------------------------------------------------------------------------
    # 🏷️ AUTOMATIC ARTIFACT CLASSIFICATION
    # -------------------------------------------------------------------------
    # Uses ICAlabel to automatically label ICA components as brain activity vs artifacts
    'ICLabel': {
        'enabled': True,
        'value': {
            # Types of artifacts to automatically remove:
            'ic_flags_to_reject': [
                'muscle',           # Muscle tension artifacts
                'heart',            # Heart rate artifacts (EKG)
                'eog',             # Eye movement artifacts  
                'ch_noise',        # Channel-specific noise
                'line_noise',      # Electrical interference
            ],
            'ic_rejection_threshold': 0.3  # Confidence threshold (0.3 = 30% confident it's an artifact)
        }
    },
    # 🎯 Threshold guide: 0.1 (very strict), 0.3 (balanced), 0.7 (conservative)
    
    # -------------------------------------------------------------------------
    # ⏱️ EPOCHING CONFIGURATION
    # -------------------------------------------------------------------------
    'epoch_settings': {
        'enabled': True,
        'value': {
            'tmin': -1,                 # Epoch start: 1 second before timepoint
            'tmax': 1                   # Epoch end: 1 second after timepoint
        },                              # 📏 Total epoch length: 2 seconds
        
        'event_id': None,               # No specific events (resting state = fixed length epochs)
        
        'remove_baseline': {
            'enabled': False,           # Baseline correction 
            'window': [None, 0]         # Baseline window 
        },
        
        'threshold_rejection': {
            'enabled': False,           # Simple voltage threshold rejection 
            'volt_threshold': {
                'eeg': 125e-6           # 125 microvolts threshold (if enabled)
            }
        }
    }
}

class RestingEyesOpen(Task):

    def run(self) -> None:
        # Import raw EEG data
        self.import_raw()
        
        # Basic preprocessing with optional export
        self.run_basic_steps(export=True)  # Export after basic steps
        
        # Create BIDS-compliant paths and filenames
        self.create_bids_path()
        
        # Channel cleaning
        self.clean_bad_channels(
            cleaning_method="interpolate", 
            reset_bads=True
        )
        
        # Re-referencing
        self.rereference_data()
        
        # Artifact detection
        self.annotate_noisy_epochs()
        self.annotate_uncorrelated_epochs()
        self.detect_dense_oscillatory_artifacts()
        
        # ICA processing with optional export
        self.run_ica(export=True)  # Export after ICA
        self.run_ICLabel()
        
        # Manual save for compatibility (can be removed once all mixins updated)
        from autoclean.io.export import save_raw_to_set
        save_raw_to_set(
            raw=self.raw,
            autoclean_dict=self.config,
            stage="post_clean_raw",
            flagged=self.flagged,
        )
        
        # Epoching with export
        self.create_regular_epochs(export=True)  # Export epochs
        
        # Prepare epochs for ICA
        self.prepare_epochs_for_ica()
        
        # Clean epochs using GFP with export
        self.gfp_clean_epochs(export=True)  # Export cleaned epochs
        
        # Generate visualization reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate quality control visualizations and reports."""
        if self.raw is None or self.original_raw is None:
            return
            
        # Plot raw vs cleaned overlay using mixin method
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)
        
        # Plot PSD topography using mixin method
        self.step_psd_topo_figure(self.original_raw, self.raw)
        
        # Additional report generation can be added here


    def __init__(self, config: Dict[str, Any]):
        """Initialize the resting state task with embedded settings.
        
        Args:
            config: Configuration dictionary from the pipeline.
        """
        # Use the embedded settings from the module-level config variable
        self.settings = globals()['config']
        
        # Initialize the base class
        super().__init__(config)