#!/usr/bin/env python
"""
Example: Advanced Configuration Options

This example demonstrates how to work with advanced configuration options in the
AutoClean pipeline, including:
1. Creating and using custom configuration files
2. Merging configuration layers
3. Using environment variables in configuration
4. Creating and using configuration profiles
5. Validating configurations
"""

import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoclean.core.pipeline import Pipeline
from autoclean.core.task import Task
from autoclean.utils.config import load_config, merge_configs, validate_config
from autoclean.step_functions.preprocessing import resample, apply_bandpass_filter, apply_notch_filter
from autoclean.step_functions.artifacts import detect_bad_channels
from autoclean.step_functions.ica import fit_ica, detect_artifact_components, apply_ica
from autoclean.step_functions.reports import generate_report


class CustomTask(Task):
    """A custom task with a specific configuration schema."""
    
    # Define schema for validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "preprocessing": {
                "type": "object",
                "properties": {
                    "resampling": {
                        "type": "object",
                        "properties": {
                            "sfreq": {"type": "number", "minimum": 100, "default": 250}
                        }
                    },
                    "filtering": {
                        "type": "object",
                        "properties": {
                            "l_freq": {"type": "number", "minimum": 0, "default": 1.0},
                            "h_freq": {"type": "number", "default": 40.0},
                            "method": {"type": "string", "enum": ["fir", "iir"], "default": "fir"},
                            "filter_length": {"type": "string", "default": "auto"},
                            "notch_freqs": {
                                "type": "array", 
                                "items": {"type": "number"},
                                "default": [50, 60]
                            }
                        }
                    }
                }
            },
            "artifact_detection": {
                "type": "object",
                "properties": {
                    "bad_channels": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string", 
                                "enum": ["correlation", "ransac", "pyprep"], 
                                "default": "correlation"
                            },
                            "threshold": {"type": "number", "default": 0.7}
                        }
                    }
                }
            },
            "ica": {
                "type": "object",
                "properties": {
                    "n_components": {"type": ["number", "string"], "default": 0.99},
                    "method": {"type": "string", "default": "fastica"},
                    "automatic_artifact_detection": {"type": "boolean", "default": True},
                    "eog_threshold": {"type": "number", "default": 3.0},
                    "ecg_threshold": {"type": "number", "default": 3.0}
                }
            },
            "reporting": {
                "type": "object",
                "properties": {
                    "generate_html": {"type": "boolean", "default": True},
                    "generate_pdf": {"type": "boolean", "default": False},
                    "include_psd": {"type": "boolean", "default": True},
                    "include_ica": {"type": "boolean", "default": True},
                    "include_evoked": {"type": "boolean", "default": False}
                }
            }
        }
    }
    
    def __init__(self, config=None, name="custom_task"):
        """Initialize the custom task."""
        super().__init__(config=config, name=name)
    
    def run(self, pipeline):
        """Run the custom processing flow.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        
        Returns
        -------
        Pipeline
            Updated pipeline instance
        """
        # Get configuration parameters
        config = self.config
        
        # Extract preprocessing settings
        preproc_config = config.get("preprocessing", {})
        resample_config = preproc_config.get("resampling", {})
        filter_config = preproc_config.get("filtering", {})
        
        sfreq = resample_config.get("sfreq", 250)
        l_freq = filter_config.get("l_freq", 1.0)
        h_freq = filter_config.get("h_freq", 40.0)
        filter_method = filter_config.get("method", "fir")
        filter_length = filter_config.get("filter_length", "auto")
        notch_freqs = filter_config.get("notch_freqs", [50, 60])
        
        # Extract artifact detection settings
        artifact_config = config.get("artifact_detection", {})
        bad_ch_config = artifact_config.get("bad_channels", {})
        bad_ch_method = bad_ch_config.get("method", "correlation")
        bad_ch_threshold = bad_ch_config.get("threshold", 0.7)
        
        # Extract ICA settings
        ica_config = config.get("ica", {})
        n_components = ica_config.get("n_components", 0.99)
        ica_method = ica_config.get("method", "fastica")
        auto_artifact = ica_config.get("automatic_artifact_detection", True)
        eog_threshold = ica_config.get("eog_threshold", 3.0)
        ecg_threshold = ica_config.get("ecg_threshold", 3.0)
        
        # Extract reporting settings
        report_config = config.get("reporting", {})
        generate_html = report_config.get("generate_html", True)
        generate_pdf = report_config.get("generate_pdf", False)
        include_psd = report_config.get("include_psd", True)
        include_ica = report_config.get("include_ica", True)
        include_evoked = report_config.get("include_evoked", False)
        
        # Log the configuration
        pipeline.logger.info(f"Running {self.name} with the following configuration:")
        pipeline.logger.info(f"  - Preprocessing:")
        pipeline.logger.info(f"    - Resampling: {sfreq} Hz")
        pipeline.logger.info(f"    - Filtering: {l_freq}-{h_freq} Hz ({filter_method})")
        pipeline.logger.info(f"    - Notch frequencies: {notch_freqs}")
        pipeline.logger.info(f"  - Artifact detection:")
        pipeline.logger.info(f"    - Bad channel method: {bad_ch_method} (threshold: {bad_ch_threshold})")
        pipeline.logger.info(f"  - ICA:")
        pipeline.logger.info(f"    - Components: {n_components}, Method: {ica_method}")
        pipeline.logger.info(f"    - Auto artifact detection: {auto_artifact}")
        pipeline.logger.info(f"  - Reporting:")
        pipeline.logger.info(f"    - HTML: {generate_html}, PDF: {generate_pdf}")
        
        # Validate the raw data
        if pipeline.raw is None:
            raise ValueError("Raw data must be loaded before running the task")
        
        # 1. Preprocessing steps
        pipeline.logger.info("Step 1: Preprocessing")
        
        # Resampling
        resample(pipeline, sfreq=sfreq)
        
        # Bandpass filtering
        apply_bandpass_filter(
            pipeline,
            l_freq=l_freq,
            h_freq=h_freq,
            method=filter_method,
            filter_length=filter_length
        )
        
        # Notch filtering for line noise
        if notch_freqs:
            apply_notch_filter(pipeline, freqs=notch_freqs)
        
        # 2. Artifact detection
        pipeline.logger.info("Step 2: Artifact detection")
        
        # Detect bad channels
        detect_bad_channels(
            pipeline,
            method=bad_ch_method,
            threshold=bad_ch_threshold
        )
        
        # 3. ICA processing
        pipeline.logger.info("Step 3: ICA processing")
        
        # Fit ICA
        fit_ica(
            pipeline,
            n_components=n_components,
            method=ica_method
        )
        
        # Detect artifacts components
        if auto_artifact:
            detect_artifact_components(
                pipeline,
                eog_threshold=eog_threshold,
                ecg_threshold=ecg_threshold
            )
        
        # Apply ICA to clean the data
        apply_ica(pipeline)
        
        # 4. Generate reports
        pipeline.logger.info("Step 4: Generating reports")
        
        if generate_html or generate_pdf:
            report_file = pipeline.get_output_path("report", extension="html" if generate_html else "pdf")
            
            generate_report(
                pipeline,
                report_type="html" if generate_html else "pdf",
                file_path=report_file,
                include_psd=include_psd,
                include_ica=include_ica,
                include_evoked=include_evoked
            )
        
        pipeline.logger.info(f"Completed {self.name}")
        return pipeline


def create_example_config_files(output_dir):
    """Create example configuration files for demonstration.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save the configuration files
    
    Returns
    -------
    dict
        Dictionary with paths to the created configuration files
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Default configuration
    default_config = {
        "preprocessing": {
            "resampling": {
                "sfreq": 250
            },
            "filtering": {
                "l_freq": 1.0,
                "h_freq": 40.0,
                "method": "fir",
                "filter_length": "auto",
                "notch_freqs": [50, 60]
            }
        },
        "artifact_detection": {
            "bad_channels": {
                "method": "correlation",
                "threshold": 0.7
            }
        },
        "ica": {
            "n_components": 0.99,
            "method": "fastica",
            "automatic_artifact_detection": True,
            "eog_threshold": 3.0,
            "ecg_threshold": 3.0
        },
        "reporting": {
            "generate_html": True,
            "generate_pdf": False,
            "include_psd": True,
            "include_ica": True,
            "include_evoked": False
        }
    }
    
    # 2. Development profile
    dev_config = {
        "preprocessing": {
            "resampling": {
                "sfreq": 100  # Lower for faster processing during development
            },
            "filtering": {
                "l_freq": 1.0,
                "h_freq": 30.0,  # Lower for faster processing
                "method": "iir",  # Faster method
                "notch_freqs": [50]  # Simplified
            }
        },
        "ica": {
            "n_components": 15,  # Fixed number for faster processing
            "automatic_artifact_detection": True
        },
        "reporting": {
            "generate_html": True,
            "generate_pdf": False,
            "include_psd": True,
            "include_ica": False,  # Simplified reporting
            "include_evoked": False
        }
    }
    
    # 3. Production profile
    prod_config = {
        "preprocessing": {
            "resampling": {
                "sfreq": 250  # Higher for better quality
            },
            "filtering": {
                "l_freq": 0.5,  # More aggressive filtering
                "h_freq": 45.0,
                "method": "fir",  # Better quality but slower
                "filter_length": "10s",  # Longer filter for better frequency response
                "notch_freqs": [50, 60, 100, 120]  # Include harmonics
            }
        },
        "artifact_detection": {
            "bad_channels": {
                "method": "pyprep",  # More sophisticated method
                "threshold": 0.8  # Stricter threshold
            }
        },
        "ica": {
            "n_components": 0.999,  # Retain more variance
            "method": "picard",  # Alternative method
            "automatic_artifact_detection": True,
            "eog_threshold": 2.5,  # More sensitive
            "ecg_threshold": 2.5
        },
        "reporting": {
            "generate_html": True,
            "generate_pdf": True,  # Generate both formats
            "include_psd": True,
            "include_ica": True,
            "include_evoked": True  # Include all components
        }
    }
    
    # 4. User-specific overrides (typically used for local development)
    user_config = {
        "preprocessing": {
            "filtering": {
                "notch_freqs": []  # Disable notch filtering for this user
            }
        },
        "reporting": {
            "generate_html": True,
            "generate_pdf": False
        }
    }
    
    # 5. Create configuration with environment variables
    env_config = {
        "preprocessing": {
            "resampling": {
                "sfreq": "${AUTOCLEAN_SAMPLING_RATE:250}"  # Use env var with default
            },
            "filtering": {
                "l_freq": "${AUTOCLEAN_LOW_FREQ:1.0}",
                "h_freq": "${AUTOCLEAN_HIGH_FREQ:40.0}"
            }
        },
        "ica": {
            "n_components": "${AUTOCLEAN_ICA_COMPONENTS:0.99}"
        }
    }
    
    # Save configurations to files
    config_files = {}
    
    default_path = output_dir / "default_config.yaml"
    with open(default_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    config_files['default'] = default_path
    
    dev_path = output_dir / "development_config.yaml"
    with open(dev_path, 'w') as f:
        yaml.dump(dev_config, f, default_flow_style=False)
    config_files['development'] = dev_path
    
    prod_path = output_dir / "production_config.yaml"
    with open(prod_path, 'w') as f:
        yaml.dump(prod_config, f, default_flow_style=False)
    config_files['production'] = prod_path
    
    user_path = output_dir / "user_config.yaml"
    with open(user_path, 'w') as f:
        yaml.dump(user_config, f, default_flow_style=False)
    config_files['user'] = user_path
    
    env_path = output_dir / "env_config.yaml"
    with open(env_path, 'w') as f:
        yaml.dump(env_config, f, default_flow_style=False)
    config_files['env'] = env_path
    
    # Create a .env file for environment variables
    env_vars = [
        "# AutoClean environment variables",
        "AUTOCLEAN_SAMPLING_RATE=200",
        "AUTOCLEAN_LOW_FREQ=0.5",
        "AUTOCLEAN_HIGH_FREQ=35",
        "AUTOCLEAN_ICA_COMPONENTS=20"
    ]
    
    env_file_path = output_dir / ".env"
    with open(env_file_path, 'w') as f:
        f.write('\n'.join(env_vars))
    
    return config_files


def create_simulated_data():
    """Create a simple simulated EEG data object for demonstration.
    
    Returns
    -------
    mne.io.Raw
        Simulated EEG data
    """
    from mne import create_info
    from mne.io import RawArray
    from mne.channels import make_standard_montage
    from mne.simulation import simulate_raw
    
    # Create simulated EEG data
    sfreq = 500  # Sampling frequency (Hz)
    duration = 300  # Duration (seconds)
    n_channels = 32  # Number of channels
    
    # Create channel names based on 10-20 system
    montage = make_standard_montage('standard_1020')
    ch_names = list(montage.ch_names)[:n_channels-1] + ['STI 014']
    
    # Create info object
    ch_types = ['eeg'] * (n_channels - 1) + ['stim']
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Generate random data
    n_samples = int(duration * sfreq)
    data = np.random.randn(n_channels, n_samples) * 1e-6  # Random noise in Volts
    
    # Set stim channel to zeros
    data[-1, :] = 0
    
    # Add alpha oscillations (8-12 Hz)
    t = np.arange(n_samples) / sfreq
    alpha = np.sin(2 * np.pi * 10 * t) * 1e-6
    # Apply to posterior channels (O1, O2, P3, P4, etc.)
    posterior_chs = [i for i, ch in enumerate(ch_names) if ch[0] in 'OP' and i < n_channels-1]
    for ch in posterior_chs:
        data[ch, :] += alpha
    
    # Add 50Hz line noise
    line_noise = np.sin(2 * np.pi * 50 * t) * 0.2 * 1e-6
    data[:-1, :] += line_noise  # Add to all EEG channels
    
    # Create RawArray object
    raw = RawArray(data, info)
    
    # Set channel locations
    raw.set_montage(montage)
    
    return raw


def demonstrate_config_merging(config_dir, output_dir):
    """Demonstrate how configuration merging works.
    
    Parameters
    ----------
    config_dir : Path
        Directory containing configuration files
    output_dir : Path
        Directory to save outputs
    """
    print("\n=== Demonstrating Configuration Merging ===")
    
    # Load the default configuration
    default_config = load_config(config_dir / "default_config.yaml")
    print("\nDefault Configuration:")
    print(yaml.dump(default_config, default_flow_style=False))
    
    # Load and merge with development profile
    dev_config = load_config(config_dir / "development_config.yaml")
    merged_dev = merge_configs(default_config, dev_config)
    print("\nMerged with Development Profile:")
    print(yaml.dump(merged_dev, default_flow_style=False))
    
    # Load and merge with user config on top of development profile
    user_config = load_config(config_dir / "user_config.yaml")
    merged_user_dev = merge_configs(merged_dev, user_config)
    print("\nMerged with User Config (over Development):")
    print(yaml.dump(merged_user_dev, default_flow_style=False))
    
    # Create visualization of the configuration merging
    plt.figure(figsize=(12, 8))
    
    # Define a simple tree structure for visualization
    def add_config_node(ax, label, x, y, width=0.2, height=0.1, color='skyblue'):
        ax.add_patch(plt.Rectangle((x - width/2, y - height/2), width, height, 
                                  color=color, alpha=0.7, ec='black'))
        ax.text(x, y, label, ha='center', va='center', fontweight='bold')
    
    def add_arrow(ax, x1, y1, x2, y2, color='black'):
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.03, head_length=0.03, 
                fc=color, ec=color, length_includes_head=True, alpha=0.6)
    
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add nodes for configurations
    add_config_node(ax, "Default Config", 0.5, 0.9, width=0.25, color='lightblue')
    add_config_node(ax, "Development Profile", 0.3, 0.7, width=0.25, color='lightgreen')
    add_config_node(ax, "Production Profile", 0.7, 0.7, width=0.25, color='salmon')
    add_config_node(ax, "User Config", 0.3, 0.5, width=0.25, color='plum')
    add_config_node(ax, "Environment Variables", 0.5, 0.3, width=0.3, color='khaki')
    add_config_node(ax, "Runtime Config", 0.7, 0.5, width=0.25, color='lightcoral')
    
    # Final merged config
    add_config_node(ax, "Final Merged Configuration", 0.5, 0.1, width=0.4, color='lightgray')
    
    # Add arrows
    add_arrow(ax, 0.5, 0.85, 0.5, 0.15)  # Default to Final
    add_arrow(ax, 0.3, 0.65, 0.45, 0.15)  # Dev to Final
    add_arrow(ax, 0.7, 0.65, 0.55, 0.15)  # Prod to Final
    add_arrow(ax, 0.3, 0.45, 0.45, 0.15)  # User to Final
    add_arrow(ax, 0.5, 0.25, 0.5, 0.15)   # Env to Final
    add_arrow(ax, 0.7, 0.45, 0.55, 0.15)  # Runtime to Final
    
    # Add precedence information
    plt.text(0.1, 0.05, "Precedence: Default < Profile < User < Environment < Runtime",
             fontsize=10, ha='left')
    
    # Save the figure
    plt.savefig(output_dir / "config_merging.png")
    plt.close()
    
    print(f"\nConfiguration merging visualization saved to {output_dir / 'config_merging.png'}")


def demonstrate_env_variables(config_dir, output_dir):
    """Demonstrate how environment variables work in configuration.
    
    Parameters
    ----------
    config_dir : Path
        Directory containing configuration files
    output_dir : Path
        Directory to save outputs
    """
    print("\n=== Demonstrating Environment Variables ===")
    
    # First, load the .env file to set environment variables
    load_dotenv(config_dir / ".env")
    
    # Load the configuration with env variables
    env_config = load_config(config_dir / "env_config.yaml")
    print("\nConfiguration with environment variables resolved:")
    print(yaml.dump(env_config, default_flow_style=False))
    
    # Show the actual environment variables
    print("\nEnvironment Variables:")
    for var in ['AUTOCLEAN_SAMPLING_RATE', 'AUTOCLEAN_LOW_FREQ', 'AUTOCLEAN_HIGH_FREQ', 'AUTOCLEAN_ICA_COMPONENTS']:
        print(f"  {var} = {os.environ.get(var, 'Not set')}")


def demonstrate_config_validation(config_dir):
    """Demonstrate configuration validation.
    
    Parameters
    ----------
    config_dir : Path
        Directory containing configuration files
    """
    print("\n=== Demonstrating Configuration Validation ===")
    
    # Load a valid configuration
    valid_config = load_config(config_dir / "default_config.yaml")
    
    # Create an invalid configuration (with invalid values)
    invalid_config = valid_config.copy()
    invalid_config['preprocessing']['resampling']['sfreq'] = 50  # Below minimum of 100
    invalid_config['preprocessing']['filtering']['method'] = 'invalid_method'  # Not in enum
    
    # Validate the configurations against the schema
    try:
        validate_config(valid_config, CustomTask.CONFIG_SCHEMA)
        print("Valid configuration passed validation.")
    except Exception as e:
        print(f"Unexpected error validating valid configuration: {e}")
    
    # Try validating the invalid configuration
    try:
        validate_config(invalid_config, CustomTask.CONFIG_SCHEMA)
        print("Invalid configuration unexpectedly passed validation.")
    except Exception as e:
        print(f"Invalid configuration correctly failed validation with error: {e}")


def main():
    """Main function demonstrating advanced configuration options."""
    # Create output directories
    output_dir = Path("./config_examples")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True, parents=True)
    
    # Create example configuration files
    print("Creating example configuration files...")
    config_files = create_example_config_files(config_dir)
    
    # Demonstrate configuration merging
    demonstrate_config_merging(config_dir, output_dir)
    
    # Demonstrate environment variables
    demonstrate_env_variables(config_dir, output_dir)
    
    # Demonstrate configuration validation
    demonstrate_config_validation(config_dir)
    
    # Now use a configuration with an actual pipeline
    print("\n=== Demonstrating Custom Task with Configuration ===")
    
    # Create a pipeline with the development configuration
    pipeline = Pipeline(verbose=True)
    
    # Load development configuration
    dev_config = load_config(config_files['development'])
    
    # Override with user configuration
    user_config = load_config(config_files['user'])
    merged_config = merge_configs(dev_config, user_config)
    
    # Create a task with this configuration
    task = CustomTask(config=merged_config)
    
    # Create simulated data
    raw = create_simulated_data()
    pipeline.raw = raw
    
    # Print summary of the raw data
    print(f"\nLoaded simulated EEG data:")
    print(f"  - {len(pipeline.raw.ch_names)} channels")
    print(f"  - {pipeline.raw.info['sfreq']} Hz sampling rate")
    print(f"  - {len(pipeline.raw.times) / pipeline.raw.info['sfreq']:.1f} seconds duration")
    
    try:
        # Run the task
        print("\nRunning custom task with merged configuration...")
        pipeline = task.run(pipeline)
        print("Processing completed successfully!")
        
        # Generate a report
        report_file = output_dir / "custom_task_report.html"
        generate_report(
            pipeline,
            report_type='html',
            file_path=report_file,
            include_psd=True,
            include_ica=True
        )
        print(f"Report generated at {report_file}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAdvanced configuration demonstration completed.")


if __name__ == "__main__":
    main() 