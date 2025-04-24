#!/usr/bin/env python
"""
Example: Creating and Using Custom Tasks

This example demonstrates how to create a custom task for processing
ERP (Event-Related Potential) data with the AutoClean pipeline.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoclean.core.task import Task
from autoclean.core.pipeline import Pipeline
from autoclean.step_functions.preprocessing import resample, apply_bandpass_filter, apply_notch_filter
from autoclean.step_functions.artifacts import detect_bad_channels
from autoclean.step_functions.ica import fit_ica, detect_artifact_components, apply_ica


class ERPTask(Task):
    """Custom task for processing Event-Related Potential (ERP) data.
    
    This task implements a standard ERP processing workflow for cognitive
    EEG experiments.
    """
    
    # Define the configuration schema for this task
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "sampling_rate": {"type": "number", "minimum": 100, "default": 250},
            "filtering": {
                "type": "object",
                "properties": {
                    "l_freq": {"type": "number", "minimum": 0, "default": 0.1},
                    "h_freq": {"type": "number", "minimum": 1, "default": 40},
                    "method": {"type": "string", "enum": ["fir", "iir"], "default": "fir"}
                }
            },
            "epochs": {
                "type": "object",
                "properties": {
                    "tmin": {"type": "number", "default": -0.2},
                    "tmax": {"type": "number", "default": 0.8},
                    "baseline": {
                        "type": "array",
                        "items": {"type": ["number", "null"]},
                        "minItems": 2,
                        "maxItems": 2,
                        "default": [null, 0]
                    },
                    "event_id": {"type": "object", "default": {}}
                }
            },
            "ica": {
                "type": "object",
                "properties": {
                    "n_components": {"type": ["number", "string"], "default": 0.99},
                    "method": {"type": "string", "default": "fastica"},
                    "random_state": {"type": "integer", "default": 42}
                }
            },
            "artifacts": {
                "type": "object",
                "properties": {
                    "amplitude_threshold_uv": {"type": "number", "default": 100},
                    "bad_channel_detection": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "enum": ["correlation", "pyprep", "ransac"], "default": "correlation"},
                            "threshold": {"type": "number", "default": 0.7}
                        }
                    }
                }
            }
        },
        "required": ["sampling_rate", "epochs"]
    }
    
    def __init__(self, config=None, name="erp_task"):
        """Initialize the ERP task.
        
        Parameters
        ----------
        config : dict, optional
            Configuration for the task
        name : str, default="erp_task"
            Name of the task
        """
        super().__init__(config=config, name=name)
    
    def run(self, pipeline):
        """Execute the ERP processing workflow.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        
        Returns
        -------
        Pipeline
            The pipeline instance with processed data
        """
        # Log task start
        pipeline.logger.info(f"Starting {self.name} with configuration: {self.config}")
        
        # Get configuration parameters
        config = self.config
        sfreq = config.get("sampling_rate", 250)
        filter_config = config.get("filtering", {})
        l_freq = filter_config.get("l_freq", 0.1)
        h_freq = filter_config.get("h_freq", 40)
        filter_method = filter_config.get("method", "fir")
        
        epochs_config = config.get("epochs", {})
        tmin = epochs_config.get("tmin", -0.2)
        tmax = epochs_config.get("tmax", 0.8)
        baseline = epochs_config.get("baseline", (None, 0))
        event_id = epochs_config.get("event_id", {})
        
        ica_config = config.get("ica", {})
        n_components = ica_config.get("n_components", 0.99)
        ica_method = ica_config.get("method", "fastica")
        random_state = ica_config.get("random_state", 42)
        
        artifacts_config = config.get("artifacts", {})
        amplitude_threshold = artifacts_config.get("amplitude_threshold_uv", 100)
        bad_ch_config = artifacts_config.get("bad_channel_detection", {})
        bad_ch_method = bad_ch_config.get("method", "correlation")
        bad_ch_threshold = bad_ch_config.get("threshold", 0.7)
        
        # Input validation
        if pipeline.raw is None:
            raise ValueError("Raw data must be loaded before running the ERP task")
        
        # Step 1: Preprocessing
        pipeline.logger.info("Step 1: Preprocessing")
        
        # Resample to target sampling rate
        resample(pipeline, sfreq=sfreq)
        
        # Apply bandpass filter
        apply_bandpass_filter(
            pipeline, 
            l_freq=l_freq, 
            h_freq=h_freq, 
            method=filter_method
        )
        
        # Apply notch filter for line noise (detect automatically)
        if pipeline.raw.info['sfreq'] > 100:  # Only if sampling rate is high enough
            line_freqs = self._detect_line_noise(pipeline)
            if line_freqs:
                apply_notch_filter(pipeline, freqs=line_freqs)
        
        # Step 2: Artifact detection and rejection
        pipeline.logger.info("Step 2: Artifact detection and rejection")
        
        # Detect bad channels
        detect_bad_channels(
            pipeline, 
            method=bad_ch_method, 
            threshold=bad_ch_threshold
        )
        
        # Step 3: ICA for artifact removal
        pipeline.logger.info("Step 3: ICA for artifact removal")
        
        # Fit ICA
        fit_ica(
            pipeline, 
            n_components=n_components, 
            method=ica_method, 
            random_state=random_state
        )
        
        # Detect artifact components
        detect_artifact_components(pipeline)
        
        # Apply ICA to remove artifacts
        apply_ica(pipeline)
        
        # Step 4: Epoching
        pipeline.logger.info("Step 4: Epoching")
        
        # Extract events if not already available
        if not hasattr(pipeline, 'events') or pipeline.events is None:
            pipeline.events = self._extract_events(pipeline)
        
        # Create epochs
        from autoclean.step_functions.epochs import create_epochs
        create_epochs(
            pipeline,
            events=pipeline.events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject={'eeg': amplitude_threshold * 1e-6},  # Convert to volts
            preload=True
        )
        
        # Step 5: Averaging
        pipeline.logger.info("Step 5: Creating evoked responses")
        
        # Create evoked responses for each condition
        from autoclean.step_functions.epochs import create_evoked
        
        # For each condition, create a separate evoked response
        for condition in event_id.keys():
            create_evoked(pipeline, condition=condition)
        
        # Also create grand average
        create_evoked(pipeline)
        
        # Step 6: Generate reports
        pipeline.logger.info("Step 6: Generating reports")
        
        # Generate HTML report
        from autoclean.step_functions.reports import generate_report
        report_file = pipeline.get_output_path('report', extension='html')
        generate_report(
            pipeline, 
            report_type='html', 
            file_path=report_file,
            include_psd=True,
            include_ica=True,
            include_evoked=True
        )
        
        # Step 7: Save results
        pipeline.logger.info("Step 7: Saving results")
        
        # Save processed epochs
        epochs_file = pipeline.get_output_path('epochs', extension='fif')
        pipeline.epochs.save(epochs_file, overwrite=True)
        
        # Save evoked data
        evoked_file = pipeline.get_output_path('evoked', extension='fif')
        pipeline.evoked.save(evoked_file, overwrite=True)
        
        # Log task completion
        pipeline.logger.info(f"Completed {self.name}")
        
        return pipeline
    
    def _detect_line_noise(self, pipeline):
        """Detect power line noise frequencies in the data.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        
        Returns
        -------
        list
            List of detected line noise frequencies
        """
        # Calculate power spectral density
        from mne.time_frequency import psd_welch
        psds, freqs = psd_welch(pipeline.raw, fmin=0, fmax=pipeline.raw.info['sfreq']/2 - 1, n_fft=8192)
        
        # Average across channels
        avg_psd = np.mean(psds, axis=0)
        
        # Find peaks in common line noise frequencies
        possible_freqs = [50, 60]  # Common power line frequencies
        line_freqs = []
        
        for freq in possible_freqs:
            # Find frequencies around the target
            freq_idx = np.argmin(np.abs(freqs - freq))
            window = slice(max(0, freq_idx - 3), min(len(freqs), freq_idx + 4))
            
            # Calculate the average power in this window
            window_power = np.mean(avg_psd[window])
            
            # Calculate the average power in surrounding windows
            left_window = slice(max(0, freq_idx - 10), max(0, freq_idx - 5))
            right_window = slice(min(len(freqs), freq_idx + 5), min(len(freqs), freq_idx + 10))
            
            surrounding_power = np.mean(np.concatenate([avg_psd[left_window], avg_psd[right_window]]))
            
            # If the power at the line frequency is much higher than surroundings, it's likely line noise
            if window_power > surrounding_power * 1.5:
                line_freqs.append(freq)
                # Also add harmonics
                for harmonic in range(2, 4):
                    line_freqs.append(freq * harmonic)
        
        pipeline.logger.info(f"Detected line noise frequencies: {line_freqs}")
        return line_freqs
    
    def _extract_events(self, pipeline):
        """Extract events from the raw data.
        
        Parameters
        ----------
        pipeline : Pipeline
            The pipeline instance
        
        Returns
        -------
        numpy.ndarray
            Events array in MNE format
        """
        from mne.event import find_events
        
        # Try to find a stim channel
        stim_channel = None
        for ch in pipeline.raw.ch_names:
            if 'stim' in ch.lower() or 'trigger' in ch.lower() or 'event' in ch.lower():
                stim_channel = ch
                break
        
        # If found, extract events
        if stim_channel:
            events = find_events(pipeline.raw, stim_channel=stim_channel)
            pipeline.logger.info(f"Extracted {len(events)} events from channel {stim_channel}")
            return events
        else:
            # Try the default find_events approach
            try:
                events = find_events(pipeline.raw)
                pipeline.logger.info(f"Extracted {len(events)} events using default settings")
                return events
            except Exception as e:
                pipeline.logger.warning(f"Could not extract events: {str(e)}")
                # Create artificial events for demonstration if none found
                n_events = 20
                events = np.zeros((n_events, 3), dtype=int)
                events[:, 0] = np.linspace(
                    0, len(pipeline.raw.times) - 1, n_events, dtype=int
                )
                events[:, 2] = 1  # All events have the same ID
                pipeline.logger.warning(f"Created {n_events} artificial events for demonstration")
                return events


# Main function to demonstrate usage
def main():
    # Path to an example EEG file - replace with a real file path
    input_file = "path/to/your/eeg_file.raw"
    
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Configuration for the ERP task
    erp_config = {
        "sampling_rate": 250,
        "filtering": {
            "l_freq": 0.1,
            "h_freq": 40,
            "method": "fir"
        },
        "epochs": {
            "tmin": -0.2,
            "tmax": 0.8,
            "baseline": [None, 0],
            "event_id": {"stimulus": 1, "target": 2, "novel": 3}
        },
        "ica": {
            "n_components": 0.95,
            "method": "fastica",
            "random_state": 42
        },
        "artifacts": {
            "amplitude_threshold_uv": 100,
            "bad_channel_detection": {
                "method": "correlation",
                "threshold": 0.7
            }
        }
    }
    
    # Create the custom task
    erp_task = ERPTask(config=erp_config)
    
    # Create the pipeline
    pipeline = Pipeline()
    
    try:
        # Load the data (if the file doesn't exist, it will simulate data)
        try:
            pipeline.raw = pipeline.load_raw(input_file)
            pipeline.logger.info(f"Loaded EEG data from {input_file}")
        except FileNotFoundError:
            # If file doesn't exist, create simulated data for demonstration
            pipeline.logger.info("File not found. Creating simulated data for demonstration.")
            from mne.simulation import simulate_raw
            from mne import create_info
            import numpy as np
            
            # Create simulated ERP data
            sfreq = 500  # Sampling frequency (Hz)
            duration = 300  # Duration (seconds)
            n_channels = 32  # Number of channels
            
            # Create channel names based on 10-20 system
            ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 
                        'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'FC1', 'FC2', 'CP1', 'CP2', 
                        'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POz', 'STIM'][:n_channels]
            
            # Create info object
            ch_types = ['eeg'] * (n_channels - 1) + ['stim']
            info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            
            # Generate random data
            n_samples = int(duration * sfreq)
            data = np.random.randn(n_channels, n_samples) * 10  # 10µV noise level
            
            # Set stim channel to zeros
            data[-1, :] = 0
            
            # Create events: 100 events distributed throughout the dataset
            n_events = 100
            event_times = np.linspace(int(sfreq * 10), n_samples - int(sfreq * 10), n_events, dtype=int)
            event_types = np.random.choice([1, 2, 3], size=n_events, p=[0.6, 0.2, 0.2])  # 60% standard, 20% target, 20% novel
            
            # Add events to stim channel
            for time, event_type in zip(event_times, event_types):
                data[-1, time] = event_type
            
            # Add ERP responses (simplified)
            for i, (time, event_type) in enumerate(zip(event_times, event_types)):
                # Skip some events randomly to simulate missed trials
                if np.random.random() < 0.1:
                    continue
                    
                # Create a template ERP waveform based on event type
                erp_duration_samples = int(0.5 * sfreq)  # 500ms ERP
                t = np.arange(erp_duration_samples) / sfreq
                
                if event_type == 1:  # Standard stimulus - small response
                    erp = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 0.1) * 5  # 5µV P300
                elif event_type == 2:  # Target stimulus - larger P300
                    erp = np.sin(2 * np.pi * 3 * t) * np.exp(-t / 0.15) * 15  # 15µV P300
                else:  # Novel stimulus - different pattern
                    erp = np.sin(2 * np.pi * 8 * t) * np.exp(-t / 0.12) * 10  # 10µV response
                
                # Add to central channels more strongly
                central_channels = [8, 9, 10, 14, 15, 16]  # C3, Cz, C4, P3, Pz, P4
                end_idx = min(time + erp_duration_samples, n_samples)
                erp_to_add = erp[:end_idx - time]
                
                for ch in central_channels:
                    data[ch, time:end_idx] += erp_to_add
                
                # Add with lower amplitude to other channels
                other_channels = [i for i in range(n_channels-1) if i not in central_channels]
                for ch in other_channels:
                    data[ch, time:end_idx] += erp_to_add * 0.3  # 30% amplitude
            
            # Add 50Hz line noise
            t = np.arange(n_samples) / sfreq
            line_noise = np.sin(2 * np.pi * 50 * t) * 2  # 2µV amplitude
            data[:-1] += line_noise  # Add to all EEG channels
            
            # Create raw object
            pipeline.raw = simulate_raw(info, data, verbose=False)
        
        # Run the ERP task
        pipeline = erp_task.run(pipeline)
        
        # Print information about the processed data
        print(f"ERP processing completed successfully:")
        print(f"  - Epochs: {len(pipeline.epochs)} ({len(pipeline.epochs.drop_log) - len(pipeline.epochs)} rejected)")
        
        if hasattr(pipeline, 'evoked'):
            if isinstance(pipeline.evoked, list):
                print(f"  - Evoked responses: {len(pipeline.evoked)}")
                for evk in pipeline.evoked:
                    print(f"    * {evk.comment if evk.comment else 'Average'}: {len(evk.times)}ms, {len(evk.info['ch_names'])} channels")
            else:
                print(f"  - Evoked response: {len(pipeline.evoked.times)}ms, {len(pipeline.evoked.info['ch_names'])} channels")
        
        # Plot the results
        if hasattr(pipeline, 'evoked') and pipeline.evoked is not None:
            # Plot evoked responses
            if isinstance(pipeline.evoked, list) and len(pipeline.evoked) > 0:
                fig = pipeline.evoked[0].plot(show=False)
                plt.savefig(output_dir / "evoked_response.png")
                print(f"Evoked response plot saved to: {output_dir / 'evoked_response.png'}")
                
                # Plot all conditions together
                from mne.viz import plot_compare_evokeds
                all_evokeds = {evk.comment if evk.comment else 'Average': evk for evk in pipeline.evoked}
                fig = plot_compare_evokeds(all_evokeds, picks='eeg', show=False)
                plt.savefig(output_dir / "compare_evokeds.png")
                print(f"Comparison plot saved to: {output_dir / 'compare_evokeds.png'}")
            elif not isinstance(pipeline.evoked, list):
                fig = pipeline.evoked.plot(show=False)
                plt.savefig(output_dir / "evoked_response.png")
                print(f"Evoked response plot saved to: {output_dir / 'evoked_response.png'}")
        
        # Show plots if not in a headless environment
        plt.show()
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 