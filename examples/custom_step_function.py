#!/usr/bin/env python
"""
Example: Creating and Using Custom Step Functions

This example demonstrates how to create custom step functions to extend
the AutoClean pipeline's functionality.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoclean.core.pipeline import Pipeline
from autoclean.step_functions.preprocessing import resample, apply_bandpass_filter


# Define a custom step function for artifact detection based on amplitude
def detect_high_amplitude_artifacts(pipeline, threshold=100, min_duration=0.1, window_size=0.2):
    """Detect artifacts with amplitude exceeding a threshold.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance containing the raw data
    threshold : float, default=100
        Amplitude threshold in µV
    min_duration : float, default=0.1
        Minimum duration to mark as artifact (seconds)
    window_size : float, default=0.2
        Time window to analyze (seconds)
    
    Returns
    -------
    None
        Adds annotations to pipeline.raw
    """
    # Log the operation
    pipeline.logger.info(f"Running detect_high_amplitude_artifacts with threshold={threshold}µV")
    
    # Input validation
    if pipeline.raw is None:
        raise ValueError("Raw data must be loaded before detecting high amplitude artifacts")
    
    # Get data and parameters
    data = pipeline.raw.get_data()
    sfreq = pipeline.raw.info['sfreq']
    window_samples = int(window_size * sfreq)
    min_samples = int(min_duration * sfreq)
    
    # Calculate RMS amplitude in windows
    n_windows = len(data[0]) // window_samples
    artifact_segments = []
    
    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples
        
        # Calculate max peak-to-peak amplitude across channels
        chunk = data[:, start_idx:end_idx]
        p2p_amplitudes = np.ptp(chunk, axis=1)
        max_p2p = np.max(p2p_amplitudes)
        
        # Check if amplitude exceeds threshold
        if max_p2p > threshold:
            # Get time in seconds
            onset = start_idx / sfreq
            duration = window_size
            
            # Add to list of artifact segments
            artifact_segments.append((onset, duration))
    
    # Merge adjacent segments
    merged_segments = []
    if artifact_segments:
        current_onset, current_duration = artifact_segments[0]
        
        for onset, duration in artifact_segments[1:]:
            # If this segment starts right after the current one
            if onset <= current_onset + current_duration + 1/sfreq:
                # Extend the current segment
                current_duration = (onset + duration) - current_onset
            else:
                # Add the current segment to the list if it's long enough
                if current_duration >= min_duration:
                    merged_segments.append((current_onset, current_duration))
                
                # Start a new segment
                current_onset, current_duration = onset, duration
        
        # Add the last segment if it's long enough
        if current_duration >= min_duration:
            merged_segments.append((current_onset, current_duration))
    
    # Add annotations to the raw object
    for onset, duration in merged_segments:
        pipeline.raw.annotations.append(
            onset=onset,
            duration=duration,
            description='high_amplitude'
        )
    
    # Log the results
    pipeline.logger.info(f"Detected {len(merged_segments)} high amplitude artifact segments")
    
    # Update pipeline metadata
    pipeline.metadata['artifacts'] = pipeline.metadata.get('artifacts', {})
    pipeline.metadata['artifacts']['high_amplitude'] = {
        'count': len(merged_segments),
        'threshold': threshold,
        'min_duration': min_duration,
        'timestamp': str(datetime.datetime.now())
    }


# Define a custom step function for filtering out specific frequency bands
def apply_notch_comb_filter(pipeline, base_freq=60, n_harmonics=5, width=1.0):
    """Apply a comb filter to remove a fundamental frequency and its harmonics.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance containing the raw data
    base_freq : float, default=60
        Fundamental frequency to remove (Hz)
    n_harmonics : int, default=5
        Number of harmonics to filter (including the fundamental)
    width : float, default=1.0
        Width of each notch (Hz)
    
    Returns
    -------
    None
        Modifies pipeline.raw in-place
    """
    # Log the operation
    pipeline.logger.info(f"Applying comb filter for {base_freq}Hz with {n_harmonics} harmonics")
    
    # Input validation
    if pipeline.raw is None:
        raise ValueError("Raw data must be loaded before applying comb filter")
    
    # Calculate frequencies to filter
    freqs = [base_freq * (i + 1) for i in range(n_harmonics)]
    
    # Apply a notch filter for each frequency
    for freq in freqs:
        pipeline.logger.info(f"Applying notch filter at {freq}Hz")
        pipeline.raw.notch_filter(
            freqs=freq, 
            notch_widths=width,
            verbose=False
        )
    
    # Update pipeline metadata
    pipeline.metadata['processing_steps'] = pipeline.metadata.get('processing_steps', [])
    pipeline.metadata['processing_steps'].append({
        'step': 'apply_notch_comb_filter',
        'parameters': {
            'base_freq': base_freq,
            'n_harmonics': n_harmonics,
            'width': width,
            'freqs': freqs
        },
        'timestamp': str(datetime.datetime.now())
    })
    
    pipeline.logger.info(f"Successfully applied comb filter")


# Define a custom visualization step function
def plot_power_in_bands(pipeline, bands=None, output_file=None):
    """Plot the power in different frequency bands.
    
    Parameters
    ----------
    pipeline : Pipeline
        The pipeline instance containing the raw data
    bands : dict, optional
        Dictionary of frequency bands. If None, uses default bands:
        {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 
         'beta': (13, 30), 'gamma': (30, 45)}
    output_file : str or Path, optional
        Path to save the plot. If None, shows the plot interactively.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Log the operation
    pipeline.logger.info("Plotting power in frequency bands")
    
    # Input validation
    if pipeline.raw is None:
        raise ValueError("Raw data must be loaded before plotting power bands")
    
    # Define default frequency bands if not provided
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    # Calculate PSD
    from mne.time_frequency import psd_welch
    psds, freqs = psd_welch(pipeline.raw, fmin=0, fmax=50, n_fft=2048)
    
    # Convert to dB
    psds_db = 10 * np.log10(psds)
    
    # Calculate average power in each band
    band_power = {}
    for band_name, (fmin, fmax) in bands.items():
        # Find frequencies within the band
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power[band_name] = np.mean(psds_db[:, idx], axis=1)
    
    # Create a figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot the average PSD
    avg_psd = np.mean(psds_db, axis=0)
    std_psd = np.std(psds_db, axis=0)
    axes[0].plot(freqs, avg_psd, 'b-')
    axes[0].fill_between(freqs, avg_psd - std_psd, avg_psd + std_psd, color='b', alpha=0.2)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power (dB)')
    axes[0].set_title('Power Spectral Density')
    
    # Plot power in bands
    band_names = list(bands.keys())
    band_means = [np.mean(band_power[band]) for band in band_names]
    band_stds = [np.std(band_power[band]) for band in band_names]
    
    axes[1].bar(band_names, band_means, yerr=band_stds, capsize=5)
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('Power (dB)')
    axes[1].set_title('Average Power in Frequency Bands')
    
    plt.tight_layout()
    
    # Save or show the figure
    if output_file:
        plt.savefig(output_file)
        pipeline.logger.info(f"Saved power band plot to {output_file}")
    
    # Update pipeline metadata
    pipeline.metadata['processing_steps'] = pipeline.metadata.get('processing_steps', [])
    pipeline.metadata['processing_steps'].append({
        'step': 'plot_power_in_bands',
        'parameters': {
            'bands': bands,
            'output_file': str(output_file) if output_file else None
        },
        'timestamp': str(datetime.datetime.now())
    })
    
    return fig


# Main function to demonstrate usage
def main():
    # Path to an example EEG file - replace with a real file path
    input_file = "path/to/your/eeg_file.raw"
    
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
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
            
            # Create simulated EEG data with 60Hz line noise
            sfreq = 500  # Sampling frequency (Hz)
            duration = 60  # Duration (seconds)
            n_channels = 19  # Number of channels
            
            # Create channel names based on 10-20 system
            ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 
                        'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'][:n_channels]
            
            # Create info object
            info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            
            # Generate random data
            n_samples = int(duration * sfreq)
            data = np.random.randn(n_channels, n_samples) * 20  # 20µV noise level
            
            # Add 10Hz alpha oscillations
            t = np.arange(n_samples) / sfreq
            alpha = np.sin(2 * np.pi * 10 * t) * 15  # 15µV amplitude
            data[2:10, :] += alpha  # Add to channels F3, Fz, F4, T3, C3, Cz, C4, T4
            
            # Add 60Hz line noise
            line_noise = np.sin(2 * np.pi * 60 * t) * 5  # 5µV amplitude
            data += line_noise  # Add to all channels
            
            # Add some high amplitude artifacts
            artifact_samples = np.random.choice(n_samples, 5)  # 5 random artifact positions
            for sample in artifact_samples:
                if sample < n_samples - int(0.2 * sfreq):  # Ensure artifact fits within data
                    data[:, sample:sample+int(0.2*sfreq)] += np.random.randn(n_channels, int(0.2*sfreq)) * 100
            
            # Create raw object
            pipeline.raw = simulate_raw(info, data, verbose=False)
        
        # Apply standard preprocessing
        resample(pipeline, sfreq=250)
        apply_bandpass_filter(pipeline, l_freq=1, h_freq=45)
        
        # Apply our custom comb filter to remove 60Hz line noise and harmonics
        apply_notch_comb_filter(pipeline, base_freq=60, n_harmonics=3)
        
        # Detect high amplitude artifacts
        detect_high_amplitude_artifacts(pipeline, threshold=80)
        
        # Plot power in frequency bands
        fig = plot_power_in_bands(pipeline, output_file=output_dir / "power_bands.png")
        
        # Display information about the processed data
        print(f"Data processed successfully:")
        print(f"  - Sampling rate: {pipeline.raw.info['sfreq']} Hz")
        print(f"  - Duration: {pipeline.raw.times[-1]:.1f} seconds")
        print(f"  - Channels: {len(pipeline.raw.ch_names)}")
        print(f"  - Annotations: {len(pipeline.raw.annotations)}")
        
        # Save processed data
        processed_file = output_dir / "processed_eeg.fif"
        pipeline.raw.save(processed_file, overwrite=True)
        print(f"Processed data saved to: {processed_file}")
        
        # Show the plot if not in a headless environment
        plt.show()
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")


if __name__ == "__main__":
    main() 