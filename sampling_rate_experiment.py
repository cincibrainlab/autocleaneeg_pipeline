"""
Sampling Rate Experiment for ITC/PLV Analysis

This script tests how different sampling rates affect ITC/PLV results by:
1. Loading epoched data files
2. Resampling to different rates (50, 100, 200, 300, 400, 500, 1000 Hz)
3. Computing ITC/PLV at each sampling rate
4. Saving plots and CSV data for comparison
5. Generating summary statistics

Expected outcomes:
- Higher sampling rates should give similar results (if original was adequate)
- Very low rates (50-100 Hz) may show artifacts or reduced quality
- Frequency resolution should be consistent (depends on epoch duration, not sampling rate)
- Power values will scale with sampling rate; PLV should be more stable
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
from plv_module import compute_itc, plot_itc, export_itc_csv

# Experimental parameters
SAMPLING_RATES = [50, 100, 200, 300, 400, 500, 1000]  # Hz
TARGET_FREQUENCIES = [10/9, 30/9]  # ~1.111 and ~3.333 Hz (word and syllable rates)
ROI = [f"E{idx + 1}" for idx in [27,19,11,4,117,116,28,12,5,111,110,35,29,6,105,104,103,30,79,54,36,86,40,102]]

# Paths
INPUT_DIR = "C:/Users/Gam9LG/Documents/Autoclean-EEG/output/Statistical_Learning/bids/derivatives/autoclean-v2.1.0/intermediate/FLAGGED_08_drop_bad_epochs/"
OUTPUT_DIR = "./sampling_rate_experiment/"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def load_epoch_files(input_dir, max_files=20):
    """Load up to max_files .set files from input directory."""
    pattern = str(Path(input_dir) / "*.set")
    files = glob.glob(pattern)
    if len(files) > max_files:
        files = files[:max_files]
        print(f"Using first {max_files} files out of {len(glob.glob(pattern))} found")
    return files

def resample_epochs(epochs, target_sfreq):
    """Resample epochs to target sampling frequency."""
    if epochs.info['sfreq'] == target_sfreq:
        return epochs.copy()
    
    # Use MNE's resample method
    epochs_resampled = epochs.copy()
    epochs_resampled.resample(target_sfreq, npad='auto')
    return epochs_resampled

def run_single_file_experiment(file_path, output_dir):
    """Run sampling rate experiment on a single file."""
    print(f"\nProcessing: {Path(file_path).name}")
    
    # Load original epochs
    try:
        epochs = mne.read_epochs_eeglab(file_path)
        original_sfreq = epochs.info['sfreq']
        print(f"Original sampling rate: {original_sfreq} Hz")
        print(f"Epoch duration: {epochs.times[-1] - epochs.times[0]:.2f} s")
        print(f"Number of epochs: {len(epochs)}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    file_stem = Path(file_path).stem
    results = []
    
    for sfreq in SAMPLING_RATES:
        print(f"  Testing {sfreq} Hz...")
        
        try:
            # Resample epochs
            epochs_resampled = resample_epochs(epochs, sfreq)
            
            # Compute ITC/PLV
            freqs, plv_roi, power_roi, info = compute_itc(
                epochs_resampled, 
                roi=ROI,
                target_frequencies=TARGET_FREQUENCIES,
                fmin=0.6,
                fmax=5.0,
                df=0.01
            )
            
            # Create plot
            fig = plot_itc(freqs, plv_roi, target_frequencies=info["target_frequencies"])
            plt.suptitle(f"{file_stem} - {sfreq} Hz sampling", y=0.98)
            
            # Save plot
            plot_path = Path(output_dir) / f"{file_stem}_sfreq_{sfreq}Hz.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save to CSV
            csv_path = Path(output_dir) / f"sampling_experiment_results.csv"
            export_itc_csv(
                str(csv_path),
                f"{file_stem}_sfreq_{sfreq}Hz",
                freqs,
                plv_roi,
                power_roi
            )
            
            # Store summary statistics
            word_freq_idx = np.argmin(np.abs(freqs - TARGET_FREQUENCIES[0]))
            syll_freq_idx = np.argmin(np.abs(freqs - TARGET_FREQUENCIES[1]))
            
            results.append({
                'file': file_stem,
                'sampling_rate': sfreq,
                'original_sfreq': original_sfreq,
                'n_epochs': len(epochs_resampled),
                'epoch_duration': epochs_resampled.times[-1] - epochs_resampled.times[0],
                'freq_resolution_theoretical': 1.0 / (epochs_resampled.times[-1] - epochs_resampled.times[0]),
                'word_freq_plv': plv_roi[word_freq_idx],
                'syll_freq_plv': plv_roi[syll_freq_idx],
                'word_freq_power': power_roi[word_freq_idx],
                'syll_freq_power': power_roi[syll_freq_idx],
                'mean_plv': np.mean(plv_roi),
                'max_plv': np.max(plv_roi),
                'mean_power': np.mean(power_roi),
                'max_power': np.max(power_roi)
            })
            
        except Exception as e:
            print(f"    Error at {sfreq} Hz: {e}")
            continue
    
    return results

def create_summary_plots(summary_df, output_dir):
    """Create summary plots comparing across sampling rates."""
    
    # PLV at target frequencies vs sampling rate
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Word frequency PLV
    axes[0,0].boxplot([summary_df[summary_df['sampling_rate']==sr]['word_freq_plv'] 
                       for sr in SAMPLING_RATES], labels=SAMPLING_RATES)
    axes[0,0].set_title('Word Frequency PLV vs Sampling Rate')
    axes[0,0].set_xlabel('Sampling Rate (Hz)')
    axes[0,0].set_ylabel('PLV')
    axes[0,0].grid(True, alpha=0.3)
    
    # Syllable frequency PLV
    axes[0,1].boxplot([summary_df[summary_df['sampling_rate']==sr]['syll_freq_plv'] 
                       for sr in SAMPLING_RATES], labels=SAMPLING_RATES)
    axes[0,1].set_title('Syllable Frequency PLV vs Sampling Rate')
    axes[0,1].set_xlabel('Sampling Rate (Hz)')
    axes[0,1].set_ylabel('PLV')
    axes[0,1].grid(True, alpha=0.3)
    
    # Mean PLV across spectrum
    axes[1,0].boxplot([summary_df[summary_df['sampling_rate']==sr]['mean_plv'] 
                       for sr in SAMPLING_RATES], labels=SAMPLING_RATES)
    axes[1,0].set_title('Mean PLV vs Sampling Rate')
    axes[1,0].set_xlabel('Sampling Rate (Hz)')
    axes[1,0].set_ylabel('Mean PLV')
    axes[1,0].grid(True, alpha=0.3)
    
    # Power scaling
    axes[1,1].boxplot([summary_df[summary_df['sampling_rate']==sr]['mean_power'] 
                       for sr in SAMPLING_RATES], labels=SAMPLING_RATES)
    axes[1,1].set_title('Mean Power vs Sampling Rate')
    axes[1,1].set_xlabel('Sampling Rate (Hz)')
    axes[1,1].set_ylabel('Mean Power')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "sampling_rate_summary.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run the complete sampling rate experiment."""
    print("Starting Sampling Rate Experiment for ITC/PLV Analysis")
    print("=" * 60)
    
    # Load files
    files = load_epoch_files(INPUT_DIR, max_files=20)
    print(f"Found {len(files)} files to process")
    
    if not files:
        print("No .set files found! Check INPUT_DIR path.")
        return
    
    # Run experiment on all files
    all_results = []
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing file...")
        results = run_single_file_experiment(file_path, OUTPUT_DIR)
        if results:
            all_results.extend(results)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_results)
    summary_path = Path(OUTPUT_DIR) / "sampling_experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Create summary plots
    if not summary_df.empty:
        create_summary_plots(summary_df, OUTPUT_DIR)
        print(f"Summary plots saved to: {OUTPUT_DIR}")
        
        # Print key findings
        print("\n" + "="*60)
        print("KEY FINDINGS:")
        print("="*60)
        
        # PLV stability across sampling rates
        word_plv_std = summary_df.groupby('file')['word_freq_plv'].std().mean()
        syll_plv_std = summary_df.groupby('file')['syll_freq_plv'].std().mean()
        
        print(f"Average PLV variability across sampling rates:")
        print(f"  Word frequency: {word_plv_std:.4f} (lower = more stable)")
        print(f"  Syllable frequency: {syll_plv_std:.4f} (lower = more stable)")
        
        # Power scaling
        power_scaling = summary_df.groupby('sampling_rate')['mean_power'].mean()
        print(f"\nPower scaling with sampling rate:")
        for sr in SAMPLING_RATES:
            print(f"  {sr} Hz: {power_scaling[sr]:.2e}")
    
    print(f"\nExperiment complete! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
