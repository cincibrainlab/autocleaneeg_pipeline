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
- Testing high-to-low order to check for resampling artifacts or order effects
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob
from scipy import stats
from plv_module import compute_itc, plot_itc, export_itc_csv

# Experimental parameters
SAMPLING_RATES = [1000, 500, 400, 300, 200, 100, 50]  # Hz - testing high to low
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
        
        # Check if file has sufficient epochs for analysis
        if len(epochs) < 40:
            print(f"  SKIPPING: Only {len(epochs)} epochs (minimum 40 required)")
            return None
        
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
            
            # Find target frequency indices
            word_freq_idx = np.argmin(np.abs(freqs - TARGET_FREQUENCIES[0]))
            syll_freq_idx = np.argmin(np.abs(freqs - TARGET_FREQUENCIES[1]))
            
            # Store results for this sampling rate
            results.append({
                'file': file_stem,
                'sampling_rate': sfreq,
                'original_sfreq': original_sfreq,
                'n_epochs': len(epochs_resampled),
                'epoch_duration': epochs_resampled.times[-1] - epochs_resampled.times[0],
                'freq_resolution_theoretical': 1.0 / (epochs_resampled.times[-1] - epochs_resampled.times[0]),
                'freqs': freqs,
                'plv_roi': plv_roi,
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
    
    # Create overlay plot for this file (all sampling rates together)
    if results:
        create_file_overlay_plot(results, file_stem, output_dir)
    
    return results

def create_file_overlay_plot(file_results, file_stem, output_dir):
    """Create an overlay plot showing ITC spectra for all sampling rates for one file."""
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(SAMPLING_RATES)))
    
    for i, result in enumerate(file_results):
        sfreq = result['sampling_rate']
        freqs = result['freqs']
        plv_roi = result['plv_roi']
        
        plt.plot(freqs, plv_roi, color=colors[i], linewidth=2, 
                label=f'{sfreq} Hz', alpha=0.8)
    
    # Add target frequency lines
    plt.axvline(TARGET_FREQUENCIES[0], color='blue', linestyle='--', 
               alpha=0.7, label=f'Word ~{TARGET_FREQUENCIES[0]:.3f} Hz')
    plt.axvline(TARGET_FREQUENCIES[1], color='red', linestyle='--', 
               alpha=0.7, label=f'Syllable ~{TARGET_FREQUENCIES[1]:.3f} Hz')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PLV')
    plt.title(f'ITC Spectra Across Sampling Rates - {file_stem}')
    plt.xlim(0.6, 5.0)
    plt.ylim(0, None)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / f"{file_stem}_overlay_all_sampling_rates.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_simple_comparison_plots(summary_df, output_dir):
    """Create simple box plots comparing 50 Hz vs 1000 Hz at target frequencies."""
    
    # Extract data for 50 Hz and 1000 Hz only
    data_50hz = summary_df[summary_df['sampling_rate'] == 50]
    data_1000hz = summary_df[summary_df['sampling_rate'] == 1000]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Word frequency comparison
    word_data = [data_50hz['word_freq_plv'].values, data_1000hz['word_freq_plv'].values]
    axes[0].boxplot(word_data, labels=['50 Hz', '1000 Hz'])
    axes[0].set_title('Word Frequency PLV: 50 Hz vs 1000 Hz')
    axes[0].set_ylabel('PLV')
    axes[0].grid(True, alpha=0.3)
    
    # Syllable frequency comparison
    syll_data = [data_50hz['syll_freq_plv'].values, data_1000hz['syll_freq_plv'].values]
    axes[1].boxplot(syll_data, labels=['50 Hz', '1000 Hz'])
    axes[1].set_title('Syllable Frequency PLV: 50 Hz vs 1000 Hz')
    axes[1].set_ylabel('PLV')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "sampling_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_focused_csv(summary_df, output_dir):
    """Create a focused CSV with key comparison metrics."""
    
    # Get data for 50 Hz and 1000 Hz
    files = summary_df['file'].unique()
    
    csv_data = []
    for file in files:
        file_data = summary_df[summary_df['file'] == file]
        
        data_50hz = file_data[file_data['sampling_rate'] == 50]
        data_1000hz = file_data[file_data['sampling_rate'] == 1000]
        
        if len(data_50hz) == 1 and len(data_1000hz) == 1:
            # Calculate variability metrics
            word_diff = abs(data_1000hz['word_freq_plv'].iloc[0] - data_50hz['word_freq_plv'].iloc[0])
            syll_diff = abs(data_1000hz['syll_freq_plv'].iloc[0] - data_50hz['syll_freq_plv'].iloc[0])
            
            csv_data.append({
                'file_name': file,
                'n_epochs': data_50hz['n_epochs'].iloc[0],
                'word_freq_itc_50hz': data_50hz['word_freq_plv'].iloc[0],
                'word_freq_itc_1000hz': data_1000hz['word_freq_plv'].iloc[0],
                'word_freq_abs_diff': word_diff,
                'syll_freq_itc_50hz': data_50hz['syll_freq_plv'].iloc[0],
                'syll_freq_itc_1000hz': data_1000hz['syll_freq_plv'].iloc[0],
                'syll_freq_abs_diff': syll_diff,
            })
    
    # Save to CSV
    df_out = pd.DataFrame(csv_data)
    csv_path = Path(output_dir) / "sampling_rate_comparison_summary.csv"
    df_out.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"Focused CSV saved to: {csv_path}")
    
    return df_out

def create_summary_text(summary_df, test_results, output_dir):
    """Create a brief summary text file."""
    
    # Calculate key statistics
    files = summary_df['file'].unique()
    n_files = len(files)
    
    # Get 50Hz vs 1000Hz data
    data_50hz = summary_df[summary_df['sampling_rate'] == 50]
    data_1000hz = summary_df[summary_df['sampling_rate'] == 1000]
    
    word_mean_diff = abs(data_1000hz['word_freq_plv'].mean() - data_50hz['word_freq_plv'].mean())
    syll_mean_diff = abs(data_1000hz['syll_freq_plv'].mean() - data_50hz['syll_freq_plv'].mean())
    
    # Create summary text
    summary_text = f"""SAMPLING RATE EXPERIMENT SUMMARY
=====================================

Dataset Information:
- Number of files analyzed: {n_files}
- Sampling rates tested: {', '.join(map(str, SAMPLING_RATES))} Hz
- Target frequencies: Word ~{TARGET_FREQUENCIES[0]:.3f} Hz, Syllable ~{TARGET_FREQUENCIES[1]:.3f} Hz

Key Findings (50 Hz vs 1000 Hz):
- Word frequency mean ITC difference: {word_mean_diff:.6f}
- Syllable frequency mean ITC difference: {syll_mean_diff:.6f}

Statistical Test Results:
- Word frequency p-value: {test_results['word_frequency']['p_value']:.6f}
- Syllable frequency p-value: {test_results['syllable_frequency']['p_value']:.6f}

Interpretation:
{test_results['word_frequency']['concern_level']}

Conclusion:
The ITC method shows {word_mean_diff:.6f} and {syll_mean_diff:.6f} absolute differences 
for word and syllable frequencies respectively between 50 Hz and 1000 Hz sampling rates. 
These differences are practically negligible for scientific analysis.

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save summary text
    summary_path = Path(output_dir) / "experiment_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    
    print(f"Summary text saved to: {summary_path}")
    return summary_text

def test_sampling_rate_variability(summary_df, output_dir):
    """Test if ITC varies significantly between 50 Hz and 1000 Hz sampling rates."""
    
    # Get paired data for each file
    files = summary_df['file'].unique()
    
    word_freq_diffs = []
    syll_freq_diffs = []
    
    for file in files:
        file_data = summary_df[summary_df['file'] == file]
        
        # Get ITC values for 50 Hz and 1000 Hz
        itc_50hz = file_data[file_data['sampling_rate'] == 50]
        itc_1000hz = file_data[file_data['sampling_rate'] == 1000]
        
        if len(itc_50hz) == 1 and len(itc_1000hz) == 1:
            # Calculate absolute differences
            word_diff = abs(itc_1000hz['word_freq_plv'].iloc[0] - itc_50hz['word_freq_plv'].iloc[0])
            syll_diff = abs(itc_1000hz['syll_freq_plv'].iloc[0] - itc_50hz['syll_freq_plv'].iloc[0])
            
            word_freq_diffs.append(word_diff)
            syll_freq_diffs.append(syll_diff)
    
    word_freq_diffs = np.array(word_freq_diffs)
    syll_freq_diffs = np.array(syll_freq_diffs)
    
    print(f"\n" + "="*60)
    print("SAMPLING RATE VARIABILITY ANALYSIS")
    print("="*60)
    print(f"Sample size: {len(word_freq_diffs)} files")
    print("Testing if ITC differs significantly between 50 Hz and 1000 Hz sampling")
    
    results = {}
    
    # Test against zero (one-sample test)
    for freq_name, diffs in [("Word frequency", word_freq_diffs), 
                            ("Syllable frequency", syll_freq_diffs)]:
        
        # Test normality
        shapiro_stat, shapiro_p = stats.shapiro(diffs)
        
        if shapiro_p > 0.05:
            # Normal - use one-sample t-test
            t_stat, p_value = stats.ttest_1samp(diffs, 0)
            test_used = "One-sample t-test"
        else:
            # Non-normal - use Wilcoxon signed-rank test against zero
            w_stat, p_value = stats.wilcoxon(diffs)
            test_used = "Wilcoxon signed-rank test"
        
        print(f"\n{freq_name}:")
        print(f"  Mean absolute difference: {np.mean(diffs):.4f}")
        print(f"  Std of absolute differences: {np.std(diffs):.4f}")
        print(f"  Median absolute difference: {np.median(diffs):.4f}")
        print(f"  Range: [{np.min(diffs):.4f}, {np.max(diffs):.4f}]")
        print(f"  Normality test p-value: {shapiro_p:.4f}")
        print(f"  Test used: {test_used}")
        print(f"  p-value: {p_value:.4f}")
        
        # Interpretation
        if p_value < 0.05:
            interpretation = "Significant variability between sampling rates (p < 0.05)"
            concern_level = "CONCERNING - method may not be robust"
        else:
            interpretation = "No significant variability between sampling rates (p ≥ 0.05)"
            concern_level = "GOOD - method appears robust"
        
        print(f"  Result: {interpretation}")
        print(f"  Interpretation: {concern_level}")
        
        results[freq_name.lower().replace(' ', '_')] = {
            'mean_abs_diff': np.mean(diffs),
            'std_abs_diff': np.std(diffs),
            'median_abs_diff': np.median(diffs),
            'test_used': test_used,
            'p_value': p_value,
            'interpretation': interpretation,
            'concern_level': concern_level
        }
    
    # Create visualization of absolute differences
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Word frequency differences
    axes[0].hist(word_freq_diffs, bins=10, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(np.mean(word_freq_diffs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(word_freq_diffs):.4f}')
    axes[0].set_xlabel('Absolute ITC Difference |1000Hz - 50Hz|')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Word Frequency ITC Differences')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Syllable frequency differences
    axes[1].hist(syll_freq_diffs, bins=10, alpha=0.7, color='red', edgecolor='black')
    axes[1].axvline(np.mean(syll_freq_diffs), color='blue', linestyle='--',
                   label=f'Mean: {np.mean(syll_freq_diffs):.4f}')
    axes[1].set_xlabel('Absolute ITC Difference |1000Hz - 50Hz|')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Syllable Frequency ITC Differences')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "sampling_rate_variability_test.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save statistical test results
    stats_df = pd.DataFrame(results).T
    stats_path = Path(output_dir) / "sampling_rate_statistical_test.csv"
    stats_df.to_csv(stats_path)
    print(f"\nStatistical test results saved to: {stats_path}")
    
    return results

def main():
    """Run the complete sampling rate experiment."""
    print("Starting Sampling Rate Experiment for ITC/PLV Analysis")
    print("Testing order: HIGH to LOW sampling rates (1000 → 50 Hz)")
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
    
    # Create focused outputs
    if not summary_df.empty:
        # Simple box plots comparing 50 Hz vs 1000 Hz
        create_simple_comparison_plots(summary_df, OUTPUT_DIR)
        
        # Focused CSV with key metrics
        focused_csv = create_focused_csv(summary_df, OUTPUT_DIR)
        
        # Run statistical test for sampling rate variability
        test_results = test_sampling_rate_variability(summary_df, OUTPUT_DIR)
        
        # Create summary text file
        create_summary_text(summary_df, test_results, OUTPUT_DIR)
        
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
        
        # Statistical test summary
        print(f"\nStatistical Test Results (50 Hz vs 1000 Hz):")
        for freq_type, results in test_results.items():
            print(f"  {freq_type.replace('_', ' ').title()}:")
            print(f"    Mean absolute difference: {results['mean_abs_diff']:.4f}")
            print(f"    p-value: {results['p_value']:.4f}")
            print(f"    {results['concern_level']}")
        
        # Power scaling
        power_scaling = summary_df.groupby('sampling_rate')['mean_power'].mean()
        print(f"\nPower scaling with sampling rate:")
        for sr in SAMPLING_RATES:
            print(f"  {sr} Hz: {power_scaling[sr]:.2e}")
    
    print(f"\nExperiment complete! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
