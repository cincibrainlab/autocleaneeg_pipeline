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
from scipy import stats
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

def create_itc_comparison_plot(summary_df, output_dir):
    """Create plots showing ITC values across all sampling rates for visual inspection."""
    
    # Create subplots for both target frequencies
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different sampling rates
    colors = plt.cm.viridis(np.linspace(0, 1, len(SAMPLING_RATES)))
    
    # Word frequency ITC
    for i, sfreq in enumerate(SAMPLING_RATES):
        sfreq_data = summary_df[summary_df['sampling_rate'] == sfreq]['word_freq_plv']
        axes[0].scatter([sfreq] * len(sfreq_data), sfreq_data, 
                       color=colors[i], alpha=0.7, s=50, label=f'{sfreq} Hz')
        # Add mean line
        mean_val = sfreq_data.mean()
        axes[0].plot([sfreq-20, sfreq+20], [mean_val, mean_val], 
                    color=colors[i], linewidth=3, alpha=0.8)
    
    axes[0].set_xlabel('Sampling Rate (Hz)')
    axes[0].set_ylabel('ITC at Word Frequency (~1.11 Hz)')
    axes[0].set_title('Word Frequency ITC Across Sampling Rates')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Syllable frequency ITC
    for i, sfreq in enumerate(SAMPLING_RATES):
        sfreq_data = summary_df[summary_df['sampling_rate'] == sfreq]['syll_freq_plv']
        axes[1].scatter([sfreq] * len(sfreq_data), sfreq_data, 
                       color=colors[i], alpha=0.7, s=50, label=f'{sfreq} Hz')
        # Add mean line
        mean_val = sfreq_data.mean()
        axes[1].plot([sfreq-20, sfreq+20], [mean_val, mean_val], 
                    color=colors[i], linewidth=3, alpha=0.8)
    
    axes[1].set_xlabel('Sampling Rate (Hz)')
    axes[1].set_ylabel('ITC at Syllable Frequency (~3.33 Hz)')
    axes[1].set_title('Syllable Frequency ITC Across Sampling Rates')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "itc_across_sampling_rates.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a line plot showing mean ITC trends
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate means and standard errors for each sampling rate
    word_means = []
    word_stds = []
    syll_means = []
    syll_stds = []
    
    for sfreq in SAMPLING_RATES:
        sfreq_data = summary_df[summary_df['sampling_rate'] == sfreq]
        
        word_vals = sfreq_data['word_freq_plv']
        word_means.append(word_vals.mean())
        word_stds.append(word_vals.std())
        
        syll_vals = sfreq_data['syll_freq_plv']
        syll_means.append(syll_vals.mean())
        syll_stds.append(syll_vals.std())
    
    # Word frequency trend
    axes[0].errorbar(SAMPLING_RATES, word_means, yerr=word_stds, 
                    marker='o', linewidth=2, markersize=8, capsize=5)
    axes[0].set_xlabel('Sampling Rate (Hz)')
    axes[0].set_ylabel('Mean ITC at Word Frequency')
    axes[0].set_title('Word Frequency ITC Trend (Mean ± SD)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)
    
    # Syllable frequency trend
    axes[1].errorbar(SAMPLING_RATES, syll_means, yerr=syll_stds, 
                    marker='s', linewidth=2, markersize=8, capsize=5, color='red')
    axes[1].set_xlabel('Sampling Rate (Hz)')
    axes[1].set_ylabel('Mean ITC at Syllable Frequency')
    axes[1].set_title('Syllable Frequency ITC Trend (Mean ± SD)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "itc_trend_across_sampling_rates.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical summary
    print(f"\n" + "="*60)
    print("ITC VALUES ACROSS SAMPLING RATES")
    print("="*60)
    
    print("\nWord Frequency ITC (~1.11 Hz):")
    print("Sampling Rate | Mean ± SD     | Min-Max       | CV")
    print("-" * 50)
    for i, sfreq in enumerate(SAMPLING_RATES):
        mean_val = word_means[i]
        std_val = word_stds[i]
        sfreq_data = summary_df[summary_df['sampling_rate'] == sfreq]['word_freq_plv']
        min_val = sfreq_data.min()
        max_val = sfreq_data.max()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        print(f"{sfreq:>4} Hz       | {mean_val:.4f} ± {std_val:.4f} | {min_val:.4f}-{max_val:.4f} | {cv:.1f}%")
    
    print("\nSyllable Frequency ITC (~3.33 Hz):")
    print("Sampling Rate | Mean ± SD     | Min-Max       | CV")
    print("-" * 50)
    for i, sfreq in enumerate(SAMPLING_RATES):
        mean_val = syll_means[i]
        std_val = syll_stds[i]
        sfreq_data = summary_df[summary_df['sampling_rate'] == sfreq]['syll_freq_plv']
        min_val = sfreq_data.min()
        max_val = sfreq_data.max()
        cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
        print(f"{sfreq:>4} Hz       | {mean_val:.4f} ± {std_val:.4f} | {min_val:.4f}-{max_val:.4f} | {cv:.1f}%")

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
        
        # Create ITC comparison plots across all sampling rates
        create_itc_comparison_plot(summary_df, OUTPUT_DIR)
        print(f"ITC comparison plots saved to: {OUTPUT_DIR}")
        
        # Run statistical test for sampling rate variability
        test_results = test_sampling_rate_variability(summary_df, OUTPUT_DIR)
        
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
