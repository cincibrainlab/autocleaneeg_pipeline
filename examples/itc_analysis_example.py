"""Example: Inter-Trial Coherence Analysis for Statistical Learning Data.

This example demonstrates how to use the InterTrialCoherenceMixin for analyzing
phase consistency in statistical learning EEG data. The ITC analysis can be
used both as a standalone function or within the AutoClean pipeline framework.
"""

import numpy as np

from autoclean import Pipeline
from autoclean.core.task import Task


class StatisticalLearningITCTask(Task):
    """Example task that combines statistical learning epochs with ITC analysis."""

    def run(self):
        """Run the complete statistical learning + ITC analysis pipeline."""
        # Import and preprocess the raw data
        self.import_raw()
        self.run_basic_steps()  # Filter, rereference, etc.

        # Create statistical learning epochs (30 syllables)
        self.create_sl_epochs(num_syllables=30)

        # Compute inter-trial coherence analysis with modern parameters
        power, itc, band_results = self.compute_itc_analysis(
            # Uses statistical learning frequencies (0.6-5 Hz) if not specified
            n_cycles=7.0,  # Good balance of temporal/frequency resolution
            baseline=(-0.5, -0.1),  # Baseline correction (applied to power only)
            picks="eeg",  # All EEG channels
            analyze_bands=True,  # Get frequency band summaries
            time_window=(1.0, 8.0),  # Focus on middle of epoch for bands
            use_multitaper=False,  # Use Morlet wavelets (default)
            calculate_wli=True,  # Calculate Word Learning Index
        )

        # The results are automatically saved and metadata is tracked
        print("ITC Analysis Complete!")
        print(f"Power shape: {power.data.shape}")
        print(f"ITC shape: {itc.data.shape}")

        if band_results:
            print("\nFrequency Band ITC Results:")
            for band, value in band_results.items():
                print(f"  {band}: {value:.3f}")

            # Optional: Test significance of ITC values
            from autoclean.functions.analysis.statistical_learning import (
                validate_itc_significance,
            )

            n_trials = len(self.epochs)
            significant_mask, threshold = validate_itc_significance(
                itc.data, n_trials, alpha=0.05, verbose=False
            )
            print("\nSignificance Testing:")
            print(f"  Threshold (α=0.05): {threshold:.4f}")
            print(
                f"  Significant time-freq points: {np.sum(significant_mask)}/{significant_mask.size}"
            )


# Configuration for statistical learning + ITC analysis
config = {
    "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
    "signal_processing": {
        "filter": {"highpass": 0.1, "lowpass": 40.0},
        "resampling": {"target_sfreq": 250},
    },
    "epoch_settings": {"enabled": True, "value": {"num_syllables": 30, "tmin": 0}},
    "itc_analysis": {
        "enabled": True,
        "value": {
            "n_cycles": 7.0,
            "baseline": [-0.5, -0.1],
            "analyze_bands": True,
            "time_window": [1.0, 8.0],
            "calculate_wli": True,
        },
    },
    "output": {"save_stages": ["raw", "epochs", "itc_analysis"]},
}


if __name__ == "__main__":
    # Example usage with pipeline
    pipeline = Pipeline(output_dir="./itc_analysis_results")
    pipeline.add_task_class(StatisticalLearningITCTask, config)

    # Process a single file
    # pipeline.process_file("path/to/your/eeg_file.raw", task="StatisticalLearningITCTask")

    print("To use this example:")
    print("1. Uncomment the process_file line above")
    print("2. Provide the path to your EEG file")
    print("3. Run: python examples/itc_analysis_example.py")

    # Alternative: Use standalone functions directly
    print("\nFor standalone usage (modern API with Word Learning Index):")
    print("from autoclean.functions.epoching import create_statistical_learning_epochs")
    print(
        "from autoclean.functions.analysis import compute_statistical_learning_itc, calculate_word_learning_index"
    )
    print(
        "from autoclean.functions.analysis.statistical_learning import validate_itc_significance"
    )
    print("")
    print("# Create epochs and compute ITC")
    print("epochs = create_statistical_learning_epochs(raw_data, num_syllables=30)")
    print(
        "power, itc = compute_statistical_learning_itc(epochs)  # Uses 0.6-5 Hz range"
    )
    print("")
    print("# Calculate Word Learning Index (WLI = ITC_word / ITC_syllable)")
    print(
        "wli_results = calculate_word_learning_index(itc, word_freq=1.11, syllable_freq=3.33)"
    )
    print("print(f'Word Learning Index: {wli_results[\"wli_mean\"]:.4f}')")
    print("")
    print("# Test significance")
    print(
        "significant_mask, threshold = validate_itc_significance(itc.data, len(epochs))"
    )
    print("print(f'Significance threshold: {threshold:.4f}')")
