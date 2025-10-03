"""Example: Using Amplitude Quality Coach for Threshold Tuning.

This example demonstrates how to use the amplitude quality coaching feature
to understand data quality, identify problematic channels, and tune voltage
thresholds appropriately for artifact rejection.

The amplitude quality coach provides interpretable diagnostics that help
answer questions like:
- Are my thresholds too strict or too loose?
- Which channels are systematically noisy?
- Should I interpolate specific channels before ICA?
- What's a reasonable threshold for my data?
"""

import mne
from autoclean.tasks import YourTaskClass  # Replace with actual task class


def example_1_automatic_coaching():
    """Example 1: Automatic amplitude coaching during epoch creation.
    
    The coach runs automatically every time you create epochs, providing
    immediate feedback on data quality and threshold settings.
    """
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    processor = YourTaskClass(raw=raw, config="config.json")
    
    # Create epochs - amplitude quality coach runs automatically
    epochs = processor.create_eventid_epochs()
    
    # Output will include:
    # 1. Event discovery table
    # 2. Epoch creation progress
    # 3. Drop log summary
    # 4. Amplitude Quality Analysis (NEW!)
    #    - Per-channel-type statistics
    #    - Channels exceeding thresholds
    #    - Actionable suggestions for threshold tuning
    #    - Overall quality metrics
    
    if epochs is not None:
        print(f"Created {len(epochs)} epochs with quality coaching")


def example_2_standalone_quality_check():
    """Example 2: Run amplitude quality analysis on existing epochs.
    
    You can also run the quality coach independently on any epochs object
    to diagnose issues or validate threshold settings.
    """
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    processor = YourTaskClass(raw=raw)
    
    # Create epochs (coaching runs automatically)
    epochs = processor.create_eventid_epochs(
        event_id={"target": 4, "standard": 5},
        tmin=-0.2,
        tmax=0.8,
    )
    
    if epochs is not None:
        # Run standalone quality analysis with different thresholds
        print("\n" + "="*80)
        print("Testing stricter threshold...")
        print("="*80)
        
        quality_df = processor.summarize_amplitude_quality(
            epochs=epochs,
            volt_threshold={"eeg": 0.00015}  # Stricter than default
        )
        
        # The DataFrame contains detailed per-channel statistics
        if quality_df is not None:
            # Find channels with high rejection rates
            problem_channels = quality_df[quality_df["flagged_pct"] > 20.0]
            print(f"\nChannels flagged in >20% of epochs:")
            print(problem_channels[["channel", "mean_amp", "flagged_pct"]])


def example_3_threshold_tuning_workflow():
    """Example 3: Iterative threshold tuning using coach feedback.
    
    Use the coach to empirically determine optimal thresholds for your data.
    """
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    processor = YourTaskClass(raw=raw)
    
    # First pass: No threshold (keep all epochs)
    print("Pass 1: Analyzing data without rejection")
    print("="*80)
    epochs_all = processor.create_eventid_epochs(
        event_id={"stimulus": 4},
        volt_threshold=None,  # No rejection
        keep_all_epochs=True,
    )
    
    if epochs_all is None:
        return
    
    # Analyze to understand amplitude distribution
    quality_df = processor.summarize_amplitude_quality(
        epochs=epochs_all,
        volt_threshold=None
    )
    
    # Based on coach output, user sees:
    # - Average mean amplitude: 0.000085 V
    # - Maximum mean amplitude: 0.000320 V
    # Coach suggests trying threshold around 0.000213 V (2.5x average)
    
    # Second pass: Apply suggested threshold
    print("\n\nPass 2: Applying coach-suggested threshold")
    print("="*80)
    suggested_threshold = float(quality_df["mean_amp"].mean() * 2.5)
    
    epochs_clean = processor.create_eventid_epochs(
        event_id={"stimulus": 4},
        volt_threshold={"eeg": suggested_threshold},
        keep_all_epochs=False,  # Now actually reject
    )
    
    # Coach will show how many epochs/channels exceeded the threshold
    # and whether further adjustment is needed


def example_4_identifying_bad_channels():
    """Example 4: Use coach output to identify channels for interpolation.
    
    The coach highlights channels that consistently exceed thresholds,
    which may indicate bad electrode contact rather than transient artifacts.
    """
    raw = mne.io.read_raw_fif("your_data.fif", preload=True)
    processor = YourTaskClass(raw=raw)
    
    epochs = processor.create_eventid_epochs(
        event_id={"target": 4, "standard": 5},
        volt_threshold={"eeg": 0.0002},
    )
    
    if epochs is None:
        return
    
    # Get the quality DataFrame
    quality_df = processor.summarize_amplitude_quality(
        epochs=epochs,
        volt_threshold={"eeg": 0.0002}
    )
    
    if quality_df is not None:
        # Find systematically bad channels (high amplitude + high rejection rate)
        bad_channels = quality_df[
            (quality_df["flagged_pct"] > 30.0) &  # Flagged in >30% epochs
            (quality_df["mean_amp"] > quality_df["mean_amp"].mean() * 2)  # 2x avg
        ]["channel"].tolist()
        
        if bad_channels:
            print(f"\n⚠️  Systematically bad channels detected: {bad_channels}")
            print("Consider interpolating these channels:")
            print(f"  raw.info['bads'] = {bad_channels}")
            print(f"  raw.interpolate_bads()")
        else:
            print("\n✓ No systematically bad channels detected")


def example_5_interpreting_coach_output():
    """Example 5: Understanding coach output and suggestions.
    
    The coach provides different suggestions based on the data pattern.
    """
    # Scenario A: Many channels flagged (>30%)
    # Coach output:
    # """
    # ⚠️  Channels exceeding threshold in >20% of epochs:
    #   • E12: 45.2% of epochs (mean: 0.000280 V)
    #   • E23: 38.7% of epochs (mean: 0.000265 V)
    #   • E45: 35.1% of epochs (mean: 0.000240 V)
    #   ... (15 more channels)
    # 
    # 💡 Suggestions:
    #   • >30% of channels flagged - threshold may be too strict
    #   • Consider increasing threshold to ~0.000212 V
    # """
    # Interpretation: Threshold is rejecting too much good data
    
    # Scenario B: Few channels flagged (≤3)
    # Coach output:
    # """
    # ⚠️  Channels exceeding threshold in >20% of epochs:
    #   • E68: 67.8% of epochs (mean: 0.000450 V)
    #   • E71: 42.3% of epochs (mean: 0.000380 V)
    # 
    # 💡 Suggestions:
    #   • Few channels flagged - likely bad electrode contact
    #   • Consider interpolating: E68, E71
    # """
    # Interpretation: Specific channels have poor contact
    
    # Scenario C: No channels flagged, low amplitudes
    # Coach output:
    # """
    # ✓ All channels within acceptable limits
    # 
    # 💡 Threshold may be too loose for this data
    #   • Consider tightening to ~0.000095 V for better artifact detection
    # """
    # Interpretation: Can afford stricter threshold for better cleaning
    
    print(__doc__)


# Use cases:

# Use Case 1: "Are my thresholds appropriate?"
# → Run create_eventid_epochs() and review coach output

# Use Case 2: "Which threshold should I use?"
# → Start with no threshold, use coach suggestions

# Use Case 3: "Which channels should I interpolate?"
# → Look for channels flagged in >30% epochs with high mean amplitude

# Use Case 4: "Why are so many epochs rejected?"
# → Coach will tell you if threshold is too strict

# Use Case 5: "Is my data quality acceptable?"
# → Check overall mean amplitude and rejection rates


if __name__ == "__main__":
    # Run examples (uncomment as needed)
    # example_1_automatic_coaching()
    # example_2_standalone_quality_check()
    # example_3_threshold_tuning_workflow()
    # example_4_identifying_bad_channels()
    example_5_interpreting_coach_output()

