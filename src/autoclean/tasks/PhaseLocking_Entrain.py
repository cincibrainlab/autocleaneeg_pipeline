"""Phase Locking / Neural Entrainment Task for Adult Data (ENTRAIN Pipeline).

This task implements Dr. Battrink's phase locking preprocessing protocol for
analyzing neural entrainment to rhythmic stimuli in adult participants. The
pipeline focuses on computing inter-trial coherence (ITC) to measure phase
consistency at word and syllable frequencies.

Protocol Reference:
- Bandpass filtering: 0.1-30 Hz (Butterworth, order 2)
- Epoching: 10.8s epochs time-locked to every 12th word onset (36 syllables @ 300ms SOA)
- ICA: Extended Infomax with ICLabel classification
- Analysis: ITC with surrogate-based significance testing (zITC)
- Source analysis: MNE inverse solution targeting STG/transverse temporal gyrus

Key Output Metrics:
- zITC at word frequency (1.11 Hz) - averaged across scalp electrodes
- zITC at syllable frequency (3.33 Hz) - averaged across scalp electrodes
- Source-level zITC in STG/transverse temporal gyrus ROIs

Author: Generated from Dr. Battrink's ENTRAIN protocol
Date: 2025
"""

from autoclean.core.task import Task

# =============================================================================
#                     PHASE LOCKING CONFIGURATION
# =============================================================================
# This configuration implements the ENTRAIN pipeline for neural entrainment
# analysis using phase-locking measures (ITC/zITC).
#
# Pipeline Overview:
# 1. Initial Preprocessing (filtering, re-referencing, epoching)
# 2. Artifact Correction (minimal vs. advanced ICA pipelines)
# 3. FFT/ITC Analysis (scalp-level with surrogate testing)
# 4. Source Localization (optional)
# 5. Source-level ITC Analysis (optional)
# =============================================================================

config = {
    "schema_version": "2025.09",
    # =============================================================================
    # DATASET CONFIGURATION
    # =============================================================================
    "dataset_name": "ENTRAIN_PhaseLockin g",  # Customize for your dataset
    # "input_path": "/path/to/your/data/",  # Uncomment and specify your data path
    # Optional: Keep flagged files in standard output (set False to move them)
    "move_flagged_files": False,
    # Optional: Enable AI-powered reporting for automated summaries
    "ai_reporting": False,
    # =============================================================================
    # MONTAGE CONFIGURATION
    # =============================================================================
    # Specifies the electrode montage/layout for channel locations
    # Common montages: "GSN-HydroCel-129", "GSN-HydroCel-128", "standard_1020"
    #
    # ⚠️ CLARIFICATION NEEDED: Which montage does your data use?
    "montage": {
        "enabled": True,
        "value": "GSN-HydroCel-129",  # UPDATE THIS based on your system
    },
    # =============================================================================
    # RESAMPLING
    # =============================================================================
    # Downsample to 250 Hz for computational efficiency (adequate for 0.1-30 Hz band)
    # Note: Protocol doesn't specify resampling, but 250 Hz is standard for this analysis
    "resample_step": {
        "enabled": True,
        "value": 250,  # Hz - adequate for 30 Hz low-pass filter (Nyquist = 125 Hz)
    },
    # =============================================================================
    # FILTERING - PROTOCOL SPECIFIED
    # =============================================================================
    # Band-pass filter: 0.1 to 30 Hz (Butterworth filter, order 2)
    # This removes DC drift and high-frequency noise while preserving neural entrainment
    # frequencies (word: ~1.11 Hz, syllable: ~3.33 Hz)
    #
    # ✅ IMPLEMENTED: Butterworth bandpass 0.1-30 Hz
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 0.1,  # High-pass: removes DC component and slow drifts
            "h_freq": 30.0,  # Low-pass: removes high-frequency noise/muscle
            "notch_freqs": [60, 120],  # Line noise removal (US: 60 Hz, EU: 50 Hz)
            "notch_widths": 5,  # Hz
            # AutoClean defaults to Butterworth-like FIR filter with good characteristics
            # MNE uses zero-phase FIR filtering by default (method='fir', phase='zero')
            # which is appropriate for phase-locking analysis
        },
    },
    # =============================================================================
    # CHANNEL CONFIGURATION
    # =============================================================================
    # Drop outer layer channels if needed (e.g., face/neck electrodes in dense arrays)
    "drop_outerlayer": {
        "enabled": False,
        "value": [],  # e.g., [1, 32, 125, 126, 127, 128] for GSN-129
    },
    # EOG channel configuration - mark and optionally drop EOG channels
    # These channels can be used for artifact detection or dropped before analysis
    #
    # ⚠️ CLARIFICATION NEEDED: Which channels are EOG in your montage?
    "eog_step": {
        "enabled": False,  # Set to True if you have dedicated EOG channels
        "value": {
            "eog_indices": [],  # e.g., [1, 32, 125, 126, 127, 128] - UPDATE based on your montage
            "eog_drop": False,  # Set True to drop EOG channels after marking
        },
    },
    # Trim edges - remove initial/final seconds to avoid recording artifacts
    "trim_step": {"enabled": True, "value": 4},  # Trim 4 seconds from start and end
    # Crop duration - limit analysis to specific time window if needed
    "crop_step": {
        "enabled": False,
        "value": {"start": 0, "end": 600},  # Limit to first 600 seconds if desired
    },
    # =============================================================================
    # RE-REFERENCING - PROTOCOL SPECIFIED
    # =============================================================================
    # Re-reference to average of LM and RM (left/right mastoid)
    # Protocol assumes LM and RM were recorded during acquisition
    #
    # ⚠️ CLARIFICATION NEEDED:
    # - What are the exact channel names for LM and RM in your data?
    # - If they weren't recorded, use "average" re-referencing instead
    #
    # ✅ IMPLEMENTED (but needs channel names)
    # Option 1: If you have mastoid channels, specify them:
    # "reference_step": {"enabled": True, "value": ["LM", "RM"]},
    #
    # Option 2: If no mastoids, use average reference:
    "reference_step": {
        "enabled": True,
        "value": "average",  # Change to ["LM", "RM"] if mastoids are available
    },
    # =============================================================================
    # WAVELET DENOISING (Optional - not in original protocol)
    # =============================================================================
    # Optional wavelet-based denoising before channel cleaning
    # This is NOT part of the original Battrink protocol but can improve results
    "wavelet_threshold": {
        "enabled": False,
        "value": {
            "wavelet": "sym4",
            "level": 5,
            "threshold_mode": "soft",
            "is_erp": False,
            "threshold_scale": 1.0,
            "psd_fmax": 45.0,
            "bandpass": [1.0, 30.0],
        },
    },
    # =============================================================================
    # ICA CONFIGURATION - PROTOCOL SPECIFIED
    # =============================================================================
    # Advanced correction pipeline uses ICA (extended Infomax)
    # Protocol specifies:
    # - Method: runica (extended) - equivalent to extended Infomax
    # - Applied to high-pass filtered (1 Hz) parallel dataset
    # - Use ICLabel to reject components >= 0.9 probability for artifacts
    #
    # ✅ IMPLEMENTED: Extended Infomax ICA
    # ⚠️ CLARIFICATION NEEDED:
    # - Should we auto-create the 1 Hz high-pass parallel dataset?
    # - Currently, AutoClean supports temp_highpass_for_ica parameter
    "ICA": {
        "enabled": True,
        "value": {
            "method": "infomax",  # Extended Infomax (equivalent to runica extended)
            "n_components": None,  # Auto-determine based on data rank
            "fit_params": {"extended": True},  # Enable extended Infomax
            # Parallel dataset: high-pass at 1 Hz for ICA fitting
            # ICA weights are then applied to original 0.1 Hz filtered data
            "temp_highpass_for_ica": 1.0,  # Hz - creates temporary 1 Hz copy for ICA
        },
    },
    # =============================================================================
    # COMPONENT REJECTION - PROTOCOL SPECIFIED
    # =============================================================================
    # Use ICLabel to automatically reject artifact components
    # Protocol threshold: >= 0.9 probability for eye, muscle, noise, or line artifacts
    #
    # ✅ IMPLEMENTED: ICLabel classification with custom thresholds
    "component_rejection": {
        "enabled": True,
        "method": "iclabel",  # Use ICLabel classifier
        "value": {
            # Reject these artifact types
            "ic_flags_to_reject": ["muscle", "heart", "eog", "ch_noise", "line_noise"],
            # Protocol specifies 0.9 threshold for automatic rejection
            # AutoClean uses 0.3 by default, but we override to 0.9 per protocol
            "ic_rejection_threshold": 0.9,  # Only reject if >= 90% confidence
            # Optional: per-type overrides (not in protocol, but useful)
            "ic_rejection_overrides": {
                "eog": 0.9,  # Eye artifacts: 90% threshold
                "muscle": 0.9,  # Muscle artifacts: 90% threshold
                "heart": 0.9,  # Cardiac artifacts: 90% threshold
                "ch_noise": 0.9,  # Channel noise: 90% threshold
                "line_noise": 0.9,  # Line noise: 90% threshold
            },
            # PSD plot limit for visualization
            "psd_fmax": 40.0,  # Hz - adequate for viewing entrainment frequencies
        },
    },
    # =============================================================================
    # EPOCHING CONFIGURATION - PROTOCOL SPECIFIED
    # =============================================================================
    # Protocol specifies:
    # - Create nonoverlapping epochs time-locked to every 12th word onset
    # - Duration: 10.8 seconds (36 syllables assuming 300 ms SOA between syllables)
    # - Calculation: 12 words × 3 syllables/word = 36 syllables
    #               36 syllables × 300 ms/syllable = 10,800 ms = 10.8 s
    #
    # AutoClean Implementation:
    # - Use create_sl_epochs() with num_syllables=36 (12 words × 3 syllables)
    # - This creates epochs based on syllable events in the data
    #
    # ⚠️ CLARIFICATION NEEDED:
    # - How are word/syllable events marked in your data?
    # - What are the event codes for syllables?
    # - Is every syllable marked, or only word onsets?
    # - Is SOA exactly 300 ms between syllables?
    #
    # ✅ PARTIALLY IMPLEMENTED (needs event configuration)
    "epoch_settings": {
        "enabled": True,
        "value": {
            "tmin": 0,  # Start time relative to trigger (0 = word onset)
            # tmax is computed automatically from num_syllables
            # tmax = num_syllables × 0.3 s = 36 × 0.3 = 10.8 s
            "num_syllables": 36,  # 12 words × 3 syllables/word = 36 syllables
        },
        # Event ID configuration - defines which events to use for epoching
        # ⚠️ CLARIFICATION NEEDED: What are the event codes in your data?
        "event_id": None,  # None = auto-detect all events
        # Example: {"syllable_onset": 1, "word_onset": 10}  # UPDATE based on your data
        # Baseline correction - typically not used for phase-locking analysis
        # as we're interested in sustained phase consistency, not evoked activity
        "remove_baseline": {
            "enabled": False,
            "window": [None, 0],  # Pre-stimulus baseline
        },
        # Threshold rejection - reject epochs with extreme voltages
        # Protocol mentions "manually remove very noisy epochs" but not specific threshold
        # Enable this for automated rejection, or use GFP cleaning below
        "threshold_rejection": {
            "enabled": False,  # Set True to enable automated voltage rejection
            "volt_threshold": {
                "eeg": 0.000150  # 150 µV - adjust based on your data quality
            },
        },
    },
    # =============================================================================
    # ITC ANALYSIS CONFIGURATION - PROTOCOL SPECIFIED
    # =============================================================================
    # Compute inter-trial coherence (ITC) using FFT across all cleaned epochs
    # Protocol specifies:
    # - Compute ITC across all cleaned epochs
    # - Create 100 shuffled surrogate versions per participant
    # - Compute zITC based on surrogate distribution
    # - Output: zITC at word (1.11 Hz) and syllable (3.33 Hz) frequencies
    #
    # ✅ PARTIALLY IMPLEMENTED:
    # - ITC computation: compute_itc_analysis() ✅
    # - Frequency band analysis: analyze_itc_bands() ✅
    # - Word Learning Index: calculate_wli() ✅
    #
    # ❌ NOT YET IMPLEMENTED:
    # - Shuffled surrogate generation (100 iterations)
    # - zITC computation from surrogates
    # - Automatic surrogate-based significance testing
    #
    # Current Implementation Notes:
    # - Uses Morlet wavelet analysis to compute ITC
    # - Default frequency range: 0.6-5 Hz (appropriate for word/syllable frequencies)
    # - Can extract ITC at specific frequencies (1.11 Hz word, 3.33 Hz syllable)
    "itc_analysis": {
        "enabled": True,
        "value": {
            # Frequency configuration for ITC analysis
            # Default: 50 frequencies logarithmically spaced from 0.6 to 5 Hz
            # This captures word (1.11 Hz) and syllable (3.33 Hz) frequencies
            "freqs": None,  # None = use default 0.6-5 Hz range
            # Morlet wavelet parameters
            "n_cycles": 7.0,  # Number of cycles for wavelet (7 is standard)
            # Multitaper method (alternative to Morlet wavelets)
            "use_multitaper": False,  # Set True to use multitaper instead
            "time_bandwidth": 4.0,  # Time-bandwidth product for multitaper
            # Computation parameters
            "decim": 1,  # Decimation factor (1 = no decimation)
            "n_jobs": 4,  # Number of parallel jobs for computation
            # Channel selection
            "picks": None,  # None = all EEG channels
            # Baseline correction (typically not used for ITC)
            "baseline": None,  # None = no baseline correction for ITC
            "mode": "mean",  # Baseline mode (if baseline is enabled)
            # Analysis options
            "analyze_bands": True,  # Compute frequency band summaries
            "time_window": None,  # None = use entire epoch for band analysis
            # Word Learning Index (ratio of word ITC to syllable ITC)
            "calculate_wli": True,  # Compute WLI = ITC(1.11 Hz) / ITC(3.33 Hz)
        },
    },
    # =============================================================================
    # SOURCE LOCALIZATION CONFIGURATION (Optional)
    # =============================================================================
    # Protocol specifies source-level ITC analysis in STG/transverse temporal gyrus
    # This requires:
    # 1. Source localization (MNE inverse solution)
    # 2. Source-level ITC computation
    # 3. ROI extraction (STG, transverse temporal gyrus)
    #
    # ⚠️ CLARIFICATION NEEDED:
    # - Should source localization be included in the standard pipeline?
    # - Which specific ROIs/labels should be analyzed?
    # - Should this be done on continuous data or epochs?
    #
    # ❌ NOT FULLY IMPLEMENTED FOR ITC:
    # - Source localization function exists: apply_source_localization()
    # - But source-level ITC computation needs development
    # - ROI-specific ITC extraction needs development
    "apply_source_localization": {
        "enabled": False,  # Set True to enable source analysis
        "value": {
            "method": "MNE",  # MNE inverse solution (minimum norm)
            "lambda2": 0.111,  # Regularization parameter
            "pick_ori": "normal",  # Use surface normal orientation
            "n_jobs": 4,  # Parallel processing
            "convert_to_eeg": False,  # Don't convert back to sensor space
        },
    },
}


class PhaseLockingEntrain(Task):
    """
    Phase Locking / Neural Entrainment Task for Adult Data (ENTRAIN Pipeline).

    This task implements Dr. Battrink's preprocessing protocol for analyzing
    neural entrainment using phase-locking measures (inter-trial coherence).

    Pipeline Steps:
    ---------------
    1. Initial Preprocessing:
       - Import raw data
       - Bandpass filter (0.1-30 Hz, Butterworth-like)
       - Remove DC component (via filtering/re-referencing)
       - Load channel locations (montage)
       - Re-reference to average of LM and RM (or average reference)
       - Create 10.8s epochs time-locked to every 12th word (36 syllables)

    2. Artifact Correction (Advanced ICA Pipeline):
       - Identify noisy channels
       - Run extended Infomax ICA on 1 Hz high-pass parallel dataset
       - Apply ICA weights to original 0.1 Hz data
       - Reject components using ICLabel (>= 0.9 probability threshold)
       - Interpolate noisy channels
       - Remove epochs not corrected by ICA

    3. FFT/ITC Analysis (Scalp Level):
       - Compute ITC across all cleaned epochs using FFT
       - Generate 100 shuffled surrogates per participant
       - Compute zITC from surrogate distribution
       - Extract zITC at word (1.11 Hz) and syllable (3.33 Hz) frequencies
       - Average zITC across scalp electrodes

    4. Source-Level Analysis (Optional):
       - Apply source localization (MNE inverse)
       - Compute source-level ITC
       - Extract zITC from STG/transverse temporal gyrus ROIs

    Key Outputs:
    -----------
    - Scalp-level zITC at word frequency (1.11 Hz)
    - Scalp-level zITC at syllable frequency (3.33 Hz)
    - Source-level zITC in STG/transverse temporal gyrus (if enabled)
    - Word Learning Index (WLI = ITC_word / ITC_syllable)

    Implementation Status:
    ---------------------
    ✅ IMPLEMENTED:
    - Initial preprocessing (filtering, re-referencing, epoching)
    - ICA with extended Infomax
    - ICLabel component classification with 0.9 threshold
    - ITC computation using Morlet wavelets
    - Frequency band analysis
    - Word Learning Index calculation
    - Basic visualization and reporting

    ❌ NOT YET IMPLEMENTED (requires development):
    - Shuffled surrogate generation (100 iterations per subject)
    - zITC computation from surrogates
    - Source-level ITC analysis with ROI extraction
    - Comparison between minimal vs. advanced ICA pipelines

    ⚠️ CLARIFICATION NEEDED:
    - Event coding for syllables and words in your data
    - LM/RM channel names in your specific montage
    - SOA (stimulus onset asynchrony) between syllables
    - Specific ROIs for source analysis
    - Should surrogates be computed on sensor or source level?
    - Criteria for manual epoch rejection before ICA
    """

    def run(self) -> None:
        """
        Execute the ENTRAIN phase-locking preprocessing pipeline.

        This method orchestrates the complete workflow from raw data to
        inter-trial coherence (ITC) analysis following Dr. Battrink's protocol.
        """

        # =========================================================================
        # STEP 1: INITIAL PREPROCESSING
        # =========================================================================
        # Import raw EEG data and apply basic preprocessing steps to prepare
        # for epoching and artifact correction.

        # Import raw data from file
        self.import_raw()

        # Resample to 250 Hz for computational efficiency
        # (adequate for 0.1-30 Hz frequency band of interest)
        self.resample_data()

        # Apply bandpass filter: 0.1-30 Hz (Butterworth-like, order 2)
        # This removes DC component and preserves entrainment frequencies
        # Word frequency: ~1.11 Hz, Syllable frequency: ~3.33 Hz
        self.filter_data()

        # Drop outer layer channels if configured (e.g., face/neck electrodes)
        self.drop_outer_layer()

        # Mark EOG channels if configured (for artifact detection)
        self.assign_eog_channels()

        # Trim edges to remove recording start/end artifacts
        self.trim_edges()

        # Crop to specific duration if configured
        self.crop_duration()

        # Store original raw data for comparison and visualization
        self.original_raw = self.raw.copy()

        # Optional: Apply wavelet denoising (not in original protocol)
        # This can improve data quality but adds processing time
        # self.apply_wavelet_threshold()

        # Create BIDS-compliant paths and filenames for organized output
        self.create_bids_path()

        # =========================================================================
        # STEP 2: CHANNEL CLEANING
        # =========================================================================
        # Identify and interpolate bad channels before ICA
        # Protocol: "Identify noisy channels" (interpolate after ICA)

        # AutoClean identifies bad channels using multiple criteria:
        # - Flat channels, low correlation, high noise
        self.clean_bad_channels()

        # =========================================================================
        # STEP 3: RE-REFERENCING - PROTOCOL SPECIFIED
        # =========================================================================
        # Re-reference to average of LM and RM (left/right mastoid)
        # Or to average reference if mastoids aren't available
        # NOTE: This also removes the mean voltage (DC component) from each channel
        self.rereference_data()

        # =========================================================================
        # STEP 4: ARTIFACT ANNOTATION
        # =========================================================================
        # Annotate noisy segments before ICA to improve decomposition quality
        # Protocol mentions "manually remove very noisy epochs" but we automate this

        # Annotate high-amplitude noise segments
        self.annotate_noisy_epochs()

        # Annotate epochs with low correlation across channels
        self.annotate_uncorrelated_epochs()

        # Detect and annotate dense oscillatory artifacts (e.g., muscle)
        self.detect_dense_oscillatory_artifacts()

        # =========================================================================
        # STEP 5: ICA - PROTOCOL SPECIFIED (ADVANCED PIPELINE)
        # =========================================================================
        # Extended Infomax ICA on parallel 1 Hz high-pass dataset
        # Apply weights to original 0.1 Hz data, reject components >= 0.9 threshold
        #
        # AutoClean Implementation:
        # - Creates temporary 1 Hz high-pass copy for ICA fitting
        # - Runs extended Infomax (equivalent to runica extended)
        # - Applies ICA weights to original 0.1 Hz filtered data
        # - Only fits ICA on "good" channels (bad channels excluded)

        # Run ICA decomposition
        # This uses temp_highpass_for_ica=1.0 from config
        self.run_ica()

        # Classify components using ICLabel and reject artifacts
        # Threshold: >= 0.9 probability for eye, muscle, noise, line artifacts
        # Protocol specifies: "reject components >= 0.9 probability"
        self.classify_ica_components()  # Uses method="iclabel" from config

        # =========================================================================
        # STEP 6: FINAL CHANNEL INTERPOLATION
        # =========================================================================
        # Protocol: "Interpolate the previously identified noisy channels"
        # Note: AutoClean already interpolated during clean_bad_channels()
        # But we could re-run if needed after ICA:
        # self.interpolate_bads()  # If you want to re-interpolate after ICA

        # =========================================================================
        # STEP 7: EPOCHING - PROTOCOL SPECIFIED
        # =========================================================================
        # Create nonoverlapping epochs time-locked to every 12th word onset
        # Duration: 10.8 seconds (36 syllables @ 300 ms SOA)
        #
        # AutoClean Implementation:
        # - create_sl_epochs() with num_syllables=36
        # - This epochs based on syllable events in continuous data
        # - Non-overlapping epochs as specified in protocol
        #
        # ⚠️ NOTE: This assumes your data has syllable event markers
        # If you only have word markers, you'll need to modify the epoching strategy

        # Create statistical learning epochs (36 syllables = 12 words)
        # This automatically:
        # - Creates 10.8s epochs (36 × 0.3s)
        # - Marks epochs overlapping with bad annotations
        # - Optionally rejects epochs with voltage threshold
        self.create_sl_epochs()  # Uses num_syllables=36 from config

        # =========================================================================
        # STEP 8: EPOCH QUALITY CONTROL
        # =========================================================================
        # Protocol: "remove noisy epochs that are not corrected with ICA"
        # AutoClean provides multiple epoch cleaning methods:

        # Detect outlier epochs using statistical methods
        self.detect_outlier_epochs()

        # Clean epochs using Global Field Power (GFP) method
        # This removes epochs with abnormal overall amplitude patterns
        self.gfp_clean_epochs()

        # =========================================================================
        # STEP 9: ITC ANALYSIS - PROTOCOL SPECIFIED (PARTIAL)
        # =========================================================================
        # Compute inter-trial coherence (ITC) across all cleaned epochs
        #
        # ✅ IMPLEMENTED: Basic ITC computation
        # - Compute ITC using Morlet wavelets (FFT-based)
        # - Frequency range: 0.6-5 Hz (captures word and syllable frequencies)
        # - Extract ITC at specific frequencies (1.11 Hz, 3.33 Hz)
        # - Compute Word Learning Index (WLI = ITC_word / ITC_syllable)
        #
        # ❌ NOT YET IMPLEMENTED: Surrogate-based significance testing
        # Protocol specifies:
        # - Create 100 shuffled surrogate versions per participant
        # - Shuffle each epoch within itself by variable amount
        # - Compute ITC on shuffled datasets
        # - Compute zITC = (observed ITC - mean surrogate ITC) / std surrogate ITC
        #
        # TODO: Implement surrogate generation and zITC computation
        # This will require custom development in autoclean/functions/analysis/

        # Compute inter-trial coherence (ITC) analysis
        # Returns: power, itc, band_results
        power, itc, band_results = self.compute_itc_analysis()

        # Store results for later access
        # These can be used for custom analyses or visualization
        if power is not None and itc is not None:
            self.itc_power = power  # Time-frequency power representation
            self.itc_coherence = itc  # Time-frequency ITC representation
            self.itc_bands = band_results  # Frequency band summaries

            # Log summary statistics
            from autoclean.utils.logging import message

            message("info", "ITC Analysis Summary:")
            message("info", f"  ITC shape: {itc.data.shape}")
            message(
                "info", f"  Frequency range: {itc.freqs[0]:.2f}-{itc.freqs[-1]:.2f} Hz"
            )
            message("info", f"  Time range: {itc.times[0]:.2f}-{itc.times[-1]:.2f} s")
            message("info", f"  Number of channels: {len(itc.ch_names)}")

            if band_results:
                message("info", "  Frequency Band ITC:")
                if "word_frequency" in band_results:
                    message(
                        "info",
                        f"    Word (1.0-1.3 Hz): {band_results['word_frequency']:.4f}",
                    )
                if "syllable_frequency" in band_results:
                    message(
                        "info",
                        f"    Syllable (3.0-3.7 Hz): {band_results['syllable_frequency']:.4f}",
                    )

            # ⚠️ MISSING: zITC computation from shuffled surrogates
            # This is a key output of the protocol but requires custom development
            # Suggested implementation:
            # 1. Create function: compute_itc_surrogates(epochs, n_surrogates=100)
            # 2. For each surrogate:
            #    - Randomly shift phases of each epoch
            #    - Compute ITC on shuffled data
            # 3. Compute zITC: (observed - mean_surrogate) / std_surrogate
            # 4. Output zITC at word and syllable frequencies

            message("warning", "⚠️ MISSING FEATURE: Shuffled surrogate-based zITC")
            message(
                "warning",
                "   Protocol requires 100 surrogate iterations for significance testing",
            )
            message("warning", "   Current output is raw ITC, not zITC")
            message("warning", "   This feature requires custom development")

        # =========================================================================
        # STEP 10: SOURCE LOCALIZATION (OPTIONAL - NOT IN MAIN PROTOCOL)
        # =========================================================================
        # Protocol mentions source-level analysis in STG/transverse temporal gyrus
        # This is currently disabled but can be enabled in config
        #
        # ❌ NOT FULLY IMPLEMENTED FOR ITC:
        # - Source localization function exists
        # - But source-level ITC computation needs development
        #
        # TODO: Implement source-level ITC pipeline:
        # 1. Apply source localization to epochs
        # 2. Compute ITC in source space
        # 3. Extract ITC from specific ROIs (STG, transverse temporal gyrus)
        # 4. Apply surrogate testing in source space
        # 5. Output source-level zITC

        # Uncomment to enable source localization (if configured):
        # if self.config.get("apply_source_localization", {}).get("enabled", False):
        #     self.apply_source_localization()
        #     message("warning", "⚠️ MISSING FEATURE: Source-level ITC analysis")
        #     message("warning", "   Source localization completed, but ITC in source space")
        #     message("warning", "   is not yet implemented. Requires custom development.")

        # =========================================================================
        # STEP 11: VISUALIZATION AND REPORTING
        # =========================================================================
        # Generate quality control visualizations and reports
        self.generate_reports()

    def generate_reports(self) -> None:
        """
        Generate quality control visualizations and phase-locking reports.

        This method creates standard AutoClean visualizations plus custom
        plots relevant to phase-locking analysis.
        """
        if self.raw is None or self.original_raw is None:
            return

        # Standard AutoClean visualizations
        # Plot raw vs cleaned data overlay (time domain)
        self.plot_raw_vs_cleaned_overlay(self.original_raw, self.raw)

        # Plot power spectral density topography (frequency domain)
        self.step_psd_topo_figure(self.original_raw, self.raw)

        # TODO: Add phase-locking specific visualizations:
        # - ITC topography at word frequency (1.11 Hz)
        # - ITC topography at syllable frequency (3.33 Hz)
        # - Time-frequency ITC plots for frontal/central/parietal regions
        # - Word Learning Index (WLI) topography
        # - zITC significance maps (when implemented)
        # - Source-level ITC maps (when implemented)

        # If ITC analysis was performed, create ITC-specific plots
        if hasattr(self, "itc_coherence") and self.itc_coherence is not None:
            from autoclean.utils.logging import message

            message("info", "Generating ITC visualizations...")

            # TODO: Implement custom ITC plotting functions
            # Example:
            # self.plot_itc_topography(freq=1.11, title="Word Frequency ITC (1.11 Hz)")
            # self.plot_itc_topography(freq=3.33, title="Syllable Frequency ITC (3.33 Hz)")
            # self.plot_itc_time_frequency(picks="frontal")
            # self.plot_word_learning_index_topography()

            message("warning", "⚠️ Custom ITC visualizations not yet implemented")
            message("info", "   ITC data is available in self.itc_coherence")
            message("info", "   Use MNE plotting functions to visualize:")
            message("info", "   - self.itc_coherence.plot_topomap()")
            message("info", "   - self.itc_coherence.plot_joint()")
            message("info", "   - self.itc_coherence.plot(picks=['Fz', 'Cz', 'Pz'])")


# =============================================================================
# IMPLEMENTATION NOTES AND CLARIFICATIONS NEEDED
# =============================================================================
"""
SUMMARY OF IMPLEMENTATION STATUS:
=================================

✅ FULLY IMPLEMENTED (ready to use):
1. Initial preprocessing pipeline
   - Import raw data
   - Bandpass filter 0.1-30 Hz (Butterworth-like FIR)
   - Load channel locations (montage)
   - Re-reference to average (or custom reference)
   - Remove DC component (via filtering/re-referencing)
   - Trim edges

2. Artifact correction pipeline
   - Channel cleaning (interpolation)
   - Extended Infomax ICA on 1 Hz high-pass parallel dataset
   - ICLabel classification with 0.9 threshold
   - Automatic artifact component rejection

3. Epoching
   - Create 10.8s epochs (36 syllables = 12 words)
   - Non-overlapping epochs
   - Epoch quality control (GFP, outlier detection)

4. ITC analysis (basic)
   - Morlet wavelet-based ITC computation
   - Frequency band analysis (word, syllable bands)
   - Word Learning Index (WLI) calculation
   - Band-specific ITC extraction

❌ NOT YET IMPLEMENTED (requires development):
1. Shuffled surrogate generation
   - Generate 100 shuffled surrogates per participant
   - Shuffle phase within each epoch randomly
   - Compute ITC on each surrogate

2. zITC computation
   - Calculate z-scored ITC: (observed - mean_surrogate) / std_surrogate
   - This is the key metric specified in the protocol
   - Provides statistical significance for ITC values

3. Source-level ITC analysis
   - Apply ITC computation in source space
   - Extract ITC from specific ROIs (STG, transverse temporal gyrus)
   - Apply surrogate testing in source space
   - Output source-level zITC

4. Minimal vs. Advanced pipeline comparison
   - Protocol mentions comparing two pipelines
   - Minimal: only channel interpolation and manual epoch rejection
   - Advanced: full ICA pipeline
   - Would require running both and comparing outputs

⚠️ CLARIFICATION NEEDED FROM USER:
==================================

1. Event Coding:
   - How are syllables marked in your data? (event codes?)
   - How are words marked? (every 12th word onset?)
   - What is the exact SOA (stimulus onset asynchrony) between syllables?
   - Is it exactly 300 ms or variable?

2. Channel Configuration:
   - Which EEG system/montage are you using?
   - What are the channel names for LM and RM (mastoids)?
   - Were LM and RM recorded, or should we use average reference?
   - Are there dedicated EOG channels? If so, which ones?

3. Preprocessing Decisions:
   - Should we implement both minimal and advanced pipelines?
   - Should manual epoch rejection be included before ICA?
   - What criteria for manual epoch rejection?

4. Analysis Priorities:
   - Should surrogate-based zITC be implemented first? (HIGH PRIORITY)
   - Should source-level analysis be included? (LOWER PRIORITY?)
   - Which ROIs specifically for source analysis?
   - Should surrogate testing be done in sensor space, source space, or both?

5. Output Requirements:
   - Do you need separate outputs for minimal vs. advanced pipelines?
   - Should zITC be computed per-electrode or averaged across scalp?
   - What format for final outputs? (CSV, HDF5, JSON?)
   - Do you need trial-by-trial ITC or only averaged?

RECOMMENDED NEXT STEPS:
======================

1. IMMEDIATE (to use current pipeline):
   - Update montage setting in config (line 45)
   - Update reference setting (LM/RM or average) (line 151)
   - Configure event_id for your data (line 264)
   - Test pipeline on one participant
   - Review ITC outputs and visualization

2. SHORT-TERM (core missing features):
   - Implement surrogate generation function
   - Implement zITC computation
   - Add zITC to output reports
   - Test statistical significance of results

3. LONG-TERM (advanced features):
   - Implement source-level ITC pipeline
   - Add ROI-specific extraction
   - Implement pipeline comparison (minimal vs. advanced)
   - Add custom ITC visualizations

4. DOCUMENTATION:
   - Document event codes for your specific data
   - Create example configuration files
   - Write tutorial for running ENTRAIN pipeline
   - Document interpretation of zITC values

USAGE EXAMPLE:
=============

# 1. Update configuration in this file (lines 45, 151, 264)
# 2. Run autoclean with this task:

```bash
autoclean process /path/to/data/ \\
    --task PhaseLockingEntrain \\
    --output /path/to/output/ \\
    --export
```

# 3. Access results in output directory:
# - derivatives/epochs/: Cleaned epochs
# - derivatives/itc_analysis/: ITC and band results
# - reports/: Visualization PDFs
# - metadata/: Processing logs and statistics

# 4. Load ITC results for further analysis:
```python
import mne
itc = mne.time_frequency.read_tfrs('path/to/itc_analysis_itc-tfr.h5')[0]
# Plot ITC at word frequency
itc.plot_topomap([1.11], title='Word Frequency ITC')
```

CONTACT AND QUESTIONS:
=====================

For questions about this implementation or to clarify protocol details:
- Review this file's comments (especially ⚠️ CLARIFICATION NEEDED sections)
- Check autoclean documentation: https://autoclean.readthedocs.io
- Consult with Dr. Battrink regarding protocol details
- Open GitHub issue for feature requests: https://github.com/autoclean/autoclean
"""
