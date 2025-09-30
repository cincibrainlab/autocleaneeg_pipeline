from autoclean.core.task import Task

# =============================================================================
#  specparam SPECTRAL PARAMETERIZATION TEST
# =============================================================================
# This task tests specparam (Fitting Oscillations & One Over F) analysis:
# 1. Source localization (MNE inverse)
# 2. specparam aperiodic parameters (1/f background)
# 3. specparam periodic parameters (oscillatory peaks)
#
# Tests:
# - apply_source_localization()
# - apply_fooof_aperiodic()
# - apply_fooof_periodic()
#
# Expected outputs:
# - derivatives/source_localization/*.h5 (STCs)
# - derivatives/fooof/*_psd-stc.h5 (vertex-level PSD)
# - derivatives/fooof/*_fooof_aperiodic.parquet, *.csv
# - derivatives/fooof/*_fooof_periodic.parquet, *.csv
#
# References:
# - Donoghue T, et al. (2020). Parameterizing neural power spectra into
#   periodic and aperiodic components. Nature Neuroscience, 23(12), 1655-1665.
# =============================================================================

config = {
    "schema_version": "2025.09",
    "resample_step": {"enabled": True, "value": 250},
    "filtering": {
        "enabled": True,
        "value": {
            "l_freq": 1,
            "h_freq": 100,
            "notch_freqs": [60, 120],
            "notch_widths": 5,
        },
    },
    "drop_outerlayer": {"enabled": False, "value": []},
    "eog_step": {"enabled": False, "value": []},
    "trim_step": {"enabled": False, "value": 0},
    "montage": {"enabled": True, "value": "standard_1020"},
    "reference_step": {"enabled": True, "value": "average"},
    "crop_step": {"enabled": True, "value": {"start": 0, "end": 60}},  # 60s for speed
    "ICA": {"enabled": False, "value": {"method": "infomax"}},
    "component_rejection": {
        "enabled": False,
        "method": "iclabel",
        "value": {
            "ic_flags_to_reject": [],
            "ic_rejection_threshold": 0.3,
        },
    },
    "epoch_settings": {
        "enabled": False,
        "value": {"tmin": -1, "tmax": 1},
        "event_id": None,
        "remove_baseline": {"enabled": False, "window": [None, 0]},
        "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 0.000125}},
    },
    "apply_source_localization": {
        "enabled": True,
        "value": {
            "method": "MNE",
            "lambda2": 0.111,
            "pick_ori": "normal",
            "n_jobs": 10,
            "convert_to_eeg": False,
        },
    },
    "apply_fooof_aperiodic": {
        "enabled": True,
        "value": {
            "fmin": 1.0,
            "fmax": 45.0,
            "n_jobs": 4,
            "aperiodic_mode": "knee",
        },
    },
    "apply_fooof_periodic": {
        "enabled": True,
        "value": {
            "freq_bands": {
                "delta": (1, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 45),
            },
            "n_jobs": 4,
            "aperiodic_mode": "knee",
        },
    },
}


class specparamAnalysisTest(Task):
    """Test specparam spectral parameterization pipeline.

    This task performs complete spectral parameterization of source-localized
    EEG data using the specparam algorithm. It separates neural power spectra into:

    1. Aperiodic component (1/f background):
       - Offset: Overall power level
       - Exponent: Slope of 1/f decay
       - Knee: Bend point (optional)

    2. Periodic component (oscillatory peaks):
       - Center frequency: Peak location in Hz
       - Power: Peak amplitude
       - Bandwidth: Peak width

    The pipeline uses vertex-level analysis for high spatial resolution across
    the cortical surface (20,484 vertices).

    Note: specparam requires continuous data (self.stc), not epochs. Source
    localization processes Raw data to create self.stc.
    """

    def run(self) -> None:
        # Import and basic preprocessing
        self.import_raw()
        self.resample_data()
        self.filter_data()
        self.crop_duration()
        self.rereference_data()

        # Source analysis pipeline (no epochs - specparam needs continuous data)
        self.apply_source_localization()  # Creates self.stc from Raw
        self.apply_fooof_aperiodic()      # Extracts 1/f parameters
        self.apply_fooof_periodic()       # Extracts oscillatory peaks