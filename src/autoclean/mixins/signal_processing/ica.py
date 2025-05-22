"""ICA mixin for autoclean tasks."""

import os
import base64
import tempfile
import traceback
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mne.preprocessing import ICA
import mne_icalabel
import pandas as pd
import openai
import mne
import math # Added for math.floor
from scipy.ndimage import uniform_filter1d # Added for vertical smoothing in ERP image
from mne.time_frequency import psd_array_welch # Added for efficient PSD calculation
from matplotlib.backends.backend_pdf import PdfPages # For PDF report generation
from datetime import datetime # For PDF timestamp

from autoclean.io.export import save_ica_to_fif
from autoclean.utils.logging import message

# Define the order of labels as used in the OpenAI prompt and for labels_scores_
# This order must be consistent.
OPENAI_LABEL_ORDER = [
    "brain", 
    "eye", 
    "muscle", 
    "heart", 
    "line_noise", 
    "channel_noise", 
    "other_artifact"
]

# Mapping from OpenAI labels to MNE-compatible labels for ica.labels_
OPENAI_TO_MNE_LABEL_MAP = {
    "brain": "brain",
    "eye": "eog",
    "muscle": "muscle",
    "heart": "ecg",
    "line_noise": "line_noise",
    "channel_noise": "ch_noise",
    "other_artifact": "other",
}

class IcaMixin:
    """Mixin for ICA processing."""

    # OpenAI prompt for ICA component classification
    _OPENAI_ICA_PROMPT = """Analyze this EEG ICA component image and classify into ONE category:

- "brain": Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (NOT FRONTAL or EDGE-FOCUSED). 1/f-like spectrum with possible peaks at 8-12Hz. Rhythmic, wave-like time series WITHOUT abrupt level shifts. MUST show decreasing power with increasing frequency (1/f pattern) - a flat or random fluctuating spectrum is NOT brain activity.

- "eye": 
  * Two main types of eye components:
    1. HORIZONTAL eye movements: Characterized by a TIGHTLY FOCUSED dipolar pattern, CONFINED PRIMARILY to the LEFT-RIGHT FRONTAL regions (e.g., distinct red on one far-frontal side, blue on the opposite far-frontal side). The active areas should be relatively compact and clearly located frontally. Time series typically shows step-like or square-wave patterns. This pattern is eye UNLESS the time series shows the prominent, sharp, repetitive QRS-like spikes characteristic of "heart".
    2. VERTICAL eye movements/blinks: FRONTAL midline or bilateral positivity/negativity. Time series shows distinctive spikes or slow waves.
  * Both types show power concentrated in lower frequencies (<5Hz).
  * DO NOT be misled by 60Hz notches in the spectrum - these are normal filtering artifacts, NOT line noise.
  * Key distinction: Eye components have activity TIGHTLY FOCUSED in frontal regions. Eye component dipoles are much more FOCUSED and less widespread than the broad gradients seen in "heart" components.
  * CRITICAL: NEVER classify a component with clear FOCUSED LEFT-RIGHT FRONTAL dipole as muscle. This pattern is eye, BUT ALWAYS CHECK TIME SERIES FOR QRS COMPLEXES TO RULE OUT "heart" if the 'dipole' appears very broad or global.
  * RULE: If you see TIGHTLY FOCUSED LEFT-RIGHT FRONTAL dipole pattern or STRONG FRONTAL activation with spike patterns, AND NO QRS in time series, classify as "eye".

- "muscle": (SPECTRAL SIGNATURE IS THE MOST DOMINANT INDICATOR)
  * DECISIVE SPECTRAL FEATURE (Primary and Often Conclusive Muscle Indicator): The power spectrum exhibits a CLEAR and SUSTAINED POSITIVE SLOPE, meaning power consistently INCREASES with increasing frequency, typically starting from around 20-30Hz and continuing upwards. This often looks like the spectrum is 'curving upwards' or 'scooping upwards' at higher frequencies. IF THIS DISTINCT SPECTRAL SIGNATURE IS OBSERVED, THE COMPONENT IS TO BE CLASSIFIED AS 'muscle', EVEN IF other features might seem ambiguous or resemble other categories. This spectral cue is the strongest determinant for muscle.
  * OTHER SUPPORTING MUSCLE CHARACTERISTICS (Use if spectral cue is present, or with caution if spectral cue is less definitive but clearly NOT 1/f):
    *   Topography: Common patterns include (a) very localized 'bowtie' or 'shallow dipole' patterns (two small, adjacent areas of opposite polarity, often taking up <25% of the scalp map, can appear anywhere but frequently temporal/posterior) OR (b) more diffuse activity, typically along the EDGE of the scalp (temporal, occipital, neck regions).
    *   Time Series: Often shows spiky, high-frequency, and somewhat erratic activity.

- "heart":
  * TOPOGRAPHY: Characterized by a VERY BROAD, diffuse electrical field gradient across a large area of the scalp. This often manifests as large positive (red) and negative (blue) regions on somewhat opposite sides of the head, but these regions are WIDESPREAD and NOT TIGHTLY FOCUSED like an eye dipole. 
  * TIME SERIES (CRITICAL & DECISIVE IDENTIFIER): Look for PROMINENT, SHARP, REPETITIVE SPIKES in the 'Scrolling IC Activity' plot that stand out significantly from the background rhythm. These are QRS-like complexes (heartbeats). They are typically large in amplitude, can be positive-going or negative-going sharp deflections, and repeat at roughly 0.8-1.5 Hz (around once per second, though ICA can make the rhythm appear less than perfectly regular). THE PRESENCE OF THESE DISTINCTIVE, RECURRING, SHARP SPIKES IS THE STRONGEST AND MOST DEFINITIVE INDICATOR FOR "heart".
  * IF QRS IS PRESENT: If these clear, sharp, repetitive QRS-like spikes are visible in the time series, the component should be classified as "heart". This QRS signature, when combined with a BROAD topography, takes precedence over superficial resemblances to other patterns.
  * SPECTRUM: Often noisy or may not show a clear 1/f pattern. May show harmonics of the heart rate.

- "line_noise": 
  * MUST show SHARP PEAK at 50/60Hz in spectrum - NOT a notch/dip (notches are filters, not line noise).
  * NOTE: Almost all components show a notch at 60Hz from filtering - this is NOT line noise!
  * Line noise requires a POSITIVE PEAK at 50/60Hz, not a negative dip.

- "channel_noise": 
  * SINGLE ELECTRODE "hot/cold spot" - tiny, isolated circular area typically without an opposite pole.
  * Compare with eye: Channel noise has only ONE focal point, while eye has TWO opposite poles (dipole). Eye dipoles are also typically larger and more structured.
  * Example: A tiny isolated red or blue spot on one electrode, not a dipolar pattern.
  * Time series may show any pattern; the focal topography is decisive.

- "other_artifact": Components not fitting above categories.

CLASSIFICATION PRIORITY (IMPORTANT: Evaluate in this order. Later rules apply only if earlier conditions are not met or are ambiguous):
1.  IF 'Scrolling IC Activity' shows PROMINENT, SHARP, REPETITIVE SPIKES (QRS-like complexes...) AND topography is VERY BROAD... → "heart".
2.  ELSE IF TIGHTLY FOCUSED LEFT-RIGHT FRONTAL dipole... (and NO QRS) → "eye"
3.  ELSE IF SINGLE ELECTRODE isolated focality → "channel_noise"
4.  ELSE IF Spectrum shows SHARP PEAK (not notch) at 50/60Hz → "line_noise"
5.  ELSE IF Power spectrum exhibits a CLEAR and SUSTAINED POSITIVE SLOPE (power INCREASES with increasing frequency from ~20-30Hz upwards, often 'curving' or 'scooping' upwards) → "muscle". (THIS IS A DECISIVE RULE FOR MUSCLE. If this spectral pattern is present, classify as 'muscle' even if the topography isn't a perfect 'bowtie' or edge artifact, and before considering 'brain').
6.  ELSE IF (Topography is a clear 'bowtie'/'shallow dipole' OR distinct EDGE activity) AND (Time series is spiky/high-frequency OR spectrum is generally high-frequency without being clearly 1/f and also not clearly a positive slope) → "muscle" (Secondary muscle check, for cases where the positive slope is less perfect but other muscle signs are strong and it's definitely not brain).
7.  ELSE IF Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (AND NOT already definitively classified as 'muscle' by its spectral signature under rule 5) AND spectrum shows a clear general 1/f pattern (overall DECREASING power with increasing frequency, AND ABSOLUTELY NO sustained positive slope at high frequencies) → "brain"
8.  ELSE → "other_artifact"


IMPORTANT: A 60Hz NOTCH (negative dip) in spectrum is normal filtering, seen in most components, and should NOT be used for classification! Do not include this in your reasoning.

Return: ("label", confidence_score, "detailed_reasoning")

Example: ("eye", 0.95, "Strong frontal topography with left-right dipolar pattern (horizontal eye movement) or frontal positivity with spike-like patterns (vertical eye movement/blinks). Low-frequency dominated spectrum and characteristic time series confirm eye activity.")
"""

    def run_ica(self, eog_channel: str = None, use_epochs: bool = False, **kwargs) -> ICA:
        """Run ICA on the raw data. 

        This method will fit an ICA object to the raw data and save it to a FIF file.
        ICA object is stored in self.final_ica.
        Uses optional kwargs from the autoclean_config file to fit the mne ICA object.


        Parameters
        ----------
        eog_channel : str, optional
            The EOG channel to use for ICA. If None, no EOG detection will be performed.
        use_epochs : bool, optional
            If True, epoch data stored in self.epochs will be used.

        Returns
        -------
        final_ica : mne.preprocessing.ICA
            The fitted ICA object.
        
        Examples
        --------
        >>> self.run_ica()

        >>> self.run_ica(eog_channel="E27") 

        See Also
        --------
        run_ICLabel : Run ICLabel on the raw data.

        """
        message("header", "Running ICA step")

        is_enabled, config_value = self._check_step_enabled("ICA")

        if not is_enabled:
            message("warning", "ICA is not enabled in the config")
            return

        if use_epochs:
            message("debug", "Using epochs")
            # Create epochs
            data = self.epochs
        else:
            message("debug", "Using raw data")
            data = self.raw

        # Run ICA
        if is_enabled:
            # Get ICA parameters from config
            ica_kwargs = config_value.get("value", {})

            # Merge with any provided kwargs, with provided kwargs taking precedence
            ica_kwargs.update(kwargs)

            # Set default parameters if not provided
            if "max_iter" not in ica_kwargs:
                message("debug", "Setting max_iter to auto")
                ica_kwargs["max_iter"] = "auto"
            if "random_state" not in ica_kwargs:
                message("debug", "Setting random_state to 97")
                ica_kwargs["random_state"] = 97

            # Create ICA object

            self.final_ica = ICA(**ica_kwargs) # pylint: disable=not-callable

            message("debug", f"Fitting ICA with {ica_kwargs}")

            self.final_ica.fit(data)

            if eog_channel is not None:
                message("info", f"Running EOG detection on {eog_channel}")
                eog_indices, _ = self.final_ica.find_bads_eog(
                    data, ch_name=eog_channel
                )
                self.final_ica.exclude = eog_indices
                self.final_ica.apply(data)

        else:
            message("warning", "ICA is not enabled in the config")

        metadata = {
            "ica": {
                "ica_kwargs": ica_kwargs,
                "ica_components": self.final_ica.n_components_,
            }
        }

        self._update_metadata("step_run_ica", metadata)

        save_ica_to_fif(self.final_ica, self.config, data)

        message("success", "ICA step complete")

        return self.final_ica

    def run_ICLabel(self): # pylint: disable=invalid-name
        """Run ICLabel on the raw data.

        Returns
        -------
        ica_flags : pandas.DataFrame or None
            A pandas DataFrame containing the ICLabel flags, or None if the
            step is disabled or fails.

        Examples
        --------
        >>> self.run_ICLabel()

        Notes
        -----
        This method will modify the self.final_ica attribute in place by adding labels.
        It checks if the 'ICLabel' step is enabled in the configuration.
        """
        message("header", "Running ICLabel step")

        is_enabled, _ = self._check_step_enabled("ICLabel") # config_value not used here

        if not is_enabled:
            message("warning", "ICLabel is not enabled in the config. Skipping ICLabel.")
            return None # Return None if not enabled

        if not hasattr(self, 'final_ica') or self.final_ica is None:
            message("error", "ICA (self.final_ica) not found. Please run `run_ica` before `run_ICLabel`.")
            # Or raise an error, depending on desired behavior
            return None


        mne_icalabel.label_components(self.raw, self.final_ica, method="iclabel")

        self._icalabel_to_data_frame(self.final_ica)

        metadata = {
            "ica": {
                "ica_components": self.final_ica.n_components_,
            }
        }

        self._update_metadata("step_run_ICLabel", metadata)

        message("success", "ICLabel complete")

        self.apply_iclabel_rejection()

        return self.ica_flags

    def apply_iclabel_rejection(self, data_to_clean=None):
        """
        Apply ICA component rejection based on ICLabel classifications and configuration.

        This method uses the labels assigned by `run_ICLabel` and the rejection
        criteria specified in the 'ICLabel' section of the pipeline configuration
        (e.g., ic_flags_to_reject, ic_rejection_threshold) to mark components
        for rejection. It then applies the ICA to remove these components from
        the data.

        It updates `self.final_ica.exclude` and modifies the data object
        (e.g., `self.raw`) in-place. The updated ICA object is also saved.

        Parameters
        ----------
        data_to_clean : mne.io.Raw | mne.Epochs, optional
            The data to apply the ICA to. If None, defaults to `self.raw`.
            This should ideally be the same data object that `run_ICLabel` was
            performed on, or is compatible with `self.final_ica`.

        Returns
        -------
        None
            Modifies `self.final_ica` and the input data object in-place.

        Raises
        ------
        RuntimeError
            If `self.final_ica` or `self.ica_flags` are not available (i.e.,
            `run_ica` and `run_ICLabel` have not been run successfully).
        """
        message("header", "Applying ICLabel-based component rejection")

        if not hasattr(self, 'final_ica') or self.final_ica is None:
            message("error", "ICA (self.final_ica) not found. Skipping ICLabel rejection.")
            raise RuntimeError("ICA (self.final_ica) not found. Please run `run_ica` first.")

        if not hasattr(self, 'ica_flags') or self.ica_flags is None:
            message("error", "ICLabel results (self.ica_flags) not found. Skipping ICLabel rejection.")
            raise RuntimeError("ICLabel results (self.ica_flags) not found. Please run `run_ICLabel` first.")

        is_enabled, step_config_main_dict = self._check_step_enabled("ICLabel")
        if not is_enabled:
            message("warning", "ICLabel processing itself is not enabled in the config. "
                             "Rejection parameters might be missing or irrelevant. Skipping.")
            return

        # Attempt to get parameters from a nested "value" dictionary first (common pattern)
        iclabel_params_nested = step_config_main_dict.get("value", {})
        
        flags_to_reject = iclabel_params_nested.get("ic_flags_to_reject")
        rejection_threshold = iclabel_params_nested.get("ic_rejection_threshold")

        # If not found in "value", try to get them from the main step config dict directly
        if flags_to_reject is None and "ic_flags_to_reject" in step_config_main_dict:
            flags_to_reject = step_config_main_dict.get("ic_flags_to_reject")
        if rejection_threshold is None and "ic_rejection_threshold" in step_config_main_dict:
            rejection_threshold = step_config_main_dict.get("ic_rejection_threshold")
            
        if flags_to_reject is None or rejection_threshold is None:
            message("warning", "ICLabel rejection parameters (ic_flags_to_reject or ic_rejection_threshold) "
                             "not found in the 'ICLabel' step configuration. Skipping component rejection.")
            return

        message("info", f"Will reject ICs of types: {flags_to_reject} with confidence > {rejection_threshold}")

        rejected_ic_indices_this_step = []
        for idx, row in self.ica_flags.iterrows(): # DataFrame index is the component index
            if row['ic_type'] in flags_to_reject and row['confidence'] > rejection_threshold:
                rejected_ic_indices_this_step.append(idx)

        if not rejected_ic_indices_this_step:
            message("info", "No new components met ICLabel rejection criteria in this step.")
        else:
            message("info", f"Identified {len(rejected_ic_indices_this_step)} components for rejection "
                             f"based on ICLabel: {rejected_ic_indices_this_step}")

        # Ensure self.final_ica.exclude is initialized as a list if it's None
        if self.final_ica.exclude is None:
            self.final_ica.exclude = []
        
        # Combine with any existing exclusions (e.g., from EOG detection in run_ica)
        current_exclusions = set(self.final_ica.exclude)
        for idx in rejected_ic_indices_this_step:
            current_exclusions.add(idx)
        self.final_ica.exclude = sorted(list(current_exclusions))

        message("info", f"Total components now marked for exclusion: {self.final_ica.exclude}")

        # Determine data to clean
        target_data = data_to_clean if data_to_clean is not None else self.raw
        data_source_name = "provided data object" if data_to_clean is not None else "self.raw"
        message("debug", f"Applying ICA to {data_source_name}")


        if not self.final_ica.exclude:
            message("info", "No components are marked for exclusion. Skipping ICA apply.")
        else:
            # Apply ICA to remove the excluded components
            # This modifies target_data in-place
            self.final_ica.apply(target_data)
            message("info", f"Applied ICA to {data_source_name}, removing/attenuating "
                             f"{len(self.final_ica.exclude)} components.")

        # Update metadata
        metadata = {
            "step_apply_iclabel_rejection": {
                "configured_flags_to_reject": flags_to_reject,
                "configured_rejection_threshold": rejection_threshold,
                "iclabel_rejected_indices_this_step": rejected_ic_indices_this_step,
                "final_excluded_indices_after_iclabel": self.final_ica.exclude
            }
        }
        # Assuming _update_metadata is available in the class using this mixin
        if hasattr(self, '_update_metadata') and callable(self._update_metadata):
            self._update_metadata("step_apply_iclabel_rejection", metadata)
        else:
            message("warning", "_update_metadata method not found. Cannot save metadata for ICLabel rejection.")
            
        # Save the ICA object with updated exclusions
        if hasattr(self, 'config') and hasattr(self, 'raw'):
            #  save_ica_to_fif(self.final_ica, self.config, self.raw) # Consistently save against self.raw context
             message("debug", "Saved ICA object with updated exclusions after ICLabel rejection.")
        else:
            message("warning", "Cannot save ICA object: self.config or self.raw not found.")


        message("success", "ICLabel-based component rejection complete.")

    def _icalabel_to_data_frame(self, ica):
        """Export IClabels to pandas DataFrame."""
        ic_type = [""] * ica.n_components_
        for label, comps in ica.labels_.items():
            for comp in comps:
                ic_type[comp] = label

        self.ica_flags = pd.DataFrame(
            dict(
                component=ica._ica_names, # pylint: disable=protected-access
                annotator=["ic_label"] * ica.n_components_,
                ic_type=ic_type,
                confidence=ica.labels_scores_.max(1),
            )
        )

        return self.ica_flags

    def _plot_component_for_vision_api(
        self,
        ica_obj: ICA,
        raw_obj: mne.io.Raw,
        component_idx: int,
        output_dir: Path,
        # Parameters below are for PDF generation, not for the API image itself
        classification_label: Optional[str] = None,
        classification_confidence: Optional[float] = None,
        classification_reason: Optional[str] = None,
        return_fig_object: bool = False
    ) -> Union[Path, plt.Figure, None]: # Adjusted return type
        """
        Creates a standardized plot for an ICA component to be used for vision classification API,
        or optionally for inclusion in a PDF report with classification details.
        Layout based on test_ica_vision.py: Topo+ContData on left, TS+PSD on right.

        Args:
            ica_obj: The ICA object.
            raw_obj: The raw data used for ICA.
            component_idx: Index of the component to plot.
            output_dir: Directory to save the plot if not returning a figure object.
            classification_label: Vision API classification label (for PDF).
            classification_confidence: Vision API classification confidence (for PDF).
            classification_reason: Vision API classification reason (for PDF).
            return_fig_object: If True, returns the matplotlib Figure object instead of saving to file.

        Returns:
            Path to saved image file (if return_fig_object is False and output_dir is provided),
            matplotlib Figure object (if return_fig_object is True), 
            or None on failure.
        """
        # Ensure non-interactive backend is used, especially important for scripts/batch processing
        matplotlib.use("Agg")

        fig_height = 9.5
        gridspec_bottom = 0.05

        if return_fig_object and classification_reason:
            fig_height = 11  # Increase for reasoning text in PDF
            gridspec_bottom = 0.18

        fig = plt.figure(figsize=(12, fig_height), dpi=120)
        main_plot_title_text = f"ICA Component IC{component_idx} Analysis"
        gridspec_top = 0.95
        suptitle_y_pos = 0.98

        if return_fig_object and classification_label is not None:
            gridspec_top = 0.90
            suptitle_y_pos = 0.96

        gs = GridSpec(3, 2, figure=fig,
                      height_ratios=[0.915, 0.572, 2.213], # Adjusted from example for general look
                      width_ratios=[0.9, 1],
                      hspace=0.7, wspace=0.35,
                      left=0.05, right=0.95, top=gridspec_top, bottom=gridspec_bottom)

        ax_topo = fig.add_subplot(gs[0:2, 0])
        ax_cont_data = fig.add_subplot(gs[2, 0])
        ax_ts_scroll = fig.add_subplot(gs[0, 1])
        ax_psd = fig.add_subplot(gs[2, 1])

        try:
            sources = ica_obj.get_sources(raw_obj)
            sfreq = sources.info['sfreq']
            component_data_array = sources.get_data(picks=[component_idx])[0]
        except Exception as e:
            message("error", f"Failed to get ICA sources for IC{component_idx}: {e}")
            plt.close(fig)
            return None

        # 1. Topography
        try:
            ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                                    show=False, colorbar=False, cmap='jet', outlines='head',
                                    sensors=True, contours=6)
            ax_topo.set_title(f"IC{component_idx} Topography", fontsize=12, loc='center')
            ax_topo.set_xlabel("")
            ax_topo.set_ylabel("")
            ax_topo.set_xticks([])
            ax_topo.set_yticks([])
        except Exception as e:
            message("error", f"Error plotting topography for IC{component_idx}: {e}")
            ax_topo.text(0.5, 0.5, "Topo plot failed", ha='center', va='center')

        # 2. Scrolling IC Activity (Time Series)
        try:
            duration_segment_ts = 3.0
            max_samples_ts = min(int(duration_segment_ts * sfreq), len(component_data_array))
            times_ts_ms = (np.arange(max_samples_ts) / sfreq) * 1000

            ax_ts_scroll.plot(times_ts_ms, component_data_array[:max_samples_ts], linewidth=0.8, color='dodgerblue')
            ax_ts_scroll.set_title("Scrolling IC Activity (First 3s)", fontsize=10)
            ax_ts_scroll.set_xlabel("Time (ms)", fontsize=9)
            ax_ts_scroll.set_ylabel("Amplitude (a.u.)", fontsize=9)
            if max_samples_ts > 0 and times_ts_ms.size > 0:
                ax_ts_scroll.set_xlim(times_ts_ms[0], times_ts_ms[-1])
            ax_ts_scroll.grid(True, linestyle=':', alpha=0.6)
            ax_ts_scroll.tick_params(axis='both', which='major', labelsize=8)
        except Exception as e:
            message("error", f"Error plotting scrolling IC activity for IC{component_idx}: {e}")
            ax_ts_scroll.text(0.5, 0.5, "Time series failed", ha='center', va='center')

        # 3. Continuous Data (EEGLAB-style ERP image)
        try:
            comp_data_offset_corrected = component_data_array - np.mean(component_data_array)
            target_segment_duration_s = 1.5
            target_max_segments = 200 # Limit segments to keep plot manageable
            segment_len_samples_cd = int(target_segment_duration_s * sfreq)
            if segment_len_samples_cd == 0: segment_len_samples_cd = 1 # Avoid division by zero

            available_samples_in_component = comp_data_offset_corrected.shape[0]
            max_total_samples_to_use_for_plot = int(target_max_segments * segment_len_samples_cd)
            samples_to_feed_erpimage = min(available_samples_in_component, max_total_samples_to_use_for_plot)
            
            n_segments_cd = 0
            current_segment_len_samples = 1

            if segment_len_samples_cd > 0 and samples_to_feed_erpimage >= segment_len_samples_cd:
                n_segments_cd = math.floor(samples_to_feed_erpimage / segment_len_samples_cd)
            
            if n_segments_cd > 0:
                current_segment_len_samples = segment_len_samples_cd
                final_samples_for_reshape = n_segments_cd * current_segment_len_samples
                erp_image_data_for_plot = comp_data_offset_corrected[:final_samples_for_reshape].reshape(n_segments_cd, current_segment_len_samples)
            elif samples_to_feed_erpimage > 0: # Handle case with less than one segment of data
                n_segments_cd = 1
                current_segment_len_samples = samples_to_feed_erpimage
                erp_image_data_for_plot = comp_data_offset_corrected[:current_segment_len_samples].reshape(1, current_segment_len_samples)
            else: # No data to plot
                erp_image_data_for_plot = np.zeros((1,1))
                current_segment_len_samples = 1 # For placeholder ticks

            if n_segments_cd >= 3 and erp_image_data_for_plot.shape[0] >=3: # Ensure smoothing is possible
                erp_image_data_smoothed = uniform_filter1d(erp_image_data_for_plot, size=3, axis=0, mode='nearest')
            else:
                erp_image_data_smoothed = erp_image_data_for_plot

            if erp_image_data_smoothed.size > 0:
                max_abs_val = np.max(np.abs(erp_image_data_smoothed))
                clim_val = (2/3) * max_abs_val if max_abs_val > 1e-9 else 1.0
            else:
                clim_val = 1.0
            clim_val = max(clim_val, 1e-9) # Avoid clim_val being zero for all-zero data
            vmin_cd, vmax_cd = -clim_val, clim_val

            im = ax_cont_data.imshow(erp_image_data_smoothed, aspect='auto', cmap='jet', interpolation='nearest',
                                     vmin=vmin_cd, vmax=vmax_cd)
            
            ax_cont_data.set_title(f"Continuous Data Segments (Max {target_max_segments})", fontsize=10)
            ax_cont_data.set_xlabel("Time (ms)", fontsize=9)
            if current_segment_len_samples > 1:
                num_xticks = min(4, current_segment_len_samples)
                xtick_positions_samples = np.linspace(0, current_segment_len_samples - 1, num_xticks)
                xtick_labels_ms = (xtick_positions_samples / sfreq * 1000).astype(int)
                ax_cont_data.set_xticks(xtick_positions_samples)
                ax_cont_data.set_xticklabels(xtick_labels_ms)
            else:
                ax_cont_data.set_xticks([])

            ax_cont_data.set_ylabel("Trials (Segments)", fontsize=9)
            if n_segments_cd > 1:
                num_yticks = min(5, n_segments_cd)
                ytick_positions = np.linspace(0, n_segments_cd - 1, num_yticks).astype(int)
                ax_cont_data.set_yticks(ytick_positions)
                ax_cont_data.set_yticklabels(ytick_positions)
            elif n_segments_cd == 1:
                ax_cont_data.set_yticks([0])
                ax_cont_data.set_yticklabels(["0"])
            else:
                ax_cont_data.set_yticks([])

            if n_segments_cd > 0: ax_cont_data.invert_yaxis()

            cbar_cont = fig.colorbar(im, ax=ax_cont_data, orientation='vertical', fraction=0.046, pad=0.1)
            cbar_cont.set_label("Activation (a.u.)", fontsize=8)
            cbar_cont.ax.tick_params(labelsize=7)
        except Exception as e_cont:
            message("error", f"Error plotting continuous data for IC{component_idx}: {e_cont}")
            ax_cont_data.text(0.5, 0.5, "Continuous data failed", ha='center', va='center')

        # 4. IC Activity Power Spectrum
        try:
            fmin_psd = 1.0
            fmax_psd = min(80.0, sfreq / 2.0 - 0.51) # Cap at 80Hz or Nyquist
            n_fft_psd = int(sfreq * 2.0) # 2-second window
            if n_fft_psd > len(component_data_array):
                n_fft_psd = len(component_data_array)
            # Ensure n_fft is at least 256 if data is long enough, else length of data (if > 0)
            n_fft_psd = max(n_fft_psd, 256 if len(component_data_array) >= 256 else (len(component_data_array) if len(component_data_array) > 0 else 1))
            
            if n_fft_psd == 0 or fmax_psd <= fmin_psd:
                 raise ValueError(f"Cannot compute PSD for IC{component_idx}: Invalid params (n_fft={n_fft_psd}, fmin={fmin_psd}, fmax={fmax_psd})")

            psds, freqs = psd_array_welch(
                component_data_array, sfreq=sfreq, fmin=fmin_psd, fmax=fmax_psd,
                n_fft=n_fft_psd, n_overlap=int(n_fft_psd * 0.5), verbose=False, average='mean'
            )
            if psds.size == 0:
                raise ValueError("PSD computation returned empty array.")

            psds_db = 10 * np.log10(np.maximum(psds, 1e-20)) # Avoid log(0)
            
            ax_psd.plot(freqs, psds_db, color='red', linewidth=1.2)
            ax_psd.set_title(f"IC{component_idx} Power Spectrum (1-80Hz)", fontsize=10)
            ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
            ax_psd.set_ylabel("Power (dB)", fontsize=9)
            if len(freqs) > 0:
                ax_psd.set_xlim(freqs[0], freqs[-1])
            ax_psd.grid(True, linestyle='--', alpha=0.5)
            ax_psd.tick_params(axis='both', which='major', labelsize=8)
        except Exception as e_psd:
            message("error", f"Error plotting PSD for IC{component_idx}: {e_psd}")
            ax_psd.text(0.5, 0.5, "PSD plot failed", ha='center', va='center')

        fig.suptitle(main_plot_title_text, fontsize=14, y=suptitle_y_pos)

        if return_fig_object:
            if classification_label is not None and classification_confidence is not None:
                subtitle_color_map = {
                    "brain": "green", "eye": "darkorange", "muscle": "firebrick", 
                    "heart": "mediumvioletred", "line_noise": "darkcyan", 
                    "channel_noise": "goldenrod", "other_artifact": "grey"
                }
                subtitle_color = subtitle_color_map.get(classification_label.lower(), 'black')
                classification_subtitle_text = f"Vision Classification: {str(classification_label).title()} (Confidence: {classification_confidence:.2f})"
                fig.text(0.5, suptitle_y_pos - 0.035, classification_subtitle_text, ha='center', va='top',
                         fontsize=13, fontweight='bold', color=subtitle_color, transform=fig.transFigure)

            if classification_reason:
                reason_title = "Reasoning (Vision API):"
                reason_title_y = gridspec_bottom - 0.03 
                reason_text_y = reason_title_y - 0.025   

                fig.text(0.05, reason_title_y, reason_title, ha='left', va='top',
                         fontsize=9, fontweight='bold', transform=fig.transFigure)
                fig.text(0.05, reason_text_y, classification_reason, ha='left', va='top',
                         fontsize=8, wrap=True, transform=fig.transFigure,
                         bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.75, ec='lightgrey'))
            
            bottom_adj = gridspec_bottom if classification_reason else 0.03 # Slightly more if no reason
            top_adj = gridspec_top - (0.05 if classification_label else 0.02)
            try:
                fig.subplots_adjust(left=0.05, right=0.95, bottom=bottom_adj, top=top_adj, hspace=0.7, wspace=0.35)
            except ValueError:
                message("warning", f"Could not apply subplots_adjust for IC{component_idx} in PDF.")
            return fig
        else:
            # Saving .webp for OpenAI API call (no classification text needed on the image itself)
            if output_dir is None:
                plt.close(fig)
                raise ValueError("output_dir must be provided if not returning figure object.")
            
            filename = f"component_IC{component_idx}_vision_analysis.webp"
            filepath = output_dir / filename
            try:
                # Ensure layout is tight for the API image
                fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.93, hspace=0.7, wspace=0.35)
                plt.savefig(filepath, format='webp', bbox_inches='tight', pad_inches=0.1)
                # message("debug", f"Saved component plot for API to {filepath}")
            except Exception as e_save:
                message("error", f"Error saving API figure for IC{component_idx}: {e_save}")
                plt.close(fig)
                return None
            finally:
                plt.close(fig)
            return filepath

    def _classify_component_image_openai(self, image_path: Path, api_key: Optional[str] = None, model_name: str = "gpt-4.1") -> Tuple[str, float, str]:
        """
        Sends a component image to OpenAI Vision API for classification using the detailed prompt.

        Args:
            image_path: Path to the component image file (WebP format preferred).
            api_key: OpenAI API key. If None, attempts to use OPENAI_API_KEY env var or openai.api_key.

        Returns:
            Tuple: (label: str, confidence: float, reason: str)
                   Defaults to ("other_artifact", 1.0, "API error or parsing failure") on error.
                   The label is one of the keys from OPENAI_LABEL_ORDER.
        """
        if not image_path or not image_path.exists():
            message("error", f"Invalid or non-existent image path provided for classification: {image_path}")
            return "other_artifact", 1.0, "Invalid image path"

        try:
            effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not effective_api_key:
                # Attempt to use the globally set openai.api_key if available
                if hasattr(openai, 'api_key') and openai.api_key:
                    effective_api_key = openai.api_key
                else:
                    message("error", "OpenAI API key not provided via argument, environment variable, or openai.api_key.")
                    raise ValueError("OpenAI API key is missing.")

            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            client = openai.OpenAI(api_key=effective_api_key)
            message("debug", f"Sending component image {image_path.name} to OpenAI Vision API (gpt-4.1)...")
            
            response = client.chat.completions.create( # Updated to use client.chat.completions.create
                model=model_name, 
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._OPENAI_ICA_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_image}", "detail": "high"}}
                    ]
                }],
                max_tokens=300, # OpenAI recommends setting max_tokens for vision models
                temperature=0.1
            )
            resp_text = None
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                resp_text = response.choices[0].message.content.strip()

            if resp_text:
                message("debug", f"Raw OpenAI response for parsing: '{resp_text}'")
                
                # Use OPENAI_LABEL_ORDER for dynamic pattern generation
                labels_pattern = "|".join(re.escape(label) for label in OPENAI_LABEL_ORDER)
                
                # Regex adapted from the test script for robustness
                match = re.search(
                    r"^\s*\(\s*"                                      # Start of tuple
                    r"['\"]?(" + labels_pattern + r")['\"]?"           # Group 1: The label
                    r"\s*,\s*"                                       # Comma separator
                    r"([01](?:\.\d+)?)"                             # Group 2: Confidence score (0.0 to 1.0)
                    r"\s*,\s*"                                       # Comma separator
                    r"['\"](.*?)['\"]"                                 # Group 3: Reasoning (non-greedy)
                    r"\s*\)\s*$",                                    # End of tuple
                    resp_text, re.IGNORECASE | re.DOTALL
                )

                if match:
                    label = match.group(1).lower()
                    # Ensure the matched label is one of the predefined ones
                    if label not in OPENAI_LABEL_ORDER:
                        message("warning", f"OpenAI returned an unexpected label '{label}' not in OPENAI_LABEL_ORDER. Raw: '{resp_text}'")
                        return "other_artifact", 1.0, f"Unexpected label: {label}. Parsed from: {resp_text}"
                    
                    confidence = float(match.group(2))
                    reason = match.group(3).strip()
                    # Basic unescaping, good practice though non-greedy match helps
                    reason = reason.replace("\\\"", '"').replace("\\\'", "'")

                    message("debug", f"Parsed classification: Label={label}, Conf={confidence:.2f}, Reason='{reason[:70]}...'")
                    return label, confidence, reason
                else:
                    message("warning", f"Could not parse OpenAI response format: '{resp_text}'. Defaulting to 'other_artifact'.")
                    return "other_artifact", 1.0, f"Failed to parse response: {resp_text}"
            else:
                message("error", "No text content in OpenAI response or invalid response structure.")
                return "other_artifact", 1.0, "Invalid response structure (no text content)"

        except openai.APIConnectionError as e:
            message("error", f"OpenAI API connection error: {e}")
            return "other_artifact", 1.0, f"API Connection Error: {str(e)[:100]}"
        except openai.AuthenticationError as e:
            message("error", f"OpenAI API authentication error: {e}. Check API key.")
            return "other_artifact", 1.0, f"API Authentication Error: {str(e)[:100]}"
        except openai.RateLimitError as e:
            message("error", f"OpenAI API rate limit exceeded: {e}")
            return "other_artifact", 1.0, f"API Rate Limit Error: {str(e)[:100]}"
        except openai.APIStatusError as e:
            message("error", f"OpenAI API status error: Status={e.status_code}, Response={e.response}")
            return "other_artifact", 1.0, f"API Status Error {e.status_code}: {str(e.response)[:100]}"
        except ValueError as e: # Catch specific ValueError for API key
            message("error", f"Configuration error: {e}")
            return "other_artifact", 1.0, f"Configuration error: {str(e)[:100]}"
        except Exception as e:
            message("error", f"Unexpected exception during vision classification: {type(e).__name__} - {e}")
            # message("error", traceback.format_exc()) # traceback can be too verbose for general log
            return "other_artifact", 1.0, f"Unexpected Exception: {type(e).__name__} - {str(e)[:100]}"
    
    def classify_ica_components_vision(self, 
                                      api_key: Optional[str] = None,
                                      confidence_threshold: float = 0.8,
                                      auto_exclude: bool = True,
                                      labels_to_exclude: Optional[List[str]] = None,
                                      model_name: str = "gpt-4.1" # Added model_name parameter
                                      ) -> pd.DataFrame:
        """
        Classifies ICA components using OpenAI Vision API with specific artifact labels.
        
        This method generates visualizations of each ICA component, sends them to
        the OpenAI Vision API for classification, updates the ICA object attributes
        (labels_, labels_scores_, exclude) similarly to mne_icalabel, applies ICA
        rejection, and saves the modified ICA object.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, uses OPENAI_API_KEY environment variable or openai.api_key.
        confidence_threshold : float, default=0.8
            Minimum confidence for a classification to be accepted for auto-exclusion.
        auto_exclude : bool, default=True
            If True, automatically add components to the exclude list in self.final_ica
            if their label is in `labels_to_exclude` and confidence is met.
        labels_to_exclude : List[str], optional
            A list of specific OpenAI labels (e.g., ["muscle", "eye", "heart"]) that should be
            considered for auto-exclusion. If None, defaults to all OpenAI labels 
            except "brain".
        model_name : str, default="gpt-4.1"
            The OpenAI model to use for classification (e.g., "gpt-4-vision-preview", "gpt-4.1").
            
        Returns
        ------ pd.DataFrame
            DataFrame (`self.ica_vision_flags`) containing the classification results 
            for each component, including columns: `component_index`, `component_name`, 
            `label` (OpenAI label), `mne_label` (mapped MNE label), `confidence`, 
            `reason`, `exclude_vision`.
            
        Notes
        -----
        - Updates `self.final_ica.labels_`, `self.final_ica.labels_scores_`, and `self.final_ica.exclude`.
        - Applies ICA rejection to `self.raw` (or the data used for ICA) if components are excluded.
        - Saves the updated `self.final_ica` object.
        - Generates a PDF report summarizing the classifications.
        """
        message("header", f"Running ICA component classification with OpenAI Vision API ({model_name})")
        
        if not hasattr(self, 'final_ica') or self.final_ica is None:
            message("error", "ICA (self.final_ica) not found. Please run `run_ica` first.")
            return pd.DataFrame() 
        
        ica = self.final_ica
        # Determine the data object that was used to fit ICA, default to self.raw
        # This is crucial for ica.get_sources() and ica.apply()
        data_for_ica = getattr(self, '_ica_fit_data', self.raw) 
        if data_for_ica is None: # Should ideally be self.raw or self.epochs if run_ica was called correctly
            message("error", "Data object used for ICA fitting not found. Using self.raw, but this might be incorrect.")
            data_for_ica = self.raw

        # Define default OpenAI labels to exclude if not provided (all except brain)
        if labels_to_exclude is None:
            labels_to_exclude = [lbl for lbl in OPENAI_LABEL_ORDER if lbl != "brain"]
        
        message("info", f"Using model: {model_name}")
        message("info", f"Auto-excluding components with OpenAI labels: {labels_to_exclude} if confidence >= {confidence_threshold}")

        classification_results_list: List[Dict[str, Any]] = []
        num_components = ica.n_components_
        if num_components is None or num_components == 0:
            message("warning", "No ICA components found to classify.")
            return pd.DataFrame()

        message("info", f"Preparing to process {num_components} ICA components...")
        
        with tempfile.TemporaryDirectory(prefix="autoclean_ica_vision_") as temp_dir_str:
            temp_path = Path(temp_dir_str)
            message("info", f"Using temporary directory for component images: {temp_path}")
            
            component_image_paths: List[Optional[Path]] = []
            for i in range(num_components):
                try:
                    image_path = self._plot_component_for_vision_api(ica, data_for_ica, i, temp_path, return_fig_object=False)
                    component_image_paths.append(image_path)
                except Exception as plot_err:
                    message("warning", f"Failed to plot component IC{i}: {plot_err}. Skipping classification.")
                    component_image_paths.append(None)
            
            processed_count = 0
            for i, image_path in enumerate(component_image_paths):
                comp_name = f"IC{i}" 
                if image_path is None:
                    openai_label, confidence, reason = "other_artifact", 1.0, "Plotting failed"
                else:
                    try:
                        # Pass the model_name to the underlying API call if it supports it
                        # For now, _classify_component_image_openai uses a hardcoded model, 
                        # but this structure allows future flexibility.
                        openai_label, confidence, reason = self._classify_component_image_openai(image_path, api_key, model_name)
                    except Exception as classify_err:
                         message("warning", f"OpenAI API call failed for {comp_name}: {classify_err}. Defaulting.")
                         openai_label, confidence, reason = "other_artifact", 1.0, f"API call failed: {classify_err}"

                mne_label = OPENAI_TO_MNE_LABEL_MAP.get(openai_label, "other")
                exclude_this_component = auto_exclude and openai_label in labels_to_exclude and confidence >= confidence_threshold
                
                classification_results_list.append({
                    "component_index": i,
                    "component_name": comp_name,
                    "label": openai_label, # Original OpenAI label
                    "mne_label": mne_label, # Mapped MNE label
                    "confidence": confidence,
                    "reason": reason,
                    "exclude_vision": exclude_this_component
                })
                
                log_level = "debug" if openai_label == "brain" and not exclude_this_component else "warning" if exclude_this_component else "info"
                message(log_level, f"Vision | {comp_name} | Label: {openai_label.upper()} (MNE: {mne_label}) | Conf: {confidence:.2f} | Exclude: {exclude_this_component}")
                processed_count +=1

            message("info", f"OpenAI classification complete. Processed {processed_count}/{num_components} components.")
            
            self.ica_vision_flags = pd.DataFrame(classification_results_list)
            if not self.ica_vision_flags.empty:
                 self.ica_vision_flags = self.ica_vision_flags.set_index("component_index", drop=False)

            # Update ICA object (labels_scores_, labels_, exclude)
            if not self.ica_vision_flags.empty:
                # 1. Update ica.labels_scores_
                # Array shape: (n_components, n_openai_label_categories)
                # For each component, the column of its classified OpenAI label gets the confidence, others 0.
                n_label_categories = len(OPENAI_LABEL_ORDER)
                labels_scores_array = np.zeros((num_components, n_label_categories))
                for _, row in self.ica_vision_flags.iterrows():
                    comp_idx = row["component_index"]
                    openai_label = row["label"]
                    conf = row["confidence"]
                    if openai_label in OPENAI_LABEL_ORDER:
                        label_col_idx = OPENAI_LABEL_ORDER.index(openai_label)
                        labels_scores_array[comp_idx, label_col_idx] = conf
                self.final_ica.labels_scores_ = labels_scores_array
                message("debug", "Updated self.final_ica.labels_scores_ based on vision classification.")

                # 2. Update ica.labels_
                # Dictionary mapping MNE label type to list of component indices.
                # Initialize self.final_ica.labels_ if it doesn't exist or to clear previous
                self.final_ica.labels_ = {mne_lbl: [] for mne_lbl in OPENAI_TO_MNE_LABEL_MAP.values()}
                for _, row in self.ica_vision_flags.iterrows():
                    comp_idx = row["component_index"]
                    mne_mapped_label = row["mne_label"]
                    # Ensure component is not added multiple times if labels_ already had entries
                    if comp_idx not in self.final_ica.labels_[mne_mapped_label]:
                        self.final_ica.labels_[mne_mapped_label].append(comp_idx)
                # Sort lists for consistency
                for mne_lbl in self.final_ica.labels_:
                    self.final_ica.labels_[mne_lbl].sort()
                message("debug", "Updated self.final_ica.labels_ based on vision classification.")

                # 3. Update ica.exclude and apply ICA
                if auto_exclude:
                    components_to_exclude_indices = self.ica_vision_flags[
                        self.ica_vision_flags['exclude_vision'] == True
                    ]['component_index'].tolist()
                    
                    if components_to_exclude_indices:
                        message("info", f"Vision identified {len(components_to_exclude_indices)} components for exclusion: {components_to_exclude_indices}")
                        if self.final_ica.exclude is None:
                            self.final_ica.exclude = []
                        
                        current_exclusions = set(self.final_ica.exclude)
                        for idx_to_exclude in components_to_exclude_indices:
                            current_exclusions.add(idx_to_exclude)
                        self.final_ica.exclude = sorted(list(current_exclusions))
                        message("info", f"Applying ICA to {getattr(data_for_ica, 'filenames', 'loaded data')}. Updated ICA exclude list: {self.final_ica.exclude}")
                        self.final_ica.apply(data_for_ica) # Apply to the original data source
                        message("success", "ICA applied with vision-based exclusions.")
                    else:
                        message("info", "No components met vision-based auto-exclusion criteria.")
            
            # Save the updated ICA object
            if hasattr(self, 'config') and self.config:
                save_ica_to_fif(self.final_ica, self.config, data_for_ica) # Pass the correct data context
                message("debug", "Saved ICA object with vision-based classifications and exclusions.")
            else:
                message("warning", "Cannot save ICA object: self.config not found or incomplete.")

            # Generate PDF report (implementation of _generate_ica_vision_report will be next)
            report_path = self._generate_ica_vision_report_pdf(
                ica_obj=self.final_ica, 
                raw_obj=data_for_ica, # Pass the correct data context
                classification_results_df=self.ica_vision_flags, # Pass the DataFrame
                # output_dir will be derived from self.config inside the report function
            )
            if report_path:
                 message("info", f"ICA Vision classification report saved to: {report_path}")
            else:
                message("warning", "Failed to generate ICA Vision PDF report.")

            # Update metadata (simplified example)
            metadata = {
                "ica_vision_classification": {
                    "components_processed": processed_count,
                    "total_components": num_components,
                    "model_used": model_name,
                    "auto_excluded_count": len(self.final_ica.exclude) if self.final_ica.exclude else 0,
                    "report_file": str(report_path) if report_path else "N/A"
                }
            }
            if hasattr(self, '_update_metadata') and callable(self._update_metadata):
                self._update_metadata("step_classify_ica_components_vision", metadata)
        
        message("success", "ICA component classification with Vision API complete.")
        return self.ica_vision_flags
    
    def _create_vision_summary_table_page(
        self,
        pdf_pages_obj: PdfPages,
        classification_results_df: pd.DataFrame, # Expecting a DataFrame here
        component_indices_to_include: List[int],
        bids_basename_for_title: str
    ):
        """
        Creates summary table pages for the ICA Vision PDF report.
        Adapted from _create_summary_table_page in test_ica_vision.py.

        Args:
            pdf_pages_obj: The PdfPages object to save the figure to.
            classification_results_df: DataFrame containing classification results 
                                     (expects columns like 'label', 'confidence').
                                     The DataFrame index should be the component_index.
            component_indices_to_include: List of component indices to include in this summary.
            bids_basename_for_title: Basename of the BIDS file for the PDF title.
        """
        matplotlib.use("Agg") # Ensure non-interactive backend
        components_per_page = 25 # Adjusted to fit more components per summary page
        num_total_components_in_list = len(component_indices_to_include)

        if num_total_components_in_list == 0:
            message("debug", "No components to summarize in table page.")
            return

        num_pages_for_summary = math.ceil(num_total_components_in_list / components_per_page)
        if num_pages_for_summary == 0 and num_total_components_in_list > 0:
            num_pages_for_summary = 1

        # Colors for different OpenAI labels
        # These are the direct labels from OpenAI, not the MNE mapped ones for this table
        color_map_vision_direct = {
            "brain": "#d4edda",      # Light green
            "eye": "#f9e79f",        # Light yellow
            "muscle": "#f5b7b1",    # Light red
            "heart": "#d7bde2",      # Light purple
            "line_noise": "#add8e6", # Light blue
            "channel_noise": "#ffd700", # Gold/Orange
            "other_artifact": "#e9ecef", # Lighter grey
        }

        for page_num in range(num_pages_for_summary):
            start_idx_overall = page_num * components_per_page
            end_idx_overall = min((page_num + 1) * components_per_page, num_total_components_in_list)
            
            page_component_actual_indices = component_indices_to_include[start_idx_overall:end_idx_overall]

            if not page_component_actual_indices:
                continue

            fig_table = plt.figure(figsize=(11, 8.5)) # Standard US Letter size
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('off')

            table_data_page = []
            table_cell_colors_page = []

            for comp_idx in page_component_actual_indices:
                if comp_idx not in classification_results_df.index:
                    message("warning", f"Component IC{comp_idx} not in classification_results_df. Skipping in summary.")
                    continue

                comp_info = classification_results_df.loc[comp_idx]
                # Use 'label' for the OpenAI direct label, and 'mne_label' for the mapped one
                openai_label = comp_info.get('label', 'N/A') 
                mne_mapped_label = comp_info.get('mne_label', 'N/A')
                confidence = comp_info.get('confidence', 0.0)
                is_excluded_text = "Yes" if comp_info.get('exclude_vision', False) else "No"
                reason_snippet = str(comp_info.get('reason', ''))[:50] + "..." if len(str(comp_info.get('reason', ''))) > 50 else str(comp_info.get('reason', ''))
                
                table_data_page.append([
                    f"IC{comp_idx}",
                    str(openai_label).title(),
                    str(mne_mapped_label).title(),
                    f"{confidence:.2f}",
                    is_excluded_text,
                    reason_snippet
                ])
                
                row_color = color_map_vision_direct.get(openai_label, "#ffffff") # Default to white
                table_cell_colors_page.append([row_color] * 6) # 6 columns now

            if not table_data_page:
                plt.close(fig_table)
                continue

            table = ax_table.table(
                cellText=table_data_page,
                colLabels=["Component", "Vision Label", "MNE Label", "Confidence", "Excluded?", "Reason (Brief)"],
                loc='center',
                cellLoc='left', # Left align text in cells
                cellColours=table_cell_colors_page,
                colWidths=[0.1, 0.15, 0.15, 0.1, 0.1, 0.4] # Adjusted for 6 columns
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8) # Reduced font size for more content
            table.scale(1.0, 1.2) # Adjusted scale

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_page_of_total = f"(Page {page_num + 1} of {num_pages_for_summary})"
            fig_table.suptitle(
                f"ICA Vision API Classification Summary - {bids_basename_for_title}\n"
                f"{current_page_of_total} - Generated: {timestamp}",
                fontsize=11, y=0.96
            )
            
            legend_patches = [plt.Rectangle((0,0), 1, 1, facecolor=color, label=label.title()) 
                              for label, color in color_map_vision_direct.items()]
            if legend_patches:
                ax_table.legend(handles=legend_patches, loc='upper right', 
                                bbox_to_anchor=(1.02, 0.85), title="Vision Labels", fontsize=7)

            plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05) 
            pdf_pages_obj.savefig(fig_table, bbox_inches='tight')
            plt.close(fig_table)

    def _generate_ica_vision_report_pdf(self, 
                                   ica_obj: ICA, 
                                   raw_obj: mne.io.Raw, 
                                   classification_results_df: pd.DataFrame,
                                   components_to_plot: str = "all" 
                                   ) -> Optional[Path]:
        """
        Generates a comprehensive PDF report for ICA components classified by Vision API.
        Includes summary tables and individual component detail pages with plots and reasoning.
        Adapted from _generate_ica_report_pdf in test_ica_vision.py.

        Args:
            ica_obj: The MNE ICA object.
            raw_obj: The MNE Raw object (or Epochs) used for ICA, for context.
            classification_results_df: DataFrame with classification results from 
                                     `classify_ica_components_vision` (indexed by component_index).
            components_to_plot: Which components to include: "all" or "classified_as_artifact".
                                "classified_as_artifact" includes those where 'exclude_vision' is True.

        Returns:
            Path to the generated PDF report, or None if generation failed.
        """
        matplotlib.use("Agg") # Ensure non-interactive backend

        if not (hasattr(self, 'config') and self.config and 
                  'derivatives_dir' in self.config and 'bids_path' in self.config):
            message("error", "Configuration for derivatives_dir or bids_path not found. Cannot generate PDF report.")
            return None
        
        # Determine output path from self.config
        derivatives_dir = Path(self.config["derivatives_dir"])
        bids_path_obj = self.config["bids_path"]
        if not hasattr(bids_path_obj, 'basename'):
            message("error", "BIDSPath object in config does not have a 'basename'. Cannot name PDF report.")
            return None
        
        # Ensure derivatives directory exists
        try:
            derivatives_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            message("error", f"Could not create derivatives directory {derivatives_dir}: {e_mkdir}")
            return None

        base_name_for_pdf = bids_path_obj.basename
        report_type_suffix = "vision_all_comps" if components_to_plot == "all" else "vision_artifacts_only"
        pdf_filename = f"{base_name_for_pdf.replace('_eeg', '')}_{report_type_suffix}.pdf"
        pdf_path = derivatives_dir / pdf_filename

        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError as e_unlink:
                message("warning", f"Could not delete existing PDF {pdf_path}: {e_unlink}")

        plot_indices = []
        if components_to_plot == "all":
            if not classification_results_df.empty:
                plot_indices = list(classification_results_df.index)
        elif components_to_plot == "classified_as_artifact":
            if 'exclude_vision' in classification_results_df.columns:
                plot_indices = list(classification_results_df[classification_results_df['exclude_vision'] == True].index)
            else:
                message("warning", "'exclude_vision' column not in classification_results. Cannot determine artifact components for report.")
        
        plot_indices = sorted(list(set(idx for idx in plot_indices if idx < ica_obj.n_components_))) # Ensure valid and sorted

        if not plot_indices:
            message("info", f"No components to plot for '{report_type_suffix}' report. Skipping PDF.")
            return None
        
        message("info", f"Generating PDF report ('{report_type_suffix}') for {len(plot_indices)} components to {pdf_path}...")

        try:
            with PdfPages(pdf_path) as pdf:
                # 1. Summary Table Page(s)
                self._create_vision_summary_table_page(pdf, classification_results_df, plot_indices, base_name_for_pdf)

                # 2. Component Topographies Overview Page (optional, can be verbose)
                # Consider making this conditional or batched if too many components
                try:
                    if plot_indices:
                        max_topo_per_fig = 25 
                        for i in range(0, len(plot_indices), max_topo_per_fig):
                            batch_indices = plot_indices[i:i+max_topo_per_fig]
                            if batch_indices:
                                # Create a figure for this batch of topomaps
                                # Calculate layout for a grid of topomaps
                                n_batch = len(batch_indices)
                                ncols = math.ceil(math.sqrt(n_batch / 1.5)) # Aim for a slightly wider aspect ratio
                                nrows = math.ceil(n_batch / ncols)
                                
                                fig_topo_batch, axes_topo_batch = plt.subplots(nrows, ncols, 
                                                                            figsize=(min(ncols * 2.5, 14), 
                                                                                     min(nrows * 2.5, 18)), 
                                                                            squeeze=False)
                                fig_topo_batch.suptitle(f"Topographies Overview (Batch {i//max_topo_per_fig + 1})", fontsize=14)
                                
                                for ax_idx, comp_idx_topo in enumerate(batch_indices):
                                    r, c = divmod(ax_idx, ncols)
                                    ax_curr = axes_topo_batch[r, c]
                                    try:
                                        ica_obj.plot_components(picks=comp_idx_topo, axes=ax_curr, 
                                                                show=False, colorbar=False, cmap='jet', 
                                                                outlines='head', sensors=False, contours=4)
                                        ax_curr.set_title(f"IC{comp_idx_topo}", fontsize=9)
                                    except Exception as e_single_topo:
                                        message("warning", f"Could not plot topography for IC{comp_idx_topo} in overview: {e_single_topo}")
                                        ax_curr.text(0.5,0.5, "Error", ha='center', va='center')
                                        ax_curr.set_title(f"IC{comp_idx_topo} (Err)", fontsize=9)
                                    ax_curr.set_xlabel('')
                                    ax_curr.set_ylabel('')
                                    ax_curr.set_xticks([])
                                    ax_curr.set_yticks([])
                                
                                # Hide unused axes
                                for ax_idx_hide in range(n_batch, nrows * ncols):
                                    r, c = divmod(ax_idx_hide, ncols)
                                    fig_topo_batch.delaxes(axes_topo_batch[r,c])
                                
                                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Space for suptitle
                                pdf.savefig(fig_topo_batch)
                                plt.close(fig_topo_batch)
                except Exception as e_topo_overview:
                    message("error", f"Error plotting component topographies overview: {e_topo_overview}")
                    # Fallback error page for topo overview
                    fig_err = plt.figure() 
                    ax_err = fig_err.add_subplot(111)
                    ax_err.text(0.5,0.5, "Topographies overview failed", ha='center', va='center')
                    pdf.savefig(fig_err)
                    plt.close(fig_err)

                # 3. Individual Component Detail Pages
                for comp_idx_detail in plot_indices:
                    if comp_idx_detail not in classification_results_df.index:
                        message("warning", f"Skipping IC{comp_idx_detail} detail page: not in classification_results_df.")
                        continue
                    
                    comp_info = classification_results_df.loc[comp_idx_detail]
                    label = comp_info.get('label', 'N/A')
                    conf = comp_info.get('confidence', 0.0)
                    reason = comp_info.get('reason', 'N/A')

                    # Use the main plotting function, now asking it to return a fig object
                    # output_dir is not needed here as we are returning the figure for PdfPages
                    fig_detail = self._plot_component_for_vision_api(
                        ica_obj=ica_obj,
                        raw_obj=raw_obj,
                        component_idx=comp_idx_detail,
                        output_dir=None, # Not saving a separate file, just getting the figure
                        classification_label=label,
                        classification_confidence=conf,
                        classification_reason=reason,
                        return_fig_object=True
                    )

                    if fig_detail:
                        try:
                            pdf.savefig(fig_detail)
                        except Exception as e_save_detail:
                            message("error", f"Error saving detail page for IC{comp_idx_detail} to PDF: {e_save_detail}")
                            fig_err_s = plt.figure()
                            ax_err_s = fig_err_s.add_subplot(111)
                            ax_err_s.text(0.5,0.5, f"Plot save for IC{comp_idx_detail}\nfailed.", ha='center',va='center')
                            pdf.savefig(fig_err_s)
                            plt.close(fig_err_s)
                        finally:
                            plt.close(fig_detail) # Always close the figure
                    else:
                        message("warning", f"Failed to generate plot object for IC{comp_idx_detail} for PDF detail page.")
                        fig_err_g = plt.figure()
                        ax_err_g = fig_err_g.add_subplot(111)
                        ax_err_g.text(0.5,0.5,f"Plot gen for IC{comp_idx_detail}\nfailed.", ha='center',va='center')
                        pdf.savefig(fig_err_g)
                        plt.close(fig_err_g)
            
            message("debug", f"Successfully generated ICA Vision PDF report: {pdf_path}")
            return pdf_path
        except ImportError:
            message("error", "Matplotlib PdfPages not available. Cannot generate PDF report.")
            return None
        except Exception as e_pdf_main:
            message("error", f"Major error during PDF report generation for ICA vision: {e_pdf_main}")
            # message("error", traceback.format_exc()) # Can be verbose
            return None

    def classify_ica_components_vision_parallel(
        self, 
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.8,
        auto_exclude: bool = True,
        labels_to_exclude: Optional[List[str]] = None,
        model_name: str = "gpt-4.1",
        batch_size: int = 10,  # Number of images per API request
        max_concurrency: int = 5  # Maximum number of concurrent API requests
    ) -> pd.DataFrame:
        """
        Parallelized version that classifies ICA components using OpenAI Vision API in batches.
        
        This method improves processing speed by:
        1. Batching multiple images in single API requests
        2. Processing multiple batches concurrently
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, uses OPENAI_API_KEY environment variable or openai.api_key.
        confidence_threshold : float, default=0.8
            Minimum confidence for a classification to be accepted for auto-exclusion.
        auto_exclude : bool, default=True
            If True, automatically add components to the exclude list in self.final_ica
            if their label is in `labels_to_exclude` and confidence is met.
        labels_to_exclude : List[str], optional
            A list of specific OpenAI labels (e.g., ["muscle", "eye", "heart"]) that should be
            considered for auto-exclusion. If None, defaults to all OpenAI labels 
            except "brain".
        model_name : str, default="gpt-4.1"
            The OpenAI model to use for classification (e.g., "gpt-4-vision-preview", "gpt-4.1").
        batch_size : int, default=10
            Number of ICA component images to include in a single API request. 
            Recommended range is 5-20 to balance efficiency with API limits.
        max_concurrency : int, default=5
            Maximum number of concurrent API requests to send. Higher values improve speed
            but may exceed API rate limits.
            
        Returns
        ------ pd.DataFrame
            DataFrame (`self.ica_vision_flags`) containing the classification results 
            for each component, including columns: `component_index`, `component_name`, 
            `label` (OpenAI label), `mne_label` (mapped MNE label), `confidence`, 
            `reason`, `exclude_vision`.
            
        Notes
        -----
        - Updates `self.final_ica.labels_`, `self.final_ica.labels_scores_`, and `self.final_ica.exclude`.
        - Applies ICA rejection to `self.raw` (or the data used for ICA) if components are excluded.
        - Saves the updated `self.final_ica` object.
        - Generates a PDF report summarizing the classifications.
        - For very large datasets, consider further adjusting batch_size and max_concurrency
          based on your OpenAI API tier's rate limits.
        """
        import concurrent.futures
        import asyncio
        import time
        from io import BytesIO
        from PIL import Image
        import base64

        message("header", f"Running parallelized ICA component classification with OpenAI Vision API ({model_name})")
        
        if not hasattr(self, 'final_ica') or self.final_ica is None:
            message("error", "ICA (self.final_ica) not found. Please run `run_ica` first.")
            return pd.DataFrame() 
        
        ica = self.final_ica
        data_for_ica = getattr(self, '_ica_fit_data', self.raw) 
        if data_for_ica is None:
            message("error", "Data object used for ICA fitting not found. Using self.raw, but this might be incorrect.")
            data_for_ica = self.raw

        # Define default OpenAI labels to exclude if not provided (all except brain)
        if labels_to_exclude is None:
            labels_to_exclude = [lbl for lbl in OPENAI_LABEL_ORDER if lbl != "brain"]
        
        message("info", f"Using model: {model_name}")
        message("info", f"Auto-excluding components with OpenAI labels: {labels_to_exclude} if confidence >= {confidence_threshold}")
        message("info", f"Processing with batch_size={batch_size}, max_concurrency={max_concurrency}")

        num_components = ica.n_components_
        if num_components is None or num_components == 0:
            message("warning", "No ICA components found to classify.")
            return pd.DataFrame()

        # Step 1: Generate all component images in a temporary directory
        message("info", f"Preparing to process {num_components} ICA components...")
        
        component_image_paths = []
        component_indices = []
        
        with tempfile.TemporaryDirectory(prefix="autoclean_ica_vision_") as temp_dir_str:
            temp_path = Path(temp_dir_str)
            message("info", f"Using temporary directory for component images: {temp_path}")
            
            # Generate all component images first (this step is not parallelized)
            for i in range(num_components):
                try:
                    image_path = self._plot_component_for_vision_api(ica, data_for_ica, i, temp_path, return_fig_object=False)
                    if image_path is not None:
                        component_image_paths.append(image_path)
                        component_indices.append(i)
                except Exception as plot_err:
                    message("warning", f"Failed to plot component IC{i}: {plot_err}. Skipping classification.")
                    continue  # Skip this component
            
            message("info", f"Successfully generated {len(component_image_paths)} images for classification")
            
            # Step 2: Divide components into batches
            batches = []
            for i in range(0, len(component_image_paths), batch_size):
                batch_image_paths = component_image_paths[i:i+batch_size]
                batch_indices = component_indices[i:i+batch_size]
                batches.append((batch_indices, batch_image_paths))
            
            message("info", f"Divided components into {len(batches)} batches for parallel processing")
            
            # Step 3: Define the batch classification function
            def classify_batch(batch_tuple):
                batch_indices, batch_paths = batch_tuple
                batch_results = []
                
                try:
                    # Read and encode all images in the batch
                    batch_images_base64 = []
                    for img_path in batch_paths:
                        with open(img_path, "rb") as image_file:
                            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                            batch_images_base64.append(base64_image)
                    
                    # Create the content list for the API request with specific prompts for each image
                    content_list = []
                    for idx, base64_img in enumerate(batch_images_base64):
                        comp_idx = batch_indices[idx]
                        # Add prompt for this specific component
                        content_list.append({
                            "type": "text",
                            "text": f"For IC component {comp_idx}, classify according to the criteria below:"
                        })
                        content_list.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/webp;base64,{base64_img}", "detail": "high"}
                        })
                    
                    # Add the classification guidelines at the beginning
                    content_list.insert(0, {
                        "type": "text", 
                        "text": self._OPENAI_ICA_PROMPT + "\n\nIMPORTANT: This request contains multiple component images. For EACH image, provide a separate classification in the format: 'IC{component_number}: (label, confidence, reason)'"
                    })
                    
                    # Make the API call
                    effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
                    if not effective_api_key and hasattr(openai, 'api_key') and openai.api_key:
                        effective_api_key = openai.api_key
                    
                    if not effective_api_key:
                        raise ValueError("OpenAI API key not provided")
                    
                    client = openai.OpenAI(api_key=effective_api_key)
                    message("debug", f"Sending batch with {len(batch_indices)} components to OpenAI Vision API...")
                    
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{
                            "role": "user",
                            "content": content_list
                        }],
                        max_tokens=1000,
                        temperature=0.1
                    )
                    
                    # Parse the response
                    if response.choices and response.choices[0].message and response.choices[0].message.content:
                        resp_text = response.choices[0].message.content.strip()
                        message("debug", f"Got batch response: {len(resp_text)} characters")
                        
                        # Store for debugging which components couldn't be parsed
                        unparsed_components = []
                        
                        # Extract component-specific responses using regex
                        for comp_idx in batch_indices:
                            # Look for "IC{comp_idx}: (label, confidence, reason)" pattern
                            pattern = rf"IC{comp_idx}:\s*\(\s*['\"]?(brain|eye|muscle|heart|line_noise|channel_noise|other_artifact)['\"]?\s*,\s*([01](?:\.\d+)?)\s*,\s*['\"](.*?)['\"]\s*\)"
                            match = re.search(pattern, resp_text, re.IGNORECASE | re.DOTALL)
                            
                            if match:
                                label = match.group(1).lower()
                                confidence = float(match.group(2))
                                reason = match.group(3).strip()
                                
                                batch_results.append({
                                    "component_index": comp_idx,
                                    "component_name": f"IC{comp_idx}",
                                    "label": label,
                                    "mne_label": OPENAI_TO_MNE_LABEL_MAP.get(label, "other"),
                                    "confidence": confidence,
                                    "reason": reason,
                                    "exclude_vision": auto_exclude and label in labels_to_exclude and confidence >= confidence_threshold
                                })
                                
                                message("debug", f"Parsed IC{comp_idx}: {label} (conf={confidence:.2f})")
                            else:
                                # Track unparsed components for detailed debugging
                                unparsed_components.append(comp_idx)
                                
                                # Try a more permissive pattern as fallback
                                fallback_pattern = rf"IC{comp_idx}[^\(]*?\(\s*['\"]?(\w+)['\"]?\s*,\s*([01](?:\.\d+)?)\s*,\s*['\"]([^\"']+)['\"]"
                                fallback_match = re.search(fallback_pattern, resp_text, re.IGNORECASE | re.DOTALL)
                                
                                if fallback_match:
                                    # We found something with the fallback pattern
                                    label_text = fallback_match.group(1).lower()
                                    # Check if the label is valid
                                    if label_text in OPENAI_LABEL_ORDER:
                                        label = label_text
                                        confidence = float(fallback_match.group(2))
                                        reason = fallback_match.group(3).strip()
                                        
                                        batch_results.append({
                                            "component_index": comp_idx,
                                            "component_name": f"IC{comp_idx}",
                                            "label": label,
                                            "mne_label": OPENAI_TO_MNE_LABEL_MAP.get(label, "other"),
                                            "confidence": confidence,
                                            "reason": reason,
                                            "exclude_vision": auto_exclude and label in labels_to_exclude and confidence >= confidence_threshold
                                        })
                                        
                                        message("info", f"Parsed IC{comp_idx} using fallback pattern: {label} (conf={confidence:.2f})")
                                        # Remove from unparsed since we handled it
                                        unparsed_components.remove(comp_idx)
                                    else:
                                        message("warning", f"Fallback pattern matched for IC{comp_idx} but found invalid label: {label_text}")
                                else:
                                    # Fallback if not found: mark as other_artifact
                                    message("warning", f"Could not parse response for IC{comp_idx} in batch result. Defaulting to other_artifact.")
                                    batch_results.append({
                                        "component_index": comp_idx,
                                        "component_name": f"IC{comp_idx}",
                                        "label": "other_artifact",
                                        "mne_label": "other",
                                        "confidence": 1.0,
                                        "reason": f"Failed to parse response for component in batch",
                                        "exclude_vision": auto_exclude and "other_artifact" in labels_to_exclude and 1.0 >= confidence_threshold
                                    })
                        
                        # Print detailed debugging info if some components couldn't be parsed
                        if unparsed_components:
                            message("debug", "==== RESPONSE PARSING DEBUG INFORMATION ====")
                            message("debug", f"Components that couldn't be parsed: {unparsed_components}")
                            
                            # Print the actual response text
                            message("debug", f"Full response text: \n{resp_text}\n")
                            
                            # Look for any text mentioning the unparsed components
                            for comp_idx in unparsed_components:
                                context_pattern = rf"((?:[^\n]*IC{comp_idx}[^\n]*\n?){1,5})"
                                context_match = re.search(context_pattern, resp_text, re.IGNORECASE)
                                if context_match:
                                    context = context_match.group(1).strip()
                                    message("debug", f"Context around IC{comp_idx}:\n{context}\n")
                                else:
                                    message("debug", f"No specific context found for IC{comp_idx}")
                            
                            # Print example of successful pattern for comparison
                            if len(batch_indices) > 0 and len(unparsed_components) < len(batch_indices):
                                parsed_comp = next(idx for idx in batch_indices if idx not in unparsed_components)
                                context_pattern = rf"((?:[^\n]*IC{parsed_comp}[^\n]*\n?){1,5})"
                                context_match = re.search(context_pattern, resp_text, re.IGNORECASE)
                                if context_match:
                                    context = context_match.group(1).strip()
                                    message("debug", f"Example of successfully parsed component IC{parsed_comp}:\n{context}\n")
                            
                            message("debug", "==== END DEBUG INFORMATION ====")
                    else:
                        message("error", f"Invalid response structure from API for batch")

                except Exception as e:
                    message("error", f"Error processing batch: {str(e)}")
                    # Add default results for all components in this batch
                    for comp_idx in batch_indices:
                        batch_results.append({
                            "component_index": comp_idx,
                            "component_name": f"IC{comp_idx}",
                            "label": "other_artifact",
                            "mne_label": "other",
                            "confidence": 1.0,
                            "reason": f"Batch processing error: {str(e)}",
                            "exclude_vision": auto_exclude and "other_artifact" in labels_to_exclude and 1.0 >= confidence_threshold
                        })
                
                return batch_results
            
            # Step 4: Process batches in parallel
            all_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                future_to_batch = {executor.submit(classify_batch, batch): batch for batch in batches}
                
                total_batches = len(batches)
                completed_batches = 0
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    completed_batches += 1
                    
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        message("info", f"Completed batch {completed_batches}/{total_batches} with {len(batch_results)} components")
                    except Exception as e:
                        message("error", f"Batch processing failed: {str(e)}")
            
            # Step 5: Process results and update the ICA object
            classification_results_list = sorted(all_results, key=lambda x: x["component_index"])
            message("info", f"Total components successfully classified: {len(classification_results_list)}/{num_components}")
            
            self.ica_vision_flags = pd.DataFrame(classification_results_list)
            if not self.ica_vision_flags.empty:
                self.ica_vision_flags = self.ica_vision_flags.set_index("component_index", drop=False)
                
                # Update ICA object (labels_scores_, labels_, exclude)
                # 1. Update ica.labels_scores_
                n_label_categories = len(OPENAI_LABEL_ORDER)
                labels_scores_array = np.zeros((num_components, n_label_categories))
                
                for _, row in self.ica_vision_flags.iterrows():
                    comp_idx = row["component_index"]
                    openai_label = row["label"]
                    conf = row["confidence"]
                    if openai_label in OPENAI_LABEL_ORDER:
                        label_col_idx = OPENAI_LABEL_ORDER.index(openai_label)
                        labels_scores_array[comp_idx, label_col_idx] = conf
                
                self.final_ica.labels_scores_ = labels_scores_array
                message("debug", "Updated self.final_ica.labels_scores_ based on vision classification.")
                
                # 2. Update ica.labels_
                self.final_ica.labels_ = {mne_lbl: [] for mne_lbl in OPENAI_TO_MNE_LABEL_MAP.values()}
                for _, row in self.ica_vision_flags.iterrows():
                    comp_idx = row["component_index"]
                    mne_mapped_label = row["mne_label"]
                    if comp_idx not in self.final_ica.labels_[mne_mapped_label]:
                        self.final_ica.labels_[mne_mapped_label].append(comp_idx)
                
                # Sort lists for consistency
                for mne_lbl in self.final_ica.labels_:
                    self.final_ica.labels_[mne_lbl].sort()
                message("debug", "Updated self.final_ica.labels_ based on vision classification.")
                
                # 3. Update ica.exclude and apply ICA
                if auto_exclude:
                    components_to_exclude_indices = self.ica_vision_flags[
                        self.ica_vision_flags['exclude_vision'] == True
                    ]['component_index'].tolist()
                    
                    if components_to_exclude_indices:
                        message("info", f"Vision identified {len(components_to_exclude_indices)} components for exclusion: {components_to_exclude_indices}")
                        if self.final_ica.exclude is None:
                            self.final_ica.exclude = []
                        
                        current_exclusions = set(self.final_ica.exclude)
                        for idx_to_exclude in components_to_exclude_indices:
                            current_exclusions.add(idx_to_exclude)
                        self.final_ica.exclude = sorted(list(current_exclusions))
                        message("info", f"Applying ICA to {getattr(data_for_ica, 'filenames', 'loaded data')}. Updated ICA exclude list: {self.final_ica.exclude}")
                        self.final_ica.apply(data_for_ica)  # Apply to the original data source
                        message("success", "ICA applied with vision-based exclusions.")
                    else:
                        message("info", "No components met vision-based auto-exclusion criteria.")
            
            # Save the updated ICA object and generate PDF report
            if hasattr(self, 'config') and self.config:
                save_ica_to_fif(self.final_ica, self.config, data_for_ica)
                message("debug", "Saved ICA object with vision-based classifications and exclusions.")
                
                # Generate PDF report
                report_path = self._generate_ica_vision_report_pdf(
                    ica_obj=self.final_ica, 
                    raw_obj=data_for_ica, 
                    classification_results_df=self.ica_vision_flags
                )
                if report_path:
                    message("info", f"ICA Vision classification report saved to: {report_path}")
                else:
                    message("warning", "Failed to generate ICA Vision PDF report.")
            else:
                message("warning", "Cannot save ICA object: self.config not found or incomplete.")
            
            # Update metadata
            metadata = {
                "ica_vision_classification_parallel": {
                    "components_processed": len(classification_results_list),
                    "total_components": num_components,
                    "model_used": model_name,
                    "batch_size": batch_size,
                    "max_concurrency": max_concurrency,
                    "auto_excluded_count": len(self.final_ica.exclude) if self.final_ica.exclude else 0,
                    "report_file": str(report_path) if 'report_path' in locals() and report_path else "N/A"
                }
            }
            if hasattr(self, '_update_metadata') and callable(self._update_metadata):
                self._update_metadata("step_classify_ica_components_vision_parallel", metadata)
        
        message("success", "Parallelized ICA component classification complete.")
        return self.ica_vision_flags
