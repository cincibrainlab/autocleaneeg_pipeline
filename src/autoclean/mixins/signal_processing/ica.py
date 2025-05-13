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

from autoclean.io.export import save_ica_to_fif
from autoclean.utils.logging import message

class IcaMixin:
    """Mixin for ICA processing."""

    # OpenAI prompt for ICA component classification
    _OPENAI_ICA_PROMPT = """Analyze this EEG ICA component image and classify it.

The image displays an Independent Component (IC) from an EEG recording. It includes:
1. A scalp topography map showing the spatial distribution of the component's activity.
2. A Power Spectral Density (PSD) plot showing the component's power at different frequencies.
3. Time series plots of the component's activity (typically a full view and a shorter 10-second segment).

Based on a comprehensive analysis of ALL these features, classify the component into ONE of the following categories:

- "brain":
    - *Topography:* Often looks dipolar (distinct positive and negative poles). A good dipole fit is suggested by a clean, bilobed pattern, and the source appears to be located within the brain. Scalp map is not excessively concentrated to a single electrode.
    - *Spectrum:* Shows a 1/f-like decrease in power with increasing frequency (i.e., power diminishes at higher frequencies). Often has distinct peaks in physiological bands (e.g., alpha around 8-12 Hz, but also other peaks between 5-30 Hz are common).
    - *Time Series:* Can show Event-Related Potentials (ERPs) if the data is epoched (activity consistently related to experimental events). Generally smoother compared to stereotyped, non-physiological artifact patterns.

- "eye": (Eye movements or blinks)
    - *Topography:* Suggests electrical sources near the eyes.
        - *Vertical eye movements/blinks:* Strong activity at frontal electrodes, often appearing unipolar (all positive or all negative deflection) due to the typical EEG recording montage.
        - *Horizontal eye movements:* Often a dipolar pattern with opposite polarities on the left and right frontal sides, suggesting a source between the eyes.
    - *Spectrum:* Most power is concentrated at low frequencies (typically < 5 Hz).
    - *Time Series:*
        - *Vertical (blinks/looking up/down):* Shows clear, relatively frequent, large-amplitude spikes (blinks) or slow rolling waves/drifts.
        - *Horizontal (saccades):* Often shows step-function-like changes (fast transitions to different stable levels) corresponding to saccadic eye movements.

- "muscle": (Muscle activity, EMG)
    - *Topography:* Can sometimes appear dipolar but is usually very focal/concentrated, indicating a shallow source outside the skull (e.g., scalp, temporal, or neck muscles). The pattern is often less organized than brain dipoles.
    - *Spectrum:* Dominated by power at higher frequencies (typically > 20 Hz). This power is often broadband, meaning it's spread across a wide range of high frequencies, rather than a sharp peak. Little low-frequency power relative to high-frequency power.
    - *Time Series:* Often shows sustained or bursting high-frequency, spiky, non-stationary activity.

- "heart": (Heartbeat artifact, ECG)
    - *Topography:* Often a near-linear gradient across the scalp or a very broad, diffuse pattern, suggesting a distant and powerful electrical source like the heart. Can sometimes have a left-to-right or inferior-to-superior orientation.
    - *Spectrum:* May not show specific defining peaks related to the heartbeat itself; the primary evidence is in the time series. Spectrum might be contaminated by other activity.
    - *Time Series:* Shows clear, regular, repetitive QRS-complex-like waveforms (sharp peak followed by slower wave). These complexes occur at a rate of approximately 1 Hz (around 60 beats per minute, but can vary).

- "line_noise": (Electrical grid interference)
    - *Topography:* Can vary and is not the primary indicator. The effect might be widespread or more localized depending on the noise source.
    - *Spectrum:* Characterized by a VERY SHARP and prominent peak at either 50 Hz or 60 Hz (depending on the local power system frequency). This peak should be significantly more powerful than surrounding frequencies. (Note: This is different from a notch filter *dip* or *absence of power* at these frequencies, which is a common preprocessing step).
    - *Time Series:* Shows continuous or intermittent sinusoidal oscillations at 50 Hz or 60 Hz.

- "channel_noise": (Noise specific to one or a few EEG channels)
    - *Topography:* Extremely focal, with almost all the component's energy concentrated on a single electrode or a very small, isolated group of adjacent electrodes.
    - *Spectrum:* Typically shows a 1/f-like spectrum (power decreasing with frequency), which can sometimes resemble brain activity. However, the extreme focality of the topography is the key differentiator. This spectral pattern helps distinguish it from muscle components, which have more high-frequency power.
    - *Time Series:* May show large, erratic, or persistent high-amplitude artifacts, sudden pops, or periods of flat or noisy signal unique to that component.

- "other_artifact": (Artifacts not fitting other categories, or unclear/mixed components)
    - *Topography:* Often non-dipolar, splotchy, messy, or not clearly interpretable according to the patterns above.
    - *Spectrum:* May lack clear features, or show a mixture of patterns that don't fit a single category well. Could have weak or unusual peaks.
    - *General:* Use for components that do not clearly fit any of the above categories. Components with very high IC numbers (e.g., IC 150 of 200, indicating low variance explained) are often in this category. Use this also if features are highly contradictory (e.g., some brain-like spectral features but a very non-brain-like, noisy topography), making a single primary classification difficult.

Return your classification in this exact format:
("label", confidence_score, "detailed_reasoning")

Where:
- "label" is one of the exact category strings listed above (e.g., "brain", "eye", "muscle", etc.).
- confidence_score is a number between 0 and 1 (e.g., 0.95), representing your confidence in the assigned label.
- "detailed_reasoning" is a concise explanation (1-2 sentences) justifying your classification based on specific visual features observed in the topography, power spectrum, AND time series, referencing the characteristics described for each category.

Example: ("eye", 0.95, "Strong bilateral frontal topography. Spectrum power concentrated below 5 Hz. Time series shows large, recurrent sharp deflections characteristic of eye blinks.")
Example: ("brain", 0.88, "Clear dipolar topography over parietal regions. Spectrum shows a prominent peak at 10 Hz and a 1/f decrease. Time series is relatively smooth.")
Example: ("muscle", 0.92, "Focal topography over temporal area, shallow appearance. Spectrum is dominated by broadband high-frequency activity (>20Hz). Time series shows sustained spiky, high-frequency bursts.")
Example: ("line_noise", 0.98, "Spectrum exhibits an extremely sharp and dominant peak at 50 Hz, far exceeding other frequencies. Topography is diffuse but the spectral peak is unequivocal.")
Example: ("channel_noise", 0.90, "Topography is highly focal, concentrated on a single frontal electrode. Spectrum shows a 1/f trend. Time series exhibits intermittent large spikes.")
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

        save_ica_to_fif(self.final_ica, self.config, self.raw)

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

        save_ica_to_fif(self.final_ica, self.config, self.raw)

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
             save_ica_to_fif(self.final_ica, self.config, self.raw) # Consistently save against self.raw context
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

    def _plot_component_for_vision(self, ica, raw, component_idx: int, output_dir: Path) -> Path:
        """
        Creates a standardized plot for an ICA component to be used for vision classification.
        This version is optimized for speed, especially in PSD calculation.

        Args:
            ica: The ICA object to plot components from
            raw: The raw data used for the ICA
            component_idx: Index of the component to plot
            output_dir: Directory to save the plot

        Returns:
            Path to the saved image file
        """
        # Force matplotlib to use non-interactive backend
        matplotlib.use("Agg") 
        
        # Create figure with multiple panels using GridSpec
        fig = plt.figure(figsize=(10, 10), dpi=120) 
        gs = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.3)
        
        # Define axes on the main figure
        ax_topo = fig.add_subplot(gs[0, 0]) 
        ax_psd = fig.add_subplot(gs[0, 1])
        ax_ts_full = fig.add_subplot(gs[1, :])  
        ax_ts_10s = fig.add_subplot(gs[2, :])  
        
        # Get ICA component source data
        sources = ica.get_sources(raw) # This returns a RawArray-like object
        sfreq = sources.info['sfreq']
        
        # Extract the time series for the specific component (as a 1D numpy array)
        # sources.get_data() returns (n_components, n_times)
        # We need to pick the specific component. Using .get_data(picks=) is safer.
        component_data_array = sources.get_data(picks=[component_idx])[0] # Ensure it's 1D
        
        # 1. Plot Topography (already efficient)
        try:
            ica.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                                show=False, colorbar=True, title="")
            ax_topo.set_title(f"IC {component_idx+1} - Topography")
        except Exception as e:
            message("error", f"Failed to plot topography for IC {component_idx+1}: {str(e)}")
            ax_topo.text(0.5, 0.5, "Topo plot failed", ha='center', va='center')
            ax_topo.set_title(f"IC {component_idx+1} - Topography (Failed)")

        # 2. Plot PSD directly and efficiently using psd_array_welch
        try:
            from mne.time_frequency import psd_array_welch
            
            fmin_psd = 1.0
            fmax_psd = min(100.0, sfreq / 2.0 - 0.5) # Up to 100Hz or Nyquist, ensure fmax < sfreq/2
            n_fft_psd = int(sfreq * 2.0) # 2-second window for Welch
            
            if fmax_psd <= fmin_psd: # Safety check if sfreq is very low
                fmax_psd = sfreq / 2.0 - 0.5
                if fmax_psd <= fmin_psd:
                     raise ValueError(f"Cannot compute PSD for IC {component_idx+1}: fmax ({fmax_psd:.2f} Hz) is not greater than fmin ({fmin_psd:.2f} Hz) for sfreq {sfreq:.2f} Hz.")

            psds, freqs = psd_array_welch(
                component_data_array, 
                sfreq=sfreq, 
                fmin=fmin_psd, 
                fmax=fmax_psd, 
                n_fft=n_fft_psd,
                n_overlap=0, # No overlap for speed
                average='mean',
                window='hann',
                verbose=False
            )
            
            psds_db = 10 * np.log10(psds) # Convert to dB
            
            ax_psd.plot(freqs, psds_db, color='black', linewidth=1)
            ax_psd.set_title("Power Spectrum")
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("Power (dB)")
            ax_psd.set_xlim(freqs[0], freqs[-1])
            ax_psd.grid(True, linestyle='--', alpha=0.5)

        except Exception as e:
            message("error", f"PSD plotting failed for IC {component_idx+1}: {str(e)}")
            ax_psd.text(0.5, 0.5, "PSD plot failed", ha='center', va='center', transform=ax_psd.transAxes)
            ax_psd.set_title("Power Spectrum (Failed)")
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("Power (dB)")

        # 3. Plot Full Component Time Series (already efficient)
        times_full = np.arange(len(component_data_array)) / sfreq
        ax_ts_full.plot(times_full, component_data_array, linewidth=0.6, color='black')
        ax_ts_full.set_xlabel("Time (s)")
        ax_ts_full.set_ylabel("Amplitude (a.u.)")
        ax_ts_full.set_title(f"IC {component_idx+1} Full Time Series")
        ax_ts_full.grid(True, linestyle=':', alpha=0.7)
        ax_ts_full.set_xlim(times_full[0], times_full[-1])

        # 4. Plot First 10 Seconds of Component Time Series (already efficient)
        duration_10s = 10.0
        max_samples_10s = min(int(duration_10s * sfreq), len(component_data_array))
        times_10s = np.arange(max_samples_10s) / sfreq
        
        ax_ts_10s.plot(times_10s, component_data_array[:max_samples_10s], linewidth=0.7, color='darkblue')
        ax_ts_10s.set_xlabel("Time (s)")
        ax_ts_10s.set_ylabel("Amplitude (a.u.)")
        ax_ts_10s.set_title(f"IC {component_idx+1} First {duration_10s:.0f}s")
        ax_ts_10s.grid(True, linestyle=':', alpha=0.7)
        if max_samples_10s > 0:
             ax_ts_10s.set_xlim(times_10s[0], times_10s[-1])
        
        # Overall figure title and layout adjustment
        fig.suptitle(f"Analysis for ICA Component {component_idx + 1}", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust rect to make space for suptitle
        
        # Save the combined figure
        filename = f"component_{component_idx+1}.webp"
        filepath = output_dir / filename
        
        try:
            plt.savefig(filepath, format='webp', bbox_inches='tight', pad_inches=0.1)
            # message("debug", f"Saved component plot to {filepath}") # Optional: can be verbose
        except Exception as e:
            message("error", f"Failed to save component plot {filepath}: {str(e)}")
            plt.close(fig) 
            raise
        finally:
            plt.close(fig) 
            
        return filepath

    def _classify_component_image_openai(self, image_path: Path, api_key: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Sends a component image to OpenAI Vision API for classification.

        Args:
            image_path: Path to the component image file (WebP format)
            api_key: OpenAI API key. If None, attempts to use OPENAI_API_KEY env var

        Returns:
            Tuple: (label: str, confidence: float, reason: str)
                   Defaults to ("other_artifact", 1.0, "API error or parsing failure") on error.
        """
        if not image_path or not image_path.exists():
            message("error", "Invalid or non-existent image path provided for classification.")
            return "other_artifact", 1.0, "Invalid image path"

        try:
            effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not effective_api_key:
                if hasattr(openai, 'api_key') and openai.api_key:
                    effective_api_key = openai.api_key
                else:
                    raise ValueError("OpenAI API key not provided via argument, environment variable (OPENAI_API_KEY), or openai.api_key.")

            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            client = openai.OpenAI(api_key=effective_api_key)
            message("debug", f"Sending component image {image_path.name} to OpenAI Vision API...")
            
            response = client.responses.create( # type: ignore
                model="gpt-4.1",
                input=[{ # type: ignore
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": self._OPENAI_ICA_PROMPT},
                        {"type": "input_image", "image_url": f"data:image/webp;base64,{base64_image}", "detail": "high"}
                    ]
                }],
                temperature=0.1
            )
            message("debug", "OpenAI API call successful.")

            resp_text = None
            if hasattr(response, 'output') and response.output and isinstance(response.output, list) and len(response.output) > 0:
                first_output = response.output[0]
                if hasattr(first_output, 'content') and isinstance(first_output.content, list) and len(first_output.content) > 0:
                    first_content = first_output.content[0]
                    if hasattr(first_content, 'text'):
                        resp_text = first_content.text.strip()
            
            if resp_text is not None:
                message("debug", f"Raw OpenAI response: '{resp_text}'")
                allowed_labels = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other_artifact"]
                labels_pattern = "|".join(allowed_labels)
                
                # Regex adjusted for correct group indexing and robust string capture.
                # Group 1: label, Group 2: confidence, Group 3: reason.
                match = re.search(
                    r"^\s*\(\s*"                                  # Start of tuple, optional leading whitespace
                    r"['\"]?(" + labels_pattern + r")['\"]?"       # Group 1: The label (e.g., "eye", "brain")
                    r"\s*,\s*"                                   # Comma and optional spaces
                    r"([01](?:\.\d+)?)"                         # Group 2: The confidence score (e.g., 0.95)
                    r"\s*,\s*"                                   # Comma and optional spaces
                    r"['\"](.*)['\"]"                             # Group 3: The reasoning string (greedy capture between quotes)
                    r"\s*\)\s*$",                                # End of tuple, optional trailing whitespace
                    resp_text, re.IGNORECASE | re.DOTALL
                )

                if match:
                    label = match.group(1).lower()
                    if label not in allowed_labels:
                        message("warning", f"OpenAI returned an unexpected label '{label}'. Defaulting to 'other_artifact'. Raw: '{resp_text}'")
                        return "other_artifact", 1.0, f"Unexpected label: {label}. Parsed from: {resp_text}"
                    confidence = float(match.group(2))
                    reason = match.group(3).strip() # No need for complex replace if captured well
                    # Basic unescaping for common cases if still needed, though greedy capture might handle many.
                    reason = reason.replace("\\\"", '"').replace("\\'", "'")

                    message("debug", f"Parsed classification: Label={label}, Conf={confidence:.2f}, Reason='{reason[:50]}...'")
                    return label, confidence, reason
                else:
                    message("warning", f"Could not parse OpenAI response format for specific labels: '{resp_text}'. Defaulting to 'other_artifact'.")
                    return "other_artifact", 1.0, f"Failed to parse response: {resp_text}"
            else:
                message("error", "Could not extract text content from OpenAI response structure.")
                return "other_artifact", 1.0, "Invalid response structure (no text)"

        except openai.APIConnectionError as e: # type: ignore
            message("error", f"OpenAI API connection error: {e}")
            return "other_artifact", 1.0, f"API Connection Error: {e}"
        except openai.AuthenticationError as e: # type: ignore
            message("error", f"OpenAI API authentication error: {e}. Check your API key.")
            return "other_artifact", 1.0, f"API Authentication Error: {e}"
        except openai.RateLimitError as e: # type: ignore
            message("error", f"OpenAI API rate limit exceeded: {e}")
            return "other_artifact", 1.0, f"API Rate Limit Error: {e}"
        except openai.APIStatusError as e: # type: ignore
            message("error", f"OpenAI API status error: Status={e.status_code}, Response={e.response}")
            return "other_artifact", 1.0, f"API Status Error {e.status_code}: {e}"
        except Exception as e:
            message("error", f"Unexpected exception during vision classification: {type(e).__name__} - {e}")
            message("error", traceback.format_exc())
            return "other_artifact", 1.0, f"Unexpected Exception: {type(e).__name__}"
    
    def classify_ica_components_vision(self, 
                                      api_key: Optional[str] = None,
                                      confidence_threshold: float = 0.8,
                                      auto_exclude: bool = True,
                                      labels_to_exclude: Optional[List[str]] = None
                                      ) -> pd.DataFrame:
        """
        Classifies ICA components using OpenAI Vision API with specific artifact labels.
        
        This method generates visualizations of each ICA component, sends them to
        the OpenAI Vision API for classification into categories (brain, muscle, eye, etc.), 
        and returns a DataFrame with the results. It can also automatically exclude 
        components based on their assigned label and confidence.
        
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
        confidence_threshold : float, default=0.8
            Minimum confidence for a classification to be accepted for auto-exclusion.
        auto_exclude : bool, default=True
            If True, automatically add components to the exclude list in self.final_ica
            if their label is in `labels_to_exclude` and confidence is met.
        labels_to_exclude : List[str], optional
            A list of specific labels (e.g., ["muscle", "eye", "heart"]) that should be
            considered for auto-exclusion if `auto_exclude` is True and confidence
            threshold is met. If None, defaults to all labels except "brain".
            
        Returns
        ------ pd.DataFrame
            DataFrame containing the classification results for each component.
            
        Notes
        -----
        This method also generates a PDF report with all component images and
        their classifications.
        """
        message("header", "Running ICA component classification with OpenAI Vision API (Detailed Labels)")
        
        if not hasattr(self, 'final_ica') or self.final_ica is None:
            message("error", "ICA (self.final_ica) not found. Please run `run_ica` first.")
            # Consider raising an error or returning an empty DataFrame
            return pd.DataFrame() # Or raise RuntimeError
        
        ica = self.final_ica
        raw = self.raw # Assuming self.raw is available and appropriate
        
        # Define default labels to exclude if not provided
        if labels_to_exclude is None:
            all_possible_labels = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other_artifact"]
            labels_to_exclude = [lbl for lbl in all_possible_labels if lbl != "brain"]
        
        message("info", f"Auto-excluding components with labels: {labels_to_exclude} if confidence >= {confidence_threshold}")

        # Results storage
        classification_results_list: List[Dict[str, Any]] = []
        processed_count = 0
        num_components = ica.n_components_
        
        message("info", f"Processing {num_components} ICA components...")
        
        # Create a temporary directory for images
        with tempfile.TemporaryDirectory(prefix="autoclean_ica_vision_") as temp_dir:
            temp_path = Path(temp_dir)
            message("info", f"Using temporary directory for images: {temp_path}")
            
            component_image_paths: List[Optional[Path]] = []
            
            # Step 1: Generate images for all components
            for i in range(num_components):
                try:
                    # Generate an image suitable for classification
                    image_path = self._plot_component_for_vision(ica, raw, i, temp_path)
                    component_image_paths.append(image_path)
                except Exception as plot_err:
                    message("warning", f"Failed to plot component {i}: {plot_err}. Skipping classification for this component.")
                    component_image_paths.append(None) # Add a placeholder
            
            # Step 2: Classify all component images
            for i, image_path in enumerate(component_image_paths):
                comp_name = f"IC{i+1}" # Component name for logging
                if image_path is None:
                    # Component plotting failed, assign default "other_artifact"
                    label, confidence, reason = "other_artifact", 1.0, "Failed to generate component image for classification."
                else:
                    try:
                        label, confidence, reason = self._classify_component_image_openai(image_path, api_key)
                    except Exception as classify_err:
                         message("warning", f"Classification call failed for component {comp_name}: {classify_err}. Defaulting to 'other_artifact'.")
                         label, confidence, reason = "other_artifact", 1.0, f"Classification call failed: {classify_err}"

                # Determine if component should be excluded
                exclude_this_component = False
                if auto_exclude and label in labels_to_exclude and confidence >= confidence_threshold:
                    exclude_this_component = True
                
                classification_results_list.append({
                    "component_index": i,
                    "component_name": comp_name,
                    "label": label,
                    "confidence": confidence,
                    "reason": reason,
                    "exclude_vision": exclude_this_component # Explicitly name this decision source
                })
                
                # Log result immediately
                log_level = "debug" # Default
                if label == "brain":
                    log_level = "success"
                elif exclude_this_component: # Highlight if it's marked for exclusion
                    log_level = "warning"
                    
                log_message = (f"Vision Result | Component: {comp_name} | Label: {label.upper()} | "
                               f"Confidence: {confidence:.2f} | Exclude: {exclude_this_component} | Reason: {reason[:30]}...")
                message(log_level, log_message)
                
                if image_path is not None: # Only count if image was processed
                    processed_count += 1
            
            message("info", f"OpenAI classification complete. Successfully processed {processed_count} of {num_components} components.")
            
            # Create DataFrame from results
            self.ica_vision_flags = pd.DataFrame(classification_results_list)
            
            # Step 3: Update ICA exclude list based on vision classification if auto_exclude=True
            if auto_exclude and not self.ica_vision_flags.empty:
                # Get components to exclude based on 'exclude_vision' flag
                components_to_exclude_indices = self.ica_vision_flags[
                    self.ica_vision_flags['exclude_vision'] == True
                ]['component_index'].tolist()
                
                if not components_to_exclude_indices:
                    message("info", "No new components met criteria for exclusion based on vision classification.")
                else:
                    message("info", f"Identified {len(components_to_exclude_indices)} components for exclusion via vision: {components_to_exclude_indices}")
                    
                    # Initialize exclude list if needed
                    if ica.exclude is None:
                        ica.exclude = []
                    
                    # Update exclude list, ensuring no duplicates and sorted
                    current_exclusions = set(ica.exclude)
                    for idx in components_to_exclude_indices:
                        current_exclusions.add(idx)
                    ica.exclude = sorted(list(current_exclusions))
                    
                    message("info", f"Updated ICA exclude list: {ica.exclude}")
                    
                    # Save updated ICA object
                    if hasattr(self, 'config') and hasattr(self, 'raw'): # Check if task has config and raw
                        save_ica_to_fif(ica, self.config, self.raw)
                        message("debug", "Saved ICA object with updated exclusions from vision.")
            
            # Step 4: Generate PDF report with classified components
            report_path_obj = self._generate_ica_vision_report(component_image_paths, classification_results_list)
            report_file_str = str(report_path_obj) if report_path_obj else "N/A"

            # Update metadata
            # Count brain vs. non-brain (artifact) components for summary
            brain_components_count = 0
            artifact_components_count = 0
            if not self.ica_vision_flags.empty:
                brain_components_count = len(self.ica_vision_flags[self.ica_vision_flags['label'] == 'brain'])
                # All other labels are considered artifacts for this summary count
                artifact_components_count = len(self.ica_vision_flags[self.ica_vision_flags['label'] != 'brain'])

            metadata = {
                "ica_vision_classification": {
                    "components_processed_successfully": processed_count,
                    "total_components": num_components,
                    "brain_components_count_vision": brain_components_count,
                    "artifact_components_count_vision": artifact_components_count,
                    "configured_confidence_threshold": confidence_threshold,
                    "configured_auto_exclude": auto_exclude,
                    "configured_labels_to_exclude": labels_to_exclude,
                    "report_file": report_file_str
                }
            }
            
            if hasattr(self, '_update_metadata') and callable(self._update_metadata):
                self._update_metadata("step_classify_ica_components_vision", metadata)
        
        message("success", "ICA component classification with detailed labels complete. See PDF report for details.")
        return self.ica_vision_flags
    
    def _generate_ica_vision_report(self, 
                                   component_image_paths: List[Optional[Path]], 
                                   classification_results: List[Dict[str, Any]]
                                   ) -> Optional[Path]:
        """
        Generates a PDF report with ICA component images and their specific classifications.
        
        Parameters
        ----------
        component_image_paths : List[Optional[Path]]
            List of paths to component images (can be None if plotting failed).
        classification_results : List[Dict[str, Any]]
            List of dictionaries containing classification results from `classify_ica_components_vision`.
            Each dict should have 'component_name', 'label', 'confidence', 'exclude_vision', 'reason'.
            
        Returns
        ------
        Optional[Path]
            Path to the generated PDF report, or None if generation failed.
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages # Local import
            from datetime import datetime # Local import
            
            # Ensure matplotlib backend is Agg for non-interactive use
            matplotlib.use("Agg")

            if not (hasattr(self, 'config') and self.config and 'derivatives_dir' in self.config and 'bids_path' in self.config):
                 message("error", "Configuration for derivatives_dir or bids_path not found. Cannot generate PDF report.")
                 return None

            derivatives_dir = Path(self.config["derivatives_dir"])
            bids_path_obj = self.config["bids_path"]
            if not hasattr(bids_path_obj, 'basename'):
                message("error", "BIDSPath object in config does not have a 'basename'. Cannot name PDF report.")
                return None

            basename = bids_path_obj.basename
            basename = basename.replace("_eeg", "_ica_vision_classification_detailed") 
            pdf_path = derivatives_dir / basename
            pdf_path = pdf_path.with_suffix(".pdf")
            
            if pdf_path.exists():
                try:
                    pdf_path.unlink()
                except OSError as e:
                    message("warning", f"Could not remove existing report {pdf_path}: {e}")

            with PdfPages(pdf_path) as pdf:
                # First page: Summary table
                fig_summary = plt.figure(figsize=(12, 9))
                ax_summary = fig_summary.add_subplot(111)
                ax_summary.axis("off")
                
                table_data = []
                row_colors = []
                color_brain = "#d4edda" 
                color_artifact = "#f8d7da" 

                for result in classification_results:
                    label = result.get("label", "N/A")
                    table_data.append([
                        result.get("component_name", "N/A"),
                        label.upper(),
                        f"{result.get('confidence', 0.0):.2f}",
                        "Yes" if result.get("exclude_vision", False) else "No",
                        result.get("reason", "N/A")[:60] + "..." if len(result.get("reason", "N/A")) > 60 else result.get("reason", "N/A")
                    ])
                    current_row_color = color_brain if label == "brain" else color_artifact
                    row_colors.append([current_row_color] * 5) 
                
                col_labels = ["Component", "Vision Label", "Confidence", "Auto-Exclude", "Reason (brief)"]
                col_widths = [0.12, 0.15, 0.10, 0.13, 0.50]

                if not table_data:
                     ax_summary.text(0.5, 0.5, "No classification results to display.", ha='center', va='center')
                else:
                    table = ax_summary.table(
                        cellText=table_data, colLabels=col_labels, loc="center",
                        cellLoc="left", rowColours=row_colors, colWidths=col_widths
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1.0, 1.3)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fig_summary.suptitle(
                    f"OpenAI Vision ICA Classification Summary - {basename}\\n"
                    f"Generated: {timestamp}", fontsize=12, y=0.97
                )
                
                from matplotlib.patches import Patch 
                legend_elements = [
                    Patch(facecolor=color_brain, edgecolor=color_brain, label='Brain Component'),
                    Patch(facecolor=color_artifact, edgecolor=color_artifact, label='Artifact Component (any type)')
                ]
                ax_summary.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 0.9), fontsize=9)
                
                # Use subplots_adjust for the summary page, tight_layout might struggle with complex tables.
                plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
                pdf.savefig(fig_summary, bbox_inches='tight')
                plt.close(fig_summary)
                
                # Add each component image to the PDF on subsequent pages
                for idx, result in enumerate(classification_results):
                    image_path = component_image_paths[idx] if idx < len(component_image_paths) else None
                    comp_name = result.get("component_name", f"IC {idx+1}")
                    label = result.get("label", "N/A").upper()
                    conf = result.get("confidence", 0.0)
                    exclude_str = "Yes" if result.get("exclude_vision", False) else "No"
                    reason = result.get("reason", "N/A")

                    fig_comp_page = plt.figure(figsize=(8.5, 11))

                    if image_path is None or not image_path.exists():
                        ax_text_page = fig_comp_page.add_subplot(111)
                        ax_text_page.axis("off")
                        info_text = (f"Component: {comp_name}\\n"
                                     f"Vision Label: {label}\\n"
                                     f"Confidence: {conf:.2f}\\n"
                                     f"Auto-Excluded: {exclude_str}\\n\\n"
                                     f"Reason: {reason}\\n\\n"
                                     f"Note: Component image could not be plotted or found.")
                        ax_text_page.text(0.05, 0.95, info_text, ha='left', va='top', fontsize=10, wrap=True)
                        fig_comp_page.suptitle(f"Details for {comp_name} (Image Unavailable)", fontsize=14, y=0.98)
                    else:
                        try:
                            ax_img = fig_comp_page.add_subplot(111)
                            img = plt.imread(image_path) 
                            ax_img.imshow(img)
                            ax_img.axis("off")
                            
                            title_color = color_brain if result.get("label") == "brain" else color_artifact
                            page_title = (f"{comp_name} - Vision Label: {label} (Conf: {conf:.2f})\\n"
                                          f"Auto-Excluded: {exclude_str}")
                            
                            fig_comp_page.suptitle(page_title, fontsize=12, fontweight="bold", 
                                                   color='black', y=0.98, 
                                                   bbox=dict(boxstyle="round,pad=0.3", fc=title_color, ec="black", alpha=0.7))
                            fig_comp_page.text(0.05, 0.05, f"Reason: {reason}", fontsize=9, wrap=True, 
                                               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                        except Exception as e_img:
                            message("warning", f"Failed to process/display image for component {comp_name} in PDF: {str(e_img)}")
                            # Add a fallback text display if image processing fails mid-way
                            ax_fallback = fig_comp_page.add_subplot(111) # Potentially overwrites ax_img if it existed
                            ax_fallback.axis("off")
                            ax_fallback.text(0.5,0.5, f"Error displaying image for {comp_name}.\nClassification: {label} (Conf: {conf:.2f})\nReason: {reason}", 
                                             ha='center', va='center', wrap=True)
                            fig_comp_page.suptitle(f"Details for {comp_name} (Image Display Error)", fontsize=14, y=0.98)
                    
                    # Removed explicit plt.tight_layout() for component pages.
                    # Relaying on bbox_inches='tight' in savefig.
                    pdf.savefig(fig_comp_page, bbox_inches='tight') 
                    plt.close(fig_comp_page)
                
                message("info", f"ICA vision classification report (detailed) saved to {pdf_path}")
                return pdf_path
                
        except ImportError:
            message("error", "Matplotlib or PdfPages not available. Cannot generate PDF report for ICA vision.")
            return None
        except Exception as e_pdf:
            message("error", f"Failed to generate ICA classification PDF report: {str(e_pdf)}")
            message("error", traceback.format_exc())
            return None
