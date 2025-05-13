import os
import base64
import tempfile
import traceback
import re
import json # Added for JSON output
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any # Added List, Dict, Any

import matplotlib
matplotlib.use("Agg") # Ensure non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import mne
from mne.preprocessing import read_ica # Correct import for reading ICA
import openai

# --- Configuration ---
# Ensure your OpenAI API key is set as an environment variable
# or set it directly here (less secure for shared scripts)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MODIFY THESE PATHS TO YOUR EXAMPLE DATA
# It's best to use a small, representative raw file and its corresponding ICA solution
EXAMPLE_RAW_PATH = "C:/Users/Gam9LG/Documents/AutocleanDev2/TestingRest/bids/derivatives/sub-141278/eeg/0101_rest_pre_ica.set"
EXAMPLE_ICA_PATH = "C:/Users/Gam9LG/Documents/AutocleanDev2/TestingRest/bids/derivatives/sub-141278/eeg/0101_rest-ica.fif" 

# --- New Batch Configuration ---
START_COMPONENT_INDEX = 0  # Starting component index for the batch
NUM_COMPONENTS_TO_BATCH = 5 # Number of components to process in this batch
OUTPUT_DIR_PATH = "./ica_vision_test_output" # Directory to save plots and JSON results
# --- End New Batch Configuration ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!          EDIT YOUR EXPERIMENTAL PROMPT BELOW        !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CUSTOM_OPENAI_ICA_PROMPT = """Analyze this EEG ICA component image and classify it.

The image displays an Independent Component (IC) from an EEG recording. It includes:
1. A scalp topography map showing the spatial distribution of the component's activity.
2. A Power Spectral Density (PSD) plot showing the component's power at different frequencies.
3. Time series plots of the component's activity over the first 30 seconds.

Based on a comprehensive analysis of ALL these features, classify the component into ONE of the following categories:

- "brain":
    - *Topography:* Often looks dipolar (distinct positive and negative poles). A good dipole fit is suggested by a clean, bilobed pattern, and the source appears to be located within the brain. Scalp map is not excessively concentrated to a single electrode. While brain components can be frontal and dipolar, always cross-reference with the time series to exclude eye movement patterns (see 'eye' component description).
    - *Spectrum:* Shows a 1/f-like decrease in power with increasing frequency (i.e., power diminishes at higher frequencies). Often has distinct peaks in physiological bands (e.g., alpha around 8-12 Hz, but also other peaks between 5-30 Hz are common).
    - *Time Series:* Often characterized by **rhythmic or quasi-rhythmic oscillations** (e.g., alpha, beta bands) or more complex, but still fundamentally **wave-like activity without sharp, sustained DC offsets or step-like shifts**. It should **not** exhibit abrupt, sustained shifts in amplitude between distinct levels, which are characteristic of eye saccades (see 'eye' time series).

- "eye": (Eye movements or blinks)
    - *Topography:* Suggests electrical sources near the eyes.
        - *Vertical eye movements/blinks:* Strong activity at frontal electrodes, often appearing unipolar (all positive or all negative deflection) due to the typical EEG recording montage.
        - *Horizontal eye movements:* Typically a clear dipolar pattern focused near the frontal regions, with distinct positive polarity on one side (e.g., left frontal) and negative polarity on the other (e.g., right frontal), or vice-versa. This pattern, suggesting a source between or just in front of the eyes, is a key indicator of horizontal/lateral eye movements, *especially when accompanied by characteristic step-like changes in the time series (see below)*.
    - *Spectrum:* Most power is concentrated at low frequencies (typically < 5 Hz).
    - *Time Series:*
        - *Vertical (blinks/looking up/down):* Shows clear, relatively frequent, large-amplitude spikes (blinks) or slow rolling waves/drifts.
        - *Horizontal (saccades):* Shows **abrupt, rapid shifts connecting distinct, relatively sustained DC levels or plateaus** (the signal 'holds' at one level, then quickly jumps and 'holds' at another, even if briefly or within a somewhat noisy signal). This creates a pattern of 'square waves' or 'boxcar shapes,' which are **non-oscillatory by nature** and lack the continuous rhythmic flow of brain waves. **Crucially, when a frontal dipolar topography (as described above) co-occurs with these non-oscillatory, step-like time series patterns featuring these characteristic sustained level changes, it is a very strong sign of lateral eye movements.**

- "muscle": (Muscle activity, EMG)
    - *Topography:* Can sometimes appear dipolar but is usually very focal/concentrated, indicating a shallow source outside the skull (e.g., scalp, temporal, or neck muscles). The pattern is often less organized than brain dipoles.
    - *Spectrum:* Dominated by power at higher frequencies (typically > 20 Hz). This power is often broadband, meaning it's spread across a wide range of high frequencies, rather than a sharp peak. Little low-frequency power relative to high-frequency power.
    - *Time Series:* Often shows sustained or bursting high-frequency, spiky, non-stationary activity.

- "heart": (Heartbeat artifact, ECG)
    - *Topography:* Often a near-linear gradient across the scalp or a very broad, diffuse pattern, suggesting a distant and powerful electrical source like the heart. Can sometimes have a left-to-right or inferior-to-superior orientation.
    - *Spectrum:* May not show specific defining peaks related to the heartbeat itself; the primary evidence is in the time series. Spectrum might be contaminated by other activity.
    - *Time Series:* Shows clear, regular, repetitive QRS-complex-like waveforms (sharp peak followed by slower wave). These complexes occur at a rate of approximately 1 Hz (around 60 beats per minute, but can vary).

- "line_noise": (Electrical grid interference)
    - *Topography:* Can vary; not the primary indicator.
    - *Spectrum:* Must show a VERY SHARP and PROMINENT **PEAK** of power at either 50 Hz or 60 Hz. This peak should be significantly more powerful than the activity at surrounding frequencies.
      **IMPORTANT CLARIFICATION:** A sharp *DIP*, *NOTCH*, or *ABSENCE OF POWER* at 50 Hz or 60 Hz is **NOT** line noise. Such a feature indicates that a notch filter was applied during preprocessing, which is a normal step to *remove* line noise. Therefore, if you see a notch or dip at these frequencies, do **NOT** classify the component as "line_noise" based on that feature. Line noise is only present if there is an *EXCESSIVE PEAK* of power.
    - *Time Series:* Often shows continuous or intermittent sinusoidal oscillations at 50 Hz or 60 Hz, corresponding to the spectral peak.

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
Example: ("brain", 0.85, "Dipolar topography primarily over prefrontal areas. Spectrum shows a 1/f trend with some beta band activity. Time series displays continuous, somewhat irregular wave-like activity WITHOUT any sharp, sustained DC level shifts or clear 'boxcar' shapes typically seen in saccadic eye movements.")
Example: ("muscle", 0.92, "Focal topography over temporal area, shallow appearance. Spectrum is dominated by broadband high-frequency activity (>20Hz). Time series shows sustained spiky, high-frequency bursts.")
Example: ("line_noise", 0.98, "Spectrum exhibits an extremely sharp and dominant PEAK at 50 Hz, far exceeding other frequencies. Topography is diffuse but the spectral PEAK is unequivocal.")
Example: ("channel_noise", 0.90, "Topography is highly focal, concentrated on a single frontal electrode. Spectrum shows a 1/f trend. Time series exhibits intermittent large spikes.")
"""


# --- Adapted Helper Functions (from your IcaMixin) ---

def _plot_component_for_vision_standalone(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    component_idx: int,
    output_dir: Path
) -> Optional[Path]:
    """
    Standalone version to create a plot for an ICA component.
    Adapted from IcaMixin._plot_component_for_vision.
    """
    fig = plt.figure(figsize=(10, 10), dpi=120)
    gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3) # Increased hspace slightly for titles
    
    ax_topo = fig.add_subplot(gs[0, 0]) 
    ax_psd_wide = fig.add_subplot(gs[0, 1]) # Renamed from ax_psd
    ax_psd_focused = fig.add_subplot(gs[1, :])  # Replaces ax_ts_full
    ax_ts_10s = fig.add_subplot(gs[2, :])  # Remains the same
    
    sources = ica_obj.get_sources(raw_obj)
    sfreq = sources.info['sfreq']
    component_data_array = sources.get_data(picks=[component_idx])[0]
    
    # 1. Topography
    try:
        ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                               show=False, colorbar=True, title="")
        ax_topo.set_title(f"IC {component_idx} - Topography")
    except Exception as e:
        print(f"Error plotting topography: {e}")
        ax_topo.text(0.5, 0.5, "Topo plot failed", ha='center', va='center')

    # 2. PSD (Wide Range, 1-100 Hz or Nyquist)
    try:
        from mne.time_frequency import psd_array_welch
        fmin_psd_wide = 1.0
        fmax_psd_wide = min(100.0, sfreq / 2.0 - 0.5)
        n_fft_psd = int(sfreq * 1.0) # 1-second window for a smoother spectrum
        if n_fft_psd == 0: 
            n_fft_psd = min(256, len(component_data_array))

        if fmax_psd_wide <= fmin_psd_wide: fmax_psd_wide = sfreq / 2.0 - 0.5

        psds_wide, freqs_wide = psd_array_welch(
            component_data_array, sfreq=sfreq, fmin=fmin_psd_wide, fmax=fmax_psd_wide, 
            n_fft=n_fft_psd, n_overlap=0, verbose=False
        )
        psds_db_wide = 10 * np.log10(psds_wide)
        ax_psd_wide.plot(freqs_wide, psds_db_wide, color='black', linewidth=1)
        ax_psd_wide.set_title("PSD (1-100 Hz)")
        ax_psd_wide.set_xlabel("Frequency (Hz)")
        ax_psd_wide.set_ylabel("Power (dB)")
        if len(freqs_wide) > 0 : ax_psd_wide.set_xlim(freqs_wide[0], freqs_wide[-1])
        ax_psd_wide.grid(True, linestyle='--', alpha=0.5)
    except Exception as e:
        print(f"Error plotting wide PSD: {e}")
        ax_psd_wide.text(0.5, 0.5, "Wide PSD plot failed", ha='center', va='center')

    # 3. Focused PSD (1-40 Hz)
    try:
        # Ensure psd_array_welch is imported (it is from above)
        fmin_psd_focused = 1.0
        fmax_psd_focused = 40.0
        # Use the same n_fft for comparable smoothness, or adjust if different resolution is desired for this plot
        # n_fft_psd is already defined from the wide PSD plot

        # Ensure fmax_psd_focused is valid given sfreq
        if fmax_psd_focused >= sfreq / 2.0:
            fmax_psd_focused = sfreq / 2.0 - 0.5 
            print(f"Warning: Requested fmax for focused PSD ({40.0} Hz) was too high for sfreq ({sfreq} Hz). Adjusted to {fmax_psd_focused:.2f} Hz.")

        if fmax_psd_focused <= fmin_psd_focused:
             print(f"Skipping focused PSD: fmax ({fmax_psd_focused:.2f} Hz) is not greater than fmin ({fmin_psd_focused:.2f} Hz).")
        else:
            psds_focused, freqs_focused = psd_array_welch(
                component_data_array, sfreq=sfreq, fmin=fmin_psd_focused, fmax=fmax_psd_focused, 
                n_fft=n_fft_psd, n_overlap=0, verbose=False # Using same n_fft as wide PSD
            )
            psds_db_focused = 10 * np.log10(psds_focused)
            ax_psd_focused.plot(freqs_focused, psds_db_focused, color='darkgreen', linewidth=1)
            ax_psd_focused.set_title("PSD (1-40 Hz)")
            ax_psd_focused.set_xlabel("Frequency (Hz)")
            ax_psd_focused.set_ylabel("Power (dB)")
            if len(freqs_focused) > 0 : ax_psd_focused.set_xlim(freqs_focused[0], freqs_focused[-1])
            ax_psd_focused.grid(True, linestyle='--', alpha=0.5)

    except Exception as e:
        print(f"Error plotting focused PSD: {e}")
        ax_psd_focused.text(0.5, 0.5, "Focused PSD plot failed", ha='center', va='center')

    # 4. Time Series Segment (e.g., 30 seconds)
    duration_segment = 30.0 # Changed from 10.0 to 30.0 seconds
    max_samples_segment = min(int(duration_segment * sfreq), len(component_data_array))
    times_segment = np.arange(max_samples_segment) / sfreq
    
    ax_ts_10s.plot(times_segment, component_data_array[:max_samples_segment], linewidth=0.7, color='darkblue')
    ax_ts_10s.set_xlabel("Time (s)"); ax_ts_10s.set_ylabel("Amplitude (a.u.)")
    ax_ts_10s.set_title(f"IC {component_idx} Time Series ({duration_segment:.0f}s Segment)") # Updated title
    if max_samples_segment > 0: ax_ts_10s.set_xlim(times_segment[0], times_segment[-1])
    ax_ts_10s.grid(True, linestyle=':', alpha=0.7)
    
    fig.suptitle(f"ICA Component {component_idx} for Vision API Test", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    filename = f"test_component_{component_idx}.webp"
    filepath = output_dir / filename
    try:
        plt.savefig(filepath, format='webp', bbox_inches='tight', pad_inches=0.1)
    except Exception as e:
        print(f"Error saving figure: {e}")
        plt.close(fig)
        return None
    finally:
        plt.close(fig)
    return filepath

def _classify_component_image_openai_standalone(
    image_path: Path,
    prompt_string: str, # Takes the custom prompt string
    api_key_to_use: str
) -> Tuple[str, float, str]:
    """
    Standalone version to classify a component image using OpenAI Vision API.
    Adapted from IcaMixin._classify_component_image_openai.
    """
    if not image_path or not image_path.exists():
        print(f"Error: Invalid image path provided: {image_path}")
        return "other_artifact", 1.0, "Invalid image path for classification"

    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        client = openai.OpenAI(api_key=api_key_to_use)
        
        response = client.chat.completions.create( # Updated to use client.chat.completions.create
            model="gpt-4.1", 
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_string},
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
            print(f"Raw OpenAI response for parsing: '{resp_text}'") # Print before parsing
            allowed_labels = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other_artifact"]
            labels_pattern = "|".join(allowed_labels)
            
            match = re.search(
                r"^\s*\(\s*"
                r"['\"]?(" + labels_pattern + r")['\"]?"
                r"\s*,\s*"
                r"([01](?:\.\d+)?)"
                r"\s*,\s*"
                r"['\"](.*)['\"]" 
                r"\s*\)\s*$",
                resp_text, re.IGNORECASE | re.DOTALL
            )
            if match:
                label = match.group(1).lower()
                if label not in allowed_labels:
                    print(f"Warning: OpenAI returned an unexpected label '{label}'. Raw: '{resp_text}'")
                    return "other_artifact", 1.0, f"Unexpected label: {label}. Parsed from: {resp_text}"
                confidence = float(match.group(2))
                reason = match.group(3).strip().replace("\\\"", "\"").replace("\\\'", "\'")
                return label, confidence, reason
            else:
                print(f"Warning: Could not parse OpenAI response: '{resp_text}'")
                return "other_artifact", 1.0, f"Failed to parse: {resp_text}"
        else:
            print("Error: No text content in OpenAI response.")
            return "other_artifact", 1.0, "Invalid response structure (no text)"

    except Exception as e:
        print(f"Error during OpenAI call or parsing: {type(e).__name__} - {e}")
        traceback.print_exc()
        return "other_artifact", 1.0, f"Exception: {type(e).__name__}"

# --- Main Testing Logic ---
def run_test():
    print("--- Starting ICA Prompt Test ---")
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set. Please set it as an environment variable or in the script.")
        return

    output_path = Path(OUTPUT_DIR_PATH)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {output_path.resolve()}")
    except Exception as e:
        print(f"Error creating output directory {output_path}: {e}")
        return

    if not Path(EXAMPLE_RAW_PATH).exists() or not Path(EXAMPLE_ICA_PATH).exists():
        print(f"Error: Example data paths not found. Please check:\n- Raw: {EXAMPLE_RAW_PATH}\n- ICA: {EXAMPLE_ICA_PATH}")
        traceback.print_exc()
        return

    try:
        # It's crucial to preload data for ICA operations if not already done
        if EXAMPLE_RAW_PATH.lower().endswith('.set'):
            raw = mne.io.read_raw_eeglab(EXAMPLE_RAW_PATH, preload=True, eog=()) # Added eog=() as common practice for .set if EOG channels are not to be auto-detected or are handled later
        elif EXAMPLE_RAW_PATH.lower().endswith('.fif'):
            raw = mne.io.read_raw_fif(EXAMPLE_RAW_PATH, preload=True)
        else:
            print(f"Error: Unsupported raw file extension for {EXAMPLE_RAW_PATH}. Please use .set or .fif.")
            return
            
        ica = read_ica(EXAMPLE_ICA_PATH) # Use mne.preprocessing.read_ica
        print(f"Successfully loaded Raw data from: {EXAMPLE_RAW_PATH}")
        print(f"Successfully loaded ICA solution from: {EXAMPLE_ICA_PATH} ({ica.n_components_} components)")
    except Exception as e:
        print(f"Error loading MNE data: {e}")
        traceback.print_exc()
        return

    all_results: List[Dict[str, Any]] = [] # Initialize list to store results for all components

    # Determine the actual number of components to process in this batch
    # to avoid going out of bounds
    end_component_index = min(START_COMPONENT_INDEX + NUM_COMPONENTS_TO_BATCH, ica.n_components_)

    print(f"Processing components from index {START_COMPONENT_INDEX} to {end_component_index - 1}")

    # Use a single temporary directory for all images if _plot_component_for_vision_standalone needs it,
    # but since we're saving plots to OUTPUT_DIR_PATH, temp_dir might not be strictly needed here
    # unless _plot_component_for_vision_standalone *requires* it for intermediate steps.
    # For now, we'll pass the final output_path to the plotting function.

    for current_component_idx in range(START_COMPONENT_INDEX, end_component_index):
        print(f"\n--- Processing ICA Component: {current_component_idx} ---")
        
        component_result: Dict[str, Any] = {
            "component_index": current_component_idx,
            "image_file": None,
            "classification_status": "Not Processed",
            "label": None,
            "confidence": None,
            "reason": None,
            "error": None
        }

        # The _plot_component_for_vision_standalone saves the image directly
        # to the provided output_dir (which is our output_path here).
        image_file_path = _plot_component_for_vision_standalone(ica, raw, current_component_idx, output_path)

        if image_file_path and image_file_path.exists():
            component_result["image_file"] = str(image_file_path.relative_to(output_path.parent)) # Store relative path
            component_result["classification_status"] = "Image Generated, Pending Classification"
            print(f"Component image generated: {image_file_path}")
            print("Sending to OpenAI for classification with custom prompt...")
            
            try:
                label, confidence, reason = _classify_component_image_openai_standalone(
                    image_file_path,
                    CUSTOM_OPENAI_ICA_PROMPT,
                    OPENAI_API_KEY
                )
                component_result["label"] = label
                component_result["confidence"] = confidence
                component_result["reason"] = reason
                component_result["classification_status"] = "Classified"
                print(f"  Label:      {label.upper()}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Reasoning:  {reason[:100]}...") # Print a snippet
            except Exception as e_classify:
                error_msg = f"OpenAI classification failed: {type(e_classify).__name__} - {e_classify}"
                print(f"Error: {error_msg}")
                traceback.print_exc()
                component_result["classification_status"] = "Classification Failed"
                component_result["error"] = error_msg
        else:
            error_msg = f"Failed to generate or find image for component {current_component_idx}."
            print(error_msg)
            component_result["classification_status"] = "Plotting Failed"
            component_result["error"] = error_msg
        
        all_results.append(component_result)

    # Save all results to a JSON file
    results_json_path = output_path / "vision_classification_results.json"
    try:
        with open(results_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print(f"\nAll classification results saved to: {results_json_path}")
    except Exception as e_json:
        print(f"Error saving JSON results: {e_json}")
        traceback.print_exc()
            
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_test()