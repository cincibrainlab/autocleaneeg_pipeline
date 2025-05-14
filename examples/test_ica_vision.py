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
NUM_COMPONENTS_TO_BATCH = 15 # Number of components to process in this batch
OUTPUT_DIR_PATH = "./ica_vision_test_output" # Directory to save plots and JSON results
# --- End New Batch Configuration ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!          EDIT YOUR EXPERIMENTAL PROMPT BELOW        !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CUSTOM_OPENAI_ICA_PROMPT = """Analyze this EEG ICA component image and classify into ONE category:

- "brain": Dipolar pattern in CENTRAL/PARIETAL/TEMPORAL regions (NOT frontal/edges). 1/f spectrum with possible peaks at 8-12Hz. Wave-like time series WITHOUT step-shifts.

- "eye": 
  * LEFT-RIGHT FRONTAL dipolar pattern (red-blue on opposite sides) = DEFINITIVE for horizontal eye movements, regardless of other features.
  * Step-like, square-wave patterns in time series are clear indicators of lateral eye movements.
  * DO NOT be misled by 60Hz notches in the spectrum - these are normal filtering artifacts, NOT line noise.
  * Important distinction: Eye components have dipolar (two-sided) frontal activity, unlike channel noise which has only a single focal point.
  * RULE: If you see LEFT-RIGHT FRONTAL dipole pattern, classify as "eye" even if time series or spectrum is ambiguous.

- "muscle": 
  * DISTRIBUTED activity along EDGE of scalp (temporal/occipital/neck regions).
  * MUST show high-frequency power (>20Hz).
  * NOT isolated to single electrode (unlike channel noise).
  * Time series often shows spiky, high-frequency activity.

- "heart": Broad gradient across scalp. Time series shows regular QRS complexes (~1Hz).

- "line_noise": 
  * MUST show SHARP PEAK at 50/60Hz in spectrum - NOT a notch/dip (notches are filters, not line noise).
  * NOTE: Almost all components show a notch at 60Hz from filtering - this is NOT line noise!
  * Line noise requires a POSITIVE PEAK at 50/60Hz, not a negative dip.

- "channel_noise": 
  * SINGLE ELECTRODE "hot/cold spot" - tiny, isolated circular area WITHOUT an opposite pole.
  * Compare with eye: Channel noise has only ONE focal point, while eye has TWO opposite poles (dipole).
  * Example: A tiny isolated red or blue spot on one electrode, not a dipolar pattern.
  * Time series may show any pattern; topography is decisive.

- "other_artifact": Components not fitting above categories.

CLASSIFICATION PRIORITY:
1. LEFT-RIGHT FRONTAL dipole → "eye" (frontal red-blue pattern overrides everything except single-electrode focality)
2. SINGLE ELECTRODE focality (one tiny spot) → "channel_noise" 
3. PEAK (not notch) at 50/60Hz → "line_noise"
4. EDGE activity WITH high-frequency → "muscle"
5. Central/parietal/temporal dipole → "brain"

IMPORTANT: A 60Hz NOTCH (negative dip) in spectrum is normal filtering, seen in most components, and should NOT be used for classification!

Return: ("label", confidence_score, "detailed_reasoning")

Example: ("eye", 0.95, "Strong bilateral frontal topography with left-right dipolar pattern (red-blue). Detail view shows characteristic step-like patterns typical of horizontal eye movements.")
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
    fig = plt.figure(figsize=(10, 10), dpi=120)  # Adjusted height back to 10
    gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3)  # Back to 3 rows
    
    ax_topo = fig.add_subplot(gs[0, 0]) 
    ax_psd_wide = fig.add_subplot(gs[0, 1]) 
    ax_ts_zoom = fig.add_subplot(gs[1, :])  # Moved up one position
    ax_ts_full = fig.add_subplot(gs[2, :])  # Moved up one position
    
    sources = ica_obj.get_sources(raw_obj)
    sfreq = sources.info['sfreq']
    component_data_array = sources.get_data(picks=[component_idx])[0]
    
    # 1. Topography - Modified to use MATLAB-style colormap and settings
    try:
        # Create a custom colormap similar to MATLAB's 'jet'
        import matplotlib.cm as cm
        
        # Use a simpler set of parameters for increased compatibility
        topomap_args = dict(
            cmap='jet',                 # Use jet colormap (MATLAB-like)
            contours=6,                 # Number of contour lines
            sensors=True,               # Show sensors
            outlines='head'             # Simple head outline
        )
        
        print(f"Plotting topography for component {component_idx} with MATLAB-style settings")
        ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                              show=False, colorbar=True, title="", **topomap_args)
        
        ax_topo.set_title(f"IC {component_idx} - Topography")
        
        # Safer colorbar adjustment
        for child in ax_topo.get_children():
            if isinstance(child, plt.matplotlib.colorbar.ColorbarBase):
                try:
                    child.ax.set_box_aspect(10)
                except:
                    pass  # Ignore errors in colorbar adjustment
                break
                
    except Exception as e:
        print(f"Error plotting topography with MATLAB style: {e}")
        print("Falling back to default MNE topography plot")
        
        try:
            # Fallback to default plotting without custom parameters
            ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                                  show=False, colorbar=True, title="")
            ax_topo.set_title(f"IC {component_idx} - Topography")
        except Exception as e2:
            print(f"Error in fallback topography plot: {e2}")
            ax_topo.text(0.5, 0.5, "Topo plot failed", ha='center', va='center')

    # 2. PSD (Wide Range, 1-100 Hz) - this is the only PSD plot now
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
        ax_psd_wide.set_title("Power Spectrum")
        ax_psd_wide.set_xlabel("Frequency (Hz)")
        ax_psd_wide.set_ylabel("Power (dB)")
        if len(freqs_wide) > 0 : ax_psd_wide.set_xlim(freqs_wide[0], freqs_wide[-1])
        ax_psd_wide.grid(True, linestyle='--', alpha=0.5)
    except Exception as e:
        print(f"Error plotting wide PSD: {e}")
        ax_psd_wide.text(0.5, 0.5, "PSD plot failed", ha='center', va='center')

    # 3. Zoomed-in Time Series (2 seconds - ideal for seeing eye movements and neural oscillations)
    try:
        # Find an interesting segment with high variance if possible
        # This helps show characteristic patterns more clearly
        segment_length = int(3.0 * sfreq)  # Changed from 2.0 to 3.0 seconds
        if len(component_data_array) > segment_length * 2:
            # Calculate variance in 3-second sliding windows with 1.5-second overlap
            variance_windows = []
            for i in range(0, len(component_data_array) - segment_length, int(segment_length/2)):
                window = component_data_array[i:i+segment_length]
                variance_windows.append((i, np.var(window)))
            
            # Select the window with highest variance for eye/muscle components
            # or a window around 25-50% through the recording for more stable components
            variance_windows.sort(key=lambda x: x[1], reverse=True)
            
            # Use high-variance segment if it's significantly higher than median
            if variance_windows[0][1] > np.median([v for _, v in variance_windows]) * 1.5:
                start_idx = variance_windows[0][0]
            else:
                start_idx = int(len(component_data_array) * 0.3)  # ~30% through the recording
        else:
            start_idx = 0
        
        zoom_end_idx = min(start_idx + segment_length, len(component_data_array))
        zoom_times = np.arange(start_idx, zoom_end_idx) / sfreq
        zoom_data = component_data_array[start_idx:zoom_end_idx]
        
        ax_ts_zoom.plot(zoom_times, zoom_data, linewidth=1.0, color='darkblue')
        ax_ts_zoom.set_title(f"IC {component_idx} Detail View (3s Segment)")  # Updated to 3s
        ax_ts_zoom.set_xlabel("Time (s)")
        ax_ts_zoom.set_ylabel("Amplitude (a.u.)")
        if len(zoom_times) > 0:
            ax_ts_zoom.set_xlim(zoom_times[0], zoom_times[-1])
        
        # Add high-frequency specific y-limits for better visualization
        # Set ylim for better visualization based on component characteristics
        data_range = np.max(zoom_data) - np.min(zoom_data)
        if data_range > 0:
            # Add some padding (20%) to the limits
            ax_ts_zoom.set_ylim(
                np.min(zoom_data) - 0.1 * data_range,
                np.max(zoom_data) + 0.1 * data_range
            )
        
        # Add grid for visibility
        ax_ts_zoom.grid(True, linestyle=':', alpha=0.7)
        
        # Add time markers at 0.5-second intervals for eye blink/movement timing
        for t in np.arange(np.ceil(zoom_times[0] * 2) / 2, zoom_times[-1], 0.5):
            ax_ts_zoom.axvline(x=t, color='lightgray', linestyle='--', alpha=0.5)
    
    except Exception as e:
        print(f"Error plotting zoomed-in time series: {e}")
        ax_ts_zoom.text(0.5, 0.5, "Zoomed time series plot failed", ha='center', va='center')

    # 4. Full Time Series Segment (30 seconds)
    duration_segment = 30.0
    max_samples_segment = min(int(duration_segment * sfreq), len(component_data_array))
    times_segment = np.arange(max_samples_segment) / sfreq
    
    ax_ts_full.plot(times_segment, component_data_array[:max_samples_segment], linewidth=0.7, color='darkblue')
    ax_ts_full.set_xlabel("Time (s)"); ax_ts_full.set_ylabel("Amplitude (a.u.)")
    ax_ts_full.set_title(f"IC {component_idx} Full Time Series (30s Segment)")
    if max_samples_segment > 0: ax_ts_full.set_xlim(times_segment[0], times_segment[-1])
    ax_ts_full.grid(True, linestyle=':', alpha=0.7)
    
    # If we have the zoomed section, highlight it in the full view
    try:
        if 'zoom_times' in locals() and len(zoom_times) > 0:
            # Add a highlight rectangle for the zoomed region
            zoom_start = zoom_times[0]
            zoom_end = zoom_times[-1]
            ylims = ax_ts_full.get_ylim()
            rect = plt.Rectangle((zoom_start, ylims[0]), zoom_end - zoom_start, ylims[1] - ylims[0], 
                                fill=True, alpha=0.2, color='red')
            ax_ts_full.add_patch(rect)
    except Exception as e:
        print(f"Error highlighting zoom region: {e}")
    
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