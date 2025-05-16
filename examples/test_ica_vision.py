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
import warnings # Added for dipole fitting warnings
from mne import EvokedArray, make_sphere_model, create_info # Added for dipole fitting
from mne.time_frequency import psd_array_welch # Ensure this is imported if not already

# --- Configuration ---
# Ensure your OpenAI API key is set as an environment variable
# or set it directly here (less secure for shared scripts)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MODIFY THESE PATHS TO YOUR EXAMPLE DATA
# It's best to use a small, representative raw file and its corresponding ICA solution
EXAMPLE_RAW_PATH = "C:/Users/Gam9LG/Documents/AutocleanDev2/TestingRest/bids/derivatives/sub-69253/eeg/0022_VDaudio_ICA_pre_ica.set"
EXAMPLE_ICA_PATH = "C:/Users/Gam9LG/Documents/AutocleanDev2/TestingRest/bids/derivatives/sub-69253/eeg/0022_VDaudio_ICA-ica.fif" 

# --- New Batch Configuration ---
START_COMPONENT_INDEX = 0  # Starting component index for the batch
NUM_COMPONENTS_TO_BATCH = 5 # Number of components to process in this batch
OUTPUT_DIR_PATH = "./ica_vision_test_output" # Directory to save plots and JSON results
# --- End New Batch Configuration ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!          EDIT YOUR EXPERIMENTAL PROMPT BELOW        !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CUSTOM_OPENAI_ICA_PROMPT = """Analyze this EEG ICA component image and classify into ONE category:

- "brain": Dipolar pattern in CENTRAL, PARIETAL, or TEMPORAL regions (NOT FRONTAL or EDGE-FOCUSED). 1/f-like spectrum with possible peaks at 8-12Hz. Rhythmic, wave-like time series WITHOUT abrupt level shifts. MUST show decreasing power with increasing frequency (1/f pattern) - a flat or random fluctuating spectrum is NOT brain activity.

- "eye": 
  * Two main types of eye components:
    1. HORIZONTAL eye movements: LEFT-RIGHT FRONTAL dipolar pattern (red-blue on opposite sides of frontal region). Detail view shows step-like or square-wave patterns. LOOK CAREFULLY for RED on one FRONTAL side and BLUE on the other FRONTAL side - this pattern is ALWAYS an eye component, never muscle.
    2. VERTICAL eye movements/blinks: FRONTAL midline or bilateral positivity/negativity. Detail view shows distinctive spikes or slow waves.
  * Both types show power concentrated in lower frequencies (<5Hz).
  * DO NOT be misled by 60Hz notches in the spectrum - these are normal filtering artifacts, NOT line noise.
  * Key distinction: Eye components have activity focused in frontal regions, and will NEVER have activity in the occipital region.
  * CRITICAL: NEVER classify a component with clear LEFT-RIGHT FRONTAL dipole (red on one frontal side, blue on opposite frontal side) as muscle - this pattern is ALWAYS eye movement.
  * RULE: If you see LEFT-RIGHT FRONTAL dipole pattern or STRONG FRONTAL activation with spike patterns, classify as "eye".

- "muscle": 
 * topography often looks like 2 small dots that are different colors and right next to each other (a very shallow dipole/ "bowtie" like pattern) that combined take up less than 25% of the topography
 * topography can look like DISTRIBUTED activity along EDGE of scalp (temporal/occipital/neck regions).
 * NOT isolated to single electrode (unlike channel noise).
 * Power spectrum will show a positive slope (usually trending upwards from around 20Hz and above), are ALWAYS muscle.
 * Time series often shows spiky, high-frequency activity.

- "heart": Broad gradient across scalp. Time series shows regular QRS complexes (~1Hz).

- "line_noise": 
  * MUST show SHARP PEAK at 50/60Hz in spectrum - NOT a notch/dip (notches are filters, not line noise).
  * NOTE: Almost all components show a notch at 60Hz from filtering - this is NOT line noise!
  * Line noise requires a POSITIVE PEAK at 50/60Hz, not a negative dip.

- "channel_noise": 
  * SINGLE ELECTRODE "hot/cold spot" - tiny, isolated circular area typically without an opposite pole.
  * Compare with eye: Channel noise has only ONE focal point, while eye has TWO opposite poles (dipole).
  * Example: A tiny isolated red or blue spot on one electrode, not a dipolar pattern.
  * Time series may show any pattern; the focal topography is decisive.

- "other_artifact": Components not fitting above categories.

CLASSIFICATION PRIORITY:
1. LEFT-RIGHT FRONTAL dipole or STRONG FRONTAL activation with spikes → "eye"
2. SINGLE ELECTRODE isolated focality activity → "channel_noise" 
3. PEAK (not notch) at 50/60Hz → "line_noise"
4. EDGE activity WITH high-frequency power → "muscle"
5. Central/parietal/temporal dipole → "brain"

IMPORTANT: A 60Hz NOTCH (negative dip) in spectrum is normal filtering, seen in most components, and should NOT be used for classification!

Return: ("label", confidence_score, "detailed_reasoning")

Example: ("eye", 0.95, "Strong frontal topography with left-right dipolar pattern (horizontal eye movement) or frontal positivity with spike-like patterns (vertical eye movement/blinks). Low-frequency dominated spectrum and characteristic time series confirm eye activity.")
"""


# --- Adapted Helper Functions (from your IcaMixin) ---

def _plot_component_for_vision_standalone(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    component_idx: int,
    output_dir: Path
) -> Optional[Path]:
    """
    Updated version to create a plot for an ICA component, matching the example image.
    Layout: Topo+ContData on left, TS+Dipole+PSD on right.
    Dipole shown as text info (RV, Pos, Ori) using a sphere model.
    """
    fig = plt.figure(figsize=(12, 9.5), dpi=120)  # Adjusted figsize for better aspect ratio & space
    # GridSpec to match example: Topo+ContData on left, TS+Dipole+PSD on right
    # height_ratios: [row0_height, row1_height, row2_height]
    # width_ratios: [col0_width, col1_width]
    gs = GridSpec(3, 2, figure=fig, 
                  height_ratios=[0.915, 0.572, 2.213],  # Adjusted for square continuous data
                  width_ratios=[0.9, 1],      # Left col slightly narrower
                  hspace=0.6, wspace=0.35)     # Increased spacing

    # Left column
    ax_topo = fig.add_subplot(gs[0:2, 0]) # Topography spans 2 rows (row 0 and 1) on left
    ax_cont_data = fig.add_subplot(gs[2, 0]) # Continuous Data below Topo (row 2 on left)
    
    # Right column
    ax_ts_scroll = fig.add_subplot(gs[0, 1]) # Scrolling IC Activity (row 0 on right)
    ax_dipole = fig.add_subplot(gs[1, 1])    # Dipole Position (row 1 on right)
    ax_psd = fig.add_subplot(gs[2, 1])       # IC Activity Power Spectrum (row 2 on right)

    sources = ica_obj.get_sources(raw_obj)
    sfreq = sources.info['sfreq']
    component_data_array = sources.get_data(picks=[component_idx])[0]

    # 1. Topography
    try:
        ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                                show=False, colorbar=False, cmap='jet', outlines='head', 
                                sensors=True, contours=6) # colorbar=False, will add custom text instead
        ax_topo.set_title(f"IC{component_idx}", fontsize=12, loc='center') # Centered title

        # Calculate and display "% scalp data var. accounted for"
        comp_var = np.var(component_data_array)
        total_source_var = np.sum(np.var(sources.get_data(), axis=1)) # Sum of variances of all IC time courses
        if total_source_var > 0:
            var_explained = (comp_var / total_source_var) * 100
            # Position text below the topography plot area
            fig.text(0.27, 0.34, f"% scalp data var. accounted for: {var_explained:.1f}%", 
                     ha='center', va='top', fontsize=8) # Used fig.text for better global positioning
        else:
            fig.text(0.27, 0.34, "% scalp data var. accounted for: N/A",
                     ha='center', va='top', fontsize=8)
        
        ax_topo.set_xlabel("")
        ax_topo.set_ylabel("")
        ax_topo.set_xticks([])
        ax_topo.set_yticks([])

    except Exception as e:
        print(f"Error plotting topography for IC{component_idx}: {e}")
        ax_topo.text(0.5, 0.5, "Topo plot failed", ha='center', va='center', transform=ax_topo.transAxes)
        # traceback.print_exc() # Optionally uncomment for detailed trace

    # 2. Scrolling IC Activity (Time Series)
    try:
        duration_segment_ts = 5.0 
        max_samples_ts = min(int(duration_segment_ts * sfreq), len(component_data_array))
        times_ts_ms = (np.arange(max_samples_ts) / sfreq) * 1000 
        
        ax_ts_scroll.plot(times_ts_ms, component_data_array[:max_samples_ts], linewidth=0.8, color='dodgerblue')
        ax_ts_scroll.set_title("Scrolling IC Activity", fontsize=10)
        ax_ts_scroll.set_xlabel("Time (ms)", fontsize=9)
        ax_ts_scroll.set_ylabel("Amplitude (a.u.)", fontsize=9)
        if max_samples_ts > 0 and times_ts_ms.size > 0: # Check if times_ts_ms is not empty
            ax_ts_scroll.set_xlim(times_ts_ms[0], times_ts_ms[-1])
        ax_ts_scroll.grid(True, linestyle=':', alpha=0.6)
        ax_ts_scroll.tick_params(axis='both', which='major', labelsize=8)
    except Exception as e:
        print(f"Error plotting scrolling IC activity for IC{component_idx}: {e}")
        ax_ts_scroll.text(0.5, 0.5, "Time series failed", ha='center', va='center', transform=ax_ts_scroll.transAxes)
        # traceback.print_exc()

    # 3. Continuous Data (EEGLAB-style ERP image)
    try:
        total_duration_disp_cd = 30.0 
        segment_duration_ms_cd = 500 
        
        segment_len_samples_cd = int(segment_duration_ms_cd / 1000 * sfreq)
        if segment_len_samples_cd == 0: # Avoid division by zero or empty segments
             raise ValueError("Segment length in samples is zero. Check sfreq and segment_duration_ms_cd.")

        n_points_total_cd = int(min(total_duration_disp_cd * sfreq, len(component_data_array)))
        
        if n_points_total_cd < segment_len_samples_cd:
            raise ValueError(f"Not enough data ({n_points_total_cd} points) for one segment of {segment_len_samples_cd} points.")

        n_segments_cd = n_points_total_cd // segment_len_samples_cd
        if n_segments_cd == 0:
            raise ValueError("Zero segments calculated for Continuous Data plot. Not enough data.")
            
        erp_image_data = component_data_array[:n_segments_cd * segment_len_samples_cd].reshape(n_segments_cd, segment_len_samples_cd)
        
        v_abs = np.percentile(np.abs(erp_image_data), 98) if erp_image_data.size > 0 else 1.0
        v_abs = max(v_abs, 1e-9) # Ensure v_abs is not zero for vmin/vmax

        im = ax_cont_data.imshow(erp_image_data, aspect='auto', cmap='jet', interpolation='nearest',
                                 vmin=-v_abs, vmax=v_abs,
                                 extent=[0, segment_duration_ms_cd, n_segments_cd, 0])
        ax_cont_data.set_title("Continuous Data", fontsize=10)
        ax_cont_data.set_xlabel("Time (ms)", fontsize=9)
        ax_cont_data.set_ylabel("Data (Segments)", fontsize=9)
        
        cbar_cont = fig.colorbar(im, ax=ax_cont_data, orientation='vertical', fraction=0.046, pad=0.1) # Adjusted pad
        cbar_cont.set_label("Activation (a.u.)", fontsize=8)
        cbar_cont.ax.tick_params(labelsize=7)
        ax_cont_data.tick_params(axis='both', which='major', labelsize=8)

    except Exception as e:
        print(f"Error plotting continuous data for IC{component_idx}: {e}")
        ax_cont_data.text(0.5, 0.5, "Continuous data failed", ha='center', va='center', transform=ax_cont_data.transAxes)
        # traceback.print_exc()

    # 4. Dipole Position (Text-based info using sphere model)
    try:
        ax_dipole.set_title("Dipole Position", fontsize=10)
        ax_dipole.set_axis_off() # Turn off axis for text display

        if raw_obj.get_montage() is None:
            ax_dipole.text(0.5, 0.5, "No montage in raw data.\nCannot fit dipole.", 
                           ha='center', va='center', fontsize=8, transform=ax_dipole.transAxes)
            raise RuntimeError("Raw object does not have a montage.")

        sphere = make_sphere_model(info=raw_obj.info, head_radius=None, verbose=False)
        component_pattern = ica_obj.get_components()[:, component_idx].reshape(-1, 1)

        # Create EvokedArray info - crucial that ch_names match raw_obj for montage and sphere model
        ev_info = create_info(ch_names=raw_obj.ch_names, sfreq=raw_obj.info['sfreq'], ch_types='eeg', verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ev_info.set_montage(raw_obj.get_montage())
        
        if component_pattern.shape[0] != len(ev_info['ch_names']):
            # This can happen if ICA was fit on a subset of channels not matching raw_obj.info
            # Attempt to use ica_obj.ch_names if they exist and match pattern
            if hasattr(ica_obj, 'ch_names') and len(ica_obj.ch_names) == component_pattern.shape[0]:
                temp_info_for_ica_channels = create_info(ch_names=list(ica_obj.ch_names), sfreq=raw_obj.info['sfreq'], ch_types='eeg', verbose=False)
                # We need a montage for these specific channels if different from raw_obj. This is tricky.
                # For now, we must assume component_pattern channels are in raw_obj.info or fitting is ill-defined for sphere model from raw_obj.info.
                # Sticking to original ev_info based on raw_obj.ch_names and hoping for subset match or error.
                # If error, it will be caught.
                print(f"Warning: Component pattern ({component_pattern.shape[0]} ch) might not match raw_obj.info channels ({len(ev_info['ch_names'])} ch). Using raw_obj.info channels for dipole fitting.")

        evoked_topo = EvokedArray(component_pattern, ev_info, tmin=0, verbose=False)
        
        # Fit dipole (cov=None implies identity matrix, common for ICs)
        dipoles, residual = mne.fit_dipole(evoked_topo, cov=None, bem=sphere, min_dist=1.0, verbose=False)
        
        if dipoles is None or len(dipoles) == 0:
            ax_dipole.text(0.5, 0.5, "Dipole fitting returned no dipoles.", 
                           ha='center', va='center', fontsize=8, transform=ax_dipole.transAxes)
            raise RuntimeError("Dipole fitting failed to return any dipoles.")
        
        dip = dipoles[0]
        rv_text = f"RV: {dip.gof[0]:.1f}%" # MNE's gof is residual variance for fit_dipole
        pos_text = f"Pos (mm): ({dip.pos[0,0]*1000:.1f}, {dip.pos[0,1]*1000:.1f}, {dip.pos[0,2]*1000:.1f})"
        ori_text = f"Ori: ({dip.ori[0,0]:.2f}, {dip.ori[0,1]:.2f}, {dip.ori[0,2]:.2f})"
        full_text = f"{rv_text}\n{pos_text}\n{ori_text}"
        
        ax_dipole.text(0.05, 0.9, full_text, ha='left', va='top', fontsize=7.5, wrap=True, 
                       bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.7),
                       transform=ax_dipole.transAxes)

    except Exception as e:
        print(f"Error in dipole processing for IC{component_idx}: {e}")
        if not ax_dipole.texts: # Only add text if not already populated by specific error
            ax_dipole.text(0.5, 0.5, "Dipole info failed", ha='center', va='center', fontsize=8, transform=ax_dipole.transAxes)
        # traceback.print_exc()

    # 5. IC Activity Power Spectrum
    try:
        fmin_psd_wide = 1.0
        fmax_psd_wide = min(120.0, sfreq / 2.0 - 0.51) # Ensure fmax < sfreq/2
        
        # Ensure n_fft is not too large for data, and not zero
        n_fft_psd = int(sfreq * 2.0)
        if n_fft_psd > len(component_data_array):
             n_fft_psd = len(component_data_array)
        n_fft_psd = max(n_fft_psd, 256 if len(component_data_array) >= 256 else len(component_data_array)) # Min sensible n_fft
        
        if n_fft_psd == 0 : raise ValueError("n_fft_psd is zero.")


        psds_wide, freqs_wide = psd_array_welch(
            component_data_array, sfreq=sfreq, fmin=fmin_psd_wide, fmax=fmax_psd_wide, 
            n_fft=n_fft_psd, n_overlap=int(n_fft_psd*0.5), verbose=False, average='mean'
        )
        if psds_wide.size == 0: raise ValueError("PSD computation returned empty array.")

        psds_db_wide = 10 * np.log10(np.maximum(psds_wide, 1e-20)) # Add small constant to avoid log(0)
        
        ax_psd.plot(freqs_wide, psds_db_wide, color='red', linewidth=1.2)
        ax_psd.set_title(f"IC{component_idx} Activity Power Spectrum", fontsize=10)
        ax_psd.set_xlabel("Frequency (Hz)", fontsize=9)
        ax_psd.set_ylabel("Power (dB)", fontsize=9)
        if len(freqs_wide) > 0:
            ax_psd.set_xlim(freqs_wide[0], freqs_wide[-1])
        ax_psd.grid(True, linestyle='--', alpha=0.5)
        ax_psd.tick_params(axis='both', which='major', labelsize=8)
    except Exception as e:
        print(f"Error plotting PSD for IC{component_idx}: {e}")
        ax_psd.text(0.5, 0.5, "PSD plot failed", ha='center', va='center', transform=ax_psd.transAxes)
        # traceback.print_exc()

    fig.suptitle(f"ICA Component {component_idx} Analysis", fontsize=14, y=0.98) # Adjusted y for suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusted rect for suptitle and bottom text space

    filename = f"component_{component_idx}_analysis.webp"
    filepath = output_dir / filename
    try:
        plt.savefig(filepath, format='webp', bbox_inches='tight', pad_inches=0.2) # Increased pad_inches slightly
        print(f"Successfully saved plot for IC{component_idx} to {filepath}")
    except Exception as e:
        print(f"Error saving figure for IC{component_idx}: {e}")
        # traceback.print_exc() # Optionally uncomment
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