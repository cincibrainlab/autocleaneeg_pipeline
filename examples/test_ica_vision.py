import os
import base64
import tempfile
import traceback
import re
import json # Added for JSON output
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any # Added List, Dict, Any
import math
from datetime import datetime # Added for PDF timestamp
import pandas as pd # Added for DataFrame handling of results

import matplotlib
matplotlib.use("Agg") # Ensure non-interactive backend for scripts
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Added for PDF output
import numpy as np
from matplotlib.gridspec import GridSpec
import mne
from mne.preprocessing import read_ica # Correct import for reading ICA
import openai
import warnings # Added for dipole fitting warnings
from mne import EvokedArray, make_sphere_model, create_info, Covariance # Added for dipole fitting, added Covariance
from mne.time_frequency import psd_array_welch # Ensure this is imported if not already
from scipy.ndimage import uniform_filter1d # Added for vertical smoothing

# --- Configuration ---
# Ensure your OpenAI API key is set as an environment variable
# or set it directly here (less secure for shared scripts)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MODIFY THESE PATHS TO YOUR EXAMPLE DATA
# It's best to use a small, representative raw file and its corresponding ICA solution
EXAMPLE_RAW_PATH = "C:/Users/Gam9LG/Documents/AutocleanDev2/TestingRest/bids/derivatives/sub-732282/eeg/0161_VD_noaudio_ICA_pre_ica.set"
EXAMPLE_ICA_PATH = "C:/Users/Gam9LG/Documents/AutocleanDev2/TestingRest/bids/derivatives/sub-732282/eeg/0161_VD_noaudio_ICA-ica.fif" 

# --- New Batch Configuration ---   
START_COMPONENT_INDEX = 0  # Starting component index for the batch
NUM_COMPONENTS_TO_BATCH =5 # Number of components to process in this batch
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


# --- Helper Functions for PDF Report Generation ---

def _create_summary_table_page(
    pdf_pages_obj: PdfPages,
    classification_results: pd.DataFrame,
    component_indices_to_include: List[int],
    bids_basename_for_title: str
):
    """
    Creates summary table pages for the PDF report.
    """
    components_per_page = 20
    num_total_components_in_list = len(component_indices_to_include)
    if num_total_components_in_list == 0:
        return # No components to summarize

    num_pages_for_summary = int(np.ceil(num_total_components_in_list / components_per_page))
    if num_pages_for_summary == 0 and num_total_components_in_list > 0: # handle case for <20 components
        num_pages_for_summary = 1


    color_map_vision = {
        "brain": "#d4edda",
        "eye": "#f9e79f",
        "muscle": "#f5b7b1",
        "heart": "#d7bde2",
        "line_noise": "#add8e6",
        "channel_noise": "#ffd700",
        "other_artifact": "#f0f0f0",
    }

    for page_num in range(num_pages_for_summary):
        start_idx_overall = page_num * components_per_page
        end_idx_overall = min((page_num + 1) * components_per_page, num_total_components_in_list)
        
        page_component_actual_indices = component_indices_to_include[start_idx_overall:end_idx_overall]

        if not page_component_actual_indices:
            continue

        fig_table = plt.figure(figsize=(11, 8.5))
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('off')

        table_data_page = []
        table_cell_colors_page = []

        for comp_idx in page_component_actual_indices:
            if comp_idx not in classification_results.index:
                print(f"Warning: Component index {comp_idx} not found in classification_results. Skipping in summary table.")
                continue

            comp_info = classification_results.loc[comp_idx]
            component_label = comp_info.get('label', 'N/A')
            component_confidence = comp_info.get('confidence', 0.0)
            
            is_rejected_text = "Yes" if component_label != 'brain' else "No"
            
            table_data_page.append([
                f"IC{comp_idx}",
                str(component_label).title(),
                f"{component_confidence:.2f}",
                is_rejected_text
            ])
            
            row_color = color_map_vision.get(component_label, "white")
            table_cell_colors_page.append([row_color] * 4)

        if not table_data_page:
             plt.close(fig_table)
             continue

        table = ax_table.table(
            cellText=table_data_page,
            colLabels=["Component", "Vision Label", "Confidence", "Is Artifact?"],
            loc='center',
            cellLoc='center',
            cellColours=table_cell_colors_page,
            colWidths=[0.2, 0.3, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.3)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig_table.suptitle(
            f"ICA Vision Classification Summary - {bids_basename_for_title}\n"
            f"(Page {page_num + 1} of {num_pages_for_summary})\n"
            f"Generated: {timestamp}",
            fontsize=12, y=0.95
        )
        
        legend_patches = [plt.Rectangle((0,0), 1, 1, facecolor=color, label=label.title()) 
                          for label, color in color_map_vision.items()]
        if legend_patches:
             ax_table.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.05, 0.85), title="Label Types", fontsize=8)

        plt.subplots_adjust(top=0.85, bottom=0.1, right=0.85) # Adjust right for legend
        pdf_pages_obj.savefig(fig_table)
        plt.close(fig_table)


def _generate_ica_report_pdf(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    classification_results: pd.DataFrame,
    output_dir: Path,
    bids_basename: str,
    duration: int = 10,
    components_to_plot: str = "all"
):
    """
    Generates a PDF report for ICA components, similar to IcaMixin._plot_ica_components.
    """
    report_name_suffix = "ica_vision_report_all"
    if components_to_plot == "classified_as_artifact":
        report_name_suffix = "ica_vision_report_artifacts"

    pdf_basename_cleaned = bids_basename.replace("_eeg.set", "").replace(".set","").replace("_eeg.fif","").replace(".fif","")
    if not pdf_basename_cleaned: 
        pdf_basename_cleaned = "untitled_eeg_data"
    pdf_filename = f"{pdf_basename_cleaned}_{report_name_suffix}.pdf"
    pdf_path = output_dir / pdf_filename

    if pdf_path.exists():
        try:
            pdf_path.unlink()
        except OSError as e:
            print(f"Warning: Could not delete existing PDF {pdf_path}: {e}")


    plot_indices = []
    if components_to_plot == "all":
        # Use component_index from the DataFrame's index
        plot_indices = list(classification_results.index)
    elif components_to_plot == "classified_as_artifact":
        if 'label' in classification_results.columns:
            plot_indices = list(classification_results[classification_results['label'] != 'brain'].index)
        else:
            print("Warning: 'label' column not in classification_results. Cannot determine artifact components.")
            plot_indices = [] # Ensure it's empty
            
        if not plot_indices:
            print(f"No components classified as artifacts. Skipping '{report_name_suffix}' PDF report.")
            return None

    if not plot_indices:
        print(f"No components to plot for '{report_name_suffix}'. Skipping PDF report.")
        return None

    with PdfPages(pdf_path) as pdf:
        # 1. Summary Table Page(s)
        _create_summary_table_page(pdf, classification_results, plot_indices, bids_basename)

        # 2. Component Topographies Overview Page
        try:
            valid_plot_indices_for_topo = [idx for idx in plot_indices if idx < ica_obj.n_components_]
            if valid_plot_indices_for_topo:
                # Plot components in batches if too many, to avoid huge figures
                max_topo_per_fig = 20 
                for i in range(0, len(valid_plot_indices_for_topo), max_topo_per_fig):
                    batch_indices = valid_plot_indices_for_topo[i:i+max_topo_per_fig]
                    if batch_indices: # Ensure batch is not empty
                        fig_topo_list = ica_obj.plot_components(picks=batch_indices, show=False)
                        if isinstance(fig_topo_list, list):
                            for fig_t in fig_topo_list:
                                pdf.savefig(fig_t)
                                plt.close(fig_t)
                        else:
                            pdf.savefig(fig_topo_list)
                            plt.close(fig_topo_list)
            else:
                print("No valid components to plot for topographies overview.")
        except Exception as e_topo:
            print(f"Error plotting component topographies overview: {e_topo}")
            fig_err_topo = plt.figure(figsize=(11, 8.5))
            ax_err_topo = fig_err_topo.add_subplot(111)
            ax_err_topo.text(0.5, 0.5, "Component topographies plot failed.", ha='center', va='center')
            pdf.savefig(fig_err_topo)
            plt.close(fig_err_topo)

        # 3. Individual Component Detail Pages
        ica_sources = ica_obj.get_sources(raw_obj)
        sfreq = raw_obj.info["sfreq"]
        # duration parameter is available from the function arguments for the time series plot

        for comp_idx in plot_indices:
            if comp_idx >= ica_obj.n_components_:
                print(f"Skipping IC{comp_idx} detail page: index out of bounds ({ica_obj.n_components_} components).")
                continue
            if comp_idx not in classification_results.index:
                print(f"Skipping IC{comp_idx} detail page: not found in classification_results.")
                continue
            
            comp_info_series = classification_results.loc[comp_idx]
            vision_label_pdf = comp_info_series.get('label')
            vision_confidence_pdf = comp_info_series.get('confidence')
            vision_reason_pdf = comp_info_series.get('reason')

            # Generate the plot using the modified _plot_component_for_vision_standalone
            fig_detail_page = _plot_component_for_vision_standalone(
                ica_obj=ica_obj,
                raw_obj=raw_obj,
                component_idx=comp_idx,
                output_dir=None, # Not saving a separate file here
                classification_label=vision_label_pdf,
                classification_confidence=vision_confidence_pdf,
                classification_reason=vision_reason_pdf,
                return_fig_object=True
            )

            if fig_detail_page:
                try:
                    pdf.savefig(fig_detail_page) # Save the returned figure object to the PDF
                except Exception as e_save:
                    print(f"Error saving detail page for IC{comp_idx} to PDF: {e_save}")
                    # Fallback: create an error page in PDF
                    fig_err_save = plt.figure()
                    ax_err_save = fig_err_save.add_subplot(111)
                    ax_err_save.text(0.5, 0.5, f"Plot for IC{comp_idx}\nsave to PDF failed.", ha='center', va='center')
                    pdf.savefig(fig_err_save)
                    plt.close(fig_err_save)
                finally:
                    plt.close(fig_detail_page) # Always close the figure after attempting to save
            else:
                # If _plot_component_for_vision_standalone returned None (e.g. error during its own plotting)
                print(f"Failed to generate plot object for IC{comp_idx} for PDF.")
                fig_err_gen = plt.figure()
                ax_err_gen = fig_err_gen.add_subplot(111)
                ax_err_gen.text(0.5, 0.5, f"Plot generation for IC{comp_idx}\nfailed internally.", ha='center', va='center')
                pdf.savefig(fig_err_gen)
                plt.close(fig_err_gen)

    print(f"ICA Vision PDF report saved to {pdf_path}")
    return pdf_path

# --- Adapted Helper Functions (from your IcaMixin) ---

def _plot_component_for_vision_standalone(
    ica_obj: mne.preprocessing.ICA,
    raw_obj: mne.io.Raw,
    component_idx: int,
    output_dir: Optional[Path] = None, # Now optional, but required if not returning fig
    classification_label: Optional[str] = None,
    classification_confidence: Optional[float] = None,
    classification_reason: Optional[str] = None,
    return_fig_object: bool = False
) -> Any: # Returns Path or Figure or None
    """
    Updated version to create a plot for an ICA component, matching the example image,
    and optionally include classification details and return the figure object.
    Layout: Topo+ContData on left, TS+Dipole+PSD on right.
    Dipole shown as text info (RV, Pos, Ori) using a sphere model.
    """
    fig_height = 9.5
    gridspec_bottom = 0.05 # Default bottom for GridSpec

    if return_fig_object and classification_reason:
        fig_height = 11  # Increase overall figure height for reasoning text
        gridspec_bottom = 0.18 # Ensure GridSpec leaves space at the bottom for reasoning

    fig = plt.figure(figsize=(12, fig_height), dpi=120)

    main_plot_title_text = f"ICA Component {component_idx} Analysis"
    gridspec_top = 0.95  # Default top for GridSpec
    suptitle_y_pos = 0.98 # Default Y for main suptitle

    if return_fig_object and classification_label is not None:
        gridspec_top = 0.90  # Lower GridSpec top to make space for classification subtitle
        suptitle_y_pos = 0.96 # Main title slightly lower to accommodate subtitle

    gs = GridSpec(3, 2, figure=fig, 
                  height_ratios=[0.915, 0.572, 2.213],
                  width_ratios=[0.9, 1],
                  hspace=0.7, wspace=0.35, # Adjusted hspace
                  left=0.05, right=0.95, top=gridspec_top, bottom=gridspec_bottom)

    ax_topo = fig.add_subplot(gs[0:2, 0])
    ax_cont_data = fig.add_subplot(gs[2, 0])
    ax_ts_scroll = fig.add_subplot(gs[0, 1])
    ax_dipole = fig.add_subplot(gs[1, 1])
    ax_psd = fig.add_subplot(gs[2, 1])

    sources = ica_obj.get_sources(raw_obj)
    sfreq = sources.info['sfreq']
    component_data_array = sources.get_data(picks=[component_idx])[0]

    # 1. Topography
    try:
        ica_obj.plot_components(picks=component_idx, axes=ax_topo, ch_type='eeg',
                                show=False, colorbar=False, cmap='jet', outlines='head', 
                                sensors=True, contours=6)
        ax_topo.set_title(f"IC{component_idx}", fontsize=12, loc='center')
        ax_topo.set_xlabel("")
        ax_topo.set_ylabel("")
        ax_topo.set_xticks([])
        ax_topo.set_yticks([])
    except Exception as e:
        print(f"Error plotting topography for IC{component_idx}: {e}")
        ax_topo.text(0.5, 0.5, "Topo plot failed", ha='center', va='center', transform=ax_topo.transAxes)

    # 2. Scrolling IC Activity (Time Series)
    try:
        duration_segment_ts = 5.0 
        max_samples_ts = min(int(duration_segment_ts * sfreq), len(component_data_array))
        times_ts_ms = (np.arange(max_samples_ts) / sfreq) * 1000 
        
        ax_ts_scroll.plot(times_ts_ms, component_data_array[:max_samples_ts], linewidth=0.8, color='dodgerblue')
        ax_ts_scroll.set_title("Scrolling IC Activity", fontsize=10)
        ax_ts_scroll.set_xlabel("Time (ms)", fontsize=9)
        ax_ts_scroll.set_ylabel("Amplitude (a.u.)", fontsize=9)
        if max_samples_ts > 0 and times_ts_ms.size > 0:
            ax_ts_scroll.set_xlim(times_ts_ms[0], times_ts_ms[-1])
        ax_ts_scroll.grid(True, linestyle=':', alpha=0.6)
        ax_ts_scroll.tick_params(axis='both', which='major', labelsize=8)
    except Exception as e:
        print(f"Error plotting scrolling IC activity for IC{component_idx}: {e}")
        ax_ts_scroll.text(0.5, 0.5, "Time series failed", ha='center', va='center', transform=ax_ts_scroll.transAxes)

    # 3. Continuous Data (EEGLAB-style ERP image)
    try:
        comp_data_offset_corrected = component_data_array - np.mean(component_data_array)
        target_segment_duration_s = 1.5
        target_max_segments = 200
        segment_len_samples_cd = int(target_segment_duration_s * sfreq)
        if segment_len_samples_cd == 0: segment_len_samples_cd = 1 

        available_samples_in_component = comp_data_offset_corrected.shape[0]
        max_total_samples_to_use_for_plot = int(target_max_segments * target_segment_duration_s * sfreq)
        samples_to_feed_erpimage = min(available_samples_in_component, max_total_samples_to_use_for_plot)
        
        n_segments_cd = 0
        erp_image_data_for_plot = np.array([[]])
        current_segment_len_samples = 1 # Default to avoid later div by zero if no data

        if segment_len_samples_cd > 0 and samples_to_feed_erpimage > 0 :
            n_segments_cd = math.floor(samples_to_feed_erpimage / segment_len_samples_cd)

        if n_segments_cd == 0:
            if samples_to_feed_erpimage > 0:
                n_segments_cd = 1
                current_segment_len_samples = samples_to_feed_erpimage
                erp_image_data_for_plot = comp_data_offset_corrected[:current_segment_len_samples].reshape(n_segments_cd, current_segment_len_samples)
            else:
                erp_image_data_for_plot = np.zeros((1,1)) # Placeholder
                current_segment_len_samples = 1 # Avoid division by zero for xticks later
        else:
            current_segment_len_samples = segment_len_samples_cd
            final_samples_for_reshape = n_segments_cd * current_segment_len_samples
            erp_image_data_for_plot = comp_data_offset_corrected[:final_samples_for_reshape].reshape(n_segments_cd, current_segment_len_samples)

        if n_segments_cd >= 3:
            erp_image_data_smoothed = uniform_filter1d(erp_image_data_for_plot, size=3, axis=0, mode='nearest')
        else:
            erp_image_data_smoothed = erp_image_data_for_plot

        if erp_image_data_smoothed.size > 0:
            max_abs_val = np.max(np.abs(erp_image_data_smoothed))
            clim_val = (2/3) * max_abs_val if max_abs_val > 1e-9 else 1.0 # Handle near-zero data
        else:
            clim_val = 1.0 
        clim_val = max(clim_val, 1e-9)
        vmin_cd, vmax_cd = -clim_val, clim_val

        im = ax_cont_data.imshow(erp_image_data_smoothed, aspect='auto', cmap='jet', interpolation='nearest',
                                 vmin=vmin_cd, vmax=vmax_cd)
        
        ax_cont_data.set_xlabel("Time (ms)", fontsize=9)
        if current_segment_len_samples > 1: # Ensure sensible ticks
            num_xticks = 4
            xtick_positions_samples = np.linspace(0, current_segment_len_samples -1 , num_xticks)
            xtick_labels_ms = (xtick_positions_samples / sfreq * 1000).astype(int)
            ax_cont_data.set_xticks(xtick_positions_samples)
            ax_cont_data.set_xticklabels(xtick_labels_ms)
        else:
            ax_cont_data.set_xticks([]) # No ticks if only one sample or empty

        ax_cont_data.set_ylabel("Trials (Segments)", fontsize=9)
        if n_segments_cd > 1:
            num_yticks = min(5, n_segments_cd)
            ytick_positions = np.linspace(0, n_segments_cd -1 , num_yticks).astype(int)
            ax_cont_data.set_yticks(ytick_positions)
            ax_cont_data.set_yticklabels(ytick_positions) 
        elif n_segments_cd == 1:
            ax_cont_data.set_yticks([0])
            ax_cont_data.set_yticklabels(["0"])
        else: # No segments
            ax_cont_data.set_yticks([])

        if n_segments_cd > 0: ax_cont_data.invert_yaxis()

        cbar_cont = fig.colorbar(im, ax=ax_cont_data, orientation='vertical', fraction=0.046, pad=0.1)
        cbar_cont.set_label("Activation (a.u.)", fontsize=8)
        cbar_cont.ax.tick_params(labelsize=7)
    except Exception as e_cont:
        print(f"Error plotting continuous data for IC{component_idx}: {e_cont}")
        ax_cont_data.text(0.5, 0.5, "Continuous data failed", ha='center', va='center', transform=ax_cont_data.transAxes)

    # 4. Dipole Position (Text-based info using sphere model)
    ax_dipole.set_title("Dipole Position", fontsize=10)
    ax_dipole.set_axis_off()
    try:
        if raw_obj.get_montage() is None:
            ax_dipole.text(0.5, 0.5, "No montage.\nCannot fit dipole.", ha='center', va='center', fontsize=8, transform=ax_dipole.transAxes)
        else:
            sphere_model = make_sphere_model(info=raw_obj.info, head_radius=0.090, verbose=False)
            component_pattern_dipole = ica_obj.get_components()[:, component_idx].reshape(-1, 1)

            # Use ica_obj.ch_names as these are the channels ICA was fit on
            ica_ch_names_for_evoked = list(ica_obj.ch_names)
            if component_pattern_dipole.shape[0] != len(ica_ch_names_for_evoked):
                 raise ValueError(f"IC{component_idx}: Dipole fitting channel mismatch. Pattern has {component_pattern_dipole.shape[0]} ch, ICA obj has {len(ica_ch_names_for_evoked)} ch.")

            ev_info_dipole = create_info(ch_names=ica_ch_names_for_evoked, sfreq=raw_obj.info['sfreq'], ch_types='eeg', verbose=False)
            
            # Get the montage that corresponds to the channels ICA was fit on.
            # This is best done by picking channels on a raw copy BEFORE getting montage if needed,
            # but here we assume raw_obj.get_montage() is compatible enough or set_montage will handle selection.
            montage_for_dipole = raw_obj.get_montage()
            if montage_for_dipole:
                 with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning) # For warnings about non-EEG channels in montage etc.
                    ev_info_dipole.set_montage(montage_for_dipole)
            
            evoked_topo_dipole = EvokedArray(component_pattern_dipole, ev_info_dipole, tmin=0, verbose=False)
            evoked_topo_dipole.set_eeg_reference('average', projection=False, verbose=False)
            
            n_channels_dipole = evoked_topo_dipole.info['nchan']
            if n_channels_dipole == 0: raise ValueError("No channels found in evoked data for dipole fitting.")

            noise_cov_data_dipole = np.eye(n_channels_dipole)
            noise_cov_dipole = Covariance(noise_cov_data_dipole, evoked_topo_dipole.info['ch_names'],
                                             bads=evoked_topo_dipole.info['bads'], projs=evoked_topo_dipole.info['projs'],
                                             nfree=1, verbose=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                dipoles_fit, _ = mne.fit_dipole(evoked_topo_dipole, noise_cov_dipole, sphere_model, min_dist=5.0, verbose=False)
            
            if dipoles_fit and len(dipoles_fit) > 0:
                dip = dipoles_fit[0]
                rv_text = f"RV: {dip.gof[0]:.1f}%"
                pos_text = f"Pos (mm): ({dip.pos[0,0]*1000:.1f}, {dip.pos[0,1]*1000:.1f}, {dip.pos[0,2]*1000:.1f})"
                ori_text = f"Ori: ({dip.ori[0,0]:.2f}, {dip.ori[0,1]:.2f}, {dip.ori[0,2]:.2f})"
                full_text = f"{rv_text}\n{pos_text}\n{ori_text}"
                ax_dipole.text(0.05, 0.9, full_text, ha='left', va='top', fontsize=7.5, wrap=True,
                               bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.7),
                               transform=ax_dipole.transAxes)
            else:
                ax_dipole.text(0.5, 0.5, "Dipole fitting returned no dipoles.", ha='center', va='center', fontsize=8, transform=ax_dipole.transAxes)
    except Exception as e:
        print(f"Error in dipole processing for IC{component_idx}: {e}")
        if not ax_dipole.texts: 
            ax_dipole.text(0.5, 0.5, "Dipole info failed", ha='center', va='center', fontsize=8, transform=ax_dipole.transAxes)

    # 5. IC Activity Power Spectrum
    try:
        component_trace_psd = component_data_array
        fmin_psd_wide = 1.0
        fmax_psd_wide = min(80.0, sfreq / 2.0 - 0.51)
        n_fft_psd = int(sfreq * 2.0)
        if n_fft_psd > len(component_trace_psd):
             n_fft_psd = len(component_trace_psd)
        n_fft_psd = max(n_fft_psd, 256 if len(component_trace_psd) >= 256 else len(component_trace_psd))
        
        if n_fft_psd == 0 : raise ValueError("n_fft_psd is zero.")

        psds_wide, freqs_wide = psd_array_welch(
            component_trace_psd, sfreq=sfreq, fmin=fmin_psd_wide, fmax=fmax_psd_wide, 
            n_fft=n_fft_psd, n_overlap=int(n_fft_psd*0.5), verbose=False, average='mean'
        )
        if psds_wide.size == 0: raise ValueError("PSD computation returned empty array.")

        psds_db_wide = 10 * np.log10(np.maximum(psds_wide, 1e-20))
        
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

    # Add main title for the component plots section
    fig.suptitle(main_plot_title_text, fontsize=14, y=suptitle_y_pos)

    # Add classification label and confidence as a subtitle (if for PDF)
    if return_fig_object and classification_label is not None and classification_confidence is not None:
        subtitle_color = 'red' if classification_label.lower() != 'brain' else 'green'
        classification_subtitle_text = f"Classification: {str(classification_label).title()} (Confidence: {classification_confidence:.2f})"
        # Position below the main suptitle
        fig.text(0.5, suptitle_y_pos - 0.035, classification_subtitle_text, ha='center', va='top',
                 fontsize=13, fontweight='bold', color=subtitle_color,
                 transform=fig.transFigure)

    # Add reasoning text box at the bottom (if for PDF and reason is available)
    if return_fig_object and classification_reason:
        reason_title = "Reasoning (Vision API):"
        fig.text(0.05, gridspec_bottom - 0.03, reason_title, ha='left', va='bottom', 
                 fontsize=9, fontweight='bold', transform=fig.transFigure)
        fig.text(0.05, 0.02, classification_reason, ha='left', va='top', 
                 fontsize=8, wrap=True, transform=fig.transFigure,
                 bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.75, ec='lightgrey'),
                 # Figure width for text box: from x=0.05 up to x=0.95 (0.9 of figure width)
                 # The text box will be constrained by these relative coordinates. Max width = 0.9 * fig_width
                 ) 

    if return_fig_object:
        # fig.tight_layout(rect=[0, (gridspec_bottom if classification_reason else 0.01), 1, (gridspec_top - 0.04 if classification_label else gridspec_top)]) # Adjust layout considering text
        # Using subplots_adjust might be more reliable with fig.text
        bottom_adj = gridspec_bottom if classification_reason else 0.01
        top_adj = gridspec_top - (0.05 if classification_label else 0.01) # reduce top slightly more if subtitle present
        try:
            fig.subplots_adjust(left=0.05, right=0.95, bottom=bottom_adj, top=top_adj, hspace=0.7, wspace=0.35)
        except ValueError: # Sometimes tight_layout/subplots_adjust can fail with complex layouts or fig.text
            print(f"Warning: Could not apply subplots_adjust for IC{component_idx} in PDF.")
        return fig # Return the figure object to be saved by PdfPages
    else:
        # This is for saving .webp for OpenAI API call
        if output_dir is None:
            plt.close(fig)
            raise ValueError("output_dir must be provided if not returning figure object and saving .webp")
        
        # No classification/reason text for .webp images sent to OpenAI
        # If they were added, they would be part of the image sent for classification.
        # The current logic correctly only adds them if return_fig_object is True.

        filename = f"component_{component_idx}_analysis.webp"
        filepath = output_dir / filename
        try:
            # Use a more constrained layout for the .webp to ensure it matches what was originally designed for OpenAI
            # fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Original tight_layout call for .webp
            # Since we are not adding text for .webp, the original gs and suptitle should be fine.
            # However, the gs might have changed if return_fig_object was accidentally true logic path was taken.
            # It's safer to ensure the .webp is generated with its intended layout directly.
            # For .webp, we don't want the extra space for classification/reasoning text.
            # The current code structure correctly separates this: if not return_fig_object, extra text is not added.
            plt.savefig(filepath, format='webp', bbox_inches='tight', pad_inches=0.2)
            print(f"Successfully saved plot for IC{component_idx} to {filepath}")
        except Exception as e:
            print(f"Error saving figure for IC{component_idx}: {e}")
            plt.close(fig)
            return None
        finally:
            plt.close(fig) # Ensure figure is closed after saving .webp
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
    # Ensure ica.n_components_ is valid before using it
    if ica.n_components_ is None or not isinstance(ica.n_components_, int) or ica.n_components_ < 0:
        print(f"Error: Invalid number of ICA components found: {ica.n_components_}. Cannot proceed.")
        return
        
    end_component_index = min(START_COMPONENT_INDEX + NUM_COMPONENTS_TO_BATCH, ica.n_components_)


    print(f"Processing components from index {START_COMPONENT_INDEX} to {end_component_index - 1}")

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

    # --- Generate PDF Reports ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        if "component_index" in results_df.columns:
            results_df = results_df.set_index("component_index")
            # Sort by index to ensure components are in order in the PDF
            results_df = results_df.sort_index()
        else:
            print("Warning: 'component_index' not found in results. PDF generation might be affected.")
            # Attempt to use default index if component_index is missing
            # This might not be ideal if order matters and isn't preserved.

        raw_filename_for_pdf = Path(EXAMPLE_RAW_PATH).name
        
        print("\n--- Generating PDF Report for ALL components (Vision API) ---")
        pdf_all_path = _generate_ica_report_pdf(
            ica_obj=ica,
            raw_obj=raw,
            classification_results=results_df,
            output_dir=output_path,
            bids_basename=raw_filename_for_pdf,
            components_to_plot="all"
        )
        if pdf_all_path:
            print(f"Full ICA Vision report saved to: {pdf_all_path}")

        print("\n--- Generating PDF Report for ARTIFACT components (Vision API) ---")
        pdf_artifact_path = _generate_ica_report_pdf(
            ica_obj=ica,
            raw_obj=raw,
            classification_results=results_df,
            output_dir=output_path,
            bids_basename=raw_filename_for_pdf,
            components_to_plot="classified_as_artifact"
        )
        if pdf_artifact_path:
            print(f"Artifact ICA Vision report saved to: {pdf_artifact_path}")
    else:
        print("No results to generate PDF report.")
            
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    run_test()