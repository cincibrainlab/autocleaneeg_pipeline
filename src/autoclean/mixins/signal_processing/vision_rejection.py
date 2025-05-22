import mne
import numpy as np
import matplotlib.pyplot as plt
import openai
import os
import tempfile
import base64
import re
from pathlib import Path
from typing import Dict, Union, List, Optional, Tuple
import traceback
import matplotlib
from matplotlib.gridspec import GridSpec
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

# Add pandas and numpy imports
import pandas as pd

from autoclean.utils.logging import message

# Attempt to load environment variables for API key
try:
    from dotenv import load_dotenv
    if load_dotenv():
        message("info", "Loaded environment variables from .env file.")
    else:
        message("info", "No .env file found or no variables loaded.")
except ImportError:
    message("warning", "python-dotenv not installed. Cannot load .env file. Ensure OPENAI_API_KEY is set manually or via environment.")

class VisionEpochRejectionMixin:
    """Mixin class for detecting bad EEG segments/epochs using vision-based classification via OpenAI.

    Provides methods to:
    1. Annotate bad segments in continuous mne.io.Raw data (annotate_bad_segments_vision).
    2. Identify bad epochs in mne.Epochs data (find_bad_epochs_vision).
    """

    # Store the prompt as a class variable
    _OPENAI_PROMPT = """
Classify the input image as "good" or "bad" based on the holistic visual characteristics of a time series plot representing a 2-second epoch of multichannel dense array EEG data. The image is a 512x512 pixel RGB WebP showing multiple EEG channels overlaid as time series traces on a white background with a grid and labeled axes. Note that individual channel cleaning has already been performed, so focus on overall epoch quality rather than isolated channel issues.

Good Image: Overall consistent pattern across the epoch; predominantly smooth, oscillatory EEG signals; minimal widespread artifacts; good signal-to-noise ratio across the majority of channels; coherent appearance without major disruptions.

Bad Image: Egregious artifacts affecting multiple channels simultaneously; widespread high-amplitude spikes (>±100 µV) across the epoch; large sections of flatlined channels; prominent muscle artifacts (high-frequency bursts); eye blinks or movement artifacts (large deflections); electrical interference patterns; abrupt baseline shifts affecting multiple channels.

Task: Analyze the overall quality of the entire epoch (not just individual channels) and output a binary label: "good" or "bad", along with a confidence score (0.0–1.0), and a brief reason.

Output Format: Return a tuple exactly like this: ('label', confidence, 'reason'), e.g., ('good', 0.95, 'consistent oscillatory patterns across the epoch with no major disruptions') or ('bad', 0.82, 'widespread high-amplitude artifact affecting multiple channels at 0.5-0.7s').

Additional Notes: Focus on major, egregious artifacts that affect the overall epoch quality rather than minor issues in individual channels. Look for patterns that would significantly contaminate downstream analysis. If the image is corrupted or unreadable, default to ('bad', 1.0, 'Image unreadable or corrupted'). Ensure the output format is strictly followed.
"""

    def annotate_bad_segments_vision(
        self,
        raw: Optional[mne.io.Raw] = None,
        api_key: Optional[str] = None,
        window_duration: float = 2.0,
        confidence_threshold: float = 0.8,
        annotation_label: str = "BAD_VISION",
        skip_existing_bad: bool = True,
        max_segments: Optional[int] = None,
        batch_size: int = 10, # Process images in batches before cleanup
    ) -> mne.Annotations:
        """
        Detects bad segments in continuous Raw data using vision classification
        and returns annotations for them. Operates non-destructively.

        Args:
            raw: The input MNE Raw object. Data should be loaded. Optional, if not provided, the raw data will be loaded from the instance.
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
            window_duration: Duration of the sliding window in seconds (default: 2.0). Matches prompt.
            confidence_threshold: Minimum confidence for a "bad" classification to be accepted (default: 0.8).
            annotation_label: Description prefix for the annotations marking bad segments (default: "BAD_VISION").
                              Confidence score will be appended (e.g., "BAD_VISION_0.85").
            skip_existing_bad: If True, segments overlapping with existing annotations whose description
                               starts with 'bad' (case-insensitive) are skipped (default: True).
            max_segments: Maximum number of segments to process (useful for debugging/cost control). Default: None (process all).
            batch_size: Number of images to generate before cleaning up the temporary directory batch. (default: 10).

        Returns:
            mne.Annotations object containing annotations for the detected bad segments.
            Returns empty Annotations if no bad segments are found or if processing fails critically.
        """

        raw = self._get_data_object(raw, use_epochs=False)

        if raw is None:
            raise ValueError("Raw data is not provided and cannot be loaded from the instance.")

        # Check if raw is an instance of mne.io.Raw or any of its subclasses
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("Input must be an mne.io.Raw object or a subclass of mne.io.BaseRaw.")
        if not raw.preload:
             message("warning", "Raw data is not preloaded. Loading data now for vision analysis.")
             try:
                  raw.load_data()
             except Exception as e:
                  message("error", f"Failed to load raw data: {e}")
                  return mne.Annotations([], [], [], orig_time=raw.info.get('meas_date'))


        message("header", f"Starting vision-based bad segment annotation (Window: {window_duration}s, Threshold: {confidence_threshold})...")

        onsets = []
        durations = []
        descriptions = []
        processed_count = 0
        classification_results = []  # Store results for PDF report

        # Create temporary epochs for analysis windows
        try:
            # Use reject_by_annotation=False to ensure all segments are created initially
            temp_epochs = mne.make_fixed_length_epochs(
                raw, duration=window_duration, overlap=0, preload=True, reject_by_annotation=False
            )
            if len(temp_epochs) == 0:
                message("warning", "No segments (epochs) could be created from the raw data.")
                return mne.Annotations([], [], [], orig_time=raw.info.get('meas_date'))

            num_to_process = len(temp_epochs) if max_segments is None else min(len(temp_epochs), max_segments)
            message("info", f"Created {len(temp_epochs)} temporary segments. Processing up to {num_to_process}.")

        except Exception as e:
             message("error", f"Failed to create temporary fixed-length epochs from Raw data: {e}")
             return mne.Annotations([], [], [], orig_time=raw.info.get('meas_date')) # Return empty annotations on failure


        # Use a single temporary directory for all images
        with tempfile.TemporaryDirectory(prefix="autoclean_vision_") as temp_dir:
            temp_path = Path(temp_dir)
            message("info", f"Using temporary directory for images: {temp_path}")

            for i in range(num_to_process):
                segment_start_time = (temp_epochs.events[i, 0] - raw.first_samp) / raw.info['sfreq'] # Start time in seconds from raw beginning

                # --- Check for overlap with existing 'bad' annotations ---
                if skip_existing_bad and raw.annotations is not None and len(raw.annotations) > 0:
                    overlaps = False
                    segment_end_time = segment_start_time + window_duration
                    for annot in raw.annotations:
                        # Check if description starts with 'bad' (case-insensitive)
                        if annot['description'].lower().startswith('bad'):
                             annot_start = annot['onset']
                             annot_end = annot['onset'] + annot['duration']
                             # Check for overlap: (StartA < EndB) and (EndA > StartB)
                             if (segment_start_time < annot_end) and (segment_end_time > annot_start):
                                 message("debug", f"Segment {i+1} ({segment_start_time:.2f}s) overlaps with existing annotation '{annot['description']}' ({annot_start:.2f}s). Skipping.")
                                 overlaps = True
                                 break # No need to check other annotations for this segment
                    if overlaps:
                        continue # Skip to the next segment

                # --- Plot the segment ---
                try:
                    # Get data for the single segment [epoch_index][channel, time]
                    segment_data = temp_epochs[i].get_data(copy=False)[0]
                    # Ensure we pass times relative to segment start (0 to window_duration)
                    segment_times = temp_epochs.times

                    image_path = self._plot_single_epoch_for_vision(
                        segment_data,
                        segment_times,
                        f"{i+1}", # Use 1-based identifier for messages/filenames
                        temp_path
                    )
                    if image_path is None: # Plotting skipped due to empty data
                         continue

                except Exception as plot_err:
                    message("warning", f"Failed to plot segment {i+1}: {plot_err}. Skipping classification for this segment.")
                    continue # Skip classification if plotting fails

                # --- Classify the image ---
                try:
                    label, confidence, reason = self._classify_epoch_image_openai(image_path, api_key)
                except Exception as classify_err: # Catch any unexpected error during classification call itself
                     message("warning", f"Classification failed for segment {i+1} ({image_path.name}): {classify_err}. Defaulting to 'bad'.")
                     label, confidence, reason = "bad", 1.0, f"Classification call failed: {classify_err}"

                # Store result for PDF report
                classification_results.append({
                    "index": i,
                    "onset": segment_start_time,
                    "label": label,
                    "confidence": confidence,
                    "reason": reason
                })

                # --- Consolidate Logging ---
                log_level = "debug" # Default level
                is_bad_above_threshold = False
                if label == "good":
                    log_level = "success"
                elif label == "bad":
                    if confidence >= confidence_threshold:
                        log_level = "warning" # Use WARNING for bad segments meeting threshold
                        is_bad_above_threshold = True
                    # else: Keep default "debug" for bad below threshold

                log_message = f"Vision Result | Segment: {i+1} | Label: {label.upper()} | Confidence: {confidence:.2f} | Reason: {reason}"
                message(log_level, log_message)

                # --- Add annotation if bad and above threshold ---
                if is_bad_above_threshold:
                    # Calculate absolute onset time, accounting for raw.first_samp
                    onsets.append(segment_start_time)
                    durations.append(window_duration)
                    # Include confidence in description for traceability
                    descriptions.append(f"{annotation_label}_{confidence:.2f}")

                processed_count += 1

        # Create the final Annotations object
        if onsets: # Only create if we found bad segments
            bad_annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=raw.annotations.orig_time if raw.annotations else raw.info.get('meas_date') # Preserve original time reference
            )
            message("info", f"Completed vision scan. Found {len(bad_annotations)} segments to annotate as '{annotation_label}'.")
        else:
             bad_annotations = mne.Annotations([], [], [], orig_time=raw.info.get('meas_date'))
             message("info", "Completed vision scan. No segments met criteria for annotation.")

        # Generate PDF report
        if hasattr(self, 'config') and self.config and 'derivatives_dir' in self.config and 'bids_path' in self.config:
            derivatives_dir = Path(self.config["derivatives_dir"])
            bids_basename = self.config["bids_path"].basename
            report_path = self._generate_vision_segments_report_pdf(
                raw_obj=raw,
                classification_results=classification_results,
                output_dir=derivatives_dir,
                bids_basename=bids_basename,
                window_duration=window_duration
            )
            if report_path:
                message("info", f"Vision segments PDF report saved to: {report_path}")
            else:
                message("warning", "Failed to generate Vision segments PDF report.")
        else:
            message("warning", "Configuration not found. Skipping PDF report generation.")

        annotated_raw = raw.copy()
        annotated_raw.set_annotations(bad_annotations)

        self._save_raw_result(annotated_raw, "post_clean_raw")

        self._update_instance_data(raw, annotated_raw, use_epochs=False)

        return bad_annotations


    def find_bad_epochs_vision(
         self,
         epochs: Optional[mne.Epochs] = None,
         api_key: Optional[str] = None,
         confidence_threshold: float = 0.8,
         metadata_label: str = "BAD_VISION", # New parameter
         add_confidence_to_metadata: bool = True, # New parameter
         max_epochs: Optional[int] = None,
         batch_size: int = 10, # Process images in batches before cleanup
     ) -> mne.Epochs: # Changed return type
        """
        Identifies bad epochs within an MNE Epochs object using vision classification
        and marks them in the Epochs metadata.

        Args:
             epochs: The input MNE Epochs object. Data should be loaded. Optional, attempts to load from instance if None.
             api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
             confidence_threshold: Minimum confidence for a "bad" classification to be accepted (default: 0.8).
             metadata_label: Name for the boolean metadata column marking bad epochs (default: "BAD_VISION").
             add_confidence_to_metadata: If True, add a column with the confidence score (default: True).
                                           Column name will be f"{metadata_label}_CONF".
             max_epochs: Maximum number of epochs to process (useful for debugging/cost control). Default: None (process all).
             batch_size: Number of images to generate before cleaning up the temporary directory batch. (default: 10).

        Returns:
            mne.Epochs: The epochs object with updated metadata marking bad epochs.
                      Returns the original epochs object if processing fails critically.
        """

        epochs = self._get_data_object(epochs, use_epochs=True)

        if not isinstance(epochs, mne.Epochs):
            raise TypeError("Input must be an mne.Epochs object.")
        if not epochs.preload:
             message("warning", "Epochs data is not preloaded. Loading data now for vision analysis.")
             try:
                 epochs.load_data()
             except Exception as e:
                  message("error", f"Failed to load epochs data: {e}")
                  return epochs # Return original epochs on failure

        message("header", f"Starting vision-based bad epoch identification (Threshold: {confidence_threshold})...")

        # Store results temporarily
        classification_results = []
        processed_count = 0
        num_to_process = len(epochs) if max_epochs is None else min(len(epochs), max_epochs)
        message("info", f"Processing up to {num_to_process} epochs...")

        # Create a temporary directory for images
        with tempfile.TemporaryDirectory(prefix="autoclean_vision_") as temp_dir:
            temp_path = Path(temp_dir)
            message("info", f"Using temporary directory for images: {temp_path}")

            for i in range(num_to_process):
                # --- Plot the epoch ---
                try:
                    epoch_data = epochs[i].get_data(copy=False)[0]
                    epoch_times = epochs.times
                    image_path = self._plot_single_epoch_for_vision(
                        epoch_data, epoch_times, f"{i}", temp_path
                    )
                    if image_path is None: continue
                except Exception as plot_err:
                    message("warning", f"Failed to plot epoch {i}: {plot_err}. Skipping.")
                    continue

                # --- Classify the image ---
                label, confidence, reason = "bad", 1.0, "Classification skipped"
                try:
                    label, confidence, reason = self._classify_epoch_image_openai(image_path, api_key)
                    # Store the result for later metadata update
                    classification_results.append({
                        "index": i,
                        "label": label,
                        "confidence": confidence,
                        "reason": reason
                    })
                except Exception as classify_err: # Catch unexpected error during classification call
                     message("warning", f"Classification failed for epoch {i} ({image_path.name}): {classify_err}. Defaulting to 'bad'.")
                     # Store a default bad result
                     classification_results.append({
                         "index": i,
                         "label": "bad",
                         "confidence": 1.0,
                         "reason": f"Classification call failed: {classify_err}"
                     })
                     # Ensure variables are set for logging
                     label, confidence, reason = "bad", 1.0, f"Classification call failed: {classify_err}"

                # --- Log Result Immediately ---
                log_level = "debug"
                is_bad_above_threshold = False # We'll re-check this in the metadata loop
                if label == "good":
                    log_level = "success"
                elif label == "bad":
                    # Check threshold *here* just for logging level, actual marking happens later
                    if confidence >= confidence_threshold:
                        log_level = "warning"
                    # else: Keep debug for bad below threshold
                
                log_message = f"Vision Result | Epoch: {i} | Label: {label.upper()} | Confidence: {confidence:.2f} | Reason: {reason}"
                message(log_level, log_message)

                processed_count += 1
                # Optional batch cleanup
                # if processed_count % batch_size == 0:
                #    message("debug", f"Cleaning up image batch at epoch {i}...")
                #    for item in temp_path.glob("*.webp"):
                #        try: os.remove(item)
                #        except OSError: pass

        message("info", f"Completed vision scan. Processed {processed_count} epochs. Updating metadata...")

        # --- Update Epochs Metadata ---
        if not classification_results:
            message("info", "No epochs were classified. Returning original epochs object.")
            return epochs

        # Ensure metadata exists
        if epochs.metadata is None:
             message("info", "Epochs metadata not found. Creating new metadata DataFrame.")
             # Create metadata with simple index matching epochs
             epochs.metadata = pd.DataFrame(index=range(len(epochs)))

        # Initialize new metadata columns
        epochs.metadata[metadata_label] = False
        confidence_col_name = f"{metadata_label}_CONF"
        if add_confidence_to_metadata:
            # Use appropriate dtype that handles NaN (float)
            epochs.metadata[confidence_col_name] = np.nan

        bad_epoch_count = 0
        # Populate metadata based on classification results
        for result in classification_results:
            idx = result["index"]
            label = result["label"]
            confidence = result["confidence"]
            reason = result["reason"]

            # Determine log level based on outcome
            log_level = "debug"
            is_bad_above_threshold = False
            if label == "good":
                log_level = "success"
            elif label == "bad":
                if confidence >= confidence_threshold:
                    log_level = "warning"
                    is_bad_above_threshold = True
                # else: Keep default "debug"

            # Log the result
            log_message = f"Vision Result | Epoch: {idx} | Label: {label.upper()} | Confidence: {confidence:.2f} | Reason: {reason}"
            message(log_level, log_message)

            # Update metadata if bad and above threshold
            if is_bad_above_threshold:
                # Check if index exists in metadata (it should, but safe check)
                if idx in epochs.metadata.index:
                    epochs.metadata.loc[idx, metadata_label] = True
                    if add_confidence_to_metadata:
                        epochs.metadata.loc[idx, confidence_col_name] = confidence
                    bad_epoch_count += 1
                else:
                     message("warning", f"Epoch index {idx} not found in metadata. Cannot mark as bad.")


        message("info", f"Marked {bad_epoch_count} epochs as '{metadata_label}' in metadata.")

        # Return the epochs object with updated metadata

        self._update_instance_data(epochs, epochs, use_epochs=True)

        return epochs

    def _plot_single_epoch_for_vision(self, epoch_data: np.ndarray, times: np.ndarray, segment_identifier: Union[int, str], output_dir: Path) -> Path:
        """
        Internal helper to create and save a standardized plot for a single epoch/segment.
        Adapted from the user's process_eeglab.py script for 512x512px output.

        Args:
            epoch_data: Data for a single epoch (n_channels, n_times). Should be in Volts.
            times: Time vector for the epoch (in seconds).
            segment_identifier: Identifier for the epoch/segment (e.g., sequence number or index).
            output_dir: Temporary directory to save the plot.

        Returns:
            Path to the saved webp image file.

        Raises:
            ValueError: If input data dimensions are incorrect.
            Exception: Propagates exceptions from Matplotlib.
        """
        if epoch_data.ndim != 2:
            raise ValueError(f"Expected 2D epoch_data (channels, times), got shape {epoch_data.shape}")

        n_channels, n_times = epoch_data.shape
        if n_channels == 0 or n_times == 0:
             message("warning", f"Skipping plot for segment {segment_identifier} due to empty data ({n_channels}x{n_times}).")
             return None # Indicate plotting was skipped

        message("debug", f"Plotting segment {segment_identifier} with {n_channels} channels, {n_times} timepoints.")

        # Convert data to microvolts for plotting
        data_uv = epoch_data * 1e6

        # Create a square figure (8x8 inches at 64 DPI = 512x512 pixels)
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='white', dpi=64)

        # Fixed offset between channels for visualization
        channel_offset = 30  # µV separation

        max_abs_val = 0 # Track max absolute value for potential scaling adjustment if needed

        # Plot each channel with offset
        for idx, channel_data in enumerate(data_uv):
            offset_data = channel_data - (idx * channel_offset)
            max_abs_val = max(max_abs_val, np.max(np.abs(channel_data))) # Track original data max amplitude

            # Use black for all channels, slightly thicker if potentially problematic
            linewidth = 0.7
            alpha = 0.8
            # Highlight channels with high amplitude (>100 µV) or near flatline (variance < 0.1 µV^2)
            if np.max(np.abs(channel_data)) > 100 or np.var(channel_data) < 0.1:
                 linewidth = 1.5
                 alpha = 1.0
            ax.plot(times, offset_data, color='black', linewidth=linewidth, alpha=alpha)

        # --- Styling ---
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3) # Make grid subtle

        # Set y-axis limits to accommodate all channels with offsets, plus some padding
        ax.set_ylim(-(n_channels * channel_offset) - 50, 50)

        # Set y-ticks to show channel numbers (show ~10 labels)
        channel_positions = [-(idx * channel_offset) for idx in range(n_channels)]
        channel_labels = [str(idx + 1) for idx in range(n_channels)] # 1-based channel numbers
        step = max(1, n_channels // 10) # Show around 10 labels
        ax.set_yticks(channel_positions[::step])
        ax.set_yticklabels(channel_labels[::step], fontsize=8)

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Channel', fontsize=10)
        # Use a generic title as specific epoch info isn't needed by the model per se
        ax.set_title(f'EEG Segment {segment_identifier}', fontsize=12)

        # Add vertical line at time zero if within plot limits
        if times[0] <= 0 <= times[-1]:
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        # Ensure x-axis limits match the provided times exactly
        ax.set_xlim(times[0], times[-1])

        # Remove figure frame/spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # --- Save plot ---
        # Use a consistent naming scheme
        filename = f"segment_{segment_identifier}.webp"
        filepath = output_dir / filename

        try:
            plt.savefig(
                filepath,
                dpi=64, # To achieve 512x512 with 8x8 fig
                bbox_inches='tight', # Minimize whitespace
                pad_inches=0.05, # Minimal padding
                format='webp' # Use webp as requested
            )
            message("debug", f"Saved plot to {filepath}")
        except Exception as e:
            message("error", f"Failed to save plot {filepath}: {e}")
            plt.close(fig) # Ensure figure is closed even on error
            raise # Re-raise the exception
        finally:
            plt.close(fig) # Ensure figure is closed after saving

        return filepath

    def _classify_epoch_image_openai(self, image_path: Path, api_key: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Internal helper to send a segment image to OpenAI API and parse the result.

        Args:
            image_path: Path to the temporary segment image file (WebP format).
            api_key: OpenAI API key. If None, attempts to use OPENAI_API_KEY env var.

        Returns:
            Tuple: (label: str, confidence: float, reason: str)
                   Defaults to ("bad", 1.0, "API error or parsing failure") on error.
        """
        if not image_path or not image_path.exists():
             message("error", "Invalid or non-existent image path provided for classification.")
             return "bad", 1.0, "Invalid image path"

        try:
            # Determine API key
            effective_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not effective_api_key:
                 # Check again directly just in case
                 if hasattr(openai, 'api_key') and openai.api_key:
                      effective_api_key = openai.api_key
                 else:
                    raise ValueError("OpenAI API key not provided via argument, environment variable (OPENAI_API_KEY), or openai.api_key.")

            # Read and encode image in base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Initialize OpenAI client (ensure API key is set for the client)
            # Setting openai.api_key globally might have side effects if autoclean is used as a library.
            # Prefer passing the key explicitly to the client instance if possible,
            # although the current openai library version might primarily rely on the global setting or env vars.
            client = openai.OpenAI(api_key=effective_api_key)

            message("debug", f"Sending image {image_path.name} to OpenAI Vision API...")
            # Use the chat completions endpoint with GPT-4 Vision model
            response = client.responses.create(
                model="gpt-4.1",  # or your chosen vision-capable model
                input=[{
                    "role": "user",
                    "content": [
                        {   
                            "type": "input_text",
                            "text": self._OPENAI_PROMPT
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/webp;base64,{base64_image}", # Assuming webp
                            "detail": "high"
                        }
                    ]
                }],
                temperature=0.1 # Low temperature for more deterministic classification output
            )

            message("debug", "OpenAI API call successful.")

            # --- Parse Response ---
            # Updated extraction logic based on new response structure
            # Check structure existence step-by-step for robustness
            resp_text = None
            if response.output and isinstance(response.output, list) and len(response.output) > 0:
                first_output = response.output[0]
                if hasattr(first_output, 'content') and isinstance(first_output.content, list) and len(first_output.content) > 0:
                    first_content = first_output.content[0]
                    if hasattr(first_content, 'text'):
                        resp_text = first_content.text.strip()

            if resp_text is not None:
                 # Changed level to DEBUG
                 message("debug", f"Raw OpenAI response: '{resp_text}'")
                 # Regex to find the tuple ('good'|'bad', float, 'reason')
                 # Handles optional quotes, whitespace, and captures reason robustly
                 match = re.search(
                    r"\(\s*['\"]?(good|bad)['\"]?\s*,\s*([01](?:\.\d+)?)\s*,\s*['\"](.*?)['\"]\s*\)",                    
                    resp_text, re.IGNORECASE | re.DOTALL # Ignore case, allow '.' to match newline
                 )
                 if match:
                    label = match.group(1).lower()
                    confidence = float(match.group(2))
                    reason = match.group(3).strip().replace("\\\\'", "'").replace('\\\\"', '"') # Clean up captured reason
                    # Changed level to DEBUG
                    message("debug", f"Parsed classification: Label={label}, Conf={confidence:.2f}, Reason='{reason[:50]}...'")
                    return label, confidence, reason
                 else:
                    message("warning", f"Could not parse OpenAI response format: '{resp_text}'. Defaulting to 'bad'.")
                    # Provide the raw text in the reason for debugging
                    return "bad", 1.0, f"Failed to parse response: {resp_text}"
            else:
                 # This case handles when resp_text could not be extracted
                 message("error", "Could not extract text content from OpenAI response structure.")
                 return "bad", 1.0, "Invalid response structure (no text)"

        except openai.APIConnectionError as e:
             message("error", f"OpenAI API connection error: {e}")
             return "bad", 1.0, f"API Connection Error: {e}"
        except openai.AuthenticationError as e:
             message("error", f"OpenAI API authentication error: {e}. Check your API key.")
             return "bad", 1.0, f"API Authentication Error: {e}"
        except openai.RateLimitError as e:
             message("error", f"OpenAI API rate limit exceeded: {e}")
             return "bad", 1.0, f"API Rate Limit Error: {e}"
        except openai.APIStatusError as e:
             message("error", f"OpenAI API status error: Status={e.status_code}, Response={e.response}")
             return "bad", 1.0, f"API Status Error {e.status_code}: {e}"
        except Exception as e:
            # Catch other potential exceptions (e.g., file reading, base64 encoding, regex)
            message("error", f"Unexpected exception during vision classification of {image_path.name}: {type(e).__name__} - {e}")
            # Log traceback for debugging if possible
            message("error", traceback.format_exc())
            return "bad", 1.0, f"Unexpected Exception: {type(e).__name__}"

    def _generate_vision_segments_report_pdf(
        self,
        raw_obj: mne.io.Raw,
        classification_results: List[Dict],
        output_dir: Path,
        bids_basename: str,
        window_duration: float
    ) -> Optional[Path]:
        """
        Generates a comprehensive PDF report for segments classified by Vision API.
        Includes summary tables and individual segment detail pages with plots and reasoning.

        Args:
            raw_obj: The MNE Raw object used for analysis.
            classification_results: List of dictionaries containing classification results.
            output_dir: Directory to save the PDF report.
            bids_basename: Base name for the output file.
            window_duration: Duration of each segment in seconds.

        Returns:
            Path to the generated PDF report, or None if generation failed.
        """
        matplotlib.use("Agg")  # Ensure non-interactive backend

        # Ensure output directory exists
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e_mkdir:
            message("error", f"Could not create output directory {output_dir}: {e_mkdir}")
            return None

        # Create PDF filename
        pdf_basename = bids_basename.replace("_eeg", "_vision_segments_report")
        pdf_filename = f"{pdf_basename}.pdf"
        pdf_path = output_dir / pdf_filename

        if pdf_path.exists():
            try:
                pdf_path.unlink()
            except OSError as e_unlink:
                message("warning", f"Could not delete existing PDF {pdf_path}: {e_unlink}")

        # Filter results to only include bad segments
        bad_segments = [r for r in classification_results if r["label"] == "bad" and r["confidence"] >= 0.8]

        if not bad_segments:
            message("info", "No bad segments found. Skipping PDF report generation.")
            return None

        message("info", f"Generating PDF report for {len(bad_segments)} bad segments...")

        try:
            with PdfPages(pdf_path) as pdf:
                # 1. Summary Table Page
                fig_table = plt.figure(figsize=(11, 8.5))
                ax_table = fig_table.add_subplot(111)
                ax_table.axis("off")

                # Prepare table data
                table_data = []
                for segment in bad_segments:
                    table_data.append([
                        f"{segment['index'] + 1}",  # Segment number (1-based)
                        f"{segment['onset']:.2f}s",  # Onset time
                        f"{window_duration:.2f}s",  # Duration
                        f"{segment['confidence']:.2f}",  # Confidence
                        segment['reason'][:70] + ('...' if len(segment['reason']) > 70 else '')  # Truncate reason
                    ])

                # Create table
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=["Segment", "Onset", "Duration", "Confidence", "Reason"],
                    loc='center',
                    cellLoc='left',
                    colWidths=[0.1, 0.15, 0.15, 0.1, 0.5]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.3)

                # Add title and timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fig_table.suptitle(
                    f"Vision-based Bad Segment Classification Summary - {bids_basename}\n"
                    f"Generated: {timestamp}",
                    fontsize=12,
                    y=0.95
                )

                plt.subplots_adjust(top=0.85, bottom=0.1)
                pdf.savefig(fig_table)
                plt.close(fig_table)

                # 2. Individual Segment Detail Pages
                for segment in bad_segments:
                    # Create figure for segment details
                    fig_detail = plt.figure()
                    fig_detail.set_size_inches(11, 8.5) # Explicitly set figure size
                    gs = GridSpec(2, 1, height_ratios=[3, 1], figure=fig_detail)

                    # Plot segment data
                    ax_data = fig_detail.add_subplot(gs[0])
                    start_sample = int(segment['onset'] * raw_obj.info['sfreq'])
                    end_sample = start_sample + int(window_duration * raw_obj.info['sfreq'])
                    # Ensure we don't try to plot beyond the raw data length
                    end_sample = min(end_sample, raw_obj.n_times)
                    data_segment = raw_obj[:, start_sample:end_sample][0]
                    times_segment = np.arange(data_segment.shape[1]) / raw_obj.info['sfreq']
                    
                    if data_segment.ndim == 1:
                        data_segment = data_segment[np.newaxis, :] # Ensure 2D for plotting multiple channels (even if 1)

                    for ch_data in data_segment:
                        ax_data.plot(times_segment, ch_data.T) # Plot each channel
                    
                    ax_data.set_title(f"Segment {segment['index'] + 1} - Bad Segment Visualization")
                    ax_data.set_xlabel("Time within Segment (s)")
                    ax_data.set_ylabel("Amplitude (µV)")
                    ax_data.set_xlim(0, window_duration) # Consistent x-axis limit
                    ax_data.grid(True, linestyle=':', alpha=0.7)

                    # Add classification details
                    ax_info = fig_detail.add_subplot(gs[1])
                    ax_info.axis("off")
                    info_text = (
                        f"Onset in File: {segment['onset']:.2f}s\n"
                        f"Duration: {window_duration:.2f}s\n"
                        f"Vision Classification: {segment['label'].upper()}\n"
                        f"Confidence: {segment['confidence']:.2f}\n"
                        f"Reason: {segment['reason']}"
                    )
                    # Add text with wrapping. Adjust x, y, width, height as needed.
                    # The width (0.9 here) controls how much horizontal space the text can occupy before wrapping.
                    ax_info.text(0.05, 0.95, info_text, va='top', ha='left', fontsize=10, wrap=True,
                                 bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.75))

                    fig_detail.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle if any, or just use tight_layout()
                    pdf.savefig(fig_detail)

            message("success", f"Vision segments PDF report saved to: {pdf_path}")
            return pdf_path

        except Exception as e:
            message("error", f"Failed to generate PDF report: {e}")
            return None
