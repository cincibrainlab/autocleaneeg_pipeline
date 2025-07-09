# ./src/autoclean/step_functions/continuous.py
"""Continuous preprocessing steps."""
# pylint: disable=not-callable
# pylint: disable=isinstance-second-argument-not-valid-type
from datetime import datetime
from typing import Any, Dict, Tuple

import mne

from autoclean.utils.bids import step_convert_to_bids
from autoclean.utils.database import manage_database_conditionally
from autoclean.utils.logging import message


def step_create_bids_path(
    raw: mne.io.Raw, autoclean_dict: Dict[str, Any]
) -> Tuple[mne.io.Raw, Dict[str, Any]]:
    """Create BIDS-compliant paths."""
    message("header", "step_create_bids_path")
    unprocessed_file = autoclean_dict["unprocessed_file"]
    task = autoclean_dict["task"]
    bids_dir = autoclean_dict["bids_dir"]
    eeg_system = autoclean_dict["eeg_system"]
    config_file = "python_task_config"  # No config file for Python tasks

    # Handle both YAML and Python task configurations
    if task in autoclean_dict.get("tasks", {}):
        # YAML-based task
        mne_task = autoclean_dict["tasks"][task]["mne_task"]
        try:
            line_freq = autoclean_dict["tasks"][task]["settings"]["filtering"]["value"][
                "notch_freqs"
            ][0]
        except Exception as e:  # pylint: disable=broad-except
            message(
                "error",
                f"Failed to load line frequency: {str(e)}. Using default value of 60 Hz.",
            )
            line_freq = 60.0
    else:
        # Python-based task - use defaults
        mne_task = task.lower()  # Use task name as default
        line_freq = 60.0  # Default line frequency

    try:
        bids_path, derivatives_dir = step_convert_to_bids(
            raw,
            output_dir=str(bids_dir),
            task=mne_task,
            participant_id=None,
            line_freq=line_freq,
            overwrite=True,
            study_name=unprocessed_file.stem,
            autoclean_dict=autoclean_dict,
        )

        autoclean_dict["bids_path"] = bids_path
        autoclean_dict["bids_basename"] = bids_path.basename
        autoclean_dict["derivatives_dir"] = derivatives_dir
        metadata = {
            "step_convert_to_bids": {
                "creationDateTime": datetime.now().isoformat(),
                "bids_subject": bids_path.subject,
                "bids_task": bids_path.task,
                "bids_run": bids_path.run,
                "bids_session": bids_path.session,
                "bids_dir": str(bids_dir),
                "bids_datatype": bids_path.datatype,
                "bids_suffix": bids_path.suffix,
                "bids_extension": bids_path.extension,
                "bids_root": str(bids_path.root),
                "eegSystem": eeg_system,
                "configFile": str(config_file),
                "line_freq": line_freq,
                "derivatives_dir": str(derivatives_dir),
            }
        }

        manage_database_conditionally(
            operation="update",
            update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
        )

        manage_database_conditionally(
            operation="update_status",
            update_record={
                "run_id": autoclean_dict["run_id"],
                "status": "bids_path_created",
            },
        )

        return raw, autoclean_dict

    except Exception as e:
        message("error", f"Error converting raw to bids: {e}")
        raise e


# def plot_bad_channels_with_topography(  # TODO: Remove this function and add it to viz mixin
#     raw_original, raw_cleaned, autoclean_dict, zoom_duration=30, zoom_start=0
# ):
#     """
#     Plot bad channels with a topographical map and time series overlays
#     for both full duration and a zoomed-in window.

#     Parameters:
#     -----------
#     raw_original : mne.io.Raw
#         Original raw EEG data before cleaning.
#     raw_cleaned : mne.io.Raw
#         Cleaned raw EEG data after interpolation of bad channels.
#     autoclean_dict : dict
#         Autoclean dictionary containing metadata.
#     zoom_duration : float, optional
#         Duration in seconds for the zoomed-in time series plot. Default is 30 seconds.
#     zoom_start : float, optional
#         Start time in seconds for the zoomed-in window. Default is 0 seconds.
#     """

#     # ----------------------------
#     # 1. Collect Bad Channels
#     # ----------------------------
#     bad_channels_info = {}

#     # Mapping from channel to reason(s)
#     for reason, channels in pipeline.flags.get("ch", {}).items():
#         for ch in channels:
#             if ch in bad_channels_info:
#                 if reason not in bad_channels_info[ch]:
#                     bad_channels_info[ch].append(reason)
#             else:
#                 bad_channels_info[ch] = [reason]

#     bad_channels = list(bad_channels_info.keys())

#     if not bad_channels:
#         print("No bad channels were identified.")
#         return

#     # Debugging: Print bad channels
#     print(f"Identified Bad Channels: {bad_channels}")

#     # ----------------------------
#     # 2. Identify Good Channels
#     # ----------------------------
#     all_channels = raw_original.ch_names
#     good_channels = [ch for ch in all_channels if ch not in bad_channels]

#     # Debugging: Print good channels count
#     print(f"Number of Good Channels: {len(good_channels)}")

#     # ----------------------------
#     # 3. Extract Data for Bad Channels
#     # ----------------------------
#     picks_bad_original = mne.pick_channels(raw_original.ch_names, bad_channels)
#     picks_bad_cleaned = mne.pick_channels(raw_cleaned.ch_names, bad_channels)

#     if len(picks_bad_original) == 0:
#         print("No bad channels found in original data.")
#         return

#     if len(picks_bad_cleaned) == 0:
#         print("No bad channels found in cleaned data.")
#         return

#     data_original, times = raw_original.get_data(
#         picks=picks_bad_original, return_times=True
#     )
#     data_cleaned = raw_cleaned.get_data(picks=picks_bad_cleaned)

#     channel_labels = [raw_original.ch_names[i] for i in picks_bad_original]
#     n_channels = len(channel_labels)

#     # Debugging: Print number of bad channels being plotted
#     print(f"Number of Bad Channels to Plot: {n_channels}")

#     # ----------------------------
#     # 4. Downsample Data if Necessary
#     # ----------------------------
#     sfreq = raw_original.info["sfreq"]
#     desired_sfreq = 100  # Target sampling rate
#     downsample_factor = int(sfreq // desired_sfreq)
#     if downsample_factor > 1:
#         data_original = data_original[:, ::downsample_factor]
#         data_cleaned = data_cleaned[:, ::downsample_factor]
#         times = times[::downsample_factor]
#         print(
#             f"Data downsampled by a factor of {downsample_factor} to {desired_sfreq} Hz."
#         )

#     # ----------------------------
#     # 5. Normalize and Scale Data
#     # ----------------------------
#     data_original_normalized = np.zeros_like(data_original)
#     data_cleaned_normalized = np.zeros_like(data_cleaned)
#     # Dynamic spacing based on number of bad channels
#     spacing = 10 + (n_channels * 2)  # Adjusted spacing

#     for idx in range(n_channels):
#         channel_data_original = data_original[idx]
#         channel_data_cleaned = data_cleaned[idx]
#         # Remove DC offset
#         channel_data_original -= np.mean(channel_data_original)
#         channel_data_cleaned -= np.mean(channel_data_cleaned)
#         # Normalize by standard deviation
#         std_orig = np.std(channel_data_original)
#         std_clean = np.std(channel_data_cleaned)
#         if std_orig == 0:
#             std_orig = 1  # Prevent division by zero
#         if std_clean == 0:
#             std_clean = 1
#         data_original_normalized[idx] = channel_data_original / std_orig
#         data_cleaned_normalized[idx] = channel_data_cleaned / std_clean

#     # Scaling factor for better visibility
#     scaling_factor = 5  # Increased scaling factor
#     data_original_scaled = data_original_normalized * scaling_factor
#     data_cleaned_scaled = data_cleaned_normalized * scaling_factor

#     # Calculate offsets
#     offsets = np.arange(n_channels) * spacing

#     # ----------------------------
#     # 6. Define Zoom Window
#     # ----------------------------
#     zoom_end = zoom_start + zoom_duration
#     if zoom_end > times[-1]:
#         zoom_end = times[-1]
#         zoom_start = max(zoom_end - zoom_duration, times[0])

#     # ----------------------------
#     # 7. Create Figure with GridSpec
#     # ----------------------------
#     fig_height = 10 + (n_channels * 0.3)
#     fig = plt.figure(constrained_layout=True, figsize=(20, fig_height))
#     gs = GridSpec(3, 2, figure=fig)

#     # ----------------------------
#     # 8. Topography Subplot
#     # ----------------------------
#     ax_topo = fig.add_subplot(gs[0, :])

#     # Plot sensors with ch_groups for good and bad channels
#     ch_groups = [
#         [int(raw_original.ch_names.index(ch)) for ch in good_channels],
#         [int(raw_original.ch_names.index(ch)) for ch in bad_channels],
#     ]
#     colors = "RdYlBu_r"

#     # Plot again for the main figure subplot
#     mne.viz.plot_sensors(
#         raw_original.info,
#         kind="topomap",
#         ch_type="eeg",
#         title="Sensor Topography: Good vs Bad Channels",
#         show_names=True,
#         ch_groups=ch_groups,
#         pointsize=75,
#         linewidth=0,
#         cmap=colors,
#         show=False,
#         axes=ax_topo,
#     )

#     ax_topo.legend(["Good Channels", "Bad Channels"], loc="upper right", fontsize=12)
#     ax_topo.set_title("Topography of Good and Bad Channels", fontsize=16)

#     # ----------------------------
#     # 9. Full Duration Time Series Subplot
#     # ----------------------------
#     ax_full = fig.add_subplot(gs[1, 0])
#     for idx in range(n_channels):
#         # Plot original data
#         ax_full.plot(
#             times,
#             data_original_scaled[idx] + offsets[idx],
#             color="red",
#             linewidth=1,
#             linestyle="-",
#         )
#         # Plot cleaned data
#         ax_full.plot(
#             times,
#             data_cleaned_scaled[idx] + offsets[idx],
#             color="black",
#             linewidth=1,
#             linestyle="-",
#         )

#     ax_full.set_xlabel("Time (seconds)", fontsize=14)
#     ax_full.set_ylabel("Bad Channels", fontsize=14)
#     ax_full.set_title(
#         "Bad Channels: Original vs Interpolated (Full Duration)", fontsize=16
#     )
#     ax_full.set_xlim(times[0], times[-1])
#     ax_full.set_ylim(-spacing, offsets[-1] + spacing)
#     ax_full.set_yticks([])  # Hide y-ticks
#     ax_full.invert_yaxis()

#     # Add legend
#     legend_elements = [
#         Line2D([0], [0], color="red", lw=2, linestyle="-", label="Original Data"),
#         Line2D([0], [0], color="black", lw=2, linestyle="-", label="Interpolated Data"),
#     ]
#     ax_full.legend(handles=legend_elements, loc="upper right", fontsize=12)

#     # ----------------------------
#     # 10. Zoomed-In Time Series Subplot
#     # ----------------------------
#     ax_zoom = fig.add_subplot(gs[1, 1])
#     for idx in range(n_channels):
#         # Plot original data
#         ax_zoom.plot(
#             times,
#             data_original_scaled[idx] + offsets[idx],
#             color="red",
#             linewidth=1,
#             linestyle="-",
#         )
#         # Plot cleaned data
#         ax_zoom.plot(
#             times,
#             data_cleaned_scaled[idx] + offsets[idx],
#             color="black",
#             linewidth=1,
#             linestyle="-",
#         )

#     ax_zoom.set_xlabel("Time (seconds)", fontsize=14)
#     ax_zoom.set_title(
#         f"Bad Channels: Original vs Interpolated (Zoom: {zoom_start}-{zoom_end} s)",
#         fontsize=16,
#     )
#     ax_zoom.set_xlim(zoom_start, zoom_end)
#     ax_zoom.set_ylim(-spacing, offsets[-1] + spacing)
#     ax_zoom.set_yticks([])  # Hide y-ticks
#     ax_zoom.invert_yaxis()

#     # Add legend
#     ax_zoom.legend(handles=legend_elements, loc="upper right", fontsize=12)

#     # ----------------------------
#     # 11. Add Channel Labels
#     # ----------------------------
#     for idx, ch in enumerate(channel_labels):
#         label = f"{ch}\n({', '.join(bad_channels_info[ch])})"
#         ax_full.text(
#             times[0] - (0.05 * (times[-1] - times[0])),
#             offsets[idx],
#             label,
#             horizontalalignment="right",
#             fontsize=10,
#             verticalalignment="center",
#         )

#     # ----------------------------
#     # 12. Finalize and Save the Figure
#     # ----------------------------
#     plt.tight_layout()

#     # Get output path for bad channels figure
#     bids_path = autoclean_dict.get("bids_path", "")
#     if bids_path:
#         derivatives_path = pipeline.get_derivative_path(bids_path)
#     else:
#         derivatives_path = "."

#     # Assuming pipeline.get_derivative_path returns a Path-like object with a copy method
#     # and update method as per the initial code
#     try:
#         target_figure = str(
#             derivatives_path.copy().update(
#                 suffix="step_bad_channels_with_map", extension=".png", datatype="eeg"
#             )
#         )
#     except AttributeError:
#         # Fallback if copy or update is not implemented
#         target_figure = os.path.join(
#             derivatives_path, "bad_channels_with_topography.png"
#         )

#     # Save the figure
#     fig.savefig(target_figure, dpi=150, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Bad channels with topography plot saved to {target_figure}")

#     metadata = {
#         "artifact_reports": {
#             "creationDateTime": datetime.now().isoformat(),
#             "plot_bad_channels_with_topography": Path(target_figure).name,
#         }
#     }

#     manage_database(
#         operation="update",
#         update_record={"run_id": autoclean_dict["run_id"], "metadata": metadata},
#     )

#     return fig
