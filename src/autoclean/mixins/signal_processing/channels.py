"""Channel operations mixin for autoclean tasks."""

from typing import Any, Dict, List, Optional, Union

import mne

from autoclean.functions.artifacts.channels import detect_bad_channels
from autoclean.utils.bad_channel_presets import (
    merge_channel_count_bins,
    resolve_bad_channel_settings,
)
from autoclean.utils.logging import message
from autoclean.utils.metadata_table import (
    load_metadata_table,
    match_recording_row,
    split_channels,
)

# Sentinel distinguishing "caller did not pass cleaning_method" (use the
# resolved preset/config value) from an explicit cleaning_method=None
# (skip interpolation/dropping and leave channels marked as bad).
_UNSET = object()


class ChannelsMixin:
    """Mixin class providing channel operations functionality for EEG data."""

    def clean_bad_channels(
        self,
        data: Union[mne.io.Raw, None] = None,
        preset: Optional[str] = None,
        correlation_thresh: Optional[float] = None,
        deviation_thresh: Optional[float] = None,
        ransac_sample_prop: Optional[float] = None,
        ransac_corr_thresh: Optional[float] = None,
        ransac_frac_bad: Optional[float] = None,
        ransac_channel_wise: Optional[bool] = None,
        ransac_enabled: Optional[bool] = None,
        max_bad_fraction: Optional[float] = None,
        channel_count_bins: Optional[Dict[str, Dict[str, Any]]] = None,
        random_state: int = 1337,
        cleaning_method: Union[str, None] = _UNSET,  # type: ignore[assignment]
        reset_bads: bool = True,
        stage_name: str = "post_bad_channels",
        manual_bad_channels: Union[List[str], None] = None,
    ) -> mne.io.Raw:
        """Detect and mark bad channels using various methods.

        This method uses the MNE NoisyChannels class to detect bad channels using SNR,
        correlation, deviation, and RANSAC methods.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            The data object to detect bad channels from. If None, uses self.raw.
        preset : str, Optional
            Montage-density preset controlling default thresholds: ``"auto"``
            (choose by EEG channel count, default), ``"low_density"``,
            ``"mid_density"``, ``"high_density"``, or ``"legacy"`` (the
            historical fixed defaults, kept for backward compatibility).
            Overrides the ``preset`` set in the ``bad_channel_detection``
            task config, if any.
        correlation_thresh : float, Optional
            Threshold for correlation-based detection. Overrides the preset.
        deviation_thresh : float, Optional
            Threshold for deviation-based detection. Overrides the preset.
        ransac_sample_prop : float, Optional
            Proportion of samples to use for RANSAC. Overrides the preset.
        ransac_corr_thresh : float, Optional
            Threshold for RANSAC-based detection. Overrides the preset.
        ransac_frac_bad : float, Optional
            Fraction of bad channels to use for RANSAC. Overrides the preset.
        ransac_channel_wise : bool, Optional
            Whether to use channel-wise RANSAC. Overrides the preset.
        ransac_enabled : bool, Optional
            Whether RANSAC-based detection runs at all. Low-density presets
            disable it by default since RANSAC is unstable with limited
            spatial redundancy. Overrides the preset.
        max_bad_fraction : float, Optional
            Fraction of bad channels above which the recording is flagged.
            Overrides the preset's montage-appropriate default.
        channel_count_bins : dict, Optional
            Custom channel-count bins for ``preset="auto"`` (or to override a
            named density preset's range/thresholds). Merged onto the
            built-in bins; see :mod:`autoclean.utils.bad_channel_presets`.
        random_state : int, Optional
            Random state for reproducibility.
        cleaning_method : str, Optional
            Method to use for cleaning bad channels.
            Options are 'interpolate' or 'drop' or None. If not passed,
            uses the resolved preset/config value (default 'interpolate').
        reset_bads : bool, Optional
            Whether to reset bad channels.
        stage_name : str, Optional
            Name for saving and metadata.
        manual_bad_channels : List[str], Optional
            Explicit list of bad channels to apply. When provided, automatic
            detection is skipped and this list takes precedence.

        Returns
        -------
        result_raw : instance of mne.io.Raw
            The raw data object with bad channels marked or cleaned

        Notes
        -----
        Thresholds are resolved in priority order: explicit method
        arguments > the ``bad_channel_detection`` task config > the
        selected preset/channel-count bin > historical defaults. The
        resolved preset, channel-count bin, and thresholds are always
        recorded in the ``step_clean_bad_channels`` metadata.

        See Also
        --------
        :py:class:`pyprep.find_noisy_channels.NoisyChannels` : For more information on the NoisyChannels class
        """
        # Determine which data to use
        data = self._get_data_object(data)

        # Type checking - use BaseRaw to support all file formats (FIFF, EEGLAB, etc.)
        if not isinstance(data, mne.io.base.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for bad channel detection")

        try:
            # Check if "eog" is in channel types and handle EOG channels if needed
            if (
                hasattr(self, "config")
                and self.config.get("task")
                and "eog" in data.get_channel_types()
            ):
                task = self.config.get("task")
                if (
                    not self.config.get("tasks", {})
                    .get(task, {})
                    .get("settings", {})
                    .get("eog_step", {})
                    .get("enabled", True)
                ):
                    # If EOG step is disabled, temporarily set EOG channels to EEG type
                    indices_dict = mne.channel_indices_by_type(data.info, picks="eog")
                    eog_picks = indices_dict.get("eog", [])
                    eog_ch_names = [data.ch_names[idx] for idx in eog_picks]
                    data.set_channel_types({ch: "eeg" for ch in eog_ch_names})

            # Create a copy of the data
            result_raw = data.copy()

            manual_bad_channels = (
                [str(ch) for ch in manual_bad_channels]
                if manual_bad_channels is not None
                else None
            )
            manual_override = bool(manual_bad_channels)
            imported_bad_channels = (
                [] if manual_override else self._apply_bad_channel_log(result_raw)
            )

            # Read the bad_channel_detection task config (ignored when the
            # step is explicitly disabled; detection itself still always
            # runs -- only the config-driven overrides are skipped).
            is_config_enabled, bad_channel_step_config = self._check_step_enabled(
                "bad_channel_detection"
            )
            config_value = (
                (bad_channel_step_config or {}).get("value") or {}
                if is_config_enabled
                else {}
            )
            config_preset = config_value.get("preset")
            config_channel_bins = config_value.get("channel_count_bins")

            eeg_channel_count = len(
                mne.pick_types(result_raw.info, eeg=True, exclude=[])
            )
            if eeg_channel_count == 0:
                eeg_channel_count = result_raw.info["nchan"]

            resolved = resolve_bad_channel_settings(
                channel_count=eeg_channel_count,
                preset=preset if preset is not None else (config_preset or "auto"),
                channel_count_bins=merge_channel_count_bins(
                    channel_count_bins
                    if channel_count_bins is not None
                    else config_channel_bins
                ),
                config_overrides=config_value,
                explicit_overrides={
                    "correlation_thresh": correlation_thresh,
                    "deviation_thresh": deviation_thresh,
                    "ransac_sample_prop": ransac_sample_prop,
                    "ransac_corr_thresh": ransac_corr_thresh,
                    "ransac_frac_bad": ransac_frac_bad,
                    "ransac_channel_wise": ransac_channel_wise,
                    "ransac_enabled": ransac_enabled,
                    "max_bad_fraction": max_bad_fraction,
                },
            )
            final_cleaning_method = (
                cleaning_method if cleaning_method is not _UNSET else resolved.cleaning_method
            )

            message(
                "info",
                f"Bad channel detection preset={resolved.preset!r} "
                f"density_bin={resolved.density_bin!r} "
                f"(EEG channels={eeg_channel_count})",
            )

            # Setup options
            options = {
                "random_state": random_state,
                **resolved.detector_options(),
            }

            if manual_override:
                message(
                    "info",
                    "Applying manual bad channel override: " f"{manual_bad_channels}",
                )
                uncorrelated_channels: List[str] = []
                deviation_channels: List[str] = []
                ransac_channels: List[str] = []
                all_bad_channels = manual_bad_channels
            else:
                # Call standalone function for bad channel detection
                bad_channels = detect_bad_channels(
                    data=result_raw,
                    correlation_thresh=options["correlation_thresh"],
                    deviation_thresh=options["deviation_thresh"],
                    ransac_sample_prop=options["ransac_sample_prop"],
                    ransac_corr_thresh=options["ransac_corr_thresh"],
                    ransac_frac_bad=options["ransac_frac_bad"],
                    ransac_channel_wise=options["ransac_channel_wise"],
                    random_state=options["random_state"],
                    return_by_method=True,
                    verbose=False,
                )

                # Extract individual method results for compatibility
                uncorrelated_channels = bad_channels["correlation"]
                deviation_channels = bad_channels["deviation"]
                ransac_channels = bad_channels["ransac"]

                # Get the overall bad channels list for backward compatibility
                all_bad_channels = bad_channels.get("combined", [])

            if imported_bad_channels and not manual_override:
                all_bad_channels = list(
                    dict.fromkeys([*imported_bad_channels, *all_bad_channels])
                )

            # Check for reference channels to exclude from bad channels
            ref_channels = []
            if hasattr(self, "config"):
                task = self.config.get("task")
                ref_step = (
                    self.config.get("tasks", {})
                    .get(task, {})
                    .get("settings", {})
                    .get("reference_step", {})
                )
                if ref_step and ref_step.get("enabled") and ref_step.get("value"):
                    ref_channels = ref_step.get("value", [])
                    message(
                        "info",
                        f"Excluding reference channel(s) from bad channels: {ref_channels}",
                    )

            # Add bad channels to info, but exclude reference channels
            filtered_bad_channels = [
                str(ch) for ch in all_bad_channels if str(ch) not in ref_channels
            ]
            result_raw.info["bads"].extend(filtered_bad_channels)

            # Remove duplicates
            bads = list(set(result_raw.info["bads"]))
            result_raw.info["bads"] = bads

            if final_cleaning_method == "interpolate":
                result_raw.interpolate_bads(reset_bads=reset_bads)
            if final_cleaning_method == "drop":
                result_raw.drop_channels(result_raw.info["bads"])
                result_raw.info["bads"] = []

            if hasattr(self.raw, "bad_channels"):
                total_bads = self.raw.bad_channels
                total_bads.extend(bads)
                total_bads = list(set(total_bads))
                self.raw.bad_channels = total_bads
            else:
                self.raw.bad_channels = bads

            bad_fraction = len(self.raw.bad_channels) / result_raw.info["nchan"]
            if bad_fraction > resolved.max_bad_fraction:
                self.flagged = True
                warning = (
                    f"WARNING: {bad_fraction:.2%} bad channels detected, exceeding the "
                    f"{resolved.preset!r} preset's max_bad_fraction "
                    f"({resolved.max_bad_fraction:.0%})"
                )
                self.flagged_reasons.append(warning)
                message("warning", f"Flagging: {warning}")

            if manual_override:
                message(
                    "info",
                    f"Applied manual bad channels ({len(bads)}): {bads}",
                )
                for channel in bads:
                    self._track_channel_removal(
                        channels=channel,
                        reason="MANUAL_OVERRIDE",
                        source_step="clean_bad_channels",
                    )
            else:
                message("info", f"Detected {len(bads)} bad channels: {bads}")

                # Track channel removals in unified metadata by detection method
                for channel in uncorrelated_channels:
                    self._track_channel_removal(
                        channels=channel,
                        reason="UNCORRELATED",
                        source_step="clean_bad_channels",
                    )
                for channel in deviation_channels:
                    self._track_channel_removal(
                        channels=channel,
                        reason="DEVIATION",
                        source_step="clean_bad_channels",
                    )
                for channel in ransac_channels:
                    self._track_channel_removal(
                        channels=channel,
                        reason="RANSAC",
                        source_step="clean_bad_channels",
                    )

            # Update metadata
            metadata = {
                "method": "ManualOverride" if manual_override else "NoisyChannels",
                "options": (
                    options
                    if not manual_override
                    else {"manual_bad_channels": manual_bad_channels}
                ),
                "preset": resolved.preset,
                "density_bin": resolved.density_bin,
                "eeg_channel_count": eeg_channel_count,
                "resolved_thresholds": resolved.as_metadata(),
                "cleaning_method": final_cleaning_method,
                "channelCount": len(result_raw.ch_names),
                "durationSec": int(result_raw.n_times) / result_raw.info["sfreq"],
                "numberSamples": int(result_raw.n_times),
                "bads": bads,
                "uncorrelated_channels": uncorrelated_channels,
                "deviation_channels": deviation_channels,
                "ransac_channels": ransac_channels,
            }

            self._update_metadata("step_clean_bad_channels", metadata)

            # Save the result
            self._save_raw_result(result_raw, stage_name)

            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)

            return result_raw
        except Exception as e:
            message("error", f"Error during bad channel detection: {str(e)}")
            raise RuntimeError(f"Failed to detect bad channels: {str(e)}") from e

    def _apply_bad_channel_log(self, raw: mne.io.Raw) -> List[str]:
        """Apply configured user-provided bad channels to ``raw.info['bads']``."""

        is_enabled, config_value = self._check_step_enabled("bad_channel_log")
        if not is_enabled:
            return []

        value = (config_value or {}).get("value") or {}
        path = value.get("path")
        if not path:
            raise ValueError("bad_channel_log is enabled but no value.path was set")

        action = value.get("action", "mark")
        if action != "mark":
            raise ValueError(
                "bad_channel_log currently supports action='mark' only; "
                "drop/interpolate/exclude require a separate tested workflow."
            )

        file_column = value.get("file_column", "file")
        channels_column = value.get("channels_column", "bad_channels")
        strict = bool(value.get("strict", False))
        field_matches = self._bad_channel_log_field_matches(value)

        rows = load_metadata_table(path, delimiter=value.get("delimiter"))
        match = match_recording_row(
            rows,
            self.config.get("unprocessed_file", ""),
            file_column=file_column,
            field_matches=field_matches,
        )

        if match is None:
            warning = (
                "No bad-channel log row matched input file "
                f"{self.config.get('unprocessed_file')!s}"
            )
            if strict:
                raise ValueError(warning)
            self._record_bad_channel_log_metadata(
                path=path,
                matched=False,
                matched_by=None,
                applied=[],
                missing=[],
                warning=warning,
            )
            message("warning", warning)
            return []

        if channels_column not in match.row:
            raise ValueError(
                f"Bad-channel log is missing channels column {channels_column!r}"
            )

        requested = split_channels(match.row.get(channels_column))
        missing = [channel for channel in requested if channel not in raw.ch_names]
        applied = [channel for channel in requested if channel in raw.ch_names]

        warning = None
        if missing:
            warning = (
                "Bad-channel log referenced channel(s) not present in raw data: "
                f"{missing}"
            )
            if strict:
                raise ValueError(warning)
            message("warning", warning)

        if applied:
            raw.info["bads"] = list(dict.fromkeys([*raw.info["bads"], *applied]))
            self._track_channel_removal(
                channels=applied,
                reason="BAD_CHANNEL_LOG",
                source_step="bad_channel_log",
            )
            message("info", f"Applied bad-channel log channels: {applied}")

        self._record_bad_channel_log_metadata(
            path=path,
            matched=True,
            matched_by=match.matched_by,
            applied=applied,
            missing=missing,
            warning=warning,
        )
        return applied

    def _bad_channel_log_field_matches(self, value: dict) -> dict[str, str]:
        """Build optional exact-match criteria for subject/session columns."""

        matches: dict[str, str] = {}
        subject_column = value.get("subject_column")
        subject = value.get("subject") or self.config.get("subject")
        if subject_column:
            if subject:
                matches[str(subject_column)] = str(subject)
            else:
                message(
                    "warning",
                    "bad_channel_log subject_column configured but no subject value was found",
                )

        session_column = value.get("session_column")
        session = value.get("session") or self.config.get("session")
        if session_column:
            if session:
                matches[str(session_column)] = str(session)
            else:
                message(
                    "warning",
                    "bad_channel_log session_column configured but no session value was found",
                )

        return matches

    def _record_bad_channel_log_metadata(
        self,
        *,
        path: str,
        matched: bool,
        matched_by: str | None,
        applied: List[str],
        missing: List[str],
        warning: str | None,
    ) -> None:
        metadata = {
            "path": str(path),
            "matched": matched,
            "matched_by": matched_by,
            "applied_channels": applied,
            "missing_channels": missing,
            "warning": warning,
        }
        self._update_metadata("step_bad_channel_log", metadata)

    def drop_channels(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        channels: List[str] = None,
        stage_name: str = "drop_channels",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Drop specified channels from the data.

        This method removes specified channels from the data.

        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs, Optional
            The data object to drop channels from. If None, uses self.raw or self.epochs.
        channels : List[str], Optional
            List of channel names to drop.
        stage_name : str, Optional
            Name for saving and metadata.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns
        -------
        result_data : instance of mne.io.Raw or mne.Epochs
            The data object with channels dropped

        See Also
        --------
        :py:meth:`mne.io.Raw.drop_channels` : For MNE's raw data channel dropping functionality
        :py:meth:`mne.Epochs.drop_channels` : For MNE's epochs channel dropping functionality
        """
        # Check if channels is provided
        if channels is None:
            is_enabled, config_value = self._check_step_enabled("drop_outerlayer")

            if not is_enabled:
                message("info", "Channel dropping is disabled in configuration")
                return data

            # Get channels from config
            channels = config_value

            if not channels:
                message("warning", "No channels specified for dropping in config")
                return data

        # Determine which data to use
        data = self._get_data_object(data, use_epochs)

        # Type checking
        if not isinstance(
            data, (mne.io.base.BaseRaw, mne.BaseEpochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError(
                "Data must be an MNE Raw or Epochs object for dropping channels"
            )

        try:
            # Drop channels
            message("header", "Dropping channels...")
            result_data = data.copy().drop_channels(channels)
            message("info", f"Dropped {len(channels)} channels: {channels}")

            # Track channel removals in unified metadata
            self._track_channel_removal(
                channels=channels,
                reason="MANUAL_EXCLUDE",
                source_step=stage_name,
            )

            # Update metadata
            metadata = {
                "channels_dropped": channels,
                "channels_remaining": len(result_data.ch_names),
            }

            self._update_metadata("step_drop_channels", metadata)

            # Save the result if it's a Raw object (use BaseRaw for all file formats)
            if isinstance(result_data, mne.io.base.BaseRaw):
                self._save_raw_result(result_data, stage_name)

            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)

            return result_data

        except Exception as e:
            message("error", f"Error during channel dropping: {str(e)}")
            raise RuntimeError(f"Failed to drop channels: {str(e)}") from e

    def set_channel_types(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        ch_types_dict: Dict[str, str] = None,
        drop: bool = False,
        stage_name: str = "set_channel_types",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Set channel types for specific channels, or optionally drop them.

        This method sets the type of specific channels (e.g., marking channels as EOG),
        or drops them entirely if drop=True. Dropping is useful for reference electrodes
        (A1/A2) that should not be included in analysis when using average reference.

        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs, Optional
            The data object to set channel types for. If None, uses self.raw or self.epochs.
        ch_types_dict : dict, Optional
            Dictionary mapping channel names to types (e.g., {'E1': 'eog'}).
            When drop=True, the types are ignored and channels are dropped instead.
        drop : bool, Optional (default: False)
            If True, drop the specified channels instead of changing their type.
            Useful for removing reference electrodes (A1/A2) or other channels
            that should not be included in analysis. Channels not present in data
            are silently skipped.
        stage_name : str, Optional
            Name for saving and metadata.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns
        -------
        result_data : instance of mne.io.Raw or mne.Epochs
            The data object with updated channel types (or channels dropped)

        """
        # Check if ch_types_dict is provided
        if ch_types_dict is None or len(ch_types_dict) == 0:
            # Check if eog_step is enabled in configuration
            is_enabled, config_value = self._check_step_enabled("eog_step")

            if not is_enabled:
                message("info", "Channel type setting is disabled in configuration")
                return data

            # Get channel types from config
            ch_types_dict = config_value

            if not ch_types_dict:
                message("warning", "No channel types specified in config")
                return data

        # Determine which data to use
        data = self._get_data_object(data, use_epochs)

        # Type checking - use BaseRaw/BaseEpochs to support all file formats (FIFF, EEGLAB, etc.)
        if not isinstance(
            data, (mne.io.BaseRaw, mne.BaseEpochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError(
                "Data must be an MNE Raw or Epochs object for setting channel types"
            )

        try:
            result_data = data.copy()

            if drop:
                # Drop specified channels instead of setting their type
                message("header", "Dropping channels...")
                channels_to_drop = [
                    ch for ch in ch_types_dict.keys() if ch in result_data.ch_names
                ]

                if channels_to_drop:
                    result_data.drop_channels(channels_to_drop)
                    message(
                        "info",
                        f"Dropped {len(channels_to_drop)} channels: {channels_to_drop}",
                    )

                    # Track channel removals in unified metadata
                    self._track_channel_removal(
                        channels=channels_to_drop,
                        reason="REFERENCE_DROPPED",
                        source_step=stage_name,
                    )

                    # Update metadata
                    metadata = {
                        "dropped_channels": channels_to_drop,
                        "reason": "Reference electrodes excluded from analysis",
                        "original_dict": ch_types_dict,
                    }
                else:
                    message(
                        "info",
                        "No matching channels found to drop (channels may already be absent)",
                    )
                    metadata = {
                        "dropped_channels": [],
                        "reason": "No matching channels present",
                        "original_dict": ch_types_dict,
                    }
            else:
                # Original behavior: Set channel types
                message("header", "Setting channel types...")
                result_data.set_channel_types(ch_types_dict)
                message("info", f"Set types for {len(ch_types_dict)} channels")

                # Update metadata
                metadata = {"channel_types": ch_types_dict}

            self._update_metadata("set_channel_types", metadata)

            # Save the result if it's a Raw object (use BaseRaw for all file formats)
            if isinstance(result_data, mne.io.base.BaseRaw):
                self._save_raw_result(result_data, stage_name)

            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)

            return result_data

        except Exception as e:
            message("error", f"Error during setting channel types: {str(e)}")
            raise RuntimeError(f"Failed to set channel types: {str(e)}") from e

    def drop_eog_channels(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        stage_name: str = "drop_eog_channels",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Drop EOG channels from EEG data after ICA processing.

        This method removes all channels marked as EOG type from the data.
        Useful for cleaning up the data after ICA artifact removal.

        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs, Optional
            The data object to drop EOG channels from. If None, uses self.raw or self.epochs.
        stage_name : str, Optional
            Name for saving and metadata.
        use_epochs : bool, Optional
            If True, operates on epochs data instead of raw data.

        Returns
        -------
        mne.io.Raw or mne.Epochs
            The data with EOG channels removed.
        """
        try:
            # Get the appropriate data object
            if data is None:
                data = self.epochs if use_epochs else self.raw
                if data is None:
                    raise ValueError("No data available to process")

            # Detect EOG channels
            indices_dict = mne.channel_indices_by_type(data.info, picks="eog")
            eog_picks = indices_dict.get("eog", [])
            eog_ch_names = [data.ch_names[idx] for idx in eog_picks]

            if not eog_ch_names:
                message("info", "No EOG channels found to drop")
                return data.copy()

            message(
                "info", f"Dropping {len(eog_ch_names)} EOG channels: {eog_ch_names}"
            )

            # Drop the EOG channels
            result_data = data.copy()
            result_data.drop_channels(eog_ch_names, on_missing="ignore")

            # Track channel removals in unified metadata
            self._track_channel_removal(
                channels=eog_ch_names,
                reason="EOG_DROPPED",
                source_step=stage_name,
            )

            # Export the result using standard pipeline saving
            if use_epochs:
                self._save_epochs_result(result_data, stage_name)
            else:
                self._save_raw_result(result_data, stage_name)

            message(
                "info", f"Exported {stage_name} data using standard pipeline method"
            )
            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)

            return result_data

        except Exception as e:
            message("error", f"Error during EOG channel dropping: {str(e)}")
            raise RuntimeError(f"Failed to drop EOG channels: {str(e)}") from e
