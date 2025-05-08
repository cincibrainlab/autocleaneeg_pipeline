"""Basic steps mixin for autoclean tasks."""

from typing import Union

import mne

from autoclean.utils.logging import message


class BasicStepsMixin:
    """Mixin class providing basic signal processing steps for autoclean tasks."""
    def run_basic_steps(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Runs all basic preprocessing steps sequentially based on configuration.

        The steps included are:
        1. Resample Data
        2. Drop Outer Layer Channels
        3. Assign EOG Channels
        4. Rereference Data
        5. Trim Edges
        6. Crop Duration

        Each step's execution depends on its 'enabled' status in the configuration.

        Parameters
        ----------
        data : Optional
            The data object to process. If None, uses self.raw or self.epochs.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns
        -------
        inst : instance of mne.io.Raw or mne.io.Epochs
            The data object after applying all enabled basic processing steps.
        """
        message("header", "Running basic preprocessing steps...")

        # Start with the correct data object
        processed_data = self._get_data_object(data, use_epochs)

        # 1. Resample
        processed_data = self.resample_data(data=processed_data, use_epochs=use_epochs)

        # 2. Filter
        processed_data = self.filter_data(data=processed_data, use_epochs=use_epochs)

        # 3. Drop Outer Layer
        processed_data = self.drop_outer_layer(
            data=processed_data, use_epochs=use_epochs
        )

        # 4. Assign EOG Channels
        processed_data = self.assign_eog_channels(
            data=processed_data, use_epochs=use_epochs
        )

        # 6. Trim Edges
        processed_data = self.trim_edges(data=processed_data, use_epochs=use_epochs)

        # 7. Crop Duration
        processed_data = self.crop_duration(data=processed_data, use_epochs=use_epochs)

        message("info", "Basic preprocessing steps completed successfully.")

        self._save_raw_result(processed_data, "post_basicsteps")

        return processed_data

    def filter_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Filter raw or epoched data based on configuration settings.

        This method can work with self.raw, self.epochs, or a provided data object.
        It checks the filtering_step toggle in the configuration if no filter_args are provided.
        
        Parameters
        ----------
        data : Optional
            The data object to filter. If None, uses self.raw or self.epochs.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns
        -------
        inst : instance of mne.io.Raw or mne.io.Epochs
            The filtered data object (same type as input)
            
        Examples
        --------
        >>> #Inside a task class that uses the autoclean framework
        >>> self.filter_data()

        See Also
        --------
        :py:meth:`mne.io.Raw.filter` : For MNE's raw data filtering functionality
        :py:meth:`mne.Epochs.filter` : For MNE's epochs filtering functionality
        """
        data = self._get_data_object(data, use_epochs)

        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        is_enabled, config_value = self._check_step_enabled("filtering")

        if not is_enabled:
            message("info", "Filtering step is disabled in configuration")
            return data

        filter_args = config_value.get("value", {})
        if not filter_args:
            message("warning", "No filter arguments provided, skipping filtering")
            return data

        message("header", "Filtering data...")
        filtered_data = data.copy()

        if "l_freq" in filter_args:
            filtered_data.filter(l_freq=filter_args["l_freq"], h_freq=None)

        if "h_freq" in filter_args:
            filtered_data.filter(l_freq=None, h_freq=filter_args["h_freq"])

        if "notch_freqs" in filter_args:
            filtered_data.notch_filter(
                freqs=filter_args["notch_freqs"],
                notch_widths=filter_args.get("notch_widths", 0.5),
            )

        self._save_raw_result(filtered_data, "post_filter")

        metadata = {
            "original_data_type": type(data).__name__,
            "filtered_data_type": type(filtered_data).__name__,
            "filter_args": filter_args,
        }

        self._update_metadata("step_filter_data", metadata)
        self._update_instance_data(data, filtered_data, use_epochs)

        return filtered_data

    def resample_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        target_sfreq: float = None,
        stage_name: str = "post_resample",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Resample raw or epoched data based on configuration settings.

        This method can work with self.raw, self.epochs, or a provided data object.
        It checks the resample_step toggle in the configuration if no target_sfreq is provided.

        Parameters
        ----------
        data : Optional
            The raw data to resample. If None, uses self.raw or self.epochs.
        target_sfreq : float, Optional
            The target sampling frequency. If None, reads from config.
        stage_name : str, Optional
            Name for saving the resampled data (default: "resampled").
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns:
            inst : instance of mne.io.Raw or mne.io.Epochs
            The resampled data object (same type as input)

        Examples
        --------
        >>> #Inside a task class that uses the autoclean framework
        >>> self.resample_data()

        See Also
        --------
        :py:meth:`mne.io.Raw.resample` : For MNE's raw data resampling functionality
        :py:meth:`mne.Epochs.resample` : For MNE's epochs resampling functionality
        """
        # Determine which data to use
        data = self._get_data_object(data, use_epochs)

        # Type checking
        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        # Access configuration if needed
        if target_sfreq is None:
            is_enabled, config_value = self._check_step_enabled("resample_step")

            if not is_enabled:
                message("info", "Resampling step is disabled in configuration")
                return data

            target_sfreq = config_value.get("value", None)

            if target_sfreq is None:
                message(
                    "warning",
                    "Target sampling frequency not specified, skipping resampling",
                )
                return data

        # Check if we need to resample (avoid unnecessary resampling)
        current_sfreq = data.info["sfreq"]
        if (
            abs(current_sfreq - target_sfreq) < 0.01
        ):  # Small threshold to account for floating point errors
            message(
                "info",
                f"Data already at target frequency ({target_sfreq} Hz), skipping resampling",
            )
            return data

        message(
            "header", f"Resampling data from {current_sfreq} Hz to {target_sfreq} Hz..."
        )

        try:
            # Resample based on data type
            if isinstance(data, mne.io.base.BaseRaw):
                resampled_data = data.copy().resample(target_sfreq)
                # Save resampled raw data if it's a Raw object
                self._save_raw_result(resampled_data, stage_name)
            else:  # Epochs
                resampled_data = data.copy().resample(target_sfreq)

            message("info", f"Data successfully resampled to {target_sfreq} Hz")

            # Update metadata
            metadata = {
                "original_sfreq": current_sfreq,
                "target_sfreq": target_sfreq,
                "data_type": "raw"
                if isinstance(data, mne.io.Raw) or isinstance(data, mne.io.base.BaseRaw)
                else "epochs",
            }

            self._update_metadata("step_resample_data", metadata)

            # Update self.raw or self.epochs if we're using those
            self._update_instance_data(data, resampled_data, use_epochs)

            return resampled_data

        except Exception as e:
            message("error", f"Error during resampling: {str(e)}")
            raise RuntimeError(f"Failed to resample data: {str(e)}") from e

    def rereference_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        ref_type: str = None,
        use_epochs: bool = False,
        stage_name: str = "post_rereference",
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Rereference raw or epoched data based on configuration settings.

        This method can work with self.raw, self.epochs, or a provided data object.
        It checks the rereference_step toggle in the configuration if no ref_type is provided.

        Parameters
        ----------
        data : Optional
            The raw data to rereference. If None, uses self.raw or self.epochs.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.
        ref_type : str, Optional
            The type of reference to use. If None, reads from config.
        stage_name : str, Optional
            Name for saving the rereferenced data (default: "post_rereference").

        Returns
        -------
        inst : instance of mne.io.Raw or mne.io.Epochs
            The rereferenced data object (same type as input)

        Examples
        --------    
        >>> #Inside a task class that uses the autoclean framework
        >>> self.rereference_data()

        See Also
        --------
        :py:meth:`mne.io.Raw.set_eeg_reference` : For MNE's raw data rereferencing functionality
        :py:meth:`mne.Epochs.set_eeg_reference` : For MNE's epochs rereferencing functionality
        """

        data = self._get_data_object(data, use_epochs)

        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        if ref_type is None:
            is_enabled, config_value = self._check_step_enabled("reference_step")

            if not is_enabled:
                message("info", "Rereferencing step is disabled in configuration")
                return data

            ref_type = config_value.get("value", None)

            if ref_type is None:
                message("warning", "Rereferencing value not specified, skipping rereferencing")
                return data

        if ref_type == "average":
            rereferenced_data = data.copy().set_eeg_reference(ref_type, projection=False)
        else:
            rereferenced_data = data.copy().set_eeg_reference(ref_type)

        self._save_raw_result(rereferenced_data, stage_name)

        metadata = {
            "new_ref_type": ref_type,
        }

        self._update_instance_data(data, rereferenced_data, use_epochs)

        self._update_metadata("step_rereference_data", metadata)

        return rereferenced_data

    def drop_outer_layer(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        stage_name: str = "post_outerlayer",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Drop outer layer channels based on configuration settings.

        Parameters
        ----------
        data : Optional
            The data object to process. If None, uses self.raw or self.epochs.
        stage_name : str, Optional
            Name for saving the processed data (default: "post_outerlayer").
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns:
            inst : instance of mne.io.Raw or mne.io.Epochs
            The data object with outer layer channels removed.
        """
        data = self._get_data_object(data, use_epochs)

        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        is_enabled, config_value = self._check_step_enabled("drop_outerlayer")

        if not is_enabled:
            message("info", "Drop Outer Layer step is disabled in configuration")
            return data

        outer_layer_channels = config_value.get("value", [])
        if not outer_layer_channels:
            message("warning", "Outer layer channels not specified, skipping step")
            return data

        # Ensure channels exist in the data before attempting to drop
        channels_to_drop = [ch for ch in outer_layer_channels if ch in data.ch_names]
        if not channels_to_drop:
            message(
                "info", "Specified outer layer channels not found in data, skipping drop."
            )
            return data

        message(
            "header", f"Dropping outer layer channels: {', '.join(channels_to_drop)}"
        )
        processed_data = data.copy().drop_channels(channels_to_drop)
        message("info", f"Channels dropped: {', '.join(channels_to_drop)}")

        if isinstance(processed_data, (mne.io.Raw, mne.io.base.BaseRaw)):
            self._save_raw_result(processed_data, stage_name)

        metadata = {
            "dropped_outer_layer_channels": channels_to_drop,
            "original_channel_count": len(data.ch_names),
            "new_channel_count": len(processed_data.ch_names),
        }
        self._update_metadata("step_drop_outerlayer", metadata)
        self._update_instance_data(data, processed_data, use_epochs)

        return processed_data

    def assign_eog_channels(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Assign EOG channel types based on configuration settings.

        Parameters
        ----------
        data : Optional
            The data object to process. If None, uses self.raw or self.epochs.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns:
            inst : instance of mne.io.Raw or mne.io.Epochs
            The data object with EOG channels assigned.
        """
        data = self._get_data_object(data, use_epochs)

        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        is_enabled, config_value = self._check_step_enabled("eog_step")

        if not is_enabled:
            message("info", "EOG Assignment step is disabled in configuration")
            return data

        eog_channel_indices = config_value.get("value", [])
        if not eog_channel_indices:
            message("warning", "EOG channel indices not specified, skipping step")
            return data

        # Assuming value is a list of indices or names, convert indices to names if needed
        # The example uses formatting `f"E{ch}"`, suggesting indices are expected.
        # Adapt this logic based on how channel names vs indices are stored in config.
        # For simplicity, assuming names or indices directly map to existing channel names for now.
        # A more robust implementation might handle various naming conventions.
        eog_channels_to_set = [
            ch
            for idx, ch in enumerate(data.ch_names)
            if idx + 1 in eog_channel_indices or ch in eog_channel_indices
        ] # Handling both indices (1-based) and names

        eog_channels_map = {
            ch: "eog" for ch in eog_channels_to_set if ch in data.ch_names
        }

        if not eog_channels_map:
            message("warning", "Specified EOG channels not found in data, skipping step.")
            return data

        message("header", f"Assigning EOG channel types for: {', '.join(eog_channels_map.keys())}")
        # Process a copy to avoid modifying the original data object directly
        processed_data = data.copy()
        processed_data.set_channel_types(eog_channels_map)
        message("info", f"EOG channel types assigned for: {', '.join(eog_channels_map.keys())}")

        # Note: set_channel_types modifies in place, but we operate on a copy.
        # No need to save intermediate step here unless explicitly required,
        # as channel type changes don't alter the data matrix itself.

        metadata = {"assigned_eog_channels": list(eog_channels_map.keys())}
        self._update_metadata("step_assign_eog_channels", metadata)

        # Even though set_channel_types modifies inplace on the copy,
        # we still call update_instance_data to potentially update self.raw/self.epochs
        self._update_instance_data(data, processed_data, use_epochs)

        return processed_data

    def trim_edges(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        stage_name: str = "post_trim",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Trim data edges based on configuration settings.

        Parameters
        ----------
        data : Optional
            The data object to process. If None, uses self.raw or self.epochs.
        stage_name : str, Optional
            Name for saving the processed data (default: "post_trim").
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns:
            inst : instance of mne.io.Raw or mne.io.Epochs
            The data object with edges trimmed.
        """
        data = self._get_data_object(data, use_epochs)

        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        is_enabled, config_value = self._check_step_enabled("trim_step")

        if not is_enabled:
            message("info", "Edge Trimming step is disabled in configuration")
            return data

        trim_duration_sec = config_value.get("value", None)
        if trim_duration_sec is None or trim_duration_sec <= 0:
            message(
                "warning",
                "Invalid or zero trim duration specified, skipping edge trimming",
            )
            return data

        original_start_time = data.times[0]
        original_end_time = data.times[-1]
        original_duration = original_end_time - original_start_time

        if 2 * trim_duration_sec >= original_duration:
            message(
                "error",
                f"Total trim duration ({2 * trim_duration_sec}s) is greater than or equal to data "
                f"duration ({original_duration}s). Cannot trim.",
            )
            # Consider raising an error or just returning data
            return data # Return original data to avoid erroring out pipeline

        tmin = original_start_time + trim_duration_sec
        tmax = original_end_time - trim_duration_sec

        message(
            "header",
            f"Trimming {trim_duration_sec}s from each end (new range: {tmin:.3f}s to {tmax:.3f}s)",
        )
        processed_data = data.copy().crop(tmin=tmin, tmax=tmax)
        new_duration = processed_data.times[-1] - processed_data.times[0]
        message("info", f"Data trimmed. New duration: {new_duration:.3f}s")

        if isinstance(processed_data, (mne.io.Raw, mne.io.base.BaseRaw)):
            self._save_raw_result(processed_data, stage_name)

        metadata = {
            "trim_duration": trim_duration_sec,
            "original_start_time": original_start_time,
            "original_end_time": original_end_time,
            "new_start_time": tmin,
            "new_end_time": tmax,
            "original_duration": original_duration,
            "new_duration": new_duration,
        }
        self._update_metadata("step_trim_edges", metadata)
        self._update_instance_data(data, processed_data, use_epochs)

        return processed_data

    def crop_duration(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        stage_name: str = "post_crop",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Crop data duration based on configuration settings.

        Parameters
        ----------
        data : Optional
            The data object to process. If None, uses self.raw or self.epochs.
        stage_name : str, Optional
            Name for saving the processed data (default: "post_crop").
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.

        Returns:
            inst : instance of mne.io.Raw or mne.io.Epochs
            The data object cropped to the specified duration.
        """
        data = self._get_data_object(data, use_epochs)

        if not isinstance(data, (mne.io.base.BaseRaw, mne.Epochs)):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        is_enabled, config_value = self._check_step_enabled("crop_step")

        if not is_enabled:
            message("info", "Duration Cropping step is disabled in configuration")
            return data

        crop_times = config_value.get("value", {})
        start_time_sec = crop_times.get("start", None)
        end_time_sec = crop_times.get("end", None)

        if start_time_sec is None and end_time_sec is None:
            message(
                "warning", "Crop start and end times not specified, skipping cropping"
            )
            return data

        # Use data's bounds if start or end is None
        tmin = start_time_sec if start_time_sec is not None else data.times[0]
        tmax = end_time_sec if end_time_sec is not None else data.times[-1]

        # Validate crop times against data bounds
        original_start = data.times[0]
        original_end = data.times[-1]

        # Adjust tmin/tmax if they fall outside the data range
        tmin = max(tmin, original_start)
        tmax = min(tmax, original_end)

        if tmin >= tmax:
            message(
                "error",
                f"Invalid crop range: start time ({tmin:.3f}s) is not before end time ({tmax:.3f}s)"
                f"after adjusting to data bounds. Skipping crop.",
            )
            return data

        message("header", f"Cropping data duration to range: {tmin:.3f}s to {tmax:.3f}s")
        processed_data = data.copy().crop(tmin=tmin, tmax=tmax)
        new_duration = processed_data.times[-1] - processed_data.times[0]
        message("info", f"Data cropped. New duration: {new_duration:.3f}s")

        if isinstance(processed_data, (mne.io.Raw, mne.io.base.BaseRaw)):
            self._save_raw_result(processed_data, stage_name)

        metadata = {
            "crop_duration": start_time_sec,
            "crop_start": tmin,
            "crop_end": tmax,
            "original_duration": original_end - original_start,
            "new_duration": new_duration,
        }
        self._update_metadata("step_crop_duration", metadata)
        self._update_instance_data(data, processed_data, use_epochs)

        return processed_data
