"""Basic steps mixin for autoclean tasks."""

from typing import List, Optional, Union

import mne

from autoclean.functions.preprocessing.filtering import (
    filter_data as standalone_filter_data,
)
from autoclean.functions.preprocessing.referencing import (
    rereference_data as standalone_rereference_data,
)
from autoclean.functions.preprocessing.resampling import (
    resample_data as standalone_resample_data,
)
from autoclean.utils.logging import message


class BasicStepsMixin:
    """Mixin class providing basic signal processing steps for autoclean tasks."""

    def run_basic_steps(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        use_epochs: bool = False,
        stage_name: str = "post_basic_steps",
        export: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Runs all basic preprocessing steps sequentially based on configuration.

        The steps included are:
        1. Resample Data
        2. Filter Data
        3. Drop Outer Layer Channels
        4. Assign EOG Channels
        5. Trim Edges
        6. Crop Duration

        Each step's execution depends on its 'enabled' status in the configuration.

        Parameters
        ----------
        data : Optional
            The data object to process. If None, uses self.raw or self.epochs.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.
        stage_name : str, Optional
            Name of the processing stage for export. Default is "post_basic_steps".
        export : bool, Optional
            If True, exports the processed data to the stage directory. Default is False.

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

        # Update instance data
        self._update_instance_data(data, processed_data, use_epochs)

        # Store a copy of the pre-cleaned raw data for comparison
        self.original_raw = self.raw.copy()

        # Export if requested
        self._auto_export_if_enabled(processed_data, stage_name, export)

        return processed_data

    def filter_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        use_epochs: bool = False,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        notch_freqs: Optional[List[float]] = None,
        notch_widths: Optional[Union[float, List[float]]] = None,
        method: Optional[str] = None,
        phase: Optional[str] = None,
        export: bool = True,
        fir_window: Optional[str] = None,
        verbose: Optional[bool] = None,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Apply filtering to EEG data within the AutoClean pipeline.

        This method wraps the standalone :func:`autoclean.filter_data` function
        with pipeline integration including configuration management, metadata
        tracking, and automatic export functionality.

        Parameters override configuration values when provided. If not provided,
        values are read from the task configuration using the existing
        ``_check_step_enabled`` system.

        Parameters
        ----------
        data : mne.io.Raw, mne.Epochs, or None, default None
            Input data. If None, uses ``self.raw`` or ``self.epochs`` based on
            ``use_epochs`` parameter.
        use_epochs : bool, default False
            If True and data is None, uses ``self.epochs`` instead of ``self.raw``.
        l_freq : float or None, optional
            Low cutoff frequency for highpass filtering in Hz. Overrides config if provided.
        h_freq : float or None, optional
            High cutoff frequency for lowpass filtering in Hz. Overrides config if provided.
        notch_freqs : list of float or None, optional
            Frequencies to notch filter in Hz. Overrides config if provided.
        notch_widths : float, list of float, or None, optional
            Width of notch filters in Hz. Overrides config if provided.
        method : str or None, optional
            Filtering method ('fir' or 'iir'). Overrides config if provided.
        phase : str or None, optional
            Filter phase ('zero', 'zero-double', 'minimum'). Overrides config if provided.
        fir_window : str or None, optional
            FIR window function. Overrides config if provided.
        verbose : bool or None, optional
            Control verbosity. Overrides config if provided.

        Returns
        -------
        filtered_data : mne.io.Raw or mne.Epochs
            Filtered data object. Also updates ``self.raw`` or ``self.epochs``
            and triggers metadata tracking and export if configured.

        See Also
        --------
        autoclean.filter_data : The underlying standalone filtering function
        """
        data = self._get_data_object(data, use_epochs)

        # Use existing config system
        is_enabled, config_value = self._check_step_enabled("filtering")
        if not is_enabled:
            message("info", "Filtering step is disabled in configuration")
            return data

        # Get config defaults
        filter_args = config_value.get("value", {})

        # Apply parameter overrides (only if explicitly provided)
        final_l_freq = l_freq if l_freq is not None else filter_args.get("l_freq")
        final_h_freq = h_freq if h_freq is not None else filter_args.get("h_freq")
        final_notch_freqs = (
            notch_freqs if notch_freqs is not None else filter_args.get("notch_freqs")
        )
        final_notch_widths = (
            notch_widths
            if notch_widths is not None
            else filter_args.get("notch_widths", 0.5)
        )
        final_method = (
            method if method is not None else filter_args.get("method", "fir")
        )
        final_phase = phase if phase is not None else filter_args.get("phase", "zero")
        final_fir_window = (
            fir_window
            if fir_window is not None
            else filter_args.get("fir_window", "hamming")
        )
        final_verbose = verbose if verbose is not None else filter_args.get("verbose")

        # Check if any filtering is requested
        if final_l_freq is None and final_h_freq is None and final_notch_freqs is None:
            message("warning", "No filter parameters provided, skipping filtering")
            return data

        message("header", "Filtering data...")

        # Call standalone function
        filtered_data = standalone_filter_data(
            data=data,
            l_freq=final_l_freq,
            h_freq=final_h_freq,
            notch_freqs=final_notch_freqs,
            notch_widths=final_notch_widths,
            method=final_method,
            phase=final_phase,
            fir_window=final_fir_window,
            verbose=final_verbose,
        )

        # Pipeline integration with result-based metadata
        self._update_instance_data(data, filtered_data, use_epochs)

        if export:
            self._save_raw_result(filtered_data, "post_filter")

        # Use actual results in metadata
        metadata = {
            "original_sfreq": data.info["sfreq"],
            "filtered_sfreq": filtered_data.info["sfreq"],
            "original_n_channels": len(data.ch_names),
            "filtered_n_channels": len(filtered_data.ch_names),
            "applied_l_freq": final_l_freq,
            "applied_h_freq": final_h_freq,
            "applied_notch_freqs": final_notch_freqs,
            "applied_notch_widths": final_notch_widths,
            "method": final_method,
            "phase": final_phase,
            "fir_window": final_fir_window,
            "original_data_type": type(data).__name__,
            "result_data_type": type(filtered_data).__name__,
        }
        self._update_metadata("step_filter_data", metadata)

        return filtered_data

    def resample_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        target_sfreq: Optional[float] = None,
        export: bool = True,
        stage_name: str = "post_resample",
        use_epochs: bool = False,
        npad: Optional[str] = None,
        window: Optional[str] = None,
        n_jobs: Optional[int] = None,
        pad: Optional[str] = None,
        verbose: Optional[bool] = None,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Apply resampling to EEG data within the AutoClean pipeline.

        This method wraps the standalone :func:`autoclean.resample_data` function
        with pipeline integration including configuration management, metadata
        tracking, and automatic export functionality.

        Parameters override configuration values when provided. If not provided,
        values are read from the task configuration using the existing
        ``_check_step_enabled`` system.

        Parameters
        ----------
        data : mne.io.Raw, mne.Epochs, or None, default None
            Input data. If None, uses ``self.raw`` or ``self.epochs`` based on
            ``use_epochs`` parameter.
        target_sfreq : float or None, optional
            Target sampling frequency in Hz. Overrides config if provided.
        stage_name : str, default "post_resample"
            Name for saving the resampled data.
        use_epochs : bool, default False
            If True and data is None, uses ``self.epochs`` instead of ``self.raw``.
        npad : str or None, optional
            Padding parameter. Overrides config if provided.
        window : str or None, optional
            Window function. Overrides config if provided.
        n_jobs : int or None, optional
            Number of parallel jobs. Overrides config if provided.
        pad : str or None, optional
            Padding mode. Overrides config if provided.
        verbose : bool or None, optional
            Control verbosity. Overrides config if provided.

        Returns
        -------
        resampled_data : mne.io.Raw or mne.Epochs
            Resampled data object. Also updates ``self.raw`` or ``self.epochs``
            and triggers metadata tracking and export if configured.

        See Also
        --------
        autoclean.resample_data : The underlying standalone resampling function
        """
        data = self._get_data_object(data, use_epochs)

        # Use existing config system
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

        # Get config defaults and apply overrides
        config_args = {}
        if hasattr(self, "config") and "resample_step" in self.config.get(
            "tasks", {}
        ).get(self.config.get("task", ""), {}).get("settings", {}):
            config_args = self.config["tasks"][self.config["task"]]["settings"][
                "resample_step"
            ].get("value", {})

        final_npad = npad if npad is not None else config_args.get("npad", "auto")
        final_window = (
            window if window is not None else config_args.get("window", "auto")
        )
        final_n_jobs = n_jobs if n_jobs is not None else config_args.get("n_jobs", 1)
        final_pad = pad if pad is not None else config_args.get("pad", "auto")
        final_verbose = verbose if verbose is not None else config_args.get("verbose")

        # Check if resampling is needed
        current_sfreq = data.info["sfreq"]
        if abs(current_sfreq - target_sfreq) < 0.01:
            message(
                "info",
                f"Data already at target frequency ({target_sfreq} Hz), skipping resampling",
            )
            return data

        message(
            "header", f"Resampling data from {current_sfreq} Hz to {target_sfreq} Hz..."
        )

        # Call standalone function
        resampled_data = standalone_resample_data(
            data=data,
            sfreq=target_sfreq,
            npad=final_npad,
            window=final_window,
            n_jobs=final_n_jobs,
            pad=final_pad,
            verbose=final_verbose,
        )

        message("info", f"Data successfully resampled to {target_sfreq} Hz")

        # Pipeline integration with result-based metadata
        self._update_instance_data(data, resampled_data, use_epochs)
        if export:
            self._save_raw_result(resampled_data, stage_name)

        # Use actual results in metadata
        metadata = {
            "original_sfreq": current_sfreq,
            "target_sfreq": target_sfreq,
            "actual_sfreq": resampled_data.info["sfreq"],
            "original_n_samples": (
                data.get_data().shape[1]
                if hasattr(data, "get_data")
                else len(data.times)
            ),
            "resampled_n_samples": (
                resampled_data.get_data().shape[1]
                if hasattr(resampled_data, "get_data")
                else len(resampled_data.times)
            ),
            "npad": final_npad,
            "window": final_window,
            "n_jobs": final_n_jobs,
            "pad": final_pad,
            "original_data_type": type(data).__name__,
            "result_data_type": type(resampled_data).__name__,
        }
        self._update_metadata("step_resample_data", metadata)

        return resampled_data

    def rereference_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        ref_type: str = None,
        use_epochs: bool = False,
        export: bool = True,
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

        message("header", "Rereferencing data...")

        data = self._get_data_object(data, use_epochs)

        if not isinstance(
            data, (mne.io.base.BaseRaw, mne.Epochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
            raise TypeError("Data must be an MNE Raw or Epochs object")

        if ref_type is None:
            is_enabled, config_value = self._check_step_enabled("reference_step")

            if not is_enabled:
                message("info", "Rereferencing step is disabled in configuration")
                return data

            ref_type = config_value.get("value", None)

            if ref_type is None:
                message(
                    "warning",
                    "Rereferencing value not specified, skipping rereferencing",
                )
                return data

        # Call standalone function
        rereferenced_data = standalone_rereference_data(
            data=data,
            ref_channels=ref_type,
            projection=False if ref_type == "average" else True,
            verbose=False,
        )

        # Pipeline integration
        self._update_instance_data(data, rereferenced_data, use_epochs)
        if export:
            self._save_raw_result(rereferenced_data, stage_name)

        metadata = {
            "new_ref_type": ref_type,
        }
        self._update_metadata("step_rereference_data", metadata)

        return rereferenced_data

    def drop_outer_layer(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        stage_name: str = "post_outerlayer",
        export: bool = True,
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

        if not isinstance(
            data, (mne.io.base.BaseRaw, mne.Epochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
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
                "info",
                "Specified outer layer channels not found in data, skipping drop.",
            )
            return data

        message(
            "header", f"Dropping outer layer channels: {', '.join(channels_to_drop)}"
        )
        processed_data = data.copy().drop_channels(channels_to_drop)
        message("info", f"Channels dropped: {', '.join(channels_to_drop)}")

        if export:
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

        if not isinstance(
            data, (mne.io.base.BaseRaw, mne.Epochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
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
        ]  # Handling both indices (1-based) and names

        eog_channels_map = {
            ch: "eog" for ch in eog_channels_to_set if ch in data.ch_names
        }

        if not eog_channels_map:
            message(
                "warning", "Specified EOG channels not found in data, skipping step."
            )
            return data

        message(
            "header",
            f"Assigning EOG channel types for: {', '.join(eog_channels_map.keys())}",
        )
        # Process a copy to avoid modifying the original data object directly
        processed_data = data.copy()
        processed_data.set_channel_types(eog_channels_map)
        message(
            "info",
            f"EOG channel types assigned for: {', '.join(eog_channels_map.keys())}",
        )

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

        if not isinstance(
            data, (mne.io.base.BaseRaw, mne.Epochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
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
            return data  # Return original data to avoid erroring out pipeline

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

        if not isinstance(
            data, (mne.io.base.BaseRaw, mne.Epochs)
        ):  # pylint: disable=isinstance-second-argument-not-valid-type
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

        message(
            "header", f"Cropping data duration to range: {tmin:.3f}s to {tmax:.3f}s"
        )
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
