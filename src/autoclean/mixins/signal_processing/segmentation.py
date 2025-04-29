"""Segmentation operations mixin for autoclean tasks."""

from typing import Optional, Union

import mne

from autoclean.utils.logging import message


class SegmentationMixin:
    """Mixin class providing segmentation operations functionality for EEG data."""

    def crop_data(
        self,
        data: Union[mne.io.Raw, mne.Epochs, None] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        stage_name: str = "crop",
        use_epochs: bool = False,
    ) -> Union[mne.io.Raw, mne.Epochs]:
        """Crop data to a specific time range.

        This method crops the data to a specific time range.

        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs, Optional
            Optional MNE Raw or Epochs object. If None, uses self.raw or self.epochs
        tmin : float, Optional
            Start time in seconds
        tmax : float, Optional
            End time in seconds
        stage_name : str, Optional
            Name for saving and metadata
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw

        Returns:
            inst : mne.io.Raw or mne.Epochs
            The cropped data object

        See Also
        --------
        :py:meth:`mne.io.Raw.crop` : For MNE's raw data cropping functionality
        :py:meth:`mne.Epochs.crop` : For MNE's epochs cropping functionality
        """
        # Check if tmin and tmax are provided
        if tmin is None or tmax is None:
            is_enabled, config_value = self._check_step_enabled("crop_step")

            if not is_enabled:
                message("info", "Crop step is disabled in configuration")
                return data

            # Get crop values from config
            if config_value:
                if isinstance(config_value, dict):
                    tmin = config_value.get("tmin", tmin)
                    tmax = config_value.get("tmax", tmax)
                elif isinstance(config_value, list) and len(config_value) == 2:
                    tmin, tmax = config_value

            if tmin is None or tmax is None:
                message("warning", "Crop times not specified in config")
                return data

        # Determine which data to use
        data = self._get_data_object(data, use_epochs)

        # Type checking
        if not isinstance(data, (mne.io.Raw, mne.Epochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object for cropping")

        try:
            # Crop data
            message("header", f"Cropping data from {tmin}s to {tmax}s...")
            result_data = data.copy().crop(tmin=tmin, tmax=tmax)
            message("info", f"Data cropped to {tmin}s - {tmax}s")

            # Update metadata
            metadata = {"tmin": tmin, "tmax": tmax, "duration": tmax - tmin}

            self._update_metadata("step_crop_data", metadata)

            # Save the result if it's a Raw object
            if isinstance(result_data, mne.io.Raw):
                self._save_raw_result(result_data, stage_name)

            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)

            return result_data

        except Exception as e:
            message("error", f"Error during cropping: {str(e)}")
            raise RuntimeError(f"Failed to crop data: {str(e)}") from e

    def trim_data_edges(
        self,
        data: Union[mne.io.Raw, None] = None,
        trim_amount: float = 1.0,
        stage_name: str = "trim",
    ) -> mne.io.Raw:
        """Trim a specified amount of time from the beginning and end of the data.

        This method trims the specified amount of time from both the beginning and end of the data.

        Parameters
        ----------
        data : mne.io.Raw, Optional
            Optional MNE Raw object. If None, uses self.raw
        trim_amount : float, Optional
            Amount of time in seconds to trim from both beginning and end
        stage_name : str, Optional
            Name for saving and metadata

        Returns:
            inst : mne.io.Raw
            The trimmed raw data object

        See Also
        --------
        :py:meth:`mne.io.Raw.crop` : For MNE's raw data cropping functionality
        """
        # Check if trim_amount is provided via config
        is_enabled, config_value = self._check_step_enabled("trim_step")

        if not is_enabled:
            message("info", "Trim step is disabled in configuration")
            return data

        # Get trim amount from config if available
        if config_value is not None:
            trim_amount = config_value

        # Determine which data to use
        data = self._get_data_object(data)

        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(
            data, mne.io.base.BaseRaw
        ):
            raise TypeError("Data must be an MNE Raw object for trimming")

        try:
            # Get data duration
            duration = data.times[-1]

            # Check if there's enough data to trim
            if duration <= 2 * trim_amount:
                message(
                    "warning",
                    f"Data duration ({duration}s) is too short to trim {trim_amount}s from both ends",
                )
                return data

            # Trim data
            message("header", f"Trimming {trim_amount}s from both ends of data...")
            result_raw = data.copy().crop(tmin=trim_amount, tmax=duration - trim_amount)
            message(
                "info", f"Data trimmed to {trim_amount}s - {duration - trim_amount}s"
            )

            # Update metadata
            metadata = {
                "trim_amount": trim_amount,
                "original_duration": duration,
                "trimmed_duration": duration - 2 * trim_amount,
            }

            self._update_metadata("step_trim_data_edges", metadata)

            # Save the result
            self._save_raw_result(result_raw, stage_name)

            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)

            return result_raw

        except Exception as e:
            message("error", f"Error during trimming: {str(e)}")
            raise RuntimeError(f"Failed to trim data: {str(e)}") from e
