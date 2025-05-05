"""Resampling mixin for autoclean tasks."""

from typing import Union

import mne

from autoclean.utils.logging import message


class ResamplingMixin:
    """Mixin class providing resampling functionality for EEG data."""

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
            if isinstance(data, mne.io.Raw) or isinstance(data, mne.io.base.BaseRaw):
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

            self._update_metadata("resample_data", metadata)

            # Update self.raw or self.epochs if we're using those
            self._update_instance_data(data, resampled_data, use_epochs)

            return resampled_data

        except Exception as e:
            message("error", f"Error during resampling: {str(e)}")
            raise RuntimeError(f"Failed to resample data: {str(e)}") from e
