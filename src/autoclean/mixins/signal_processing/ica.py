"""ICA mixin for autoclean tasks."""

import mne_icalabel
import pandas as pd
from mne.preprocessing import ICA

from autoclean.io.export import save_ica_to_fif
from autoclean.utils.logging import message

class IcaMixin:
    """Mixin for ICA processing."""

    def run_ica(
        self,
        eog_channel: str = None,
        use_epochs: bool = False,
        stage_name: str = "post_ica",
        **kwargs,
    ) -> ICA:
        """Run ICA on the raw data.

        This method will fit an ICA object to the raw data and save it to a FIF file.
        ICA object is stored in self.final_ica.
        Uses optional kwargs from the autoclean_config file to fit the mne ICA object.

        Parameters
        ----------
        eog_channel : str, optional
            The EOG channel to use for ICA. If None, no EOG detection will be performed.
        use_epochs : bool, optional
            If True, epoch data stored in self.epochs will be used.
        stage_name : str, optional
            Name of the processing stage for export. Default is "post_ica".
        export : bool, optional
            If True, exports the processed data to the stage directory. Default is False.

        Returns
        -------
        final_ica : mne.preprocessing.ICA
            The fitted ICA object.

        Examples
        --------
        >>> self.run_ica()
        >>> self.run_ica(eog_channel="E27", export=True)

        See Also
        --------
        run_ICLabel : Run ICLabel on the raw data.

        """
        message("header", "Running ICA step")

        is_enabled, config_value = self._check_step_enabled("ICA")

        if not is_enabled:
            message("warning", "ICA is not enabled in the config")
            return

        data = self._get_data_object(data=None, use_epochs=use_epochs)

        # Run ICA
        if is_enabled:
            # Get ICA parameters from config
            ica_kwargs = config_value.get("value", {})

            # Merge with any provided kwargs, with provided kwargs taking precedence
            ica_kwargs.update(kwargs)

            # Set default parameters if not provided
            if "max_iter" not in ica_kwargs:
                message("debug", "Setting max_iter to auto")
                ica_kwargs["max_iter"] = "auto"
            if "random_state" not in ica_kwargs:
                message("debug", "Setting random_state to 97")
                ica_kwargs["random_state"] = 97

            # Create ICA object

            self.final_ica = ICA(**ica_kwargs)  # pylint: disable=not-callable

            message("debug", f"Fitting ICA with {ica_kwargs}")

            self.final_ica.fit(data)

            if eog_channel is not None:
                message("info", f"Running EOG detection on {eog_channel}")
                eog_indices, _ = self.final_ica.find_bads_eog(data, ch_name=eog_channel)
                self.final_ica.exclude = eog_indices
                self.final_ica.apply(data)

        else:
            message("warning", "ICA is not enabled in the config")

        metadata = {
            "ica": {
                "ica_kwargs": ica_kwargs,
                "ica_components": self.final_ica.n_components_,
            }
        }

        self._update_metadata("step_run_ica", metadata)

        save_ica_to_fif(self.final_ica, self.config, data)

        message("success", "ICA step complete")

        return self.final_ica

    def run_ICLabel(
        self, stage_name: str = "post_component_removal", export: bool = False
    ):  # pylint: disable=invalid-name
        """Run ICLabel on the raw data.

        Returns
        -------
        ica_flags : pandas.DataFrame or None
            A pandas DataFrame containing the ICLabel flags, or None if the
            step is disabled or fails.

        Examples
        --------
        >>> self.run_ICLabel()

        Notes
        -----
        This method will modify the self.final_ica attribute in place by adding labels.
        It checks if the 'ICLabel' step is enabled in the configuration.
        """
        message("header", "Running ICLabel step")

        is_enabled, _ = self._check_step_enabled(
            "ICLabel"
        )  # config_value not used here

        if not is_enabled:
            message(
                "warning", "ICLabel is not enabled in the config. Skipping ICLabel."
            )
            return None  # Return None if not enabled

        if not hasattr(self, "final_ica") or self.final_ica is None:
            message(
                "error",
                "ICA (self.final_ica) not found. Please run `run_ica` before `run_ICLabel`.",
            )
            # Or raise an error, depending on desired behavior
            return None

        mne_icalabel.label_components(self.raw, self.final_ica, method="iclabel")

        self._icalabel_to_data_frame(self.final_ica)

        metadata = {
            "ica": {
                "ica_components": self.final_ica.n_components_,
            }
        }

        self._update_metadata("step_run_ICLabel", metadata)

        message("success", "ICLabel complete")

        self.apply_iclabel_rejection()

        # Export if requested
        self._auto_export_if_enabled(self.raw, stage_name, export)

        return self.ica_flags

    def apply_iclabel_rejection(self, data_to_clean=None):
        """
        Apply ICA component rejection based on ICLabel classifications and configuration.

        This method uses the labels assigned by `run_ICLabel` and the rejection
        criteria specified in the 'ICLabel' section of the pipeline configuration
        (e.g., ic_flags_to_reject, ic_rejection_threshold) to mark components
        for rejection. It then applies the ICA to remove these components from
        the data.

        It updates `self.final_ica.exclude` and modifies the data object
        (e.g., `self.raw`) in-place. The updated ICA object is also saved.

        Parameters
        ----------
        data_to_clean : mne.io.Raw | mne.Epochs, optional
            The data to apply the ICA to. If None, defaults to `self.raw`.
            This should ideally be the same data object that `run_ICLabel` was
            performed on, or is compatible with `self.final_ica`.

        Returns
        -------
        None
            Modifies `self.final_ica` and the input data object in-place.

        Raises
        ------
        RuntimeError
            If `self.final_ica` or `self.ica_flags` are not available (i.e.,
            `run_ica` and `run_ICLabel` have not been run successfully).
        """
        message("header", "Applying ICLabel-based component rejection")

        if not hasattr(self, "final_ica") or self.final_ica is None:
            message(
                "error", "ICA (self.final_ica) not found. Skipping ICLabel rejection."
            )
            raise RuntimeError(
                "ICA (self.final_ica) not found. Please run `run_ica` first."
            )

        if not hasattr(self, "ica_flags") or self.ica_flags is None:
            message(
                "error",
                "ICA results (self.ica_flags) not found. Skipping ICLabel rejection.",
            )
            raise RuntimeError(
                "ICA results (self.ica_flags) not found. Please run `run_ICLabel` first."
            )

        is_enabled, step_config_main_dict = self._check_step_enabled("ICLabel")
        if not is_enabled:
            message(
                "warning",
                "ICLabel processing itself is not enabled in the config. "
                "Rejection parameters might be missing or irrelevant. Skipping.",
            )
            return

        # Attempt to get parameters from a nested "value" dictionary first (common pattern)
        iclabel_params_nested = step_config_main_dict.get("value", {})

        flags_to_reject = iclabel_params_nested.get("ic_flags_to_reject")
        rejection_threshold = iclabel_params_nested.get("ic_rejection_threshold")

        # If not found in "value", try to get them from the main step config dict directly
        if flags_to_reject is None and "ic_flags_to_reject" in step_config_main_dict:
            flags_to_reject = step_config_main_dict.get("ic_flags_to_reject")
        if (
            rejection_threshold is None
            and "ic_rejection_threshold" in step_config_main_dict
        ):
            rejection_threshold = step_config_main_dict.get("ic_rejection_threshold")

        if flags_to_reject is None or rejection_threshold is None:
            message(
                "warning",
                "ICLabel rejection parameters (ic_flags_to_reject or ic_rejection_threshold) "
                "not found in the 'ICLabel' step configuration. Skipping component rejection.",
            )
            return

        message(
            "info",
            f"Will reject ICs of types: {flags_to_reject} with confidence > {rejection_threshold}",
        )

        rejected_ic_indices_this_step = []
        for (
            idx,
            row,
        ) in self.ica_flags.iterrows():  # DataFrame index is the component index
            if (
                row["ic_type"] in flags_to_reject
                and row["confidence"] > rejection_threshold
            ):
                rejected_ic_indices_this_step.append(idx)

        if not rejected_ic_indices_this_step:
            message(
                "info", "No new components met ICLabel rejection criteria in this step."
            )
        else:
            message(
                "info",
                f"Identified {len(rejected_ic_indices_this_step)} components for rejection "
                f"based on ICLabel: {rejected_ic_indices_this_step}",
            )

        # Ensure self.final_ica.exclude is initialized as a list if it's None
        if self.final_ica.exclude is None:
            self.final_ica.exclude = []

        # Combine with any existing exclusions (e.g., from EOG detection in run_ica)
        current_exclusions = set(self.final_ica.exclude)
        for idx in rejected_ic_indices_this_step:
            current_exclusions.add(idx)
        self.final_ica.exclude = sorted(list(current_exclusions))

        message(
            "info",
            f"Total components now marked for exclusion: {self.final_ica.exclude}",
        )

        # Determine data to clean
        target_data = data_to_clean if data_to_clean is not None else self.raw
        data_source_name = (
            "provided data object" if data_to_clean is not None else "self.raw"
        )
        message("debug", f"Applying ICA to {data_source_name}")

        if not self.final_ica.exclude:
            message(
                "info", "No components are marked for exclusion. Skipping ICA apply."
            )
        else:
            # Apply ICA to remove the excluded components
            # This modifies target_data in-place
            self.final_ica.apply(target_data)
            message(
                "info",
                f"Applied ICA to {data_source_name}, removing/attenuating "
                f"{len(self.final_ica.exclude)} components.",
            )

        # Update metadata
        metadata = {
            "step_apply_iclabel_rejection": {
                "configured_flags_to_reject": flags_to_reject,
                "configured_rejection_threshold": rejection_threshold,
                "iclabel_rejected_indices_this_step": rejected_ic_indices_this_step,
                "final_excluded_indices_after_iclabel": self.final_ica.exclude,
            }
        }
        # Assuming _update_metadata is available in the class using this mixin
        if hasattr(self, "_update_metadata") and callable(self._update_metadata):
            self._update_metadata("step_apply_iclabel_rejection", metadata)
        else:
            message(
                "warning",
                "_update_metadata method not found. Cannot save metadata for ICLabel rejection.",
            )

        message("success", "ICLabel-based component rejection complete.")

    def _icalabel_to_data_frame(self, ica):
        """Export IClabels to pandas DataFrame."""
        ic_type = [""] * ica.n_components_
        for label, comps in ica.labels_.items():
            for comp in comps:
                ic_type[comp] = label

        self.ica_flags = pd.DataFrame(
            dict(
                component=ica._ica_names,  # pylint: disable=protected-access
                annotator=["ic_label"] * ica.n_components_,
                ic_type=ic_type,
                confidence=ica.labels_scores_.max(1),
            )
        )

        return self.ica_flags