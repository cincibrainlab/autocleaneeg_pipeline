"""ICA mixin for autoclean tasks."""

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
        classify_ica_components : Classify ICA components with ICLabel or ICVision.

        """
        message("header", "Running ICA step")

        is_enabled, config_value = self._check_step_enabled("ICA")

        if not is_enabled:
            message("warning", "ICA is not enabled in the config")
            return

        data = self._get_data_object(data=None, use_epochs=use_epochs)

        # Run ICA using standalone function
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

            message("debug", f"Fitting ICA with {ica_kwargs}")

            # Call standalone function for ICA fitting
            from autoclean.functions.ica.ica_processing import fit_ica

            self.final_ica = fit_ica(raw=data, **ica_kwargs)

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

    def classify_ica_components(
        self,
        method: str | None = None,
        reject: bool = True,
        psd_fmax: float | None = None,
        stage_name: str = "post_ica",
        export: bool = False,
    ):
        """Classify ICA components and optionally reject artifact components.

        This method classifies ICA components using either ICLabel or ICVision methods
        and can automatically reject components identified as artifacts.

        Parameters
        ----------
        method : str or None, optional
            Classification method to use. Options: "iclabel", "icvision".
            If None, will try to read method from component_rejection config,
            falling back to "iclabel" if not found.
        reject : bool, default True
            If True, automatically reject components identified as artifacts.
        psd_fmax : float or None, optional
            Upper frequency limit (Hz) for PSD plots when using icvision method.
            If None, uses default (80 Hz or Nyquist frequency).
        stage_name : str, optional
            Name of the processing stage for export. Default is "post_component_removal".
        export : bool, optional
            If True, exports the processed data to the stage directory. Default is False.

        Returns
        -------
        ica_flags : pandas.DataFrame or None
            A pandas DataFrame containing the classification results, or None if the
            step fails.

        Examples
        --------
        >>> # Classify with ICLabel and auto-reject
        >>> self.classify_ica_components(method="iclabel", reject=True)
        >>> # Classify with ICVision without rejection
        >>> self.classify_ica_components(method="icvision", reject=False)
        >>> # Classify with ICVision and limit PSD plots to 40 Hz
        >>> self.classify_ica_components(method="icvision", reject=True, psd_fmax=40.0)
        >>> # Read psd_fmax from config
        >>> psd_fmax = self.config.get("ICLabel", {}).get("psd_fmax")
        >>> self.classify_ica_components(method="icvision", reject=True, psd_fmax=psd_fmax)

        Notes
        -----
        This method will modify the self.final_ica attribute in place by adding labels.
        If reject=True, it will also apply component rejection.
        """
        # Auto-detect method from config if not specified
        if method is None:
            is_enabled, step_config = self._check_step_enabled("component_rejection")
            if is_enabled:
                config_value = step_config.get("value", {})
                method = config_value.get("method", "iclabel")
                message(
                    "info",
                    f"Auto-detected method from component_rejection config: {method}",
                )
            else:
                method = "iclabel"
                message(
                    "info",
                    f"No component rejection config found, defaulting to: {method}",
                )

        message("header", f"Running ICA component classification with {method}")

        if not hasattr(self, "final_ica") or self.final_ica is None:
            message(
                "error",
                "ICA (self.final_ica) not found. Please run `run_ica` before `classify_ica_components`.",
            )
            return None

        # Call standalone function for ICA component classification
        from autoclean.functions.ica.ica_processing import classify_ica_components

        # If psd_fmax not explicitly provided, try to get it from config
        if psd_fmax is None:
            is_enabled, step_config_main_dict = self._check_step_enabled(
                "component_rejection"
            )

            if is_enabled and step_config_main_dict:
                # Check nested value dict first (common pattern)
                config_params_nested = step_config_main_dict.get("value", {})
                psd_fmax = config_params_nested.get("psd_fmax")

                # If not found in nested, check main dict
                if psd_fmax is None and "psd_fmax" in step_config_main_dict:
                    psd_fmax = step_config_main_dict.get("psd_fmax")

                if psd_fmax is not None:
                    message("info", f"Using psd_fmax={psd_fmax} Hz from config")

        # Build kwargs dict, only including psd_fmax if provided
        extra_kwargs = {}
        if psd_fmax is not None:
            extra_kwargs["psd_fmax"] = psd_fmax

        if method == "icvision":
            extra_kwargs["generate_report"] = False
            extra_kwargs["output_dir"] = self.config.get("derivatives_dir", {})

        self.ica_flags = classify_ica_components(
            self.raw, self.final_ica, method=method, **extra_kwargs
        )

        metadata = {
            "ica": {
                "classification_method": method,
                "ica_components": self.final_ica.n_components_,
            }
        }

        self._update_metadata("classify_ica_components", metadata)

        message("success", f"ICA component classification with {method} complete")

        # Apply rejection if requested
        if reject:
            self.apply_ica_component_rejection()

        # Export if requested
        self._auto_export_if_enabled(self.raw, stage_name, export)

        return self.ica_flags

    def apply_ica_component_rejection(self, data_to_clean=None):
        """
        Apply ICA component rejection based on component classifications and configuration.

        This method uses the labels assigned by `classify_ica_components` and the rejection
        criteria specified in the 'component_rejection' section of the pipeline configuration
        (e.g., ic_flags_to_reject, ic_rejection_threshold) to mark components
        for rejection. It then applies the ICA to remove these components from
        the data.

        It updates `self.final_ica.exclude` and modifies the data object
        (e.g., `self.raw`) in-place. The updated ICA object is also saved.

        Parameters
        ----------
        data_to_clean : mne.io.Raw | mne.Epochs, optional
            The data to apply the ICA to. If None, defaults to `self.raw`.
            This should ideally be the same data object that classification was
            performed on, or is compatible with `self.final_ica`.

        Returns
        -------
        None
            Modifies `self.final_ica` and the input data object in-place.

        Raises
        ------
        RuntimeError
            If `self.final_ica` or `self.ica_flags` are not available (i.e.,
            `run_ica` and `classify_ica_components` have not been run successfully).
        """
        message("header", "Applying ICA component rejection")

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
                "ICA results (self.ica_flags) not found. Skipping component rejection.",
            )
            raise RuntimeError(
                "ICA results (self.ica_flags) not found. Please run `classify_ica_components` first."
            )

        # Get component_rejection config
        is_enabled, step_config_main_dict = self._check_step_enabled(
            "component_rejection"
        )
        config_source = "component_rejection"

        if not is_enabled:
            message(
                "warning",
                "component_rejection config not enabled. "
                "Rejection parameters might be missing. Skipping.",
            )
            return

        # Attempt to get parameters from a nested "value" dictionary first (common pattern)
        config_params_nested = step_config_main_dict.get("value", {})

        flags_to_reject = config_params_nested.get("ic_flags_to_reject")
        rejection_threshold = config_params_nested.get("ic_rejection_threshold")
        threshold_overrides = config_params_nested.get("ic_rejection_overrides", {})

        # If not found in "value", try to get them from the main step config dict directly
        if flags_to_reject is None and "ic_flags_to_reject" in step_config_main_dict:
            flags_to_reject = step_config_main_dict.get("ic_flags_to_reject")
        if (
            rejection_threshold is None
            and "ic_rejection_threshold" in step_config_main_dict
        ):
            rejection_threshold = step_config_main_dict.get("ic_rejection_threshold")
        if (
            not threshold_overrides
            and "ic_rejection_overrides" in step_config_main_dict
        ):
            threshold_overrides = step_config_main_dict.get(
                "ic_rejection_overrides", {}
            )

        if flags_to_reject is None or rejection_threshold is None:
            message(
                "warning",
                f"ICA rejection parameters (ic_flags_to_reject or ic_rejection_threshold) "
                f"not found in the '{config_source}' step configuration. Skipping component rejection.",
            )
            return

        # Warn about unused overrides
        if threshold_overrides:
            unused_overrides = set(threshold_overrides.keys()) - set(flags_to_reject)
            if unused_overrides:
                message(
                    "warning",
                    f"Threshold overrides specified for types not in rejection list: {unused_overrides}",
                )

            # Show per-type thresholds when overrides are present
            threshold_info = {
                ic_type: rejection_threshold for ic_type in flags_to_reject
            }
            threshold_info.update(threshold_overrides)
            message(
                "info", f"Will reject ICs with per-type thresholds: {threshold_info}"
            )
        else:
            message(
                "info",
                f"Will reject ICs of types: {flags_to_reject} with confidence > {rejection_threshold}",
            )

        # Determine data to clean
        target_data = data_to_clean if data_to_clean is not None else self.raw
        data_source_name = (
            "provided data object" if data_to_clean is not None else "self.raw"
        )
        message("debug", f"Applying ICA to {data_source_name}")

        # Call standalone function for component rejection
        from autoclean.functions.ica.ica_processing import apply_ica_component_rejection

        _, rejected_ic_indices_this_step = apply_ica_component_rejection(
            raw=target_data,
            ica=self.final_ica,
            labels_df=self.ica_flags,
            ic_flags_to_reject=flags_to_reject,
            ic_rejection_threshold=rejection_threshold,
            ic_rejection_overrides=threshold_overrides,
            verbose=True,
        )

        if not rejected_ic_indices_this_step:
            message("info", "No new components met rejection criteria in this step.")
        else:
            message(
                "info",
                f"Identified {len(rejected_ic_indices_this_step)} components for rejection: "
                f"{rejected_ic_indices_this_step}",
            )

        message(
            "info",
            f"Total components now marked for exclusion: {self.final_ica.exclude}",
        )

        # Update metadata
        metadata = {
            "ica": {
                "configured_flags_to_reject": flags_to_reject,
                "configured_rejection_threshold": rejection_threshold,
                "configured_threshold_overrides": threshold_overrides,
                "rejected_indices_this_step": rejected_ic_indices_this_step,
                "final_excluded_indices": self.final_ica.exclude,
            }
        }
        # Assuming _update_metadata is available in the class using this mixin
        if hasattr(self, "_update_metadata") and callable(self._update_metadata):
            self._update_metadata("step_apply_ica_component_rejection", metadata)
        else:
            message(
                "warning",
                "_update_metadata method not found. Cannot save metadata for component rejection.",
            )

        message("success", "ICA component rejection complete.")
