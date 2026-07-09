"""ICA mixin for autoclean tasks."""

from typing import Dict

import mne
from mne.preprocessing import ICA

from autoclean.functions.ica.ica_processing import (
    apply_ica_component_rejection,
    classify_ica_components,
    fit_ica,
    normalize_ic_type,
    update_ica_control_sheet,
)
from autoclean.io.export import save_ica_to_fif
from autoclean.utils.logging import message

# Import cache invalidation function if available
try:
    from autoclean.functions.visualization._ica_sources_cache import (
        invalidate_ica_cache,
    )

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


class IcaMixin:
    """Mixin for ICA processing."""

    def _get_ica_fit_data(self):
        """Return the data object ICA was fit on (self.raw or self.epochs).

        Tracks the ``use_epochs`` flag passed to the most recent `run_ica`
        call so that `classify_ica_components` and
        `apply_ica_component_rejection` operate on the same data type ICA was
        actually fit on, instead of assuming `self.raw`. Falls back to
        whichever of `self.raw` / `self.epochs` is available if the
        fit-time preference is no longer set.
        """
        use_epochs = getattr(self, "_ica_used_epochs", False)
        attrs = ("epochs", "raw") if use_epochs else ("raw", "epochs")

        for i, attr in enumerate(attrs):
            obj = getattr(self, attr, None)
            if obj is not None:
                if i > 0:
                    message(
                        "warning",
                        f"ICA was fit with use_epochs={use_epochs}, but self.{attrs[0]} "
                        f"is unavailable; falling back to self.{attr}, which may not "
                        "be the data ICA was actually fit on.",
                    )
                return obj

        raise AttributeError(
            "No data object (self.raw or self.epochs) available for ICA "
            "operations. Please run `run_ica` first."
        )

    def run_ica(
        self,
        eog_channel: str = None,
        use_epochs: bool = False,
        stage_name: str = "ica_fit",
        temp_highpass_for_ica: float = None,
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
            Name of the processing stage for export. Default is "ica_fit" (ICA fitted, not yet applied).
        temp_highpass_for_ica : float, optional
            Temporary high-pass filter frequency (Hz) to apply only for ICA decomposition.
            Commonly set to 1.0 Hz for better ICA performance by reducing low-frequency
            drifts and artifacts. The ICA object fitted on filtered data can be directly
            applied to the original unfiltered data via ica.apply(). If None, uses data as-is.
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
        self._ica_used_epochs = use_epochs

        # Run ICA using standalone function
        if is_enabled:
            # Get ICA parameters from config
            ica_kwargs = config_value.get("value") or {}

            # Check for temp_highpass_for_ica in config if not provided
            if temp_highpass_for_ica is None:
                temp_highpass_for_ica = ica_kwargs.pop("temp_highpass_for_ica", None)

            # Merge with any provided kwargs, with provided kwargs taking precedence
            ica_kwargs.update(kwargs)

            # Set default parameters if not provided
            if "max_iter" not in ica_kwargs:
                message("debug", "Setting max_iter to auto")
                ica_kwargs["max_iter"] = "auto"
            if "random_state" not in ica_kwargs:
                message("debug", "Setting random_state to 97")
                ica_kwargs["random_state"] = 97

            # Prepare data for ICA fitting - always copy to avoid modifying original
            data_for_ica = data.copy()
            if temp_highpass_for_ica is not None:
                message(
                    "info",
                    f"Applying temporary {temp_highpass_for_ica} Hz high-pass filter for ICA decomposition",
                )
                data_for_ica.filter(
                    l_freq=temp_highpass_for_ica, h_freq=None, verbose=False
                )

            message("info", f"Fitting ICA with {ica_kwargs}")

            # Call standalone function for ICA fitting on (potentially filtered) data
            self.final_ica = fit_ica(raw=data_for_ica, **ica_kwargs)

            # No refit or matrix manipulation needed - MNE handles applying ICA
            # fitted on filtered data to original data seamlessly via ica.apply()

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
                "temp_highpass_for_ica": temp_highpass_for_ica,
                "ica_fit_data_type": "epochs" if use_epochs else "raw",
            }
        }

        self._update_metadata("step_run_ica", metadata)

        save_ica_to_fif(self.final_ica, self.config, data)

        # Invalidate any cached sources since ICA object has changed
        if (
            CACHE_AVAILABLE
            and hasattr(self, "final_ica")
            and self.final_ica is not None
        ):
            invalidate_ica_cache(self.final_ica)

        message("success", "ICA step complete")

        return self.final_ica

    def classify_ica_components(
        self,
        method: str | None = None,
        reject: bool = True,
        psd_fmax: float | None = None,
        icvision_n_components: int | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        stage_name: str = "ica_clean",
        export: bool = False,
    ):
        """Classify ICA components and optionally reject artifact components.

        This method classifies ICA components using ICLabel, ICVision, or a
        hybrid of both, and can automatically reject components identified as
        artifacts.

        Parameters
        ----------
        method : str or None, optional
            Classification method to use. Options: "iclabel", "icvision",
            "hybrid". If None, will try to read method from
            component_rejection config, falling back to "iclabel" if not
            found.
        reject : bool, default True
            If True, automatically reject components identified as artifacts.
        psd_fmax : float or None, optional
            Upper frequency limit (Hz) for PSD plots when using icvision method.
            If None, uses default (80 Hz or Nyquist frequency).
        icvision_n_components : int or None, optional
            Number of leading components to reclassify with ICVision when using
            the "hybrid" method. If None, uses config value or defaults to 20.
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
        >>> # Hybrid: ICLabel followed by ICVision on first 10 components
        >>> self.classify_ica_components(method="hybrid", icvision_n_components=10)
        >>> # Read psd_fmax from config
        >>> psd_fmax = self.config.get("ICLabel", {}).get("psd_fmax")
        >>> self.classify_ica_components(method="icvision", reject=True, psd_fmax=psd_fmax)

        Notes
        -----
        This method will modify the self.final_ica attribute in place by adding labels.
        If reject=True, it will also apply component rejection.
        """

        # Check if ICA is enabled in the configuration
        is_ica_enabled, _ = self._check_step_enabled("ICA")
        if not is_ica_enabled:
            message(
                "warning",
                "ICA is not enabled in the config. Skipping both ICA and component classification.",
            )
            return None

        # Auto-detect method from config if not specified
        if method is None:
            is_enabled, step_config = self._check_step_enabled("component_rejection")
            if is_enabled:
                method = step_config.get(
                    "method", "iclabel"
                )  # Check 'method' directly in step_config
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
        # If psd_fmax or icvision_n_components not explicitly provided, try to get them from config
        if psd_fmax is None or icvision_n_components is None:
            is_enabled, step_config_main_dict = self._check_step_enabled(
                "component_rejection"
            )

            if is_enabled and step_config_main_dict:
                # Check nested value dict first (common pattern)
                config_params_nested = step_config_main_dict.get("value") or {}

                if psd_fmax is None:
                    psd_fmax = config_params_nested.get("psd_fmax")
                    if psd_fmax is None and "psd_fmax" in step_config_main_dict:
                        psd_fmax = step_config_main_dict.get("psd_fmax")
                    if psd_fmax is not None:
                        message("info", f"Using psd_fmax={psd_fmax} Hz from config")

                if icvision_n_components is None:
                    icvision_n_components = config_params_nested.get(
                        "icvision_n_components"
                    )
                    if (
                        icvision_n_components is None
                        and "icvision_n_components" in step_config_main_dict
                    ):
                        icvision_n_components = step_config_main_dict.get(
                            "icvision_n_components"
                        )
                    if icvision_n_components is not None:
                        message(
                            "info",
                            f"Using icvision_n_components={icvision_n_components} from config",
                        )

                if base_url is None:
                    base_url = config_params_nested.get("base_url")
                    if base_url is None and "base_url" in step_config_main_dict:
                        base_url = step_config_main_dict.get("base_url")
                    if base_url is not None:
                        message("info", "Using base_url from config")

                if model_name is None:
                    model_name = config_params_nested.get("model_name")
                    if model_name is None and "model_name" in step_config_main_dict:
                        model_name = step_config_main_dict.get("model_name")
                    if model_name is not None:
                        message("info", f"Using model_name={model_name} from config")

        # Build kwargs dict, including optional parameters if provided
        extra_kwargs: Dict[str, object] = {}
        if psd_fmax is not None:
            extra_kwargs["psd_fmax"] = psd_fmax
        if icvision_n_components is not None:
            extra_kwargs["icvision_n_components"] = icvision_n_components
        if base_url is not None:
            extra_kwargs["base_url"] = base_url
        if model_name is not None:
            extra_kwargs["model_name"] = model_name

        if method in {"icvision", "hybrid"}:
            extra_kwargs["generate_report"] = True
            extra_kwargs["output_dir"] = self.config.get("derivatives_dir", {})

        fit_data = self._get_ica_fit_data()

        self.ica_flags = classify_ica_components(
            fit_data, self.final_ica, method=method, **extra_kwargs
        )

        vision_attr = None
        if hasattr(self.ica_flags, "attrs"):
            vision_attr = self.ica_flags.attrs.get("icvision_df")

        if vision_attr is not None:
            self.ica_vision_flags = vision_attr.copy()
        elif method == "icvision":
            self.ica_vision_flags = self.ica_flags.copy()
        else:
            self.ica_vision_flags = None

        # Persist psd_fmax for downstream plotting/reporting paths
        setattr(self, "_ica_plot_psd_fmax", psd_fmax)

        metadata = {
            "ica": {
                "classification_method": method,
                "ica_components": self.final_ica.n_components_,
            }
        }
        if psd_fmax is not None:
            metadata["ica"]["psd_fmax"] = float(psd_fmax)

        self.ica_classification_method = method
        self._update_metadata("classify_ica_components", metadata)

        message("success", f"ICA component classification with {method} complete")

        # Apply rejection if requested
        if reject:
            self.apply_ica_component_rejection()

            # Generate ICA reports after rejected components are finalized.
            if method == "iclabel":
                try:
                    if hasattr(self, "generate_ica_reports"):
                        self.generate_ica_reports()
                except Exception as e:
                    message("warning", f"Failed to generate ICA component report: {e}")

        # Export if requested
        self._auto_export_if_enabled(fit_data, stage_name, export)

        return self.ica_flags

    def apply_ica_component_rejection(
        self,
        data_to_clean=None,
        manual_rejected_components: list[int] | None = None,
    ):
        """
        Apply ICA component rejection based on component classifications and configuration.

        This method uses the labels assigned by `classify_ica_components` and the rejection
        criteria specified in the 'component_rejection' section of the pipeline configuration
        (e.g., ic_flags_to_reject, ic_rejection_threshold) to mark components
        for rejection. It then applies the ICA to remove these components from
        the data.

        It updates `self.final_ica.exclude` and modifies the data object
        (`self.raw` or `self.epochs`) in-place. The updated ICA object is
        also saved.

        Parameters
        ----------
        data_to_clean : mne.io.Raw | mne.Epochs, optional
            The data to apply the ICA to. If None, defaults to whichever of
            `self.raw` / `self.epochs` ICA was fit on (see `run_ica`).
            This should ideally be the same data object that classification was
            performed on, or is compatible with `self.final_ica`.
        manual_rejected_components : list[int], optional
            Explicit component indices to reject. When provided (even an empty
            list), automatic thresholds and control sheet processing are
            skipped in favor of these indices.

        Returns
        -------
        None
            Modifies `self.final_ica` and the input data object in-place.

        Raises
        ------
        RuntimeError
            If `self.final_ica` is not available. When running in automatic
            mode (no manual override), `self.ica_flags` must also be present,
            meaning both `run_ica` and `classify_ica_components` have run.
        """
        message("header", "Applying ICA component rejection")

        if not hasattr(self, "final_ica") or self.final_ica is None:
            message(
                "error", "ICA (self.final_ica) not found. Skipping ICLabel rejection."
            )
            raise RuntimeError(
                "ICA (self.final_ica) not found. Please run `run_ica` first."
            )

        manual_override = manual_rejected_components is not None

        if manual_override:
            # Normalize and deduplicate while preserving deterministic ordering
            manual_rejected_components = sorted(
                {int(component) for component in manual_rejected_components}
            )

        if not manual_override:
            if not hasattr(self, "ica_flags") or self.ica_flags is None:
                message(
                    "error",
                    "ICA results (self.ica_flags) not found. Skipping component rejection.",
                )
                raise RuntimeError(
                    "ICA results (self.ica_flags) not found. Please run `classify_ica_components` first."
                )

        # Get component_rejection config
        if not manual_override:
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
            config_params_nested = step_config_main_dict.get("value") or {}

            flags_to_reject = config_params_nested.get("ic_flags_to_reject")
            rejection_threshold = config_params_nested.get("ic_rejection_threshold")
            threshold_overrides = config_params_nested.get("ic_rejection_overrides", {})

            # If not found in "value", try to get them from the main step config dict directly
            if (
                flags_to_reject is None
                and "ic_flags_to_reject" in step_config_main_dict
            ):
                flags_to_reject = step_config_main_dict.get("ic_flags_to_reject")
            if (
                rejection_threshold is None
                and "ic_rejection_threshold" in step_config_main_dict
            ):
                rejection_threshold = step_config_main_dict.get(
                    "ic_rejection_threshold"
                )
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

        if not manual_override:
            # Warn about unused overrides
            if threshold_overrides:
                normalized_flags_to_reject = {
                    normalize_ic_type(flag) for flag in flags_to_reject
                }
                unused_overrides = {
                    override
                    for override in threshold_overrides
                    if normalize_ic_type(override) not in normalized_flags_to_reject
                }
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
                    "info",
                    f"Will reject ICs with per-type thresholds: {threshold_info}",
                )
            else:
                message(
                    "info",
                    f"Will reject ICs of types: {flags_to_reject} with confidence > {rejection_threshold}",
                )

        # Determine data to clean - defaults to whichever object ICA was fit on
        target_data = (
            data_to_clean if data_to_clean is not None else self._get_ica_fit_data()
        )

        # Snapshot pre-rejection data so ICA reports can show original component
        # activity/PSD even for components that get rejected below. Only do this
        # when we're operating on self.raw (not an arbitrary externally-provided
        # data object, and not self.epochs - ICA reports are raw-only), and only
        # take the snapshot once per task instance so a second call can't
        # overwrite it with already-partially-rejected data.
        if (
            data_to_clean is None
            and target_data is getattr(self, "raw", None)
            and not hasattr(self, "raw_prerejection")
        ):
            self.raw_prerejection = self.raw.copy()

        data_source_name = (
            "provided data object"
            if data_to_clean is not None
            else (
                "self.epochs" if isinstance(target_data, mne.BaseEpochs) else "self.raw"
            )
        )
        message("debug", f"Applying ICA to {data_source_name}")

        if manual_override:
            old_exclude = (
                self.final_ica.exclude.copy() if self.final_ica.exclude else []
            )
            self.final_ica.exclude = manual_rejected_components

            if CACHE_AVAILABLE and old_exclude != manual_rejected_components:
                invalidate_ica_cache(self.final_ica)
                message(
                    "debug", "Cache invalidated due to manual ICA exclude list change"
                )

            if manual_rejected_components:
                self.final_ica.apply(target_data)
                message(
                    "info",
                    "Applied manual ICA exclusions: " f"{manual_rejected_components}",
                )
            else:
                message("info", "Manual ICA override provided no components to remove.")

            metadata = {
                "ica": {
                    "method": "ManualOverride",
                    "configured_flags_to_reject": None,
                    "configured_rejection_threshold": None,
                    "configured_threshold_overrides": {},
                    "rejected_indices_this_step": [],
                    "auto_excluded_indices": [],
                    "final_excluded_indices": manual_rejected_components,
                }
            }
        else:
            # Run automatic rejection on a copy to get suggested components
            temp_data = target_data.copy()
            _, rejected_ic_indices_this_step = apply_ica_component_rejection(
                raw=temp_data,
                ica=self.final_ica,
                labels_df=self.ica_flags,
                ic_flags_to_reject=flags_to_reject,
                ic_rejection_threshold=rejection_threshold,
                ic_rejection_overrides=threshold_overrides,
                verbose=True,
            )

            auto_exclude = sorted(self.final_ica.exclude)

            if not rejected_ic_indices_this_step:
                message(
                    "info", "No new components met rejection criteria in this step."
                )
            else:
                message(
                    "info",
                    f"Identified {len(rejected_ic_indices_this_step)} components for rejection: "
                    f"{rejected_ic_indices_this_step}",
                )

            # Use control sheet to finalize exclusions
            final_exclude = update_ica_control_sheet(self.config, auto_exclude)
            old_exclude = (
                self.final_ica.exclude.copy() if self.final_ica.exclude else []
            )
            self.final_ica.exclude = final_exclude

            # Invalidate cache if exclude list changed
            if CACHE_AVAILABLE and old_exclude != final_exclude:
                invalidate_ica_cache(self.final_ica)
                message("debug", "Cache invalidated due to ICA exclude list change")

            if not final_exclude:
                message(
                    "info",
                    "No ICA components marked for exclusion after control sheet processing.",
                )
            else:
                self.final_ica.apply(target_data)
                message("info", f"Applied ICA, removing components: {final_exclude}")

            # Update metadata
            metadata = {
                "ica": {
                    "configured_flags_to_reject": flags_to_reject,
                    "configured_rejection_threshold": rejection_threshold,
                    "configured_threshold_overrides": threshold_overrides,
                    "rejected_indices_this_step": rejected_ic_indices_this_step,
                    "auto_excluded_indices": auto_exclude,
                    "final_excluded_indices": final_exclude,
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

        # Export post-ICA data after components are removed
        self._auto_export_if_enabled(target_data, "ica_clean", True)

        message("success", "ICA component rejection complete.")
