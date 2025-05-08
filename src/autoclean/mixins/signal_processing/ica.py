"""ICA mixin for autoclean tasks."""

from mne.preprocessing import ICA
import mne_icalabel
import pandas as pd

from autoclean.io.export import save_ica_to_fif
from autoclean.utils.logging import message

class IcaMixin:
    """Mixin for ICA processing."""

    def run_ica(self, eog_channel: str = None, use_epochs: bool = False, **kwargs) -> ICA:
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

        Returns
        -------
        final_ica : mne.preprocessing.ICA
            The fitted ICA object.
        
        Examples
        --------
        >>> self.run_ica()

        >>> self.run_ica(eog_channel="E27") 

        See Also
        --------
        run_ICLabel : Run ICLabel on the raw data.

        """
        message("header", "Running ICA step")

        is_enabled, config_value = self._check_step_enabled("ICA")

        if not is_enabled:
            message("warning", "ICA is not enabled in the config")
            return

        if use_epochs:
            message("debug", "Using epochs")
            # Create epochs
            data = self.epochs
        else:
            message("debug", "Using raw data")
            data = self.raw

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

            self.final_ica = ICA(**ica_kwargs) # pylint: disable=not-callable

            message("debug", f"Fitting ICA with {ica_kwargs}")

            self.final_ica.fit(data)

            if eog_channel is not None:
                message("info", f"Running EOG detection on {eog_channel}")
                eog_indices, _ = self.final_ica.find_bads_eog(
                    data, ch_name=eog_channel
                )
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

        save_ica_to_fif(self.final_ica, self.config, self.raw)

        message("success", "ICA step complete")

        return self.final_ica

    def run_ICLabel(self): # pylint: disable=invalid-name
        """Run ICLabel on the raw data.

        Returns
        -------
        ica_flags : pandas.DataFrame
            A pandas DataFrame containing the ICLabel flags.

        Examples
        --------
        >>> self.run_ICLabel()

        Notes
        -----
        This method will modify the self.final_ica attribute in place.
        """
        message("header", "Running ICLabel step")

        # is_enabled, config_value = self._check_step_enabled("ICLabel")

        # if not is_enabled:
        #     message("warning", "ICLabel is not enabled in the config")
        #     return

        mne_icalabel.label_components(self.raw, self.final_ica, method="iclabel")

        self._icalabel_to_data_frame(self.final_ica)

        metadata = {
            "ica": {
                "ica_components": self.final_ica.n_components_,
            }
        }

        self._update_metadata("step_run_ICLabel", metadata)

        save_ica_to_fif(self.final_ica, self.config, self.raw)

        message("success", "ICLabel step complete")

        return self.ica_flags

    def _icalabel_to_data_frame(self, ica):
        """Export IClabels to pandas DataFrame."""
        ic_type = [""] * ica.n_components_
        for label, comps in ica.labels_.items():
            for comp in comps:
                ic_type[comp] = label

        self.ica_flags = pd.DataFrame(
            dict(
                component=ica._ica_names, # pylint: disable=protected-access
                annotator=["ic_label"] * ica.n_components_,
                ic_type=ic_type,
                confidence=ica.labels_scores_.max(1),
            )
        )

        return self.ica_flags
