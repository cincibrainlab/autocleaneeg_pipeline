"""ICA mixin for autoclean tasks."""

from typing import Optional

import mne
from mne.preprocessing import ICA
import mne_icalabel
import pandas as pd

from autoclean.io.export import save_ica_to_fif
from autoclean.utils.logging import message

class IcaMixin:
    """Mixin for ICA processing."""

    def run_ica(self, eog_channel: str = None, use_epochs: bool = False, **kwargs) -> None:
        """Run ICA on the raw data.

        Parameters
        ----------
        eog_channel : str, optional
            The EOG channel to use for ICA. If None, the EOG channel will be detected automatically.

        Returns
        -------
        None
        
        Examples
        --------
        >>> self.one_stage_ica()

        >>> self.one_stage_ica(eog_channel="EOG") 

        Notes
        -----
        This method will modify the self.raw attribute.

        """
        message("header", "Running ICA step")

        is_enabled, config_value = self._check_step_enabled("ICA")

        if not is_enabled:
            message("warning", "ICA is not enabled in the config")
            return

        if use_epochs:
            message("debug", "Using epochs")
            # Create epochs
            data = mne.Epochs(self.raw, self.events, event_id=self.event_id, tmin=0, tmax=0.5, picks=self.picks)
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

            self.final_ica = ICA(**ica_kwargs)    

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

        self._update_metadata("ica", metadata)

        save_ica_to_fif(self.final_ica, self.config, self.raw)

        message("success", "ICA step complete")

        return self.final_ica
    
    def run_ICLabel(self):
        """Run ICLabel on the raw data."""
        message("header", "Running ICLabel step")

        # is_enabled, config_value = self._check_step_enabled("ICLabel")

        # if not is_enabled:
        #     message("warning", "ICLabel is not enabled in the config")
        #     return

        self.final_ica = mne_icalabel.label_components(self.raw, self.final_ica, method="iclabel")

        self._icalabel_to_data_frame(self.final_ica)



    def _icalabel_to_data_frame(self, ica):
        """Export IClabels to pandas DataFrame."""
        ic_type = [""] * ica.n_components_
        for label, comps in ica.labels_.items():
            for comp in comps:
                ic_type[comp] = label

        self.ica_flags = pd.DataFrame(
            dict(
                component=ica._ica_names,
                annotator=["ic_label"] * ica.n_components_,
                ic_type=ic_type,
                confidence=ica.labels_scores_.max(1),
            )
        )

        return self.ica_flags





