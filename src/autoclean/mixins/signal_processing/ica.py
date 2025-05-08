"""ICA mixin for autoclean tasks."""

from typing import Optional

import mne

from autoclean.utils.logging import message

from autoclean.mixins.signal_processing.main import SignalProcessingMixin


class IcaMixin(SignalProcessingMixin):
    """Mixin for ICA processing."""

    def one_stage_ica(self, eog_channel: str = None, **kwargs) -> None:
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
        
        # Detect EOG channel if not provided
        if eog_channel is None:
            eog_channel = self.detect_eog_channel()

        # Run ICA
        self.final_ica = self.run_ica(eog_channel=eog_channel, **kwargs)





