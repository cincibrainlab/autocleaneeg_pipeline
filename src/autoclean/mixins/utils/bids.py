"""Mixin for BIDS functions."""

from datetime import datetime
from typing import Any, Dict, Tuple

import mne

from autoclean.utils.bids import step_convert_to_bids
from autoclean.utils.logging import message


class BIDSMixin:
    """Mixin for BIDS functions."""

    def create_bids_path(
        self, use_epochs: bool = False
    ) -> Tuple[mne.io.Raw, Dict[str, Any]]:
        """Create BIDS-compliant paths."""

        message("header", "step_create_bids_path")
        unprocessed_file = self.config["unprocessed_file"]
        task = self.config["task"]
        mne_task = self.config["tasks"][task]["mne_task"]
        bids_dir = self.config["bids_dir"]
        eeg_system = self.config["eeg_system"]
        config_file = self.config["config_file"]

        try:
            line_freq = self.config["tasks"][task]["settings"]["filtering"]["value"][
                "notch_freqs"
            ][0]
        except Exception as e:  # pylint: disable=broad-except
            message(
                "error",
                f"Failed to load line frequency: {str(e)}. Using default value of 60 Hz.",
            )
            line_freq = 60.0

        if use_epochs:
            epochs_data = self._get_data_object(self.epochs, use_epochs=True)
            data = self.create_mock_raw_from_epochs(epochs_data)
        else:
            data = self._get_data_object(self.raw, use_epochs=False)

        try:
            bids_path, derivatives_dir = step_convert_to_bids(
                data,
                output_dir=str(bids_dir),
                task=mne_task,
                participant_id=None,
                line_freq=line_freq,
                overwrite=True,
                study_name=unprocessed_file.stem,
                autoclean_dict=self.config,
            )

            self.config["bids_path"] = bids_path
            self.config["bids_basename"] = bids_path.basename
            self.config["derivatives_dir"] = derivatives_dir

            metadata = {
                "creationDateTime": datetime.now().isoformat(),
                "bids_subject": bids_path.subject,
                "bids_task": bids_path.task,
                "bids_run": bids_path.run,
                "bids_session": bids_path.session,
                "bids_dir": str(bids_dir),
                "bids_datatype": bids_path.datatype,
                "bids_suffix": bids_path.suffix,
                "bids_extension": bids_path.extension,
                "bids_root": str(bids_path.root),
                "eegSystem": eeg_system,
                "configFile": str(config_file),
                "line_freq": line_freq,
                "derivatives_dir": str(derivatives_dir),
            }

            self._update_metadata("step_create_bids_path", metadata)

            return

        except Exception as e:
            message("error", f"Error converting raw to bids: {e}")
            raise e

    def create_mock_raw_from_epochs(self, epochs: mne.Epochs) -> mne.io.Raw:
        """Create a mock Raw object from Epochs data for BIDS conversion.
        
        The BIDS conversion functions expect mne.io.Raw objects, but we have epoched data.
        This method creates a synthetic Raw object that contains the concatenated epoch
        data and mimics the required attributes for BIDS conversion.
        
        Args:
            epochs: The epochs data to convert
            
        Returns:
            Mock Raw object suitable for BIDS conversion
        """
        import numpy as np
        
        message("info", "Creating mock Raw object from Epochs for BIDS conversion")
        
        # Concatenate all epoch data along the time axis
        # epochs.get_data() returns (n_epochs, n_channels, n_times)
        epoch_data = epochs.get_data()
        n_epochs, n_channels, n_times = epoch_data.shape
        
        # Flatten epochs into continuous data: (n_channels, n_epochs * n_times)
        continuous_data = epoch_data.transpose(1, 0, 2).reshape(n_channels, -1)
        
        # Create info object (copy from epochs to preserve channel info)
        info = epochs.info.copy()
        
        # Create the mock Raw object
        mock_raw = mne.io.RawArray(continuous_data, info, verbose=False)
        
        # Handle the filename/filenames attribute difference
        # Epochs has 'filename', Raw expects 'filenames' (list)
        if hasattr(epochs, 'filename') and epochs.filename:
            mock_raw.filenames = [epochs.filename]
        elif hasattr(epochs, 'filenames') and epochs.filenames:
            mock_raw.filenames = epochs.filenames
        else:
            # Fallback - create a dummy filename
            mock_raw.filenames = ['epoched_data.set']
            
        # Copy any annotations if they exist
        if hasattr(epochs, 'annotations') and epochs.annotations:
            mock_raw.set_annotations(epochs.annotations)
            
        message("success", f"Created mock Raw object: {n_channels} channels, {continuous_data.shape[1]} samples")
        
        return mock_raw
