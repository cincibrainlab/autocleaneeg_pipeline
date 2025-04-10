"""Channel operations mixin for autoclean tasks."""

from typing import Union, Dict, List, Optional
import mne
from pyprep.find_noisy_channels import NoisyChannels
import numpy as np

from autoclean.utils.logging import message

class ChannelsMixin:
    """Mixin class providing channel operations functionality for EEG data."""

    def clean_bad_channels(self, pylossless_pipeline=None, data: Union[mne.io.Raw, None] = None,
                           correlation_thresh: float = 0.35,
                           deviation_thresh: float = 2.5,
                           ransac_sample_prop: float = 0.35,
                           ransac_corr_thresh: float = 0.65,
                           ransac_frac_bad: float = 0.25,
                           ransac_channel_wise: bool = False,
                           random_state: int = 1337,
                           cleaning_method: Union[str, None] = None,
                           reset_bads: bool = False,
                           stage_name: str = "post_bad_channels") -> mne.io.Raw:
        """Detect and mark bad channels using various methods.
        
        This method uses the MNE NoisyChannels class to detect bad channels using SNR,
        correlation, deviation, and RANSAC methods.
        
        Parameters
        ----------
        pipeline : Optional
            The pipeline object to add flags to. If None, does not add flags to pylosslesspipeline.
        data : mne.io.Raw, Optional
            The data object to detect bad channels from. If None, uses self.raw.
        correlation_thresh : float, Optional
            Threshold for correlation-based detection.
        deviation_thresh : float, Optional
            Threshold for deviation-based detection.
        ransac_sample_prop : float, Optional
            Proportion of samples to use for RANSAC.
        ransac_corr_thresh : float, Optional
            Threshold for RANSAC-based detection.
        ransac_frac_bad : float, Optional
            Fraction of bad channels to use for RANSAC.
        ransac_channel_wise : bool, Optional
            Whether to use channel-wise RANSAC.
        random_state : int, Optional
            Random state for reproducibility.
        cleaning_method : str, Optional
            Method to use for cleaning bad channels. Options are 'interpolate' or 'drop' or None(default).
        reset_bads : bool, Optional
            Whether to reset bad channels.
        stage_name : str, Optional
            Name for saving and metadata.
            
        Returns
        -------
        result_raw : instance of mne.io.Raw
            The raw data object with bad channels marked or cleaned
            
        See Also
        --------
        :py:class:`pyprep.find_noisy_channels.NoisyChannels` : For more information on the NoisyChannels class
        """
        # Determine which data to use
        data = self._get_data_object(data)
        
        # Type checking
        if not isinstance(data, mne.io.Raw) and not isinstance(data, mne.io.base.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for bad channel detection")
            
        # Check configuration for rejection policy
        if hasattr(self, 'config'):
            task = self.config.get("task")
            
            # Get rejection policy from config
            rejection_policy = self.config.get("tasks", {}).get(task, {}).get("rejection_policy", {})
            
            if rejection_policy:
                # Update parameters from rejection policy if available
                for key in rejection_policy:
                    if key == "ic_rejection_threshold":
                        correlation_thresh = rejection_policy[key]
                    elif key == "ch_cleaning_mode":
                        ransac_channel_wise = rejection_policy[key] == "ransac"
            
        try:
            # Check if "eog" is in channel types and handle EOG channels if needed
            if hasattr(self, 'config') and self.config.get("task") and "eog" in data.get_channel_types():
                task = self.config.get("task")
                if not self.config.get("tasks", {}).get(task, {}).get("settings", {}).get("eog_step", {}).get("enabled", True):
                    # If EOG step is disabled, temporarily set EOG channels to EEG type
                    eog_picks = mne.pick_types(data.info, eog=True)
                    eog_ch_names = [data.ch_names[idx] for idx in eog_picks]
                    data.set_channel_types({ch: "eeg" for ch in eog_ch_names})
            
            # Create a copy of the data
            result_raw = data.copy()
            
            # Setup options
            options = {
                "random_state": random_state,
                "correlation_thresh": correlation_thresh,
                "deviation_thresh": deviation_thresh,
                "ransac_sample_prop": ransac_sample_prop,
                "ransac_corr_thresh": ransac_corr_thresh,
                "ransac_frac_bad": ransac_frac_bad,
                "ransac_channel_wise": ransac_channel_wise,
            }
            
            # Run noisy channels detection
            message("header", "Detecting bad channels...")
            cleaned_raw = NoisyChannels(result_raw, random_state=options["random_state"])
            cleaned_raw.find_bad_by_correlation(
                correlation_secs=5.0,
                correlation_threshold=options["correlation_thresh"],
                frac_bad=0.01
            )
            cleaned_raw.find_bad_by_deviation(deviation_threshold=options["deviation_thresh"])
            if options["ransac_corr_thresh"] > 0:
                cleaned_raw.find_bad_by_ransac(
                    n_samples=100,
                    sample_prop=options["ransac_sample_prop"],
                    corr_thresh=options["ransac_corr_thresh"],
                    frac_bad=options["ransac_frac_bad"],
                    corr_window_secs=5.0,
                    channel_wise=options["ransac_channel_wise"],
                    max_chunk_size=None,
                )
            
            # Get bad channels and add them to the raw object
            uncorrelated_channels = cleaned_raw.get_bads(as_dict=True)["bad_by_correlation"]
            deviation_channels = cleaned_raw.get_bads(as_dict=True)["bad_by_deviation"]
            ransac_channels = cleaned_raw.get_bads(as_dict=True)["bad_by_ransac"]
            bad_channels = cleaned_raw.get_bads(as_dict=True)

            # Check for reference channels to exclude from bad channels
            ref_channels = []
            if hasattr(self, 'config'):
                task = self.config.get("task")
                ref_step = self.config.get("tasks", {}).get(task, {}).get("settings", {}).get("reference_step", {})
                if ref_step and ref_step.get("enabled") and ref_step.get("value"):
                    ref_channels = ref_step.get("value", [])
                    message("info", f"Excluding reference channel(s) from bad channels: {ref_channels}")

            # Add bad channels to info, but exclude reference channels
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_ransac"] if str(ch) not in ref_channels])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_deviation"] if str(ch) not in ref_channels])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_correlation"] if str(ch) not in ref_channels])
            
            # Remove duplicates
            bads = list(set(result_raw.info["bads"]))
            result_raw.info["bads"] = bads

            if cleaning_method == 'interpolate':
                result_raw.interpolate_bads(reset_bads=reset_bads)
            if cleaning_method == 'drop':
                result_raw.drop_channels(result_raw.info["bads"])
                result_raw.info["bads"] = []

            if hasattr(self.raw, 'bad_channels'):
                total_bads = self.raw.bad_channels
                total_bads.extend(bads)
                total_bads = list(set(total_bads))
                self.raw.bad_channels = total_bads
            else:
                self.raw.bad_channels = bads

            if len(self.raw.bad_channels)/result_raw.info['nchan'] > self.BAD_CHANNEL_THRESHOLD:
                self.flagged = True
                warning = f"WARNING: {len(self.raw.bad_channels)/result_raw.info['nchan']:.2%} bad channels detected"
                self.flagged_reasons.append(warning)
                message("warning", f"Flagging: {warning}")

            message("info", f"Detected {len(bads)} bad channels: {bads}")
            
            # Update metadata
            metadata = {
                "method": "NoisyChannels",
                "options": options,
                "channelCount": len(result_raw.ch_names),
                "durationSec": int(result_raw.n_times) / result_raw.info["sfreq"],
                "numberSamples": int(result_raw.n_times),
                "bads": bads,
                "uncorrelated_channels": uncorrelated_channels,
                "deviation_channels": deviation_channels,
                "ransac_channels": ransac_channels,
            }
            
            self._update_metadata("step_clean_bad_channels", metadata)
            
            # Save the result
            self._save_raw_result(result_raw, stage_name)
            
            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)

            if pylossless_pipeline is not None:
                pylossless_pipeline.flags["ch"].add_flag_cat(kind="noisy", bad_ch_names=np.array(bad_channels["bad_by_ransac"]))
                pylossless_pipeline.flags["ch"].add_flag_cat(kind="noisy", bad_ch_names=np.array(bad_channels["bad_by_deviation"]))
                pylossless_pipeline.flags["ch"].add_flag_cat(kind="uncorrelated", bad_ch_names=np.array(bad_channels["bad_by_correlation"]))
                pylossless_pipeline.raw = result_raw
                message("info", "Added bad channels to pipeline flags")
                return pylossless_pipeline
            
            return result_raw
        except Exception as e:
            message("error", f"Error during bad channel detection: {str(e)}")
            raise RuntimeError(f"Failed to detect bad channels: {str(e)}") from e
    
    def drop_channels(self, data: Union[mne.io.Raw, mne.Epochs, None] = None,
                      channels: List[str] = None,
                      stage_name: str = "drop_channels",
                      use_epochs: bool = False) -> Union[mne.io.Raw, mne.Epochs]:
        """Drop specified channels from the data.
        
        This method removes specified channels from the data.
        
        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs, Optional
            The data object to drop channels from. If None, uses self.raw or self.epochs.
        channels : List[str], Optional
            List of channel names to drop.
        stage_name : str, Optional
            Name for saving and metadata.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.
            
        Returns
        -------
        result_data : instance of mne.io.Raw or mne.Epochs
            The data object with channels dropped
            
        See Also
        --------
        :py:meth:`mne.io.Raw.drop_channels` : For MNE's raw data channel dropping functionality
        :py:meth:`mne.Epochs.drop_channels` : For MNE's epochs channel dropping functionality
        """
        # Check if channels is provided
        if channels is None:
            is_enabled, config_value = self._check_step_enabled("drop_outerlayer")
                
            if not is_enabled:
                message("info", "Channel dropping is disabled in configuration")
                return data
                
            # Get channels from config
            channels = config_value
                
            if not channels:
                message("warning", "No channels specified for dropping in config")
                return data
            
        # Determine which data to use
        data = self._get_data_object(data, use_epochs)
        
        # Type checking
        if not isinstance(data, (mne.io.Raw, mne.Epochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object for dropping channels")
            
        try:
            # Drop channels
            message("header", "Dropping channels...")
            result_data = data.copy().drop_channels(channels)
            message("info", f"Dropped {len(channels)} channels: {channels}")
            
            # Update metadata
            metadata = {
                "channels_dropped": channels,
                "channels_remaining": len(result_data.ch_names)
            }
            
            self._update_metadata("step_drop_channels", metadata)
            
            # Save the result if it's a Raw object
            if isinstance(result_data, mne.io.Raw):
                self._save_raw_result(result_data, stage_name)
            
            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during channel dropping: {str(e)}")
            raise RuntimeError(f"Failed to drop channels: {str(e)}") from e
            
    def set_channel_types(self, data: Union[mne.io.Raw, mne.Epochs, None] = None,
                           ch_types_dict: Dict[str, str] = None,
                           stage_name: str = "set_channel_types",
                           use_epochs: bool = False) -> Union[mne.io.Raw, mne.Epochs]:
        """Set channel types for specific channels.
        
        This method sets the type of specific channels (e.g., marking channels as EOG).
        
        Parameters
        ----------
        data : mne.io.Raw or mne.Epochs, Optional
            The data object to set channel types for. If None, uses self.raw or self.epochs.
        ch_types_dict : dict, Optional
            Dictionary mapping channel names to types (e.g., {'E1': 'eog'})
        stage_name : str, Optional
            Name for saving and metadata.
        use_epochs : bool, Optional
            If True and data is None, uses self.epochs instead of self.raw.
            
        Returns
        -------
        result_data : instance of mne.io.Raw or mne.Epochs
            The data object with updated channel types
            
        """
        # Check if ch_types_dict is provided
        if ch_types_dict is None or len(ch_types_dict) == 0:
            # Check if eog_step is enabled in configuration
            is_enabled, config_value = self._check_step_enabled("eog_step")
                
            if not is_enabled:
                message("info", "Channel type setting is disabled in configuration")
                return data
                
            # Get channel types from config
            ch_types_dict = config_value
                
            if not ch_types_dict:
                message("warning", "No channel types specified in config")
                return data
            
        # Determine which data to use
        data = self._get_data_object(data, use_epochs)
        
        # Type checking
        if not isinstance(data, (mne.io.Raw, mne.Epochs)):
            raise TypeError("Data must be an MNE Raw or Epochs object for setting channel types")
            
        try:
            # Set channel types
            message("header", "Setting channel types...")
            result_data = data.copy().set_channel_types(ch_types_dict)
            message("info", f"Set types for {len(ch_types_dict)} channels")
            
            # Update metadata
            metadata = {
                "channel_types": ch_types_dict
            }
            
            self._update_metadata("set_channel_types", metadata)
            
            # Save the result if it's a Raw object
            if isinstance(result_data, mne.io.Raw):
                self._save_raw_result(result_data, stage_name)
            
            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during setting channel types: {str(e)}")
            raise RuntimeError(f"Failed to set channel types: {str(e)}") from e
            

