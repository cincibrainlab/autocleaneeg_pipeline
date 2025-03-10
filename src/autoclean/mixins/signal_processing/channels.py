"""Channel operations mixin for autoclean tasks."""

from typing import Union, Dict, List, Optional
import mne
from pyprep.find_noisy_channels import NoisyChannels
import numpy as np

from autoclean.utils.logging import message

class ChannelsMixin:
    """Mixin class providing channel operations functionality for EEG data."""
    
    def drop_channels(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None] = None,
                      channels: List[str] = None,
                      stage_name: str = "drop_channels",
                      use_epochs: bool = False) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        """Drop specified channels from the data.
        
        This method removes specified channels from the data.
        
        Args:
            data: Optional MNE Raw or Epochs object. If None, uses self.raw or self.epochs
            channels: List of channel names to drop
            stage_name: Name for saving and metadata
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The data object with channels dropped
            
        Raises:
            AttributeError: If self.raw or self.epochs doesn't exist when needed
            TypeError: If data is not a Raw or Epochs object
            ValueError: If channels is None or empty
            RuntimeError: If channel dropping fails
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
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
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
            if isinstance(result_data, mne.io.BaseRaw):
                self._save_raw_result(result_data, stage_name)
            
            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during channel dropping: {str(e)}")
            raise RuntimeError(f"Failed to drop channels: {str(e)}") from e
            
    def set_channel_types(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, None] = None,
                           ch_types_dict: Dict[str, str] = None,
                           stage_name: str = "set_channel_types",
                           use_epochs: bool = False) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        """Set channel types for specific channels.
        
        This method sets the type of specific channels (e.g., marking channels as EOG).
        
        Args:
            data: Optional MNE Raw or Epochs object. If None, uses self.raw or self.epochs
            ch_types_dict: Dictionary mapping channel names to types (e.g., {'E1': 'eog'})
            stage_name: Name for saving and metadata
            use_epochs: If True and data is None, uses self.epochs instead of self.raw
            
        Returns:
            The data object with updated channel types
            
        Raises:
            AttributeError: If self.raw or self.epochs doesn't exist when needed
            TypeError: If data is not a Raw or Epochs object
            ValueError: If ch_types_dict is None or empty
            RuntimeError: If setting channel types fails
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
        if not isinstance(data, (mne.io.BaseRaw, mne.BaseEpochs)):
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
            if isinstance(result_data, mne.io.BaseRaw):
                self._save_raw_result(result_data, stage_name)
            
            # Update self.raw or self.epochs
            self._update_instance_data(data, result_data, use_epochs)
                
            return result_data
            
        except Exception as e:
            message("error", f"Error during setting channel types: {str(e)}")
            raise RuntimeError(f"Failed to set channel types: {str(e)}") from e
            
    def clean_bad_channels(self, pipeline=None, data: Union[mne.io.BaseRaw, None] = None,
                           correlation_thresh: float = 0.35,
                           deviation_thresh: float = 5.0,
                           ransac_sample_prop: float = 0.35,
                           ransac_corr_thresh: float = 0.65,
                           ransac_frac_bad: float = 0.45,
                           ransac_channel_wise: bool = False,
                           random_state: int = 1337,
                           stage_name: str = "post_bad_channels") -> mne.io.BaseRaw:
        """Detect and mark bad channels using various methods.
        
        This method uses the MNE NoisyChannels class to detect bad channels using SNR,
        correlation, deviation, and RANSAC methods.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            correlation_thresh: Threshold for correlation-based detection
            deviation_thresh: Threshold for deviation-based detection
            ransac_corr_thresh: Threshold for RANSAC-based detection
            ransac_channel_wise: Whether to use channel-wise RANSAC
            random_state: Random state for reproducibility
            stage_name: Name for saving and metadata
            
        Returns:
            The raw data object with bad channels marked
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If bad channel detection fails
        """
        # Determine which data to use
        data = self._get_data_object(data)
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
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
            bad_channels = cleaned_raw.get_bads(as_dict=True)
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_ransac"]])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_deviation"]])
            result_raw.info["bads"].extend([str(ch) for ch in bad_channels["bad_by_correlation"]])
            
            # Remove duplicates
            result_raw.info["bads"] = list(set(result_raw.info["bads"]))
            
            message("info", f"Detected {len(result_raw.info['bads'])} bad channels: {result_raw.info['bads']}")
            
            # Update metadata
            metadata = {
                "method": "NoisyChannels",
                "options": options,
                "bads": result_raw.info["bads"],
                "channelCount": len(result_raw.ch_names),
                "durationSec": int(result_raw.n_times) / result_raw.info["sfreq"],
                "numberSamples": int(result_raw.n_times),
            }
            
            self._update_metadata("step_clean_bad_channels", metadata)
            
            # Save the result
            self._save_raw_result(result_raw, stage_name)
            
            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)

            if pipeline is not None:
                pipeline.flags["ch"].add_flag_cat(kind="noisy", bad_ch_names=np.array(bad_channels["bad_by_ransac"]))
                pipeline.flags["ch"].add_flag_cat(kind="noisy", bad_ch_names=np.array(bad_channels["bad_by_deviation"]))
                pipeline.flags["ch"].add_flag_cat(kind="uncorrelated", bad_ch_names=np.array(bad_channels["bad_by_correlation"]))
                pipeline.raw = result_raw
                message("info", "Added bad channels to pipeline flags")
                return pipeline
            
            return result_raw
        except Exception as e:
            message("error", f"Error during bad channel detection: {str(e)}")
            raise RuntimeError(f"Failed to detect bad channels: {str(e)}") from e
