"""Reference operations mixin for autoclean tasks."""

from typing import Union
import mne

from autoclean.utils.logging import message

class ReferenceMixin:
    """Mixin class providing reference operations functionality for EEG data."""
    
    def set_eeg_reference(self, data: Union[mne.io.BaseRaw, None] = None,
                          ref_type: str = "average", 
                          projection: bool = False,
                          stage_name: str = "post_reference") -> mne.io.BaseRaw:
        """Apply EEG reference to the data.
        
        This method applies a reference to the EEG data, such as average reference.
        
        Args:
            data: Optional MNE Raw object. If None, uses self.raw
            ref_type: Type of reference to apply (e.g., 'average')
            projection: Whether to use projection (for average reference)
            stage_name: Name for saving and metadata
            
        Returns:
            The raw data object with reference applied
            
        Raises:
            AttributeError: If self.raw doesn't exist when needed
            TypeError: If data is not a Raw object
            RuntimeError: If reference application fails
        """
        # Check if this step is enabled in the configuration
        is_enabled, config_value = self._check_step_enabled("reference_step")
            
        if not is_enabled:
            message("info", "Reference step is disabled in configuration")
            return data
            
        # Get reference type from config if available
        if config_value is not None:
            ref_type = config_value["value"]
            
        # Determine which data to use
        data = self._get_data_object(data)
        
        # Type checking
        if not isinstance(data, mne.io.BaseRaw):
            raise TypeError("Data must be an MNE Raw object for referencing")
            
        try:
            # Apply reference
            message("header", f"Applying {ref_type} reference...")
            if ref_type == "average":
                result_raw = data.copy().set_eeg_reference(ref_type, projection=projection)
            else:
                result_raw = data.copy().set_eeg_reference(ref_type)
                
            message("info", f"Applied {ref_type} reference")
            
            # Update metadata
            metadata = {
                "reference_type": ref_type,
                "projection": projection
            }
            
            self._update_metadata("step_set_eeg_reference", metadata)
            
            # Save the result
            self._save_raw_result(result_raw, stage_name)
            
            # Update self.raw if we're using it
            self._update_instance_data(data, result_raw)
                
            return result_raw
            
        except Exception as e:
            message("error", f"Error during referencing: {str(e)}")
            raise RuntimeError(f"Failed to apply reference: {str(e)}") from e
