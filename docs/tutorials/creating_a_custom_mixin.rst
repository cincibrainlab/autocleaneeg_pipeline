Creating a Custom Mixin (Processing Step)
===========================================

This tutorial explains how to create custom **Mixin** classes to add new, reusable processing steps to the Autoclean pipeline.

What are Mixins?
----------------

Mixins are Python classes designed to add specific functionalities (methods) to other classes through inheritance. In Autoclean, they are used to encapsulate individual EEG processing steps, such as:

*   Filtering
*   Resampling
*   Artifact detection algorithms
*   Channel interpolation
*   Custom epoching logic

By creating a Mixin for a new processing step, you make that step easily reusable across different Tasks.

*(This tutorial assumes you understand how to create a custom Task. See* :doc:`creating_custom_task` *if needed.)*

Steps to Create a Custom Mixin
------------------------------

1.  **Create the Mixin File:**
    Create a new Python file for your mixin. It's good practice to place it within the `src/autoclean/mixins/` directory, possibly in a subdirectory (e.g., `custom`). Example: `src/autoclean/mixins/custom/my_artifact_detector.py`.

2.  **Define the Mixin Class:**
    Define a Python class. Inheriting from `autoclean.mixins.signal_processing.base.SignalProcessingMixin` is recommended as it provides useful helper methods like `_get_data_object`, `_check_step_enabled`, `_update_metadata`, and `_update_instance_data`.

    .. code-block:: python

       # src/autoclean/mixins/custom/my_artifact_detector.py
       import mne
       from typing import Any, Dict, Optional, Union
       from autoclean.utils.logging import message
       from autoclean.mixins.signal_processing.base import SignalProcessingMixin as BaseSignalProcessingMixin

       class MyArtifactDetectorMixin(BaseSignalProcessingMixin):
           """Provides a custom artifact detection method."""

           # Method definition for the processing step
           def detect_my_custom_artifacts(self,
                                          data: Optional[Union[mne.io.Raw, mne.Epochs]] = None,
                                          use_epochs: bool = False
                                         ) -> Optional[Union[mne.io.Raw, mne.Epochs]]:
               """Detects and annotates custom artifacts based on config."""
               # ... implementation follows ...

3.  **Implement the Mixin Method - Boilerplate:**
    Most processing steps require standard code to get the data object (`self.raw`/`self.epochs` or provided `data`) and check the configuration for enablement and parameters.

    .. code-block:: python

       # Inside detect_my_custom_artifacts method...

               # --- Boilerplate: Get data object and check config ---
               data_obj = self._get_data_object(data, use_epochs)
               if data_obj is None:
                   message("warning", f"No data object found for 'detect_my_custom_artifacts'. Skipping.")
                   return None

               # Check configuration using a unique key for this step
               step_name = "my_custom_artifact_detection_step"
               is_enabled, step_config = self._check_step_enabled(step_name)
               if not is_enabled:
                   message("info", f"Step '{step_name}' is disabled in configuration.")
                   return data_obj # Return unmodified data

               # Get parameters from the step's config dictionary
               threshold = step_config.get("threshold", 5.0)
               message("info", f"Running custom artifact detection with threshold: {threshold}...")
               # --- End Boilerplate ---

               # IMPORTANT: Work on a copy
               processed_data = data_obj.copy()

4.  **Implement the Mixin Method - Custom Logic:**
    Implement your core algorithm, operating on the `processed_data` copy.

    .. code-block:: python

       # Inside detect_my_custom_artifacts method, after boilerplate...

               # --- Custom Logic ---
               # Implement your artifact detection algorithm here.
               if not isinstance(processed_data, mne.io.Raw):
                    message("warning", "Custom artifact detection example only supports Raw data.")
                    return processed_data # Return unmodified copy

               detected_onsets = []
               detected_durations = []
               # ... your algorithm that finds artifact times ...
               # Placeholder example:
               if len(processed_data.times) > 10 * processed_data.info['sfreq']:
                   artifact_time = processed_data.times[int(5 * processed_data.info['sfreq'])]
                   detected_onsets.append(artifact_time)
                   detected_durations.append(2.0)
               # --- End Custom Logic ---

5.  **Implement the Mixin Method - Annotations/Metadata:**
    Annotate the `processed_data` if needed (e.g., with `mne.Annotations`). Update the run's metadata using `self._update_metadata`. Update the instance's data (`self.raw` or `self.epochs`) using `self._update_instance_data` if the original instance data was used (`data=None`). Finally, return the `processed_data`.

    .. code-block:: python

       # Inside detect_my_custom_artifacts method, after custom logic...

               # --- Annotate and Update Metadata ---
               if isinstance(processed_data, mne.io.Raw) and detected_onsets:
                   message("info", f"Detected {len(detected_onsets)} custom artifacts.")
                   try:
                       annotations = mne.Annotations(onset=detected_onsets,
                                                     duration=detected_durations,
                                                     description=['BAD_custom_artifact'] * len(detected_onsets),
                                                     orig_time=processed_data.info.get('meas_date'))
                       current_annotations = processed_data.annotations if processed_data.annotations else mne.Annotations([],[],[])
                       processed_data.set_annotations(current_annotations + annotations)
                   except Exception as e:
                       message("error", f"Failed to set annotations for custom artifacts: {e}")
               elif detected_onsets:
                    message("warning", "Custom artifact detection found artifacts but cannot annotate non-Raw data.")
               else:
                   message("info", "No custom artifacts detected.")

               # Update metadata for this run
               metadata = {"threshold": threshold, "artifacts_found": len(detected_onsets)}
               self._update_metadata(step_name, metadata)

               # Update self.raw or self.epochs if the original object was used
               self._update_instance_data(data_obj, processed_data, use_epochs)
               # --- End Annotate and Update Metadata ---

               return processed_data # Return the processed copy

Using the Custom Mixin in a Task
--------------------------------

To use your new processing step, add the Mixin class to the inheritance list of your custom `Task`.

1.  **Import the Mixin:**
    In your custom Task file (e.g., `src/autoclean/tasks/my_visual_paradigm.py`), import your Mixin class.

    .. code-block:: python

       # src/autoclean/tasks/my_visual_paradigm.py
       from autoclean.core.task import Task
       # ... other imports
       from autoclean.mixins.custom.my_artifact_detector import MyArtifactDetectorMixin

2.  **Add Mixin to Task Inheritance:**
    Include your Mixin in the list of base classes for your Task.

    .. code-block:: python

       # Add the mixin to the list of base classes
       class MyVisualParadigm(Task, MyArtifactDetectorMixin):
           # ... __init__, _validate_task_config, etc. ...

           def run(self) -> None:
               # ... (previous steps like import_raw, resample_data) ...

               # --- Call your custom step --- 
               message("info", "Running custom artifact detection step...")
               self.detect_my_custom_artifacts() # Method is now available via inheritance

               # ... (subsequent steps like epoching, reporting) ...

Configuring the Custom Step
---------------------------

Update your `autoclean_config.yaml` to include parameters for your custom step within the relevant Task's `settings` section. The structure must match the actual config file.

*   The key for the step's configuration (e.g., `my_custom_artifact_detection_step`) **must** match the `step_name` used in your Mixin method (`self._check_step_enabled(step_name)`).

.. code-block:: yaml

   tasks:
     MyVisualParadigm: # Task that uses the custom mixin
       # ... other task config like description, mne_task,...
       settings:
         # ... config for standard steps ...
         resample_step:
           enabled: true
           value: 250

         # --- Configuration for your custom step --- 
         my_custom_artifact_detection_step: # Key MUST match step_name in mixin
           enabled: true
           # Parameters for the step are typically placed directly under the step key
           # (unless the step expects a nested 'value' dictionary)
           threshold: 7.5 # Parameter read by step_config.get("threshold") in the mixin
           another_param: "example"
           # If the step expects a nested 'value':
           # value: 
           #  threshold: 7.5 
           #  another_param: "example"

   # Stage file config is global, not per-task
   stage_files:
     # ... other stages ... 
     post_custom_artifacts: { enabled: true, suffix: "_custom_artifact_annot" }

Summary
-------

*   Create Mixin classes (ideally inheriting `BaseSignalProcessingMixin`) in `src/autoclean/mixins/` to encapsulate reusable processing steps.
*   Implement methods for your steps, including boilerplate for data handling and configuration checks.
*   Mixin methods should work on data copies, update metadata, update instance data, and return the processed copy.
*   Add your Mixin to a Task's inheritance list to make the step available.
*   Configure the step in `autoclean_config.yaml` under the Task's `settings`, using a key that matches the `step_name` checked in the Mixin method.
*   `stage_files` is configured globally. 