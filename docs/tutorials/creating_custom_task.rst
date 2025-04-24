Creating a Custom Task
======================

This tutorial shows how to create a custom Task class based on the structure found in `src/autoclean/tasks/TEMPLATE.py`.

When to Create a Custom Task?
-----------------------------

Create a custom Task to define a specific sequence of processing steps for your experimental paradigm, combining steps from `autoclean.step_functions` and methods from built-in Mixins (like `SignalProcessingMixin`, `ReportingMixin`).

*(For creating entirely new processing *steps*, see the :doc:`creating_a_custom_mixin` tutorial.)*

How to Create a Custom Task
---------------------------
The easiest way to create a custom task is to copy and adjust the `TEMPLATE.py` file.

1.  **Copy and Rename:**
    Copy `src/autoclean/tasks/TEMPLATE.py` to a new file (e.g., `my_paradigm.py`). Rename the class `TemplateTask` to your task name (e.g., `MyParadigm`) using CamelCase.

    .. code-block:: python

       # src/autoclean/tasks/my_paradigm.py
       from autoclean.core.task import Task

       class MyParadigm(Task):
           """Task definition for MyParadigm."""
           # ... Implement methods below ...

2.  **Implement `__init__`:**
    Usually minimal: initialize instance variables to `None` and call `super().__init__(config)`.

    .. code-block:: python

       class MyParadigm(Task):

           def __init__(self, config: Dict[str, Any]) -> None:
               """Initialize the task instance."""
               self.raw: Optional[mne.io.Raw] = None
               self.pipeline: Optional[Any] = None
               self.epochs: Optional[mne.Epochs] = None
               self.original_raw: Optional[mne.io.Raw] = None

                # Stages that should be configured in the autoclean_config.yaml file
                self.required_stages = [
                    "post_import",
                    "post_prepipeline",
                    "post_pylossless",
                    "post_rejection_policy",
                    "post_clean_raw",
                    "post_epochs",
                    "post_comp",
                ]

               super().__init__(config) # Calls _validate_task_config


3.  **Implement the `run` Method:**
    Define your specific sequence of processing steps, calling imported `step_` functions and `self.` methods from mixins. Call `save_raw_to_set` or `save_epochs_to_set` as needed after specific stages.

    .. code-block:: python

       # Inside MyParadigm class...

           def run(self) -> None:
               """Execute the processing pipeline for MyParadigm."""
               message("header", f"Starting MyParadigm pipeline for {self.config['unprocessed_file'].name}")

               # 1. Import
               self.import_raw()
               if self.raw is None: return # Early exit if import fails
               self.original_raw = self.raw.copy()

               # 2. Preprocessing Step Function
               self.raw = step_pre_pipeline_processing(self.raw, self.config)
               save_raw_to_set(self.raw, self.config, "post_prepipeline", self.flagged)

               # 3. BIDS Path Step Function
               self.raw, self.config = step_create_bids_path(self.raw, self.config)

               # 4. PyLossless & Artifact Detection (Example using Mixin Methods)
               # self.pipeline, self.raw = self.step_custom_pylossless_pipeline(self.config)
               # self.detect_dense_oscillatory_artifacts()

               # 5. Rejection Policy (Example using Step Function)
               # if self.pipeline: 
               #    self.pipeline.raw = self.raw
               #    self.pipeline, self.raw = step_run_ll_rejection_policy(self.pipeline, self.config)
               #    save_raw_to_set(self.raw, self.config, "post_rejection_policy", self.flagged)

               # 6. Channel Cleaning (Example using Mixin Method)
               self.clean_bad_channels(cleaning_method="interpolate") # Reads config
               save_raw_to_set(self.raw, self.config, "post_clean_raw", self.flagged)

               # 7. Epoching (Example using Mixin Methods)
               self.create_eventid_epochs() # Reads config
               if self.epochs: 
                   self.prepare_epochs_for_ica() # Reads config
                   self.gfp_clean_epochs() # Reads config
                   # save_epochs_to_set(self.epochs, self.config, "post_comp", self.flagged)

               # 8. Generate Reports
               self._generate_reports()

               message("header", f"MyParadigm pipeline finished.")

4.  **Implement `_generate_reports`:**
    Call plotting methods provided by mixins (like `ReportingMixin`). Check if the necessary data exists before plotting.

    .. code-block:: python

       # Inside MyParadigm class...

           def _generate_reports(self) -> None:
                """Generate standard reports."""
                if self.raw is None or self.original_raw is None:
                    return

                # Example calls (adapt based on steps run)
                # if self.pipeline:
                #    self.plot_ica_full(self.pipeline, self.config)
                #    self.generate_ica_reports(self.pipeline, self.config)
                #    self.step_psd_topo_figure(self.original_raw, self.raw, self.pipeline, self.config)

                # if self.epochs:
                #    self.plot_epochs_image(self.epochs)

                message("info", "Finished generating reports.")

5.  **Configure the Task:**
    In `autoclean_config.yaml`, add a section under `tasks:` with a key matching your class name (e.g., `MyParadigm`). Configure the `settings` needed by the steps in your `run` method.

    .. code-block:: yaml

       # In autoclean_config.yaml
       tasks:
         MyParadigm:
           description: "Processing for MyParadigm"
           settings:
             # Config for step_pre_pipeline_processing 
             resample_step: { enabled: true, value: 250 }
             filter_step: { enabled: true, value: { l_freq: 0.1, h_freq: 40 } }
             # Config for clean_bad_channels 
             bad_channel_step: { enabled: true, cleaning_method: "interpolate" }
             # Config for epoching methods 
             epoch_settings: { enabled: true, event_id: { Stim: 1 }, value: { tmin: -0.1, tmax: 0.5 } }
             # Config for gfp_clean_epochs 
             gfp_cleaning_step: { enabled: true, threshold: 3.0 }
             # Task-specific config checked in _validate_task_config
             my_required_setting: "value"


7.  **Run the Task:**
    Use the class name when running the pipeline.

    .. code-block:: python

       pipeline.process_file(..., task="MyParadigm")

Summary
-------

*   Create Task classes in `src/autoclean/tasks/` inheriting `autoclean.core.task.Task`.
*   Implement `__init__`, `_validate_task_config`, `run`, and `_generate_reports` based on `TEMPLATE.py`.
*   The `run` method calls a mix of imported `step_` functions and inherited `self.` mixin methods.
*   Processing methods often read parameters directly from `self.config`.
*   `_validate_task_config` checks top-level config, global `stage_files`, and task-specific settings.
*   Configure the Task in `autoclean_config.yaml` using its class name.
*   Run the pipeline using the Task's class name. 