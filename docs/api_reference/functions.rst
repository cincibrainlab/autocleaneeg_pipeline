.. _api_functions:

====================================
Standalone Functions *(autoclean.functions)*
====================================

This section covers the standalone functions that provide modular EEG data processing capabilities in AutoClean.
These functions can be used independently without the Task/Mixin framework, making them ideal for custom workflows,
scripting, and integration with other EEG analysis pipelines.

Standalone functions are designed to be stateless and operate directly on MNE data objects:

.. code-block:: python

   from autoclean.functions import filter_data, detect_bad_channels, fit_ica
   
   # Direct function usage - no task object required
   filtered_raw = filter_data(raw, l_freq=1.0, h_freq=40.0)
   bad_channels = detect_bad_channels(filtered_raw)
   ica = fit_ica(filtered_raw)

*Note:* Standalone functions return processed data objects and do not modify the input data in-place.
They provide a functional programming approach as an alternative to the object-oriented Task/Mixin pattern.

Available Function Categories
-----------------------------

Preprocessing Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.preprocessing

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   filtering.filter_data
   referencing.rereference_data
   resampling.resample_data
   basic_ops.crop_data
   basic_ops.drop_channels

Artifact Detection Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.artifacts

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   channels.detect_bad_channels
   channels.interpolate_bad_channels

Epoching Functions
~~~~~~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.epoching

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   regular.create_regular_epochs
   statistical.create_sl_epochs
   eventid.create_eventid_epochs
   quality.detect_outlier_epochs
   quality.gfp_clean_epochs

ICA Functions
~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.ica

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   ica_processing.fit_ica
   ica_processing.classify_ica_components
   ica_processing.apply_ica_rejection
   ica_processing.apply_iclabel_rejection

Segment Rejection Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.segment_rejection

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   segment_rejection.annotate_noisy_segments
   segment_rejection.annotate_uncorrelated_segments
   dense_oscillatory.detect_dense_oscillatory_artifacts

Advanced Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.advanced

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   autoreject.repair_epochs
   autoreject.compute_rejection_thresholds

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: autoclean.functions.visualization

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:

   plotting.plot_raw_comparison
   plotting.plot_ica_components
   plotting.plot_psd_topography
   reports.generate_processing_report
   reports.create_processing_summary