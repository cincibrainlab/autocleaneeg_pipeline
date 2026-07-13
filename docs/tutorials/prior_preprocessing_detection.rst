Prior Preprocessing Detection
=============================

AutoClean can inspect imported EEG data and available EEGLAB provenance for
signs of earlier filtering, referencing, epoching, ICA, and related processing.
The check is opt-in. Add this top-level configuration alongside your existing
task settings:

.. code-block:: yaml

   prior_preprocessing_detection:
     enabled: true
     strict: false

``enabled`` writes per-file JSON and Markdown reports plus a dataset summary.
``strict`` labels detected conflicts in ``strict_violations`` for downstream
review. It does not stop the import or preprocessing run.

Detection is conservative: documented provenance has priority, while signal
inspection produces confidence labels rather than proof of prior processing.
