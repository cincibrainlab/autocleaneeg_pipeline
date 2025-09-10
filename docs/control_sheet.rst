ICA Control Sheet
=================

The ICA control sheet is a lightweight CSV file that stores decisions about
which independent components are removed from each recording. It captures both
automatic selections and any manual adjustments, enabling a repeatable and
transparent workflow for revisiting ICA outcomes.

Columns
-------

``original_file``
    Path to the raw recording that was processed.
``ica_fif``
    Location of the saved ICA decomposition.
``auto_initial``
    Components automatically flagged for removal on the initial run (never
    modified).
``final_removed``
    Current truth of components that will be excluded when cleaning the data.
``manual_add`` / ``manual_drop``
    Temporary columns where reviewers can request additional components to add
    or remove. These are cleared after processing.
``status``
    ``auto`` for untouched rows, ``pending`` when manual edits are waiting to be
    applied, and ``applied`` once changes have been processed.
``last_run_iso``
    ISO 8601 timestamp of the last successful update.

Usage
-----

1. Run the automatic ICA step to populate ``auto_initial`` and ``final_removed``.
2. Review the sheet and enter component numbers in ``manual_add`` or
   ``manual_drop`` for any file that needs adjustment.
3. Re-run the pipeline. Only rows with ``pending`` changes are processed. The
   sheet is updated with the new ``final_removed`` list and ``manual_*`` fields
   are cleared.

The sheet can be committed to version control to provide an auditable history of
manual ICA decisions.
