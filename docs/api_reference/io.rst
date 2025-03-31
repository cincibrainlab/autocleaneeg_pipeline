.. _api_io:

=============
IO
=============

This section covers the input/output classes that provide functionality for reading and writing data in AutoClean.

*Note* These IO functions may not have to be used directly by the user as most mixin functions will use an internal function to save the result of the mixin function.
However, they are still useful if you need to save data at a specific stage of the pipeline.

.. currentmodule:: autoclean.io

Raw
---

.. autosummary::
   :toctree: generated/
   :template: autosummary/function.rst
   :nosignatures:
   
   import.import_eeg
   export.save_raw_to_set
   export.save_epochs_to_set
   export.save_stc_to_file
   

