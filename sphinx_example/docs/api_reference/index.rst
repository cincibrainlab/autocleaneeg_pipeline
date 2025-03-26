=============
API Reference
=============

This page provides an overview of the AutoClean API.

Main Components
--------------

.. toctree::
   :maxdepth: 2
   
   core
   step_functions
   tasks
   utils

Core Classes
-----------

.. currentmodule:: autoclean

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   Pipeline

Core Task Framework
-----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   core.task.Task

Step Functions
------------

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   step_functions.io.import_eeg
   step_functions.io.save_raw_to_set
   step_functions.io.save_epochs_to_set

Task Implementations
-----------------

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   tasks.resting_eyes_open.RestingEyesOpen
   tasks.assr_default.AssrDefault
   tasks.chirp_default.ChirpDefault 