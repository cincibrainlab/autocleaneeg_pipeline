.. _api_task:

====
Task
====

.. currentmodule:: autoclean.core.task

The Task class is an abstract base class that defines the interface for all EEG processing tasks in AutoClean.

.. autoclass:: Task
   :members:
   :member-order: groupwise
   :exclude-members: __init__, __weakref__, _validate_task_config
   
   .. rubric:: Initialization
   
   .. automethod:: __init__
   
   .. rubric:: Core Methods
   
   .. automethod:: run
   .. automethod:: import_raw
   .. automethod:: validate_config
   
   .. rubric:: Getter Methods
   
   .. automethod:: get_raw
   .. automethod:: get_epochs
   .. automethod:: get_flagged_status

Creating Custom Tasks
--------------------

To create a custom task, subclass the Task class and implement the required methods:

.. code-block:: python

   from autoclean.core.task import Task
   
   class MyCustomTask(Task):
       """Custom task implementation for special EEG preprocessing."""
       
       def __init__(self, config):
           self.raw = None
           self.pipeline = None
           self.epochs = None
           super().__init__(config)
       
       def run(self):
           """Run the complete processing pipeline."""
           # Implement your custom processing steps here
           pass
           
       def _validate_task_config(self, config):
           """Validate task-specific configuration."""
           # Perform validation of task-specific parameters
           return config 