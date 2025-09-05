Creating a Custom Task
======================

Learn how to create custom EEG processing workflows tailored to your specific experimental needs. This guide covers both simple task creation and advanced customization.

🎯 When Do You Need a Custom Task?
----------------------------------

Create a custom task when:

**Your experimental paradigm is unique:**
- Novel stimulus types or timing
- Special preprocessing requirements  
- Specific artifact patterns to address

**Built-in tasks don't fit your data:**
- Different electrode montages
- Unusual sampling rates or filters
- Special event marker requirements

**You need specific processing steps:**
- Custom artifact detection
- Specialized epoching windows
- Unique frequency filtering

**Research-specific requirements:**
- Clinical populations needing gentler artifact removal
- High-gamma analysis requiring different filtering
- Connectivity studies needing longer epochs

🚀 Quick Start: Simple Custom Task
----------------------------------

**Step 1: Create your task file**

Create a new Python file in your workspace tasks folder:

.. code-block:: python

   # ~/Documents/Autoclean-EEG/tasks/my_custom_task.py
   from autoclean.core.task import Task

   # Configuration embedded in the file
   config = {
       'resample_step': {'enabled': True, 'value': 500},  # Higher sampling rate
       'filtering': {
           'enabled': True, 
           'value': {
               'l_freq': 0.5,    # Lower frequency cutoff
               'h_freq': 150,    # Higher frequency cutoff  
               'notch_freqs': [60, 120],
               'notch_widths': 5
           }
       },
       'ICA': {
           'enabled': True,
           'value': {
               'method': 'infomax',
               'n_components': None
           }
       },
       'epoch_settings': {
           'enabled': True,
           'value': {
               'tmin': -0.5,    # Different epoch window
               'tmax': 2.0      # Longer epochs for your analysis
           }
       }
   }

   class MyCustomTask(Task):
       def __init__(self, config):
           self.settings = globals()['config']  # Use embedded config
           super().__init__(config)
       
       def run(self) -> None:
           # Standard processing workflow
           self.import_raw()
           self.resample_data()
           self.filter_data()
           self.clean_bad_channels()
           self.rereference_data()
           self.run_ica()
           self.create_regular_epochs(export=True)
           self.generate_reports()

**Step 2: Add your task to workspace**

.. code-block:: bash

   # Option 1: Drop the file into your workspace tasks folder
   # ~/Documents/Autoclean-EEG/tasks/my_custom_task.py
   
   # Option 2: Use CLI to add the task
   autoclean task add my_custom_task.py
   
   # AutoClean automatically discovers it
   autoclean task list
   
   # Use your custom task
   autoclean process MyCustomTask my_data.raw

That's it! AutoClean will automatically find and use your custom task.

⚙️ Understanding Configuration Options
--------------------------------------

Your task configuration controls every aspect of processing. Here are the key sections:

.. note::

   To keep flagged runs in the standard output folders, add ``config['move_flagged_files'] = False`` to your task file. By default, flagged outputs are moved to ``FLAGGED_*`` directories.

**Basic Preprocessing:**

.. code-block:: python

   config = {
       # Resample data for efficiency
       'resample_step': {
           'enabled': True, 
           'value': 250  # Hz - balance between quality and file size
       },
       
       # Frequency filtering
       'filtering': {
           'enabled': True,
           'value': {
               'l_freq': 1,      # High-pass: remove slow drifts
               'h_freq': 100,    # Low-pass: remove high-frequency noise
               'notch_freqs': [60, 120],  # Remove line noise
               'notch_widths': 5
           }
       },
       
       # Channel management
       'drop_outerlayer': {
           'enabled': False,     # Manually specify bad channels
           'value': []           # List channels to exclude
       }
   }

**Advanced Artifact Removal:**

.. code-block:: python

   config = {
       # Independent Component Analysis
       'ICA': {
           'enabled': True,
           'value': {
               'method': 'infomax',        # Algorithm: 'infomax', 'fastica', 'infomax'
               'n_components': None,      # Auto-determine number of components
               'fit_params': {
                   'ortho': False,        # Orthogonality constraint
                   'extended': True       # Extended ICA for mixed distributions
               },
               'temp_highpass_for_ica': 1.0,  # Temporary 1 Hz high-pass for ICA only
           }
       },
       
       # Automatic artifact classification  
       'ICLabel': {
           'enabled': True,
           'value': {
               'ic_flags_to_reject': [
                   'muscle',     # Muscle tension
                   'heart',      # Heartbeat  
                   'eog',        # Eye movements
                   'ch_noise',   # Channel noise
                   'line_noise'  # Electrical interference
               ],
               'ic_rejection_threshold': 0.3  # Confidence threshold
           }
       }
   }

The ``temp_highpass_for_ica`` option applies a temporary high-pass filter during ICA fitting to reduce low-frequency drifts without permanently modifying the data.

**Epoching and Analysis Preparation:**

.. code-block:: python

   config = {
       'epoch_settings': {
           'enabled': True,
           'value': {
               'tmin': -1,           # Epoch start (seconds)
               'tmax': 1,            # Epoch end (seconds)
           },
           'event_id': None,         # For resting state (no events)
           'remove_baseline': {
               'enabled': False,     # Baseline correction
               'window': [None, 0]   # Baseline window
           },
           'threshold_rejection': {
               'enabled': False,     # Simple amplitude rejection
               'volt_threshold': {
                   'eeg': 125e-6     # Rejection threshold (microvolts)
               }
           }
       }
   }

🔧 Common Customization Examples
--------------------------------

**High-Gamma Analysis Task:**

.. code-block:: python

   # For studying high-frequency brain activity
   config = {
       'resample_step': {'enabled': True, 'value': 1000},  # Higher sampling rate
       'filtering': {
           'enabled': True,
           'value': {
               'l_freq': 30,     # Focus on gamma frequencies  
               'h_freq': 200,    # Capture high-gamma
               'notch_freqs': [60, 120, 180],  # Multiple harmonics
               'notch_widths': 2
           }
       },
       'epoch_settings': {
           'enabled': True,
           'value': {
               'tmin': -0.2,     # Shorter epochs for high-freq analysis
               'tmax': 0.8
           }
       }
   }

**Clinical/Pediatric Populations:**

.. code-block:: python

   # Gentler processing for clinical data
   config = {
       'filtering': {
           'enabled': True,
           'value': {
               'l_freq': 0.5,    # Preserve more low frequencies
               'h_freq': 50,     # Conservative high-frequency cutoff
               'notch_freqs': [60],
               'notch_widths': 3
           }
       },
       'ICLabel': {
           'enabled': True,
           'value': {
               'ic_flags_to_reject': ['line_noise'],  # Only remove clear artifacts
               'ic_rejection_threshold': 0.7  # Higher confidence required
           }
       }
   }

**Connectivity Analysis:**

.. code-block:: python

   # Optimized for connectivity studies
   config = {
       'resample_step': {'enabled': True, 'value': 250},
       'filtering': {
           'enabled': True,
           'value': {
               'l_freq': 1,
               'h_freq': 45,     # Avoid muscle contamination
               'notch_freqs': [60, 120],
               'notch_widths': 2
           }
       },
       'epoch_settings': {
           'enabled': True,
           'value': {
               'tmin': -2,       # Longer epochs for connectivity
               'tmax': 2
           }
       }
   }

🔄 Advanced Workflow Customization
----------------------------------

**Custom Processing Steps:**

.. code-block:: python

   class AdvancedCustomTask(Task):
       def __init__(self, config):
           self.settings = globals()['config']
           super().__init__(config)
       
       def run(self) -> None:
           # Standard preprocessing
           self.import_raw()
           self.resample_data()
           self.filter_data()
           
           # Custom preprocessing step
           self.custom_artifact_detection()
           
           # Continue with standard workflow
           self.clean_bad_channels()
           self.rereference_data()
           
           # Custom ICA approach
           self.run_custom_ica()
           
           # Standard epoching and reports
           self.create_regular_epochs(export=True)
           self.generate_reports()
       
       def custom_artifact_detection(self):
           """Custom method for artifact detection."""
           # Your custom artifact detection code here
           # This could include specialized algorithms for your data type
           pass
       
       def run_custom_ica(self):
           """Custom ICA implementation."""
           # Run standard ICA first
           self.run_ica()
           
           # Add custom post-ICA processing
           # e.g., manual component review, custom classification
           pass

**Event-Related Potential (ERP) Task:**

.. code-block:: python

   # Configuration for ERP analysis
   config = {
       'resample_step': {'enabled': True, 'value': 500},
       'filtering': {
           'enabled': True,
           'value': {
               'l_freq': 0.1,    # Preserve slow ERPs
               'h_freq': 30,     # Avoid muscle artifacts
               'notch_freqs': [60],
               'notch_widths': 2
           }
       },
       'epoch_settings': {
           'enabled': True,
           'value': {
               'tmin': -0.2,     # Pre-stimulus baseline
               'tmax': 1.0,      # Post-stimulus response
           },
           'event_id': {         # Specific event types
               'target': 1,
               'standard': 2
           },
           'remove_baseline': {
               'enabled': True,
               'window': [-0.2, 0]  # Remove pre-stimulus activity
           }
       }
   }

   class ERPTask(Task):
       def __init__(self, config):
           self.settings = globals()['config']
           super().__init__(config)
       
       def run(self) -> None:
           self.import_raw()
           self.resample_data()
           self.filter_data()
           self.clean_bad_channels()
           self.rereference_data()
           
           # Find events in the data
           self.find_events()
           
           # Run ICA on continuous data
           self.run_ica()
           
           # Create event-locked epochs
           self.create_eventid_epochs(export=True)
           
           # Generate ERP-specific reports
           self.generate_reports()

📊 Testing and Validation
-------------------------

**Test your custom task:**

.. code-block:: bash

   # Test with a small file first
   autoclean process MyCustomTask test_data.raw --dry-run
   
   # Run actual processing
   autoclean process MyCustomTask test_data.raw
   
   # Check the results
   autoclean config show

**Validate processing quality:**

1. **Review quality reports:** Check that artifact removal worked appropriately
2. **Compare with built-in tasks:** Ensure your custom approach improves results
3. **Test with multiple files:** Verify consistency across participants
4. **Check analysis compatibility:** Ensure outputs work with your analysis pipeline

🎯 Best Practices
-----------------

**Start Simple:**
- Begin with minimal changes to existing tasks
- Test each modification before adding complexity
- Document your parameter choices

**Version Control:**
- Save different versions of your task files
- Document what each version is designed for
- Keep notes on what works well

**Share with Your Lab:**
- Custom tasks can be shared by copying the .py file
- Document the intended use case
- Include example usage commands

**Parameter Documentation:**
- Comment your config thoroughly
- Explain why you chose specific values
- Note any data-specific requirements

🆘 Troubleshooting Custom Tasks
------------------------------

**Task not found:**

.. code-block:: bash

   # Check task was discovered
   autoclean task list --include-custom
   
   # Verify file is in correct location
   autoclean config show
   
   # List files in tasks directory
   ls ~/Documents/Autoclean-EEG/tasks/

**Processing errors:**

.. code-block:: bash

   # Check logs for detailed error messages
   autoclean config show
   # Look in output/*/logs/ folder

**Poor results:**
- Review configuration parameters
- Compare with built-in task outputs
- Check that your processing steps are appropriate for your data

**Python syntax errors:**
- Verify proper indentation (Python is picky!)
- Check that all quotes and brackets match
- Test your Python file syntax: `python -m py_compile your_task.py`

🎉 Next Steps
-------------

Now that you can create custom tasks:

1. **Experiment with parameters:** Find the optimal settings for your data
2. **Share with colleagues:** Collaborate on task development
3. **Advanced features:** Explore custom mixins for novel processing methods
4. **Integration:** Connect your tasks to analysis pipelines

**Recommended follow-up tutorials:**
- :doc:`creating_a_custom_mixin` - Build entirely new processing methods
- :doc:`understanding_results` - Working with AutoClean outputs
- :doc:`first_time_processing` - Basic processing workflows
 