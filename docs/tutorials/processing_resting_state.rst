Processing Resting State EEG Data
==================================

Resting state EEG is one of the most common paradigms in neuroscience research. This guide shows you how to process resting state data effectively with AutoClean.

🎯 What is Resting State EEG?
-----------------------------

Resting state EEG records brain activity while participants are at rest, typically:
- Sitting quietly with eyes open or closed
- Not performing any specific task
- Usually 5-10 minutes of recording
- Used to study baseline brain activity and connectivity

**Common research applications:**
- Brain connectivity analysis
- Power spectral analysis  
- Individual differences in brain activity
- Clinical assessments
- Biomarker research

📊 The RestingEyesOpen Task
---------------------------

AutoClean includes a specialized task for resting state data that:
- Removes common artifacts (eye blinks, muscle tension, line noise)
- Applies optimal filtering for resting state analysis
- Creates clean epochs suitable for connectivity analysis
- Generates comprehensive quality control reports

**What it does automatically:**
1. **Filtering:** Removes slow drifts and high-frequency noise
2. **Artifact Detection:** Identifies bad channels and time periods
3. **ICA Cleaning:** Removes eye movements, muscle artifacts, and heartbeat
4. **Epoching:** Creates analysis-ready data segments
5. **Quality Control:** Provides detailed processing reports

🚀 Quick Start: Process Resting State Data
------------------------------------------

**For a single file:**

.. code-block:: bash

   autoclean process RestingEyesOpen subject001_rest.raw

**For multiple files in a folder:**

.. code-block:: bash

   autoclean process RestingEyesOpen data_folder/

**With custom output location:**

.. code-block:: bash

   autoclean process RestingEyesOpen subject001_rest.raw --output results/

⚙️ Understanding the Processing Steps
-------------------------------------

The RestingEyesOpen task performs these steps automatically:

**1. Data Import and Validation**
   - Loads your EEG file
   - Validates data integrity
   - Sets up electrode positions

**2. Basic Preprocessing**
   - Resamples to 250 Hz (reduces file size, maintains quality)
   - Applies bandpass filter (1-100 Hz) to remove drifts and noise
   - Removes line noise (50/60 Hz) automatically

**3. Channel and Artifact Detection**
   - Identifies bad channels with poor signal quality
   - Detects noisy time periods
   - Flags excessive artifacts for review

**4. Advanced Cleaning**
   - Runs Independent Component Analysis (ICA)
   - Automatically classifies components as brain activity vs. artifacts
   - Removes eye movements, muscle tension, and heartbeat artifacts

**5. Epoching and Quality Control**
   - Creates 2-second epochs for analysis
   - Removes epochs with residual artifacts
   - Generates comprehensive quality reports

📈 Interpreting Your Results
----------------------------

After processing, you'll find these key outputs:

**Quality Control Report (metadata/run_report.pdf)**

Look for these important metrics:

*Data Quality Indicators:*
- **Channels kept:** Should be >90% for good quality data
- **Epochs kept:** Should be >70% for reliable results  
- **ICA components removed:** Typically 3-8 artifact components

*Red flags to watch for:*
- ❌ <70% channels kept = poor electrode contact
- ❌ <50% epochs kept = very noisy data
- ❌ >15 ICA components removed = possible over-cleaning

**Processed Data Files**

*bids/derivatives/cleaned_data/*
- **continuous_clean.fif:** Artifact-free continuous data
- **epochs_clean.fif:** Clean 2-second epochs ready for analysis
- **ica_solution.fif:** ICA decomposition for review

*stage/* (intermediate files)
- Raw data at different processing stages
- Useful for troubleshooting or custom analysis

🔧 Customizing for Your Research
--------------------------------

The default RestingEyesOpen task works well for most studies, but you can customize it:

**Common modifications needed:**

**Different epoch length:**
- Default: 2-second epochs  
- For connectivity: Often want 4-8 second epochs
- For spectral analysis: 2-4 second epochs work well

**Different frequency bands:**
- Default: 1-100 Hz bandpass
- For alpha analysis: Might want 0.5-40 Hz
- For gamma: Might want 1-150 Hz

**Stricter/looser artifact rejection:**
- Default: Balanced for most studies
- Clinical data: Often needs looser criteria
- High-precision research: Might need stricter criteria

**To customize:** See :doc:`creating_custom_task` for detailed instructions.

📊 Best Practices for Resting State
-----------------------------------

**Data Collection Tips:**
- Record at least 5 minutes (8-10 minutes preferred)
- Consistent instructions across participants
- Note eyes open vs. eyes closed conditions
- Minimize environmental distractions

**Processing Recommendations:**
- Always review quality control reports
- Check that >70% of epochs are retained
- Verify ICA removed appropriate artifacts
- Document any custom processing parameters

**Analysis Considerations:**
- First/last minute often noisier - consider excluding
- Eyes open vs. closed have different spectral profiles
- Individual differences in alpha frequency are normal
- Connectivity measures sensitive to residual artifacts

🆘 Troubleshooting Common Issues
-------------------------------

**"Too many bad channels" warning:**

.. code-block:: bash

   # Check your electrode montage and impedances
   # Bad channels usually indicate:
   # - Poor electrode contact
   # - Broken electrodes  
   # - Wrong montage specification

*Solutions:*
- Verify electrode positions were set correctly
- Check original data quality
- Consider manual bad channel marking before processing

**"Insufficient clean epochs" error:**

.. code-block:: bash

   # This means >50% of data was marked as artifactual
   # Common causes:
   # - Very noisy environment during recording
   # - Participant movement/talking
   # - Equipment malfunction

*Solutions:*
- Review original recording quality
- Consider looser artifact detection settings
- Check if data is actually resting state (not task)

**ICA removes too many/few components:**

*Too many (>10):*
- Data might be very noisy
- ICA may be over-fitting
- Consider pre-cleaning steps

*Too few (<2):*
- Participant had very little eye movement
- Very clean data (this is good!)
- Verify eye artifacts are actually removed

**Processing takes very long:**

*Normal processing time:*
- 10 minutes data: ~3-5 minutes processing
- 30 minutes data: ~8-12 minutes processing

*If much slower:*
- Computer may be low on memory
- Other programs using CPU
- Very large file size

📋 Batch Processing Multiple Participants
-----------------------------------------

**Organize your data:**

.. code-block::

   data/
   ├── sub-001_rest.raw
   ├── sub-002_rest.raw  
   ├── sub-003_rest.raw
   └── ...

**Process all files:**

.. code-block:: bash

   autoclean process RestingEyesOpen data/

**Monitor progress:**
- AutoClean will process each file sequentially
- Check the command window for progress updates
- Each file gets its own results folder

**Quality control for batches:**
- Review summary statistics across participants
- Flag participants with unusual processing metrics
- Check for systematic issues across the dataset

🎯 Advanced Analysis Integration
-------------------------------

**For Python users:**

.. code-block:: python

   import mne
   from autoclean import Pipeline
   
   # Process data
   pipeline = Pipeline()
   pipeline.process_file("subject001_rest.raw", "RestingEyesOpen")
   
   # Load results for analysis
   epochs = mne.read_epochs("output/subject001_rest_*/bids/derivatives/epochs_clean.fif")
   
   # Your analysis code here
   psd = epochs.compute_psd()
   connectivity = mne_connectivity.spectral_connectivity_epochs(epochs)

**For MATLAB/EEGLAB users:**

.. code-block:: matlab

   % Load AutoClean results
   EEG = pop_loadset('epochs_clean.set', 'output/subject001_rest_*/bids/derivatives/');
   
   % Continue with your analysis
   [spectra, freqs] = spectopo(EEG.data, 0, EEG.srate);

**For R users:**

.. code-block:: r

   library(eegUtils)
   
   # Load processed data
   eeg_data <- import_set("output/subject001_rest_*/bids/derivatives/epochs_clean.set")
   
   # Continue analysis
   psd <- compute_psd(eeg_data)

🎉 Next Steps
-------------

Now that you can process resting state data:

1. **Quality Control:** Learn to systematically review processing quality
2. **Batch Processing:** Scale up to process entire datasets  
3. **Custom Tasks:** Create specialized workflows for your research
4. **Integration:** Connect AutoClean to your analysis pipeline

**Recommended follow-up tutorials:**
- :doc:`quality_control_best_practices`
- :doc:`batch_processing_datasets`  
- :doc:`python_integration`
- :doc:`creating_custom_task`