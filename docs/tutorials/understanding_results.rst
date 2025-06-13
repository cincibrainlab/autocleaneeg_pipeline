Understanding AutoClean Results
===============================

AutoClean creates a structured output directory for each processing run. This guide explains the organization and contents of your results.

Output Directory Structure
---------------------------

AutoClean creates a folder named after the task used for processing:

.. code-block::

   output/
   └── RestingEyesOpen/          # Named after the task
       ├── bids/                 # BIDS-compliant processed data
       ├── flagged/              # Data flagged for quality issues
       ├── metadata/             # Processing metadata and logs
       ├── logs/                 # Detailed processing logs
       └── stage_files/          # Exported intermediate data

Directory Contents
------------------

**bids/** - Primary Output Data

Contains the main processed EEG data in BIDS format:
- Cleaned continuous data
- Epoched data (if applicable)
- Channel information and metadata
- BIDS-compliant file naming and structure

**flagged/** - Quality Control Flagged Data

Contains data that triggered quality control flags:
- Files with less than 50% epochs remaining after artifact rejection
- Data flagged for excessive channel rejection
- Processing runs that failed quality thresholds

**metadata/** - Processing Information

Contains processing metadata and summary information:
- Run summaries and statistics
- Processing parameters used
- Quality metrics and reports

**logs/** - Processing Logs

Detailed logs of the processing pipeline:
- Step-by-step processing information
- Error messages and warnings
- Timing and performance data

**stage_files/** - Exported Data

Contains any data marked for export during processing:
- Intermediate processing stages (if export=True was used)
- Custom export data specified in task configuration
- Stage-specific outputs for analysis

**derivatives/** - Reports and Visualizations

Quality control reports and visualizations:
- Processing reports (PDF/HTML)
- ICA component analysis
- Data quality visualizations
- Before/after comparison plots

Quality Control Assessment
--------------------------

**Processing Summary**

Key metrics to evaluate:
- Original file information and recording parameters
- Channel retention percentage (target: >90%)
- Epoch retention percentage (target: >70%)  
- ICA components removed (typical: 3-8 components)

**Data Quality Visualization**

Before/after processing comparisons show:
- Original EEG traces with artifacts
- Cleaned EEG traces post-processing
- Frequency domain improvements
- Artifact removal effectiveness

**ICA Component Analysis**

Components are categorized as:
- Neural activity (retained)
- Eye movement artifacts (removed)
- Muscle artifacts (removed)
- Cardiac artifacts (removed)

Quality Thresholds
------------------

**Acceptable Quality:**
- Greater than 90% channels retained
- Greater than 70% epochs retained
- 3-8 ICA components removed
- Visible artifact reduction

**Review Required:**
- 85-90% channels retained
- 60-70% epochs retained
- Less than 3 or more than 12 ICA components removed
- Excessive data loss

**Flagged Data:**
- Less than 85% channels retained
- Less than 60% epochs retained
- Poor artifact removal
- Data moved to flagged/ directory

Processed Data Files
--------------------

**Continuous Data**

File: continuous_clean.fif
- Artifact-removed continuous EEG
- Preserves temporal structure
- Suitable for connectivity and time-frequency analysis

**Epoched Data**

File: epochs_clean.fif  
- Segmented data with bad epochs removed
- Ready for spectral analysis
- Optimized for statistical comparisons
**ICA Solution**

The ICA decomposition file contains:
- Component weights and topographies
- Classification of neural vs. artifact components  
- Basis for artifact removal decisions

Data Quality Metrics
---------------------

**Channel Metrics**

- Channels interpolated: Typically 0-5% of total channels
- Channel noise levels: Post-cleaning noise measurements
- Bad channel detection: Automated identification of problematic electrodes

**Temporal Metrics**

- Epoch rejection rate: Percentage of data segments removed
- Artifact detection: Types and quantities of artifacts identified
- Data retention: Amount of usable data remaining after cleaning

**Spectral Metrics**

- Frequency band power across standard EEG bands
- Line noise reduction at 50/60 Hz
- Spectral quality improvements post-processing

Analysis Considerations
-----------------------

**Quality Indicators**

Successful processing typically shows:
- Clear artifact component identification in ICA
- Reasonable data retention (>70% epochs, >90% channels)  
- Visible improvement in data quality
- Appropriate artifact removal without over-cleaning

**Potential Issues**

Review data if you observe:
- Excessive component removal (>15 ICA components)
- Poor data retention (<60% epochs or <85% channels)
- Residual artifacts in cleaned data
- Over-smoothed or unrealistic signal characteristics

Using Processed Data
--------------------

**Loading in Python (MNE)**

.. code-block:: python

   import mne
   
   # Load continuous cleaned data
   raw = mne.io.read_raw_fif('bids/continuous_clean.fif')
   
   # Load epoched data
   epochs = mne.read_epochs('bids/epochs_clean.fif')
   
   # Perform your analysis
   psd = epochs.compute_psd()

**Loading Data in MATLAB (EEGLAB):**

.. code-block:: matlab

   % AutoClean can export .set files for EEGLAB
   EEG = pop_loadset('epochs_clean.set', 'bids/derivatives/');
   
   % Continue with EEGLAB analysis
   [spectra, freqs] = spectopo(EEG.data, 0, EEG.srate);

**Loading Data in R:**

.. code-block:: r

   library(eegUtils)
   
   # Load processed data
   eeg_data <- import_set("bids/derivatives/epochs_clean.set")
   
   # Continue analysis
   psd <- compute_psd(eeg_data)

Quality Control Checklist
-------------------------

Before proceeding with analysis, verify:

**Data Integrity**
- Processing completed without errors
- Output files created successfully
- File sizes are appropriate

**Quality Metrics**  
- Greater than 70% of epochs retained
- Greater than 85% of channels retained
- Reasonable number of ICA components removed

**Visual Inspection**
- Clean data exhibits brain-like characteristics
- Artifacts successfully removed
- Amplitude ranges are physiologically reasonable

**Log Review**
- No critical errors in processing logs
- All steps completed successfully
- Parameters applied correctly

Troubleshooting Issues
----------------------

**Excessive Data Loss**
- Review original data quality
- Verify appropriate task selection
- Consider parameter adjustments

**Poor Artifact Removal**
- Examine ICA component classifications
- Check electrode positioning accuracy
- Review preprocessing parameters

**Processing Errors**
- Examine log files in logs/ directory
- Verify input data format compatibility
- Ensure adequate disk space

Documentation for Publication
-----------------------------

Record the following information:
- AutoClean version number
- Task name and configuration
- Quality metrics and data retention
- Custom processing modifications

**Example Methods Description:**
"EEG data were preprocessed using AutoClean v2.0.0 with the RestingEyesOpen task. Data were filtered (1-100 Hz), bad channels interpolated (mean: 2.3%), and artifacts removed using ICA. On average, 78% of epochs were retained after artifact rejection."

**Recommended tutorials:**
- :doc:`batch_processing_datasets` - Process multiple files efficiently
- :doc:`quality_control_best_practices` - Systematic QC procedures
- :doc:`python_integration` - Advanced analysis workflows