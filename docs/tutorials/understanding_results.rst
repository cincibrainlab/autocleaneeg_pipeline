Understanding Your AutoClean Results
===================================

After AutoClean processes your EEG data, it creates a comprehensive set of outputs. This guide helps you understand what everything means and how to use your results effectively.

📁 Overview: What AutoClean Creates
-----------------------------------

When AutoClean finishes processing, you'll find a timestamped folder in your workspace output directory:

.. code-block::

   output/
   └── subject001_rest_2025-06-11_14-30-25/
       ├── bids/                 # Processed data (what you need!)
       ├── logs/                 # Processing details  
       ├── metadata/             # Reports and summaries
       └── stage/                # Intermediate files

**The most important folder is `bids/` - this contains your cleaned, analysis-ready data.**

🎯 Key Files You Need to Know
------------------------------

**Start Here: Quality Control Report**

.. code-block::

   metadata/run_report.pdf

This PDF report is your first stop! It contains:
- Visual before/after comparisons
- Data quality metrics  
- Processing summary
- Any warnings or issues

**Your Cleaned Data**

.. code-block::

   bids/derivatives/
   ├── continuous_clean.fif      # Continuous cleaned EEG
   ├── epochs_clean.fif          # Epoched data ready for analysis
   └── ica_solution.fif          # ICA artifact removal details

**Processing Details**

.. code-block::

   metadata/
   ├── run_report.pdf            # Visual quality report
   ├── processing_summary.json   # Technical details
   └── data_quality_metrics.json # Quantitative measures

📊 Reading Your Quality Control Report
--------------------------------------

**Page 1: Processing Summary**

*Data Overview:*
- Original file information
- Total recording duration  
- Number of channels
- Sampling rate

*Processing Statistics:*
- **Channels kept:** Should be >90% for good data
- **Data segments kept:** Should be >70% for reliable results
- **Artifacts removed:** Shows what was cleaned

**Page 2: Before/After Comparison**

*Raw Data Visualization:*
- Shows original EEG traces
- Highlights problematic channels/periods
- Displays frequency content

*Clean Data Visualization:*  
- Shows processed EEG traces
- Demonstrates artifact removal
- Confirms data quality improvement

**Page 3: Artifact Analysis**

*ICA Components:*
- Brain activity components (kept)
- Eye movement artifacts (removed)
- Muscle artifacts (removed)  
- Heart artifacts (removed)

*Quality Metrics:*
- Signal-to-noise improvement
- Artifact detection statistics
- Data retention percentages

🚦 Interpreting Quality Indicators
----------------------------------

**Green Light (Excellent Quality):**
- ✅ >95% channels retained
- ✅ >80% epochs retained  
- ✅ 3-8 ICA components removed
- ✅ Clear artifact removal visible

**Yellow Light (Good Quality - Check These):**
- ⚠️ 85-95% channels retained
- ⚠️ 70-80% epochs retained
- ⚠️ 2-3 or 8-12 ICA components removed
- ⚠️ Some residual artifacts visible

**Red Light (Needs Attention):**
- ❌ <85% channels retained
- ❌ <70% epochs retained
- ❌ <2 or >12 ICA components removed
- ❌ Poor artifact removal

🧠 Understanding Your Processed Data
------------------------------------

**Continuous Clean Data (continuous_clean.fif)**

*What it is:*
- Your original EEG with artifacts removed
- Maintains original time structure
- Ready for time-domain analysis

*Use it for:*
- Connectivity analysis
- Time-frequency analysis
- Event-related potential analysis
- Custom epoching

**Epoched Clean Data (epochs_clean.fif)**

*What it is:*
- Data split into short segments (usually 2-4 seconds)
- Bad epochs already removed
- Ready for spectral analysis

*Use it for:*
- Power spectral density analysis
- Frequency domain connectivity
- Statistical comparisons
- Machine learning features

**ICA Solution (ica_solution.fif)**

*What it is:*
- Mathematical decomposition of your data
- Separates brain activity from artifacts
- Shows what was removed and why

*Use it for:*
- Verifying artifact removal was appropriate
- Understanding data cleaning decisions
- Advanced troubleshooting

📈 Data Quality Metrics Explained
---------------------------------

**Channel Quality Metrics**

*Channels Interpolated:*
- Bad channels that were reconstructed
- Normal: 0-5% of channels
- Concerning: >10% of channels

*Channel Noise Levels:*
- Average noise after cleaning
- Lower values = cleaner data
- Typical range: 10-50 μV

**Temporal Quality Metrics**

*Epochs Rejected:*
- Percentage of data removed as artifacts
- Normal: 10-30% rejection
- Concerning: >50% rejection

*Artifact Types Detected:*
- Eye movements: Usually 1-3 components
- Muscle artifacts: Usually 1-2 components  
- Heart artifacts: Usually 0-1 components

**Spectral Quality Metrics**

*Frequency Band Power:*
- Delta (0.5-4 Hz): Slow wave activity
- Theta (4-8 Hz): Attention/memory
- Alpha (8-13 Hz): Relaxed awareness
- Beta (13-30 Hz): Active thinking
- Gamma (30+ Hz): High-level processing

*Line Noise Reduction:*
- 50/60 Hz artifact removal
- Should show significant reduction
- Residual noise <5% of original

🔍 Advanced Analysis: What to Look For
-------------------------------------

**Successful Artifact Removal Signs:**

*ICA Components Removed:*
- Clear eye movement patterns in frontal components
- Muscle artifacts with high-frequency, localized patterns
- Heart artifacts with regular, rhythmic patterns

*Clean Data Characteristics:*
- Smooth, brain-like oscillations
- Reasonable amplitude ranges (10-100 μV)
- No obvious periodic artifacts
- Consistent noise levels across channels

**Potential Issues to Watch For:**

*Over-cleaning:*
- Too many ICA components removed (>15)
- Unrealistically low noise levels
- Loss of natural brain oscillations

*Under-cleaning:*
- Obvious eye blinks still visible
- Muscle artifacts in temporal regions
- 50/60 Hz line noise still present

🛠️ Using Your Results in Analysis Software
-------------------------------------------

**Loading Data in Python (MNE):**

.. code-block:: python

   import mne
   
   # Load continuous data
   raw = mne.io.read_raw_fif('bids/derivatives/continuous_clean.fif')
   
   # Load epoched data  
   epochs = mne.read_epochs('bids/derivatives/epochs_clean.fif')
   
   # Your analysis here
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

📋 Quality Control Checklist
----------------------------

Before using your data for analysis, check:

**✅ Data Integrity**
- [ ] Processing completed without errors
- [ ] Output files were created successfully  
- [ ] File sizes are reasonable (not empty or huge)

**✅ Quality Metrics**
- [ ] >70% of epochs retained
- [ ] >85% of channels retained
- [ ] Appropriate number of ICA components removed

**✅ Visual Inspection**
- [ ] Clean data looks like brain activity
- [ ] No obvious artifacts remain
- [ ] Reasonable amplitude ranges

**✅ Processing Log Review**
- [ ] No critical errors in logs
- [ ] All processing steps completed
- [ ] Parameter settings were appropriate

🆘 When Results Look Wrong
-------------------------

**High Data Loss (>50% epochs rejected):**
- Check original data quality
- Verify appropriate task was used
- Consider adjusting artifact detection sensitivity

**Poor Artifact Removal:**
- Review ICA component classifications
- Check if additional preprocessing needed
- Verify electrode positions were correct

**Unexpected Processing Errors:**
- Check log files in logs/ folder
- Verify input data format is supported
- Ensure sufficient disk space available

**File Format Issues:**
- AutoClean outputs standard formats (.fif, .set)
- Use conversion tools if needed for your analysis software
- Contact support for format-specific questions

📊 Documenting Your Processing
-----------------------------

**For Research Papers:**
Keep records of:
- AutoClean version used
- Task name and parameters
- Data quality metrics
- Any custom processing steps

**Example Methods Text:**
"EEG data were preprocessed using AutoClean v1.4.1 with the RestingEyesOpen task. Data were filtered (1-100 Hz), bad channels interpolated (mean: 2.3%), and artifacts removed using ICA. On average, 78% of epochs were retained after artifact rejection."

🎉 Next Steps
-------------

Now that you understand your results:

1. **Start Analysis:** Use your clean data for research questions
2. **Quality Control:** Develop systematic QC procedures  
3. **Optimization:** Fine-tune processing for your specific needs
4. **Automation:** Scale up to process larger datasets

**Recommended tutorials:**
- :doc:`batch_processing_datasets` - Process multiple files efficiently
- :doc:`quality_control_best_practices` - Systematic QC procedures
- :doc:`python_integration` - Advanced analysis workflows