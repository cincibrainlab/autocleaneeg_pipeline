Certainly! Here's the text formatted in markdown that you can copy directly into a file:

````markdown
# PyLossless

PyLossless is an automated EEG processing pipeline that annotates continuous EEG data[^1][^9]. It is designed to be fast, open-source, and built on Python, with a focus on non-destructive processing that keeps EEG data continuous while noting bad channels, times, and independent components[^9].

## Installation

PyLossless can be installed directly from its GitHub repository:

```bash
git clone git@github.com:lina-usc/pylossless.git
pip install --editable ./pylossless
```

Alternatively, for a static version:

```bash
pip install git+https://github.com/lina-usc/pylossless.git
```

## Pipeline Object and Core Functionality

The main component of PyLossless is the `LosslessPipeline` object. Here's a basic example of how to use it:

```python
import pylossless as ll
import mne

# Load raw EEG data
fname = mne.datasets.sample.data_path() / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(fname, preload=True)

# Initialize and configure the pipeline
config = ll.config.Config()
config.load_default()
pipeline = ll.LosslessPipeline(config=config)

# Run the pipeline
pipeline.run_with_raw(raw)

# Access flagged channels and epochs
print(pipeline.flagged_chs)
print(pipeline.flagged_epochs)

# Save the processed data
pipeline.save(pipeline.get_derivative_path(bids_path), overwrite=True)
```

## Key Pipeline Steps

The PyLossless pipeline includes several key steps[^5]:

1. Flagging noisy channels
2. Flagging noisy time periods
3. Filtering
4. Flagging uncorrelated channels
5. Flagging bridged channels
6. Flagging the rank channel
7. Flagging uncorrelated epochs
8. Running initial ICA
9. Flagging time periods with noisy ICs
10. Running final ICA and ICLabel

## ICA and Artifact Detection

PyLossless uses Independent Component Analysis (ICA) for artifact detection and removal. It performs two ICA runs:

1. Initial ICA: Used for preliminary artifact detection
2. Final ICA: Performed after removing noisy periods identified in the initial run

The pipeline uses the Extended Infomax ICA algorithm[^5].

## BIDS Integration

PyLossless is designed to work with BIDS (Brain Imaging Data Structure) datasets. The pipeline can be run on BIDS-formatted data using the `run_with_raw` method, which takes an MNE `Raw` object as input[^5].

## Configuration

The pipeline's behavior can be customized using a `Config` object. Default configurations can be loaded and modified:

```python
config = ll.config.Config()
config.load_default()
# Modify specific parameters as needed
config.ica_method = 'fastica'
```

## Integration with MNE-Python

PyLossless is built to work seamlessly with MNE-Python, a popular Python package for processing MEG and EEG data. It uses MNE-Python objects for data handling and some processing steps[^1][^5].

## Output and Review

After processing, PyLossless provides:

- Flagged channels and time periods
- ICA components
- Annotations for various types of artifacts

A web dashboard is available for reviewing the output and making informed decisions about the preprocessing results[^9].

For more detailed documentation and examples, refer to the official PyLossless documentation at [https://pylossless.readthedocs.io/](https://pylossless.readthedocs.io/)[^9].

---

# How does Pylossless handle epoching and events? PyLossless Continuous EEG Data Processing

PyLossless is designed to maintain continuous EEG data throughout the preprocessing pipeline, avoiding the need for early epoching[^1]. This approach offers several advantages:

1. **Flexibility**: It allows researchers to epoch their data at a later stage, according to their specific analysis needs[^3].

2. **Reusability**: The continuous data can be used for multiple analyses with different epoching strategies[^3].

3. **Non-destructive processing**: PyLossless annotates artifacts and bad segments without altering the original continuous data structure[^1].

## Event Handling

Regarding events, PyLossless does not directly process or manipulate event information. Instead, it focuses on identifying and annotating artifacts in the continuous EEG data. The pipeline flags noisy channels, time periods, and independent components without segmenting the data into epochs based on events[^1][^3].

However, PyLossless does use a concept of "epochs" internally for some of its processing steps. These are not traditional event-related epochs, but rather time segments used for computational purposes. For example:

1. **Flagging uncorrelated epochs**: The pipeline identifies periods where many sensors are uncorrelated with neighboring sensors[^1].

2. **Initial ICA and flagging of outlying independent components**: PyLossless uses epochs to detect non-stationary periods in the ICA decomposition[^1].

After processing, researchers can use the cleaned continuous data and apply their own epoching based on events as needed for their specific analyses. The `RejectionPolicy` class provided by PyLossless can be used to apply the pipeline's decisions to the data, which could then be epoched using other tools like MNE-Python[^3].

# PyLossless and MNE-Tools Integration

PyLossless integrates closely with the MNE-Tools ecosystem, particularly MNE-Python, to provide a comprehensive and non-destructive EEG processing pipeline. Here are the key aspects of this integration:

1. **Data Handling**: PyLossless uses MNE-Python objects for data handling, allowing seamless interaction with MNE-Python's data structures[^1].

2. **Continuous EEG Structure**: The pipeline maintains the continuous EEG structure using MNE-Python's native support for annotating continuous EEG data[^1].

3. **BIDS Compatibility**: PyLossless integrates with MNE-BIDS for handling BIDS-compliant datasets, ensuring compatibility with standardized neuroimaging data formats[^1].

4. **ICA Processing**: The pipeline utilizes MNE-Python's built-in ICA routines for artifact detection and removal. It uses MNE-ICALabel for automatically labeling independent components[^1].

5. **Artifact Annotation**: PyLossless adds flagged sensors as bad sensors to the MNE-Python info structure of the continuous raw EEG[^1].

6. **Data Transformation**: While PyLossless doesn't alter the input EEG data, it provides a `RejectionPolicy` class that can be used to apply the pipeline's decisions, such as removing specific ICs from the EEG data using MNE-Python functionality[^1].

7. **Visualization**: Although PyLossless has its own QCR dashboard, it can potentially be extended to use MNE-Python's visualization capabilities for additional data inspection and quality control[^1].

This integration allows researchers to leverage the strengths of both PyLossless and MNE-Python, combining PyLossless's non-destructive preprocessing approach with MNE-Python's extensive analysis and visualization tools[^1][^7][^8].

## Citations

[^1]: [https://github.com/lina-usc/pylossless](https://github.com/lina-usc/pylossless)
[^2]: [https://github.com/lina-usc/pylossless/blob/main/setup.py](https://github.com/lina-usc/pylossless/blob/main/setup.py)
[^3]: [https://pmc.ncbi.nlm.nih.gov/articles/PMC2895624/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895624/)
[^4]: [https://github.com/sappelhoff/pyprep](https://github.com/sappelhoff/pyprep)
[^5]: [https://pylossless.readthedocs.io/en/latest/auto_examples/plot_10_run_pipeline.html](https://pylossless.readthedocs.io/en/latest/auto_examples/plot_10_run_pipeline.html)
[^6]: [https://pylossless.readthedocs.io/en/latest/install.html](https://pylossless.readthedocs.io/en/latest/install.html)
[^7]: [https://www.biorxiv.org/content/biorxiv/early/2024/10/27/2024.03.19.585723.full.pdf](https://www.biorxiv.org/content/biorxiv/early/2024/10/27/2024.03.19.585723.full.pdf)
[^8]: [https://github.com/lina-usc/q1k_sync_loss](https://github.com/lina-usc/q1k_sync_loss)
[^9]: [https://pylossless.readthedocs.io/en/latest/](https://pylossless.readthedocs.io/en/latest/)
````