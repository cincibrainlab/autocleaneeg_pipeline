# Signal Processing Mixins

The Signal Processing Mixins provide a comprehensive set of methods for processing EEG data. These mixins are designed to be used in conjunction with the Task classes to provide a modular and flexible approach to EEG data processing.

## Overview

The Signal Processing Mixins are organized into specialized categories, each focusing on a specific aspect of EEG data processing:

- **Base**: Core functionality shared by all signal processing mixins
- **Resampling**: Methods for changing the sampling rate of EEG data
- **Artifacts**: Detection and removal of various types of artifacts
- **Channels**: Operations related to EEG channels (dropping, interpolation)
- **Segmentation**: Methods for segmenting continuous data
- **Reference**: Techniques for re-referencing EEG data
- **Epochs**: Creation and processing of epoched data

## Base Mixin

::: autoclean.mixins.signal_processing.base.SignalProcessingMixin
    options:
      show_root_heading: true
      show_source: true

## Resampling Mixin

::: autoclean.mixins.signal_processing.resampling.ResamplingMixin
    options:
      show_root_heading: true
      show_source: true

## Artifacts Mixin

::: autoclean.mixins.signal_processing.artifacts.ArtifactsMixin
    options:
      show_root_heading: true
      show_source: true

## Channels Mixin

::: autoclean.mixins.signal_processing.channels.ChannelsMixin
    options:
      show_root_heading: true
      show_source: true

## Segmentation Mixin

::: autoclean.mixins.signal_processing.segmentation.SegmentationMixin
    options:
      show_root_heading: true
      show_source: true

## Reference Mixin

::: autoclean.mixins.signal_processing.reference.ReferenceMixin
    options:
      show_root_heading: true
      show_source: true

## Epochs Mixins

### Main Epochs Mixin

::: autoclean.mixins.signal_processing.epochs.EpochsMixin
    options:
      show_root_heading: true
      show_source: true

### Regular Epochs Mixin

::: autoclean.mixins.signal_processing.regular_epochs.RegularEpochsMixin
    options:
      show_root_heading: true
      show_source: true

### Event ID Epochs Mixin

::: autoclean.mixins.signal_processing.eventid_epochs.EventIDEpochsMixin
    options:
      show_root_heading: true
      show_source: true

### Prepare Epochs ICA Mixin

::: autoclean.mixins.signal_processing.prepare_epochs_ica.PrepareEpochsICAMixin
    options:
      show_root_heading: true
      show_source: true

### GFP Clean Epochs Mixin

::: autoclean.mixins.signal_processing.gfp_clean_epochs.GFPCleanEpochsMixin
    options:
      show_root_heading: true
      show_source: true

### AutoReject Epochs Mixin

::: autoclean.mixins.signal_processing.autoreject_epochs.AutoRejectEpochsMixin
    options:
      show_root_heading: true
      show_source: true
