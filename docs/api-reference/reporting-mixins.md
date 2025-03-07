# Reporting Mixins

The Reporting Mixins provide a comprehensive set of methods for visualizing and reporting EEG data processing results. These mixins are designed to be used in conjunction with the Task classes to provide a modular and flexible approach to generating visualizations, reports, and summaries from EEG data.

## Overview

The Reporting Mixins are organized into specialized categories, each focusing on a specific aspect of EEG data reporting:

- **Base**: Core functionality shared by all reporting mixins
- **Visualization**: Methods for generating visualizations of EEG data
- **ICA**: Visualizations and reports specific to ICA components
- **Reports**: Generation of comprehensive processing reports
- **Main**: Combined mixin that provides all reporting functionality

## Main Reporting Mixin

::: autoclean.mixins.reporting.ReportingMixin
    options:
      show_root_heading: true
      show_source: true

## Base Mixin

::: autoclean.mixins.reporting.base.ReportingMixin
    options:
      show_root_heading: true
      show_source: true

## Visualization Mixin

::: autoclean.mixins.reporting.visualization.VisualizationMixin
    options:
      show_root_heading: true
      show_source: true

## ICA Reporting Mixin

::: autoclean.mixins.reporting.ica.ICAReportingMixin
    options:
      show_root_heading: true
      show_source: true

## Report Generation Mixin

::: autoclean.mixins.reporting.reports.ReportGenerationMixin
    options:
      show_root_heading: true
      show_source: true
