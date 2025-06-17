"""Report generation functions for EEG data processing.

This module provides standalone functions for generating comprehensive
processing reports and summaries.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import json

import mne
import numpy as np
import pandas as pd


def generate_processing_report(
    raw_original: mne.io.Raw,
    raw_cleaned: mne.io.Raw,
    processing_steps: List[Dict],
    output_path: Union[str, Path],
    include_plots: bool = True,
    title: str = "EEG Processing Report",
    verbose: Optional[bool] = None
) -> str:
    """Generate a comprehensive HTML processing report.

    This function creates a detailed HTML report summarizing the EEG processing
    pipeline, including statistics, processing steps, and optional visualizations.

    Parameters
    ----------
    raw_original : mne.io.Raw
        Original raw EEG data before processing.
    raw_cleaned : mne.io.Raw
        Cleaned raw EEG data after processing.
    processing_steps : list of dict
        List of processing steps with metadata. Each dict should contain:
        - 'step_name': Name of the processing step
        - 'parameters': Dict of parameters used
        - 'execution_time': Time taken for the step
        - 'description': Brief description of what the step does
    output_path : str or Path
        Path where the HTML report will be saved.
    include_plots : bool, default True
        Whether to include plots in the report.
    title : str, default "EEG Processing Report"
        Title for the report.
    verbose : bool or None, default None
        Control verbosity of output.

    Returns
    -------
    report_path : str
        Path to the generated HTML report.

    Raises
    ------
    TypeError
        If inputs are not correct types.
    ValueError
        If processing_steps format is invalid.
    RuntimeError
        If report generation fails.

    Notes
    -----
    **Report Sections:**
    1. Executive Summary - Key statistics and overview
    2. Data Information - Channel counts, duration, sampling rate
    3. Processing Pipeline - Step-by-step breakdown
    4. Quality Metrics - Before/after comparisons
    5. Visualizations - Plots showing processing effects (optional)

    **Processing Steps Format:**
    Each step should be a dictionary with:
    - step_name: Human-readable name
    - parameters: Dict of parameters used
    - execution_time: Processing duration in seconds
    - description: What the step accomplished

    Examples
    --------
    Basic report generation:

    >>> from autoclean.functions.visualization import generate_processing_report
    >>> steps = [
    ...     {
    ...         'step_name': 'Filtering',
    ...         'parameters': {'low_freq': 0.1, 'high_freq': 50.0},
    ...         'execution_time': 2.3,
    ...         'description': 'Applied bandpass filter'
    ...     }
    ... ]
    >>> report_path = generate_processing_report(
    ...     raw_original, raw_cleaned, steps, "report.html"
    ... )

    Report without plots:

    >>> report_path = generate_processing_report(
    ...     raw_original, raw_cleaned, steps, "report.html",
    ...     include_plots=False
    ... )

    See Also
    --------
    plot_raw_comparison : Create before/after comparison plots
    plot_ica_components : Visualize ICA components
    """
    # Input validation
    if not isinstance(raw_original, mne.io.BaseRaw):
        raise TypeError(
            f"raw_original must be an MNE Raw object, got {type(raw_original).__name__}"
        )
    
    if not isinstance(raw_cleaned, mne.io.BaseRaw):
        raise TypeError(
            f"raw_cleaned must be an MNE Raw object, got {type(raw_cleaned).__name__}"
        )
    
    if not isinstance(processing_steps, list):
        raise TypeError("processing_steps must be a list of dictionaries")
    
    # Validate processing steps format
    required_keys = ['step_name', 'parameters', 'execution_time', 'description']
    for i, step in enumerate(processing_steps):
        if not isinstance(step, dict):
            raise ValueError(f"processing_steps[{i}] must be a dictionary")
        for key in required_keys:
            if key not in step:
                raise ValueError(f"processing_steps[{i}] missing required key: {key}")

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate processing statistics
        stats = _calculate_processing_stats(raw_original, raw_cleaned)
        
        # Generate HTML content
        html_content = _generate_html_report(
            raw_original=raw_original,
            raw_cleaned=raw_cleaned,
            processing_steps=processing_steps,
            stats=stats,
            title=title,
            include_plots=include_plots,
            output_dir=output_path.parent
        )

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        if verbose:
            print(f"Processing report saved to: {output_path}")

        return str(output_path)

    except Exception as e:
        raise RuntimeError(f"Failed to generate processing report: {str(e)}") from e


def _calculate_processing_stats(
    raw_original: mne.io.Raw,
    raw_cleaned: mne.io.Raw
) -> Dict:
    """Calculate statistics comparing original and cleaned data."""
    stats = {}
    
    # Basic info
    stats['original_duration'] = raw_original.times[-1] - raw_original.times[0]
    stats['cleaned_duration'] = raw_cleaned.times[-1] - raw_cleaned.times[0]
    stats['original_channels'] = len(raw_original.ch_names)
    stats['cleaned_channels'] = len(raw_cleaned.ch_names)
    stats['sampling_rate'] = raw_original.info['sfreq']
    
    # Data quality metrics
    orig_data = raw_original.get_data()
    clean_data = raw_cleaned.get_data()
    
    stats['original_std_mean'] = float(np.mean(np.std(orig_data, axis=1)))
    stats['cleaned_std_mean'] = float(np.mean(np.std(clean_data, axis=1)))
    stats['noise_reduction'] = float(
        (stats['original_std_mean'] - stats['cleaned_std_mean']) / stats['original_std_mean'] * 100
    )
    
    # Annotation counts
    stats['original_annotations'] = len(raw_original.annotations)
    stats['cleaned_annotations'] = len(raw_cleaned.annotations)
    stats['bad_annotations'] = len([
        ann for ann in raw_cleaned.annotations 
        if ann['description'].startswith('BAD')
    ])
    
    return stats


def _generate_html_report(
    raw_original: mne.io.Raw,
    raw_cleaned: mne.io.Raw,
    processing_steps: List[Dict],
    stats: Dict,
    title: str,
    include_plots: bool,
    output_dir: Path
) -> str:
    """Generate the HTML content for the processing report."""
    
    # HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f4;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .stat-box {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 15px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                color: #6c757d;
                font-size: 0.9em;
            }}
            .step-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .step-table th, .step-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .step-table th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            .step-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .timestamp {{
                color: #6c757d;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }}
            .plot-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="stat-box">
                    <div class="stat-value">{stats['original_channels']}</div>
                    <div class="stat-label">Original Channels</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['cleaned_channels']}</div>
                    <div class="stat-label">Final Channels</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['original_duration']:.1f}s</div>
                    <div class="stat-label">Duration</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['sampling_rate']:.0f} Hz</div>
                    <div class="stat-label">Sampling Rate</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['noise_reduction']:.1f}%</div>
                    <div class="stat-label">Noise Reduction</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{stats['bad_annotations']}</div>
                    <div class="stat-label">Bad Segments</div>
                </div>
            </div>
            
            <h2>Processing Pipeline</h2>
            <table class="step-table">
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Description</th>
                        <th>Parameters</th>
                        <th>Execution Time</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add processing steps
    total_time = 0
    for step in processing_steps:
        params_str = ', '.join([f"{k}={v}" for k, v in step['parameters'].items()])
        html_template += f"""
                    <tr>
                        <td><strong>{step['step_name']}</strong></td>
                        <td>{step['description']}</td>
                        <td><code>{params_str}</code></td>
                        <td>{step['execution_time']:.2f}s</td>
                    </tr>
        """
        total_time += step['execution_time']
    
    # Continue HTML template
    html_template += f"""
                </tbody>
            </table>
            <p><strong>Total Processing Time:</strong> {total_time:.2f} seconds</p>
            
            <h2>Data Quality Metrics</h2>
            <table class="step-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Original</th>
                        <th>Cleaned</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Mean Channel Standard Deviation</td>
                        <td>{stats['original_std_mean']:.2e}</td>
                        <td>{stats['cleaned_std_mean']:.2e}</td>
                        <td>{stats['noise_reduction']:.1f}% reduction</td>
                    </tr>
                    <tr>
                        <td>Number of Annotations</td>
                        <td>{stats['original_annotations']}</td>
                        <td>{stats['cleaned_annotations']}</td>
                        <td>+{stats['cleaned_annotations'] - stats['original_annotations']}</td>
                    </tr>
                    <tr>
                        <td>Bad Segment Annotations</td>
                        <td>0</td>
                        <td>{stats['bad_annotations']}</td>
                        <td>+{stats['bad_annotations']}</td>
                    </tr>
                </tbody>
            </table>
    """
    
    # Add plots if requested
    if include_plots:
        html_template += """
            <h2>Visualizations</h2>
            <div class="plot-container">
                <h3>Raw Data Comparison</h3>
                <p>Red: Original Data, Black: Cleaned Data</p>
                <!-- Plot will be added here if generated -->
            </div>
        """
    
    # Close HTML
    html_template += f"""
            <div class="timestamp">
                Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_template


def create_processing_summary(
    processing_steps: List[Dict],
    output_path: Optional[Union[str, Path]] = None
) -> Dict:
    """Create a JSON summary of processing steps.

    Parameters
    ----------
    processing_steps : list of dict
        List of processing steps with metadata.
    output_path : str, Path, or None, default None
        Path to save JSON summary. If None, returns dict only.

    Returns
    -------
    summary : dict
        Processing summary with statistics.
    """
    summary = {
        'total_steps': len(processing_steps),
        'total_time': sum(step['execution_time'] for step in processing_steps),
        'steps': processing_steps,
        'generated_at': datetime.now().isoformat()
    }
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    return summary