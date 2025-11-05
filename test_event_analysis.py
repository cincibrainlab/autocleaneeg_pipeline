#!/usr/bin/env python3
"""
EEG Event Analysis and Report Generator

Comprehensive analysis of raw events in EEG files to understand event structure
before configuring epoching parameters.

Supported formats:
- BDF (BioSemi)
- EGI (.raw, .mff)
- BrainVision (.vhdr)
- EDF
- FIF

Features:
- Complete event extraction from status/trigger channels
- Event timing and interval analysis
- Visual timeline of events
- Statistical summaries
- HTML report generation

Usage: uv run test_event_analysis.py [eeg_file] [output_dir]
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import base64
from io import BytesIO

import numpy as np
import pandas as pd

try:
    import mne
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: uv pip install mne matplotlib seaborn rich pandas")
    sys.exit(1)

# Optional Neo import for additional format support
try:
    import neo
    NEO_AVAILABLE = True
except ImportError:
    NEO_AVAILABLE = False

# Styling
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')
console = Console()


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def detect_file_format(file_path: str) -> str:
    """Detect EEG file format from extension."""
    ext = Path(file_path).suffix.lower()
    format_map = {
        '.bdf': 'bdf',
        '.edf': 'edf',
        '.raw': 'egi',
        '.mff': 'egi_mff',
        '.vhdr': 'brainvision',
        '.fif': 'fif',
        '.set': 'eeglab',
        '.xdat': 'neuronexus_xdat',  # Neo-supported format
    }
    return format_map.get(ext, 'unknown')


def load_xdat_via_neo(file_path: str) -> mne.io.Raw:
    """Load .xdat file using Neo and convert to MNE Raw format.

    Args:
        file_path: Path to .xdat file (or .xdat.json metadata file)

    Returns:
        MNE Raw object
    """
    console.print("[info]Loading .xdat file via Neo...[/info]")

    # Neo NeuroNexusIO expects the JSON metadata file
    # If given the data file, look for companion JSON
    file_path = Path(file_path)
    if file_path.suffix == '.xdat':
        # Look for the JSON metadata file
        # Try two naming patterns:
        # 1. filename_data.xdat -> filename.xdat.json
        # 2. filename.xdat -> filename.xdat.json

        # First, try removing _data suffix if present
        stem = file_path.stem
        if stem.endswith('_data'):
            # Remove _data suffix
            base_stem = stem[:-5]  # Remove '_data'
            json_file = file_path.parent / f"{base_stem}.xdat.json"
        elif stem.endswith('_timestamp'):
            # If given timestamp file, also look for base json
            base_stem = stem[:-10]  # Remove '_timestamp'
            json_file = file_path.parent / f"{base_stem}.xdat.json"
        else:
            # Standard pattern
            json_file = Path(str(file_path) + '.json')

        if not json_file.exists():
            # Try the simple pattern as fallback
            json_file = Path(str(file_path) + '.json')
            if not json_file.exists():
                raise FileNotFoundError(
                    f"Neo NeuroNexusIO requires a JSON metadata file.\n"
                    f"Looked for: {json_file}\n"
                    f"Make sure both the .xdat data file and .xdat.json metadata file are present."
                )
        reader_file = str(json_file)
    else:
        reader_file = str(file_path)

    console.print(f"[info]Using metadata file:[/info] {Path(reader_file).name}")

    # Read with Neo
    reader = neo.io.NeuroNexusIO(filename=reader_file)
    block = reader.read_block()

    # Get the first segment (assumes single recording segment)
    segment = block.segments[0]

    # Extract all analog signals (EEG + aux + digital inputs)
    all_data = []
    all_ch_names = []
    ch_types_list = []
    sfreq = None

    for analog_signal in segment.analogsignals:
        sig_data = analog_signal.magnitude.T
        all_data.append(sig_data)

        # Get channel names
        ch_names_raw = analog_signal.array_annotations.get('channel_names', None)
        if ch_names_raw is not None:
            sig_ch_names = [str(name) for name in ch_names_raw]
        else:
            sig_ch_names = [f'ch_{i}' for i in range(sig_data.shape[0])]

        all_ch_names.extend(sig_ch_names)

        # Determine channel types based on names
        for ch_name in sig_ch_names:
            if 'din' in ch_name.lower() or 'digital' in ch_name.lower() or 'stim' in ch_name.lower():
                ch_types_list.append('stim')
            elif 'aux' in ch_name.lower():
                ch_types_list.append('misc')
            else:
                ch_types_list.append('eeg')

        # Get sampling frequency (should be same for all signals)
        if sfreq is None:
            sfreq = float(analog_signal.sampling_rate.magnitude)

    # Concatenate all data
    if len(all_data) > 0:
        data = np.vstack(all_data)
    else:
        raise ValueError("No analog signals found in the .xdat file")

    console.print(f"[info]Loaded {len(all_ch_names)} channels:[/info] {', '.join(all_ch_names[:10])}{'...' if len(all_ch_names) > 10 else ''}")

    # Create MNE info structure
    info = mne.create_info(ch_names=all_ch_names, sfreq=sfreq, ch_types=ch_types_list)

    # Create Raw object
    raw = mne.io.RawArray(data, info)

    console.print(f"[success]✓[/success] Loaded via Neo: {len(all_ch_names)} channels, {raw.times[-1]:.2f} seconds")

    return raw


def load_raw_file(file_path: str) -> mne.io.Raw:
    """Load EEG file using appropriate MNE reader based on format."""
    file_format = detect_file_format(file_path)

    console.print(f"[info]Detected format:[/info] {file_format.upper()}")

    if file_format == 'bdf':
        raw = mne.io.read_raw_bdf(
            file_path,
            preload=True,
            stim_channel="auto",
            exclude=[],
            verbose=False
        )
    elif file_format == 'edf':
        raw = mne.io.read_raw_edf(
            file_path,
            preload=True,
            stim_channel="auto",
            exclude=[],
            verbose=False
        )
    elif file_format == 'egi':
        raw = mne.io.read_raw_egi(
            file_path,
            preload=True,
            verbose=False
        )
    elif file_format == 'egi_mff':
        raw = mne.io.read_raw_egi(
            file_path,
            preload=True,
            verbose=False
        )
    elif file_format == 'brainvision':
        raw = mne.io.read_raw_brainvision(
            file_path,
            preload=True,
            verbose=False
        )
    elif file_format == 'fif':
        raw = mne.io.read_raw_fif(
            file_path,
            preload=True,
            verbose=False
        )
    elif file_format == 'eeglab':
        raw = mne.io.read_raw_eeglab(
            file_path,
            preload=True,
            verbose=False
        )
    elif file_format == 'neuronexus_xdat':
        if not NEO_AVAILABLE:
            raise ImportError(
                "Neo package is required to read .xdat files.\n"
                "Install with: uv pip install neo"
            )
        # Use Neo to read the .xdat file and convert to MNE Raw
        raw = load_xdat_via_neo(file_path)
    else:
        # Check if there's companion JSON metadata for unsupported formats
        json_file = Path(str(file_path) + '.json')
        error_msg = f"Unsupported file format: {file_format} (extension: {Path(file_path).suffix})"

        if json_file.exists():
            error_msg += f"\n\nFound metadata file: {json_file.name}"
            error_msg += "\nThis appears to be a proprietary format with JSON metadata."
            error_msg += "\n\nSupported formats: BDF, EDF, EGI (.raw/.mff), BrainVision, FIF, EEGLAB, XDAT (via Neo)"
            error_msg += "\n\nSuggestion: Install Neo (uv pip install neo) or convert to supported format"
        else:
            error_msg += f"\n\nSupported formats: BDF, EDF, EGI (.raw/.mff), BrainVision, FIF, EEGLAB, XDAT (via Neo)"

        raise ValueError(error_msg)

    return raw


def extract_events_from_bdf(bdf_file: str) -> Tuple[np.ndarray, Dict, pd.DataFrame, mne.io.Raw, Dict]:
    """Extract events from EEG file using MNE.

    Args:
        bdf_file: Path to EEG file (supports BDF, EGI, BrainVision, EDF, FIF, etc.)

    Returns:
        events: MNE events array (n_events, 3) with [sample, duration, code]
        event_id: Dictionary mapping event names to codes
        events_df: DataFrame with detailed event information
        raw: MNE Raw object for metadata extraction
        status_info: Dictionary with status channel diagnostic information
    """
    console.print(f"[info]Loading file:[/info] {Path(bdf_file).name}")

    # Load file with appropriate reader
    raw = load_raw_file(bdf_file)

    console.print(f"[success]✓[/success] Loaded {len(raw.ch_names)} channels, {raw.times[-1]:.2f} seconds")

    # Inspect status channel
    status_info = {}
    status_ch_names = [ch for ch in raw.ch_names if 'Status' in ch or 'STIM' in ch or 'STI' in ch]
    if status_ch_names:
        console.print(f"[info]Found status channels:[/info] {', '.join(status_ch_names)}")
        # Get status channel data
        status_ch_name = status_ch_names[0]
        status_data, _ = raw[status_ch_name, :]
        status_data = status_data.flatten()

        unique_values = np.unique(status_data)
        status_info['channel_name'] = status_ch_name
        status_info['unique_values'] = unique_values
        status_info['n_unique'] = len(unique_values)
        status_info['min_value'] = status_data.min()
        status_info['max_value'] = status_data.max()
        status_info['non_zero_samples'] = np.count_nonzero(status_data)

        console.print(f"[info]Status channel info:[/info]")
        console.print(f"  - Name: {status_ch_name}")
        console.print(f"  - Unique values: {len(unique_values)}")
        console.print(f"  - Range: {status_data.min():.0f} to {status_data.max():.0f}")
        console.print(f"  - Non-zero samples: {np.count_nonzero(status_data)}")
    else:
        console.print("[warning]⚠ No status channel found![/warning]")
        status_info['channel_name'] = None

    # Extract events from annotations
    console.print("[info]Extracting events from annotations...[/info]")
    events = np.array([])
    event_id = {}

    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        console.print(f"[success]✓[/success] Found {len(events)} events with {len(event_id)} unique event types")
    except ValueError as e:
        # Annotations exist but are not valid event markers
        console.print(f"[warning]Annotation extraction failed:[/warning] {e}")
        events = np.array([])
        event_id = {}
    except Exception as e:
        console.print(f"[warning]Unexpected error in annotation extraction:[/warning] {e}")
        events = np.array([])
        event_id = {}

    # If no events from annotations, try finding events directly from status channel
    if len(events) == 0 and status_ch_names:
        console.print("[info]No events from annotations, trying direct status channel extraction...[/info]")
        # Try using mne.find_events
        try:
            events = mne.find_events(raw, stim_channel=status_ch_names[0], verbose=False)
            if len(events) > 0:
                # Create event_id from unique event codes
                unique_codes = np.unique(events[:, 2])
                event_id = {f"Event_{code}": code for code in unique_codes if code != 0}
                console.print(f"[success]✓[/success] Found {len(events)} events via direct extraction")
                console.print(f"[info]Event codes:[/info] {sorted(unique_codes)}")
            else:
                console.print("[warning]No events found in status channel[/warning]")
        except Exception as e:
            console.print(f"[warning]Direct extraction failed:[/warning] {e}")

    # Create detailed events DataFrame
    if len(events) > 0:
        events_df = pd.DataFrame({
            'sample': events[:, 0],
            'time_sec': events[:, 0] / raw.info['sfreq'],
            'duration': events[:, 1],
            'code': events[:, 2],
        })

        # Add event type names
        # Create reverse mapping from code to name
        code_to_name = {v: k for k, v in event_id.items()}
        events_df['type'] = events_df['code'].map(code_to_name)

        # Calculate inter-event intervals
        events_df['interval_sec'] = events_df['time_sec'].diff()

        # Add time in minutes for better readability
        events_df['time_min'] = events_df['time_sec'] / 60
    else:
        # Create empty DataFrame with correct columns
        events_df = pd.DataFrame(columns=['sample', 'time_sec', 'duration', 'code', 'type', 'interval_sec', 'time_min'])

    return events, event_id, events_df, raw, status_info


def analyze_event_patterns(events_df: pd.DataFrame, event_id: Dict) -> Dict:
    """Analyze patterns and statistics in event data."""

    analysis = {}

    # Overall statistics
    analysis['total_events'] = len(events_df)
    analysis['unique_types'] = len(event_id)
    analysis['duration_sec'] = events_df['time_sec'].max() - events_df['time_sec'].min()
    analysis['duration_min'] = analysis['duration_sec'] / 60

    # Per-type statistics
    type_stats = []
    for event_type, event_code in sorted(event_id.items(), key=lambda x: x[1]):
        type_events = events_df[events_df['code'] == event_code]

        if len(type_events) > 0:
            # Calculate intervals between events of the same type
            same_type_intervals = type_events['time_sec'].diff().dropna()

            type_stats.append({
                'type': event_type,
                'code': event_code,
                'count': len(type_events),
                'percentage': (len(type_events) / len(events_df)) * 100,
                'first_time': type_events['time_sec'].iloc[0],
                'last_time': type_events['time_sec'].iloc[-1],
                'mean_interval': same_type_intervals.mean() if len(same_type_intervals) > 0 else np.nan,
                'std_interval': same_type_intervals.std() if len(same_type_intervals) > 0 else np.nan,
                'min_interval': same_type_intervals.min() if len(same_type_intervals) > 0 else np.nan,
                'max_interval': same_type_intervals.max() if len(same_type_intervals) > 0 else np.nan,
            })

    analysis['type_stats'] = pd.DataFrame(type_stats)

    # Overall inter-event intervals
    all_intervals = events_df['interval_sec'].dropna()
    analysis['mean_interval'] = all_intervals.mean()
    analysis['std_interval'] = all_intervals.std()
    analysis['min_interval'] = all_intervals.min()
    analysis['max_interval'] = all_intervals.max()

    # Event density (events per minute)
    analysis['events_per_minute'] = analysis['total_events'] / analysis['duration_min']

    # Detect potential sequences or patterns
    # Look for repeated event sequences
    if len(events_df) >= 3:
        event_sequence = events_df['code'].values
        # Find 3-event sequences
        sequences = {}
        for i in range(len(event_sequence) - 2):
            seq = tuple(event_sequence[i:i+3])
            sequences[seq] = sequences.get(seq, 0) + 1

        # Get most common sequences
        common_sequences = sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis['common_sequences'] = common_sequences
    else:
        analysis['common_sequences'] = []

    return analysis


def create_event_timeline(events_df: pd.DataFrame, event_id: Dict, title: str) -> str:
    """Create visual timeline of events."""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    if len(events_df) == 0 or len(event_id) == 0:
        # No events - create placeholder
        ax1.text(0.5, 0.5, 'No events found in recording',
                ha='center', va='center', fontsize=16, transform=ax1.transAxes)
        ax1.set_title('Event Timeline (All Events)', fontsize=14, fontweight='bold', pad=15)
        ax2.text(0.5, 0.5, 'No events to display',
                ha='center', va='center', fontsize=16, transform=ax2.transAxes)
        ax2.set_title('Event Density Over Time', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        return fig_to_base64(fig)

    # Create color map for event types
    unique_codes = sorted(event_id.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_codes)))
    code_to_color = dict(zip(unique_codes, colors))

    # Plot 1: Event scatter plot over time
    ax1.set_title('Event Timeline (All Events)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Event Code', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Reverse mapping for labels
    code_to_name = {v: k for k, v in event_id.items()}

    for code in unique_codes:
        code_events = events_df[events_df['code'] == code]
        if len(code_events) > 0:
            ax1.scatter(
                code_events['time_sec'],
                [code] * len(code_events),
                c=[code_to_color[code]],
                label=f"{code_to_name[code]} (n={len(code_events)})",
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1
            )

    if len(unique_codes) > 0:
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

    max_time = events_df['time_sec'].max()
    if not np.isnan(max_time) and not np.isinf(max_time):
        ax1.set_xlim(0, max_time * 1.05)

    # Plot 2: Event histogram (binned by time)
    ax2.set_title('Event Density Over Time', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Number of Events', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Create histogram with 50 bins
    n_bins = min(50, max(1, len(events_df) // 2))
    ax2.hist(events_df['time_sec'], bins=n_bins, color='steelblue',
             alpha=0.7, edgecolor='black', linewidth=1)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_interval_analysis(events_df: pd.DataFrame, analysis: Dict) -> str:
    """Create interval analysis plots."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Event Interval Analysis', fontsize=16, fontweight='bold')

    # Check if we have events
    has_events = len(events_df) > 0

    # Plot 1: Histogram of all inter-event intervals
    ax1.set_title('Distribution of Inter-Event Intervals (All Events)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Interval (seconds)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.grid(True, alpha=0.3)

    if has_events:
        intervals = events_df['interval_sec'].dropna()
        if len(intervals) > 0:
            ax1.hist(intervals, bins=30, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
            ax1.axvline(intervals.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {intervals.mean():.3f}s')
            ax1.legend(fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Insufficient data for intervals',
                    ha='center', va='center', fontsize=14, transform=ax1.transAxes)
    else:
        ax1.text(0.5, 0.5, 'No events found',
                ha='center', va='center', fontsize=14, transform=ax1.transAxes)

    # Plot 2: Intervals over time (to detect changes in event timing)
    ax2.set_title('Inter-Event Intervals Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Event Number', fontsize=11)
    ax2.set_ylabel('Interval (seconds)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    if has_events:
        intervals = events_df['interval_sec'].dropna()
        if len(intervals) > 0:
            ax2.plot(range(len(intervals)), intervals, 'o-', alpha=0.6, markersize=4)
            ax2.axhline(intervals.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for intervals',
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No events found',
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)

    # Plot 3: Per-type event counts (bar chart)
    ax3.set_title('Event Type Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Event Type', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')

    type_stats = analysis['type_stats']
    if len(type_stats) > 0:
        bars = ax3.bar(range(len(type_stats)), type_stats['count'],
                       color='steelblue', alpha=0.7, edgecolor='black', linewidth=1)
        ax3.set_xticks(range(len(type_stats)))
        ax3.set_xticklabels(type_stats['type'], rotation=45, ha='right', fontsize=9)

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, type_stats['count'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No event types found',
                ha='center', va='center', fontsize=14, transform=ax3.transAxes)

    # Plot 4: Cumulative event count over time
    ax4.set_title('Cumulative Event Count', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('Cumulative Events', fontsize=11)
    ax4.grid(True, alpha=0.3)

    if has_events:
        ax4.plot(events_df['time_sec'], range(len(events_df)),
                 linewidth=2, color='steelblue')
        ax4.fill_between(events_df['time_sec'], 0, range(len(events_df)),
                         alpha=0.3, color='steelblue')
    else:
        ax4.text(0.5, 0.5, 'No events to display',
                ha='center', va='center', fontsize=14, transform=ax4.transAxes)

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_html_report(
    bdf_file: str,
    events_df: pd.DataFrame,
    event_id: Dict,
    analysis: Dict,
    timeline_b64: str,
    interval_b64: str,
    raw: mne.io.Raw,
    status_info: Dict,
    output_file: Path
):
    """Generate comprehensive HTML report for event analysis."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data for display
    type_stats = analysis['type_stats']

    # Get first 100 events for detailed table
    events_sample = events_df.head(100)

    # Helper function to format nullable floats
    def fmt_float(val, decimals=3):
        if pd.isna(val):
            return "N/A"
        return f"{val:.{decimals}f}"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BDF Event Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Times New Roman', Times, serif;
            background: #f5f5f5;
            padding: 20px;
            color: #000;
            line-height: 1.6;
            font-size: 13px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}

        .header {{
            background: #fff;
            border-bottom: 3px solid #000;
            padding: 25px 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 22px;
            margin-bottom: 8px;
            font-weight: bold;
            color: #000;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .header .subtitle {{
            font-size: 14px;
            color: #333;
            font-style: italic;
            margin-bottom: 15px;
        }}

        .header .meta {{
            font-size: 14px;
            color: #333;
            font-weight: 600;
        }}

        .summary-panel {{
            margin: 0;
            padding: 20px 30px;
            background: #e3f2fd;
            border-left: 8px solid #1976d2;
            border-bottom: 3px solid #ddd;
        }}

        .summary-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #0d47a1;
            text-transform: uppercase;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 10px 0;
        }}

        .stat-card {{
            background: white;
            padding: 12px 14px;
            border: 1px solid #1976d2;
            border-radius: 4px;
        }}

        .stat-card .label {{
            font-size: 11px;
            color: #666;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}

        .stat-card .value {{
            font-size: 18px;
            font-weight: bold;
            color: #1976d2;
        }}

        .section {{
            margin: 0;
            padding: 25px 30px;
            border-bottom: 1px solid #ddd;
        }}

        .section:last-child {{
            border-bottom: none;
        }}

        .section h2 {{
            color: #000;
            margin-bottom: 15px;
            font-size: 16px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #000;
            padding-bottom: 8px;
        }}

        .section h3 {{
            font-size: 14px;
            font-weight: bold;
            margin: 15px 0 10px 0;
            color: #333;
        }}

        .image-container {{
            margin: 15px 0;
            text-align: center;
        }}

        .image-container img {{
            max-width: 100%;
            border: 1px solid #ccc;
        }}

        .image-caption {{
            font-size: 11px;
            margin-top: 6px;
            color: #666;
            font-style: italic;
        }}

        .table-container {{
            overflow-x: auto;
            margin: 15px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            font-size: 11px;
            border: 1px solid #000;
        }}

        th {{
            background: #f0f0f0;
            color: #000;
            padding: 8px 6px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #000;
            text-transform: uppercase;
            font-size: 10px;
        }}

        td {{
            padding: 6px;
            border: 1px solid #ddd;
            vertical-align: top;
        }}

        tr:nth-child(even) {{
            background: #fafafa;
        }}

        tbody tr:hover {{
            background: #f0f0f0;
        }}

        .info-box {{
            background: #f9f9f9;
            border: 1px solid #ccc;
            padding: 12px;
            margin: 12px 0;
            font-size: 12px;
        }}

        .code-box {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            border: 1px solid #000;
            overflow-x: auto;
        }}

        .footer {{
            background: #f5f5f5;
            color: #666;
            text-align: center;
            padding: 20px;
            font-size: 11px;
            border-top: 2px solid #000;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: bold;
            border-radius: 3px;
            margin: 2px;
        }}

        .badge.primary {{ background: #1976d2; color: white; }}
        .badge.success {{ background: #4caf50; color: white; }}
        .badge.warning {{ background: #ff9800; color: white; }}

        p {{
            margin: 10px 0;
        }}

        ul {{
            margin: 10px 0 10px 25px;
        }}

        li {{
            margin: 6px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>EEG Event Analysis Report</h1>
            <div class="subtitle">Raw Event Extraction and Pattern Analysis</div>
            <div class="meta">
                File: {Path(bdf_file).name} &nbsp;|&nbsp; Format: {detect_file_format(bdf_file).upper()} &nbsp;|&nbsp; Generated: {timestamp}
            </div>
        </div>

        <!-- SUMMARY PANEL -->
        <div class="summary-panel">
            <div class="summary-title">Analysis Summary</div>
            {'<div class="info-box" style="background: #ffebee; border: 2px solid #f44336; margin-bottom: 15px;"><strong>⚠ NO EVENTS FOUND</strong><br>No events were extracted from the status channel. See diagnostic information below.</div>' if analysis['total_events'] == 0 else ''}
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">Total Events</div>
                    <div class="value">{analysis['total_events']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Unique Event Types</div>
                    <div class="value">{analysis['unique_types']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Recording Duration</div>
                    <div class="value">{analysis.get('duration_min', raw.times[-1]/60):.2f} min</div>
                </div>
                <div class="stat-card">
                    <div class="label">Events Per Minute</div>
                    <div class="value">{analysis.get('events_per_minute', 0):.2f}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Mean Interval</div>
                    <div class="value">{analysis.get('mean_interval', 0):.3f} s</div>
                </div>
                <div class="stat-card">
                    <div class="label">Sampling Rate</div>
                    <div class="value">{raw.info['sfreq']:.0f} Hz</div>
                </div>
            </div>
        </div>

        <!-- STATUS CHANNEL DIAGNOSTICS -->
        {''.join(f'''<div class="section">
            <h2>Status Channel Diagnostics</h2>
            <div class="info-box">
                <strong>Status Channel Information:</strong>
                <ul>
                    <li><strong>Channel Name:</strong> {status_info.get('channel_name', 'Not found')}</li>
                    <li><strong>Unique Values:</strong> {status_info.get('n_unique', 0)}</li>
                    <li><strong>Value Range:</strong> {status_info.get('min_value', 0):.0f} to {status_info.get('max_value', 0):.0f}</li>
                    <li><strong>Non-Zero Samples:</strong> {status_info.get('non_zero_samples', 0):,} ({status_info.get('non_zero_samples', 0) / raw.n_times * 100:.2f}% of recording)</li>
                </ul>
            </div>
            {f'<div class="info-box"><strong>Unique values in status channel:</strong><br>{", ".join(str(int(v)) for v in status_info.get("unique_values", [])[:50])}{" ..." if len(status_info.get("unique_values", [])) > 50 else ""}</div>' if status_info.get('unique_values') is not None and len(status_info.get('unique_values', [])) > 1 else ''}
            <div class="warning-box">
                <strong>Possible Reasons for No Events:</strong>
                <ul>
                    <li>The status channel contains only zeros (no triggers recorded)</li>
                    <li>Events are encoded in a non-standard way</li>
                    <li>This is a resting-state recording with no task events</li>
                    <li>Event markers need to be added during preprocessing</li>
                </ul>
            </div>
        </div>''' if analysis['total_events'] == 0 else '')}

        <!-- EVENT TYPE STATISTICS -->
        {''.join(f'''<div class="section">
            <h2>Event Type Statistics</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Detailed breakdown of each event type found in the recording.
                Use this information to configure epoching parameters.
            </p>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Event Type</th>
                            <th>Code</th>
                            <th>Count</th>
                            <th>% of Total</th>
                            <th>First (s)</th>
                            <th>Last (s)</th>
                            <th>Mean Interval (s)</th>
                            <th>Std Interval (s)</th>
                            <th>Min Interval (s)</th>
                            <th>Max Interval (s)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'<tr><td><strong>{row["type"]}</strong></td><td><span class="badge primary">{row["code"]}</span></td><td><strong>{row["count"]}</strong></td><td>{row["percentage"]:.1f}%</td><td>{row["first_time"]:.2f}</td><td>{row["last_time"]:.2f}</td><td>{fmt_float(row["mean_interval"])}</td><td>{fmt_float(row["std_interval"])}</td><td>{fmt_float(row["min_interval"])}</td><td>{fmt_float(row["max_interval"])}</td></tr>' for _, row in type_stats.iterrows())}
                    </tbody>
                </table>
            </div>
        </div>''' if len(type_stats) > 0 else '')}

        <!-- CONFIGURATION EXAMPLES -->
        {''.join(f'''<div class="section">
            <h2>MNE Epoching Configuration Examples</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Copy these examples to configure your epoching pipeline based on the discovered events.
            </p>

            <h3>Example 1: Epoch All Event Types</h3>
            <div class="code-box">event_id = {{{', '.join(f'"{name}": {code}' for name, code in sorted(event_id.items(), key=lambda x: x[1]))}}}</div>

            <h3>Example 2: Python Dictionary (Single Line)</h3>
            <div class="code-box">event_id = {dict(sorted(event_id.items(), key=lambda x: x[1]))}</div>

            <h3>Example 3: MNE Epochs Creation</h3>
            <div class="code-box">import mne

# Define event mapping
event_id = {dict(sorted(event_id.items(), key=lambda x: x[1]))}

# Extract events
events, _ = mne.events_from_annotations(raw)

# Create epochs
epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=-0.2,      # Start 200ms before event
    tmax=0.8,       # End 800ms after event
    baseline=(None, 0),
    preload=True
)</div>
        </div>''' if len(event_id) > 0 else '')}

        <!-- EVENT TIMELINE -->
        <div class="section">
            <h2>Event Timeline</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Visual representation of when events occur throughout the recording.
            </p>
            <div class="image-container">
                <img src="{timeline_b64}" alt="Event Timeline">
                <div class="image-caption">Event distribution and density over time</div>
            </div>
        </div>

        <!-- INTERVAL ANALYSIS -->
        <div class="section">
            <h2>Interval Analysis</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Statistical analysis of timing between events.
            </p>
            <div class="image-container">
                <img src="{interval_b64}" alt="Interval Analysis">
                <div class="image-caption">Inter-event intervals and patterns</div>
            </div>
        </div>

        <!-- PATTERN DETECTION -->
        {''.join(f'''<div class="section">
            <h2>Common Event Sequences</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Most frequently occurring 3-event sequences in the data.
            </p>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Sequence</th>
                            <th>Occurrences</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'<tr><td>{i}</td><td>{" → ".join(str(code) for code in seq)}</td><td><strong>{count}</strong></td></tr>' for i, (seq, count) in enumerate(analysis['common_sequences'], 1))}
                    </tbody>
                </table>
            </div>
        </div>''' if analysis['common_sequences'] else '')}

        <!-- DETAILED EVENT TABLE -->
        {''.join(f'''<div class="section">
            <h2>Detailed Event Table (First 100 Events)</h2>
            <p style="color: #666; margin-bottom: 15px;">
                Chronological listing of events with precise timing information.
            </p>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Sample</th>
                            <th>Time (s)</th>
                            <th>Time (min)</th>
                            <th>Event Type</th>
                            <th>Code</th>
                            <th>Interval (s)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'<tr><td>{i}</td><td>{row["sample"]}</td><td>{row["time_sec"]:.3f}</td><td>{row["time_min"]:.3f}</td><td><strong>{row["type"]}</strong></td><td><span class="badge primary">{row["code"]}</span></td><td>{fmt_float(row["interval_sec"])}</td></tr>' for i, (_, row) in enumerate(events_sample.iterrows(), 1))}
                    </tbody>
                </table>
            </div>
            {f'<div class="info-box"><strong>Note:</strong> Showing first 100 of {len(events_df)} total events. Full event data available in CSV export.</div>' if len(events_df) > 100 else ''}
        </div>''' if len(events_df) > 0 else '')}

        <!-- FILE METADATA -->
        <div class="section">
            <h2>File Metadata</h2>
            <div class="table-container">
                <table style="max-width: 800px;">
                    <tr>
                        <th style="width: 200px;">Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td><strong>File Name</strong></td>
                        <td>{Path(bdf_file).name}</td>
                    </tr>
                    <tr>
                        <td><strong>Total Channels</strong></td>
                        <td>{len(raw.ch_names)}</td>
                    </tr>
                    <tr>
                        <td><strong>Sampling Rate</strong></td>
                        <td>{raw.info['sfreq']:.2f} Hz</td>
                    </tr>
                    <tr>
                        <td><strong>Duration</strong></td>
                        <td>{raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)</td>
                    </tr>
                    <tr>
                        <td><strong>Total Samples</strong></td>
                        <td>{raw.n_times:,}</td>
                    </tr>
                    <tr>
                        <td><strong>Recording Date</strong></td>
                        <td>{raw.info['meas_date'].strftime('%Y-%m-%d %H:%M:%S') if raw.info.get('meas_date') else 'Not available'}</td>
                    </tr>
                </table>
            </div>
        </div>

        <div class="footer">
            <div style="font-weight: bold; margin-bottom: 5px;">EEG Event Analysis Report v2.0</div>
            <div>Generated by AutoClean EEG Pipeline Event Diagnosis Tool</div>
            <div style="margin-top: 8px; font-size: 10px;">
                This report provides comprehensive event analysis to support epoching configuration<br>
                Supports: BDF, EGI (.raw/.mff), BrainVision, EDF, FIF, EEGLAB
            </div>
        </div>
    </div>
</body>
</html>"""

    output_file.write_text(html)
    console.print(f"\n✅ [bold green]HTML report saved:[/bold green] {output_file}")


def main():
    """Main execution."""

    # Default file path
    bdf_file = "/Users/ernie/Downloads/Example EEGs/st101as.bdf"
    output_dir = "."

    if len(sys.argv) > 1:
        bdf_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]         EEG Event Analysis Tool v2.0[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    # Extract events
    events, event_id, events_df, raw, status_info = extract_events_from_bdf(bdf_file)

    # Analyze patterns
    console.print("[info]Analyzing event patterns...[/info]")
    analysis = analyze_event_patterns(events_df, event_id)

    # Generate visualizations
    console.print("[info]Generating visualizations...[/info]")
    timeline_b64 = create_event_timeline(
        events_df, event_id,
        f"Event Timeline - {Path(bdf_file).name}"
    )
    interval_b64 = create_interval_analysis(events_df, analysis)

    # Generate HTML report
    html_file = output_dir / f"{Path(bdf_file).stem}_event_analysis.html"
    console.print("[info]Generating HTML report...[/info]")
    generate_html_report(
        bdf_file, events_df, event_id, analysis,
        timeline_b64, interval_b64, raw, status_info, html_file
    )

    # Export CSV for detailed analysis (only if there are events)
    if len(events_df) > 0:
        csv_file = output_dir / f"{Path(bdf_file).stem}_events.csv"
        events_df.to_csv(csv_file, index=False)
        console.print(f"✅ [bold green]CSV data saved:[/bold green] {csv_file}")
    else:
        console.print("[warning]⚠ No events to export to CSV[/warning]")

    # Print summary to console
    console.print("\n[bold green]════════════════════════════════════════════════════════════[/bold green]")
    console.print(f"[bold green]  Analysis Complete![/bold green]")
    console.print(f"[bold green]════════════════════════════════════════════════════════════[/bold green]\n")

    # Create summary table
    table = Table(title="Event Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Events", str(analysis['total_events']))
    table.add_row("Unique Types", str(analysis['unique_types']))
    table.add_row("Duration", f"{analysis.get('duration_min', raw.times[-1]/60):.2f} minutes")
    table.add_row("Events/Minute", f"{analysis.get('events_per_minute', 0):.2f}")
    table.add_row("Mean Interval", f"{analysis.get('mean_interval', 0):.3f} seconds")

    console.print(table)

    # Print event type breakdown (only if events exist)
    if len(analysis['type_stats']) > 0:
        console.print("\n[bold cyan]Event Type Breakdown:[/bold cyan]\n")
        type_table = Table(show_header=True, header_style="bold magenta")
        type_table.add_column("Type", style="yellow")
        type_table.add_column("Code", style="cyan")
        type_table.add_column("Count", justify="right", style="green")
        type_table.add_column("Percentage", justify="right", style="blue")

        for _, row in analysis['type_stats'].iterrows():
            type_table.add_row(
                row['type'],
                str(row['code']),
                str(row['count']),
                f"{row['percentage']:.1f}%"
            )

        console.print(type_table)
    else:
        console.print("\n[bold yellow]⚠ No events found - see HTML report for diagnostic information[/bold yellow]")

    console.print(f"\n  📊 [cyan]HTML Report:[/cyan] {html_file}")
    if len(events_df) > 0:
        console.print(f"  📁 [cyan]CSV Export:[/cyan] {csv_file}")
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
