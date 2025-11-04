#!/usr/bin/env python3
"""
BDF Import Visual Diagnostic Report - v5 (EEG Research Optimized)

Enhanced for EEG data quality workflow:
- Improved readability (larger fonts: 11-13px body, 11px+ tables)
- Side-by-side multi-column channel tables (efficient use of horizontal space)
- Information hierarchy optimized for EEG quality control workflow
- Prominent display of unmatched channels and quality issues
- Actionable verdict panel with clear pass/fail indicators
- Academic styling maintained (Times New Roman, black/white/gray)

Usage: uv run test_bdf_visual_v5.py [bdf_file] [montage_name] [output_dir]
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import base64
from io import BytesIO
import math

import numpy as np

try:
    import mne
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge, FancyBboxPatch
    import seaborn as sns
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
except ImportError as e:
    print(f"ERROR: {e}")
    print("Install: uv pip install mne matplotlib seaborn rich")
    sys.exit(1)

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


def extract_file_metadata(raw: mne.io.Raw, original_filename: str) -> Dict:
    """Extract comprehensive metadata from MNE Raw object.

    Works across all file formats supported by MNE.
    """
    info = raw.info
    metadata = {}

    # File information
    metadata['filename'] = Path(original_filename).name
    metadata['file_format'] = Path(original_filename).suffix.upper().replace('.', '')

    # Recording information
    metadata['n_channels'] = info['nchan']
    metadata['sampling_rate'] = f"{info['sfreq']:.2f} Hz"
    metadata['n_samples'] = raw.n_times
    metadata['duration'] = f"{raw.times[-1]:.2f} seconds ({raw.times[-1]/60:.2f} minutes)"

    # Date/time
    if info['meas_date'] is not None:
        metadata['recording_date'] = info['meas_date'].strftime("%Y-%m-%d %H:%M:%S")
    else:
        metadata['recording_date'] = "Not available"

    # Device information
    if info.get('device_info'):
        dev_info = info['device_info']
        metadata['device_type'] = dev_info.get('type', 'Unknown')
        metadata['device_model'] = dev_info.get('model', 'Unknown')
        metadata['device_serial'] = dev_info.get('serial', 'Unknown')
    else:
        metadata['device_type'] = "Not available"
        metadata['device_model'] = "Not available"
        metadata['device_serial'] = "Not available"

    # Subject information
    if info.get('subject_info'):
        subj_info = info['subject_info']
        metadata['subject_id'] = subj_info.get('his_id', 'Not available')
        metadata['subject_first_name'] = subj_info.get('first_name', 'Not available')
        metadata['subject_last_name'] = subj_info.get('last_name', 'Not available')
    else:
        metadata['subject_id'] = "Not available"
        metadata['subject_first_name'] = "Not available"
        metadata['subject_last_name'] = "Not available"

    # Filter settings
    metadata['highpass_filter'] = f"{info['highpass']:.2f} Hz" if info.get('highpass') else "None"
    metadata['lowpass_filter'] = f"{info['lowpass']:.2f} Hz" if info.get('lowpass') else "None"

    # Description
    metadata['description'] = info.get('description', 'Not available')

    # Channel type breakdown
    ch_types = {}
    for ch in info['chs']:
        ch_kind = mne.io.constants.FIFF.get(ch['kind'], 'unknown')
        ch_types[ch_kind] = ch_types.get(ch_kind, 0) + 1

    metadata['channel_types'] = ch_types

    # Reference info
    if hasattr(info, 'get_montage') and info.get_montage():
        metadata['reference'] = "Custom montage applied"
    else:
        metadata['reference'] = "Not specified"

    # Line frequency
    metadata['line_freq'] = f"{info.get('line_freq', 'Not specified')} Hz" if info.get('line_freq') else "Not specified"

    return metadata


def extract_channel_info(raw: mne.io.Raw, rename_map: Dict = None, montage_channels: set = None) -> List[Dict]:
    """Extract detailed channel-by-channel information.

    Works across all file formats supported by MNE.

    Args:
        raw: MNE Raw object
        rename_map: Optional dict mapping original names to renamed names
        montage_channels: Optional set of channel names in the montage (for match status)
    """
    channel_info = []
    rename_map = rename_map or {}

    for idx, ch_name in enumerate(raw.ch_names):
        ch = raw.info['chs'][idx]

        # Get channel type as human-readable string
        ch_kind = ch['kind']
        ch_type_name = {
            mne.io.constants.FIFF.FIFFV_EEG_CH: 'EEG',
            mne.io.constants.FIFF.FIFFV_MEG_CH: 'MEG',
            mne.io.constants.FIFF.FIFFV_STIM_CH: 'STIM',
            mne.io.constants.FIFF.FIFFV_EOG_CH: 'EOG',
            mne.io.constants.FIFF.FIFFV_ECG_CH: 'ECG',
            mne.io.constants.FIFF.FIFFV_EMG_CH: 'EMG',
            mne.io.constants.FIFF.FIFFV_MISC_CH: 'MISC'
        }.get(ch_kind, f'Unknown ({ch_kind})')

        # Get position
        loc = ch['loc'][:3]
        has_position = not np.any(np.isnan(loc))

        # Determine if this channel will be renamed
        raw_name = ch_name
        will_be_renamed = ch_name in rename_map
        standard_name = rename_map.get(ch_name, ch_name)

        # Check if standard name matches montage
        matched_montage = standard_name in montage_channels if montage_channels else None

        channel_info.append({
            'raw_name': raw_name,
            'standard_name': standard_name if will_be_renamed else None,
            'will_rename': will_be_renamed,
            'matched_montage': matched_montage,
            'index': idx,
            'type': ch_type_name,
            'has_position': has_position,
            'x': f"{loc[0]:.4f}" if has_position else "N/A",
            'y': f"{loc[1]:.4f}" if has_position else "N/A",
            'z': f"{loc[2]:.4f}" if has_position else "N/A",
        })

    return channel_info


def analyze_channels(raw: mne.io.Raw, montage: mne.channels.DigMontage) -> Dict:
    """Comprehensive channel analysis."""

    file_chs = set(ch for ch in raw.ch_names if ch != 'Status')
    montage_chs = set(montage.get_positions()['ch_pos'].keys())

    matched = file_chs & montage_chs
    in_file_not_montage = file_chs - montage_chs
    in_montage_not_file = montage_chs - file_chs

    # Channel-by-channel analysis
    channel_data = {}
    for ch_name in raw.ch_names:
        if ch_name == 'Status':
            continue

        ch_idx = raw.ch_names.index(ch_name)
        ch_type = raw.info['chs'][ch_idx]['kind']

        if ch_type == mne.io.constants.FIFF.FIFFV_EEG_CH:
            loc = raw.info['chs'][ch_idx]['loc'][:3]
            has_pos = not np.any(np.isnan(loc))

            channel_data[ch_name] = {
                'in_file': True,
                'in_montage': ch_name in montage_chs,
                'has_position': has_pos,
                'position': loc if has_pos else None,
                'matched': ch_name in matched,
                'distance_from_origin': np.linalg.norm(loc) if has_pos else None
            }

    # Position quality checks
    positions = np.array([d['position'] for d in channel_data.values() if d['has_position']])

    if len(positions) > 0:
        distances = np.linalg.norm(positions, axis=1)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        # Find outliers (>3 std from mean)
        outliers = [ch for ch, d in channel_data.items()
                   if d['distance_from_origin'] and
                   abs(d['distance_from_origin'] - mean_dist) > 3 * std_dist]

        # Find duplicates
        position_map = {}
        for ch, d in channel_data.items():
            if d['has_position']:
                pos_key = tuple(np.round(d['position'], 6))
                position_map.setdefault(pos_key, []).append(ch)
        duplicates = {k: v for k, v in position_map.items() if len(v) > 1}
    else:
        mean_dist = 0
        std_dist = 0
        outliers = []
        duplicates = {}

    return {
        'file_channels': file_chs,
        'montage_channels': montage_chs,
        'matched': matched,
        'unmatched_file': in_file_not_montage,
        'unmatched_montage': in_montage_not_file,
        'channel_data': channel_data,
        'match_pct': len(matched) / len(file_chs) * 100 if file_chs else 0,
        'n_positioned': len(positions),
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'outliers': outliers,
        'duplicates': duplicates,
        'positions': positions
    }


def suggest_montages(file_channels: Set[str], top_n: int = 10) -> List[Tuple[str, int, float]]:
    """Test and rank alternative montages."""

    test_montages = [
        'biosemi16', 'biosemi32', 'biosemi64', 'biosemi128', 'biosemi256',
        'standard_1005', 'standard_1020', 'standard_postfixed',
        'GSN-HydroCel-32', 'GSN-HydroCel-64', 'GSN-HydroCel-65',
        'GSN-HydroCel-124', 'GSN-HydroCel-128', 'GSN-HydroCel-129',
        'GSN-HydroCel-256', 'GSN-HydroCel-257',
        'easycap-M1', 'easycap-M10',
    ]

    results = []
    for name in test_montages:
        try:
            m = mne.channels.make_standard_montage(name)
            m_chs = set(m.get_positions()['ch_pos'].keys())
            matched = file_channels & m_chs
            pct = len(matched) / len(file_channels) * 100 if file_channels else 0
            results.append((name, len(matched), pct))
        except:
            continue

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def create_elegant_topomap(analysis: Dict, title: str) -> str:
    """Create elegant 2D topomap with enhanced styling."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Top view (X-Y)
    ax1.set_title('Top View (X-Y Plane)', fontsize=14, pad=15)
    ax1.set_xlabel('X (meters)', fontsize=11)
    ax1.set_ylabel('Y (meters)', fontsize=11)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Aesthetic head outline
    head = Circle((0, 0), 0.095, fill=False, edgecolor='#2c3e50', linewidth=3, alpha=0.7)
    ax1.add_patch(head)
    # Nose indicator
    nose = Wedge((0, 0.095), 0.02, 60, 120, facecolor='#2c3e50', alpha=0.5)
    ax1.add_patch(nose)

    # Plot channels by category
    for ch, data in analysis['channel_data'].items():
        if data['has_position']:
            pos = data['position']

            if data['matched']:
                color, marker, size, alpha, edge = '#27ae60', 'o', 120, 0.8, '#1e8449'
                label = 'Matched' if ax1.get_legend_handles_labels()[1].count('Matched') == 0 else ''
            else:
                color, marker, size, alpha, edge = '#e67e22', 's', 100, 0.7, '#d35400'
                label = 'Unmatched' if ax1.get_legend_handles_labels()[1].count('Unmatched') == 0 else ''

            ax1.scatter(pos[0], pos[1], c=color, marker=marker, s=size,
                       alpha=alpha, edgecolors=edge, linewidths=2, label=label, zorder=3)
            ax1.text(pos[0], pos[1], ch, fontsize=7, ha='center', va='center',
                    fontweight='bold', zorder=4)

    ax1.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax1.set_xlim(-0.12, 0.12)
    ax1.set_ylim(-0.12, 0.12)

    # Plot 2: Side view (Y-Z)
    ax2.set_title('Side View (Y-Z Plane)', fontsize=14, pad=15)
    ax2.set_xlabel('Y (meters)', fontsize=11)
    ax2.set_ylabel('Z (meters)', fontsize=11)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2, linestyle='--')

    head2 = Circle((0, 0), 0.095, fill=False, edgecolor='#2c3e50', linewidth=3, alpha=0.7)
    ax2.add_patch(head2)

    for ch, data in analysis['channel_data'].items():
        if data['has_position']:
            pos = data['position']
            color = '#27ae60' if data['matched'] else '#e67e22'
            marker = 'o' if data['matched'] else 's'
            size = 120 if data['matched'] else 100
            edge = '#1e8449' if data['matched'] else '#d35400'

            ax2.scatter(pos[1], pos[2], c=color, marker=marker, s=size,
                       alpha=0.8, edgecolors=edge, linewidths=2, zorder=3)
            ax2.text(pos[1], pos[2], ch, fontsize=7, ha='center', va='center',
                    fontweight='bold', zorder=4)

    ax2.set_xlim(-0.12, 0.12)
    ax2.set_ylim(-0.12, 0.12)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_3d_plot(analysis: Dict, title: str) -> str:
    """Create elegant 3D position plot with multiple views."""

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    views = [
        (90, 90, 'Top View', 221),
        (0, 90, 'Back View', 222),
        (0, 0, 'Right Side', 223),
        (30, 45, '3D Perspective', 224)
    ]

    for elev, azim, view_title, subplot in views:
        ax = fig.add_subplot(subplot, projection='3d')
        ax.set_title(view_title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        # Plot channels
        for ch, data in analysis['channel_data'].items():
            if data['has_position']:
                pos = data['position']
                color = '#27ae60' if data['matched'] else '#e67e22'
                marker = 'o' if data['matched'] else 's'
                size = 60 if data['matched'] else 50

                ax.scatter(*pos, c=color, marker=marker, s=size, alpha=0.8,
                         edgecolors='black', linewidths=0.5)

        # Head sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = 0.095 * np.outer(np.cos(u), np.sin(v))
        y = 0.095 * np.outer(np.sin(u), np.sin(v))
        z = 0.095 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.15, color='gray')

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim([-0.11, 0.11])
        ax.set_ylim([-0.11, 0.11])
        ax.set_zlim([-0.11, 0.11])
        ax.set_box_aspect([1,1,1])

    plt.tight_layout()
    return fig_to_base64(fig)


def create_stats_plot(analysis: Dict) -> str:
    """Create statistical analysis plots."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Match statistics pie chart
    ax1.set_title('Channel Matching Overview', fontsize=13, fontweight='bold')
    sizes = [len(analysis['matched']), len(analysis['unmatched_file'])]
    colors = ['#27ae60', '#e67e22']
    labels = [f"Matched\n{len(analysis['matched'])} channels",
              f"Unmatched\n{len(analysis['unmatched_file'])} channels"]
    explode = (0.05, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.axis('equal')

    # Plot 2: Distance distribution histogram
    if len(analysis['positions']) > 0:
        distances = np.linalg.norm(analysis['positions'], axis=1)
        ax2.set_title('Distance Distribution from Origin', fontsize=13, fontweight='bold')
        ax2.hist(distances, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.axvline(analysis['mean_distance'], color='#e74c3c', linestyle='--',
                   linewidth=2, label=f'Mean: {analysis["mean_distance"]:.4f}m')
        ax2.set_xlabel('Distance (meters)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Position scatter (X-Y with distance color)
    ax3.set_title('Position Distribution (colored by distance)', fontsize=13, fontweight='bold')
    if len(analysis['positions']) > 0:
        x = analysis['positions'][:, 0]
        y = analysis['positions'][:, 1]
        distances = np.linalg.norm(analysis['positions'], axis=1)
        scatter = ax3.scatter(x, y, c=distances, cmap='viridis', s=100,
                            alpha=0.7, edgecolors='black', linewidths=1)
        plt.colorbar(scatter, ax=ax3, label='Distance (m)')
        ax3.set_xlabel('X (meters)', fontsize=11)
        ax3.set_ylabel('Y (meters)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')

        # Add head circle
        circle = Circle((0, 0), 0.095, fill=False, edgecolor='black', linewidth=2)
        ax3.add_patch(circle)

    # Plot 4: Match percentage gauge
    ax4.set_title('Match Quality Score', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    pct = analysis['match_pct']

    # Gauge background
    gauge_bg = FancyBboxPatch((0.1, 0.3), 0.8, 0.3, boxstyle="round,pad=0.05",
                              facecolor='#ecf0f1', edgecolor='#95a5a6', linewidth=3)
    ax4.add_patch(gauge_bg)

    # Gauge fill
    if pct >= 95:
        color = '#27ae60'
        status = 'EXCELLENT'
    elif pct >= 85:
        color = '#f39c12'
        status = 'GOOD'
    elif pct >= 70:
        color = '#e67e22'
        status = 'FAIR'
    else:
        color = '#e74c3c'
        status = 'POOR'

    gauge_fill = FancyBboxPatch((0.1, 0.3), 0.8 * (pct/100), 0.3,
                               boxstyle="round,pad=0.05",
                               facecolor=color, alpha=0.7)
    ax4.add_patch(gauge_fill)

    # Text
    ax4.text(0.5, 0.75, f"{pct:.1f}%", ha='center', va='center',
            fontsize=36, fontweight='bold', color=color)
    ax4.text(0.5, 0.15, status, ha='center', va='center',
            fontsize=24, fontweight='bold', color=color)

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_html_report(
    bdf_file: str,
    montage_name: str,
    analysis: Dict,
    suggestions: List,
    output_file: Path,
    topomap_b64: str,
    plot3d_b64: str,
    stats_b64: str,
    metadata: Dict,
    channel_info: List[Dict],
    rename_map: Dict
):
    """Generate enhanced HTML report optimized for EEG researchers."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine verdict and generate recommendations
    issues = []
    recommendations = []

    if analysis['match_pct'] < 95:
        issues.append(f"Low match percentage ({analysis['match_pct']:.1f}%)")
        recommendations.append("Review alternative montage suggestions below")
        recommendations.append("Verify correct montage selection for this recording system")

    if analysis['match_pct'] < 90:
        recommendations.append("CRITICAL: Less than 90% match - likely wrong montage selected")

    if analysis['duplicates']:
        issues.append(f"{len(analysis['duplicates'])} duplicate position(s) - WRONG MONTAGE LIKELY")
        recommendations.append("Duplicate positions indicate incorrect montage - check alternative suggestions")

    if analysis['outliers']:
        issues.append(f"{len(analysis['outliers'])} position outlier(s)")
        recommendations.append("Check electrode placement for outlier channels")

    if analysis['unmatched_file']:
        issues.append(f"{len(analysis['unmatched_file'])} unmatched channel(s) in file")
        recommendations.append("Verify channel naming conventions match expected format")

    verdict_class = "success" if not issues else "error" if analysis['match_pct'] < 90 or analysis['duplicates'] else "warning"
    verdict_icon = "PASS" if not issues else "FAIL" if analysis['match_pct'] < 90 or analysis['duplicates'] else "WARNING"
    verdict_title = "DATA QUALITY: VALIDATED" if not issues else "DATA QUALITY: FAILED" if analysis['match_pct'] < 90 or analysis['duplicates'] else "DATA QUALITY: NEEDS REVIEW"

    # Calculate number of columns for channel table
    # Show channels in 2 or 3 columns depending on total count
    total_channels = len(channel_info)
    if total_channels <= 40:
        n_columns = 2
    elif total_channels <= 90:
        n_columns = 3
    else:
        n_columns = 4

    channels_per_column = math.ceil(total_channels / n_columns)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EEG Channel Validation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        @page {{
            margin: 0.75in;
            size: letter;
        }}

        body {{
            font-family: 'Times New Roman', Times, serif;
            background: #f5f5f5;
            padding: 20px;
            color: #000;
            line-height: 1.5;
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
            margin-bottom: 12px;
        }}

        .header .meta {{
            font-size: 11px;
            color: #666;
        }}

        /* ENHANCED VERDICT PANEL - Priority 1 */
        .verdict {{
            margin: 0;
            padding: 20px 30px;
            text-align: center;
            font-size: 14px;
            border-bottom: 3px solid #ddd;
        }}

        .verdict.success {{
            background: #e8f5e9;
            color: #1b5e20;
            border-left: 8px solid #4caf50;
        }}

        .verdict.warning {{
            background: #fff8e1;
            color: #7a5c00;
            border-left: 8px solid #ff9800;
        }}

        .verdict.error {{
            background: #ffebee;
            color: #8b0000;
            border-left: 8px solid #f44336;
        }}

        .verdict-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
            letter-spacing: 1px;
        }}

        .verdict-status {{
            font-size: 15px;
            font-weight: bold;
            margin-bottom: 12px;
        }}

        .verdict-details {{
            font-size: 13px;
            font-weight: normal;
            margin-bottom: 8px;
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
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        /* TWO-COLUMN GRID for metadata sections */
        .two-col-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
            margin: 15px 0;
        }}

        .col-section {{
            min-width: 0;
        }}

        .col-section h3 {{
            font-size: 13px;
            font-weight: bold;
            margin-bottom: 12px;
            color: #333;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        /* ENHANCED STAT CARDS */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin: 15px 0;
        }}

        .stat-card {{
            background: #fafafa;
            padding: 12px 14px;
            border: 1px solid #ddd;
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
            font-size: 16px;
            font-weight: bold;
            color: #000;
        }}

        .stat-card.success .value {{ color: #2d5016; }}
        .stat-card.warning .value {{ color: #7a5c00; }}
        .stat-card.error .value {{ color: #8b0000; }}

        /* SIDE-BY-SIDE IMAGES */
        .image-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }}

        .image-container {{
            margin: 0;
            text-align: center;
            page-break-inside: avoid;
        }}

        .image-container img {{
            max-width: 100%;
            border: 1px solid #ccc;
        }}

        .image-container.full {{
            grid-column: 1 / -1;
        }}

        .image-caption {{
            font-size: 11px;
            margin-top: 6px;
            color: #666;
            font-style: italic;
        }}

        /* ENHANCED TABLES - Improved readability */
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
            letter-spacing: 0.3px;
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

        /* MULTI-COLUMN CHANNEL TABLE */
        .multi-column-table {{
            display: grid;
            grid-template-columns: repeat({n_columns}, 1fr);
            gap: 15px;
            margin: 15px 0;
        }}

        .channel-column {{
            min-width: 0;
        }}

        .channel-column table {{
            font-size: 11px;
        }}

        .channel-column th {{
            font-size: 10px;
            padding: 6px 4px;
        }}

        .channel-column td {{
            padding: 5px 4px;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 6px;
            font-size: 9px;
            font-weight: bold;
            border: 1px solid;
        }}

        .badge.success {{ background: #e8f5e9; color: #2d5016; border-color: #4caf50; }}
        .badge.warning {{ background: #fff8e1; color: #7a5c00; border-color: #ff9800; }}
        .badge.error {{ background: #ffebee; color: #8b0000; border-color: #f44336; }}

        /* ENHANCED ISSUE BOXES */
        .channel-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin: 10px 0;
        }}

        .channel-chip {{
            background: #f0f0f0;
            color: #000;
            padding: 4px 8px;
            font-size: 11px;
            font-weight: 600;
            border: 1px solid #999;
        }}

        .channel-chip.unmatched {{
            background: #ff9800;
            color: white;
            border-color: #e65100;
            font-weight: bold;
        }}

        .footer {{
            background: #f5f5f5;
            color: #666;
            text-align: center;
            padding: 20px;
            font-size: 11px;
            border-top: 2px solid #000;
        }}

        .info-box {{
            background: #f9f9f9;
            border: 1px solid #ccc;
            padding: 12px;
            margin: 12px 0;
            font-size: 12px;
        }}

        .warning-box {{
            background: #fff8e1;
            border: 2px solid #ff9800;
            padding: 15px;
            margin: 15px 0;
            font-size: 13px;
        }}

        .error-box {{
            background: #ffebee;
            border: 2px solid #f44336;
            padding: 15px;
            margin: 15px 0;
            font-size: 13px;
            font-weight: bold;
        }}

        p {{
            margin: 10px 0;
            font-size: 13px;
        }}

        ul {{
            margin: 10px 0 10px 25px;
            font-size: 13px;
        }}

        li {{
            margin: 6px 0;
        }}

        /* COMPACT INFO ROWS */
        .info-row {{
            display: grid;
            grid-template-columns: 140px 1fr;
            gap: 12px;
            padding: 6px 0;
            border-bottom: 1px solid #eee;
        }}

        .info-row:last-child {{
            border-bottom: none;
        }}

        .info-label {{
            font-weight: bold;
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}

        .info-value {{
            font-size: 13px;
            color: #000;
        }}

        .recommendations-box {{
            background: #e3f2fd;
            border: 2px solid #1976d2;
            padding: 15px;
            margin: 15px 0;
        }}

        .recommendations-box h3 {{
            color: #0d47a1;
            margin-bottom: 10px;
            font-size: 14px;
        }}

        .recommendations-box ul {{
            margin-left: 20px;
        }}

        .recommendations-box li {{
            margin: 8px 0;
            font-size: 13px;
        }}

        @media print {{
            body {{ padding: 0; background: white; }}
            .container {{ box-shadow: none; max-width: none; }}
            .section {{ page-break-inside: avoid; }}
            .image-grid {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>EEG Channel Validation Report</h1>
            <div class="subtitle">Electrode Position and Montage Analysis - ACNS Standards</div>
            <div class="meta">
                <strong>File:</strong> {Path(bdf_file).name} | <strong>Montage:</strong> {montage_name} | <strong>Generated:</strong> {timestamp}
            </div>
        </div>

        <!-- PRIORITY 1: VERDICT & MATCH ANALYSIS -->
        <div class="verdict {verdict_class}">
            <div class="verdict-title">{verdict_icon}</div>
            <div class="verdict-status">{verdict_title}</div>
            <div class="verdict-details">
                Match Rate: <strong>{analysis['match_pct']:.1f}%</strong> ({len(analysis['matched'])}/{len(analysis['file_channels'])} channels) |
                Positioned Channels: <strong>{analysis['n_positioned']}/{len(analysis['file_channels'])}</strong>
            </div>
            {'<div class="verdict-details" style="color: #8b0000; margin-top: 8px;">Duplicate positions detected - WRONG MONTAGE likely!</div>' if analysis['duplicates'] else ''}
        </div>

        <div class="section">
            <h2>1. Match Analysis & Quality Issues</h2>
            <div class="stats-grid">
                <div class="stat-card {'success' if analysis['match_pct'] >= 95 else 'error' if analysis['match_pct'] < 90 else 'warning'}">
                    <div class="label">Match Percentage</div>
                    <div class="value">{analysis['match_pct']:.1f}%</div>
                </div>
                <div class="stat-card success">
                    <div class="label">Matched Channels</div>
                    <div class="value">{len(analysis['matched'])}</div>
                </div>
                <div class="stat-card {'error' if len(analysis['unmatched_file']) > 0 else 'success'}">
                    <div class="label">Unmatched (File)</div>
                    <div class="value">{len(analysis['unmatched_file'])}</div>
                </div>
                <div class="stat-card {'success' if analysis['n_positioned'] == len(analysis['file_channels']) else 'warning'}">
                    <div class="label">With Positions</div>
                    <div class="value">{analysis['n_positioned']}</div>
                </div>
                <div class="stat-card {'error' if analysis['duplicates'] else 'success'}">
                    <div class="label">Duplicate Positions</div>
                    <div class="value">{len(analysis['duplicates'])}</div>
                </div>
                <div class="stat-card {'warning' if analysis['outliers'] else 'success'}">
                    <div class="label">Position Outliers</div>
                    <div class="value">{len(analysis['outliers'])}</div>
                </div>
            </div>

            {f'''<div class="error-box">
                <strong>CRITICAL ERROR: DUPLICATE POSITIONS DETECTED</strong><br><br>
                Multiple channels share the same position coordinates. This usually indicates the <strong>WRONG MONTAGE</strong> was selected for this recording!<br><br>
                <strong>Action Required:</strong> Review alternative montage suggestions below and re-run validation.
            </div>''' if analysis['duplicates'] else ''}

            {f'''<div class="warning-box">
                <strong>UNMATCHED CHANNELS IN FILE ({len(analysis['unmatched_file'])} channels):</strong><br><br>
                These channels exist in the recording but not in the selected montage. This may indicate:
                <ul style="margin: 10px 0 0 20px;">
                    <li>Wrong montage selected</li>
                    <li>Channel naming convention mismatch</li>
                    <li>Non-standard electrode placement</li>
                </ul>
                <div class="channel-list" style="margin-top: 12px;">
                    {' '.join(f'<span class="channel-chip unmatched">{ch}</span>' for ch in sorted(analysis['unmatched_file']))}
                </div>
            </div>''' if analysis['unmatched_file'] else ''}

            {f'''<div class="warning-box">
                <strong>POSITION OUTLIERS DETECTED ({len(analysis['outliers'])} channels):</strong><br><br>
                These channels have positions more than 3 standard deviations from the mean distance:<br>
                <strong>{', '.join(analysis['outliers'])}</strong><br><br>
                This may indicate electrode placement errors or coordinate system issues.
            </div>''' if analysis['outliers'] else ''}

            {f'''<div class="recommendations-box">
                <h3>RECOMMENDATIONS - ACTIONS REQUIRED:</h3>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                </ul>
            </div>''' if recommendations else '<div class="info-box"><strong>ALL CHECKS PASSED:</strong> No quality issues detected. Data is ready for analysis.</div>'}
        </div>

        <!-- PRIORITY 2: VISUAL COMPARISON -->
        <div class="section">
            <h2>2. Electrode Position Visualizations</h2>
            <p style="font-size: 12px; color: #666; margin-bottom: 15px;">
                Visual verification of electrode placement using 10-20 system coordinates.
                <strong>Green circles = matched channels</strong> | <strong>Orange squares = unmatched channels</strong>
            </p>
            <div class="image-grid">
                <div class="image-container">
                    <img src="{topomap_b64}" alt="2D Topomap">
                    <div class="image-caption">2D Electrode Positions (Top & Side Views)</div>
                </div>
                <div class="image-container">
                    <img src="{plot3d_b64}" alt="3D Positions">
                    <div class="image-caption">3D Electrode Positions (Multiple Perspectives)</div>
                </div>
            </div>
        </div>

        <!-- PRIORITY 2: STATISTICAL ANALYSIS -->
        <div class="section">
            <h2>3. Statistical Analysis</h2>
            <div class="image-container full">
                <img src="{stats_b64}" alt="Statistical Analysis" style="max-width: 95%;">
                <div class="image-caption">Position distribution, match statistics, and quality metrics</div>
            </div>
        </div>

        <!-- PRIORITY 3: RECORDING METADATA -->
        <div class="section">
            <h2>4. Recording Metadata</h2>
            <div class="two-col-grid">
                <div class="col-section">
                    <h3>Acquisition Parameters</h3>
                    <div class="info-row">
                        <div class="info-label">Sampling Rate</div>
                        <div class="info-value"><strong>{metadata['sampling_rate']}</strong></div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Duration</div>
                        <div class="info-value">{metadata['duration']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Total Samples</div>
                        <div class="info-value">{metadata['n_samples']:,}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Highpass Filter</div>
                        <div class="info-value">{metadata['highpass_filter']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Lowpass Filter</div>
                        <div class="info-value">{metadata['lowpass_filter']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Line Frequency</div>
                        <div class="info-value">{metadata['line_freq']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Reference</div>
                        <div class="info-value">{metadata['reference']}</div>
                    </div>
                </div>

                <div class="col-section">
                    <h3>File & Device Information</h3>
                    <div class="info-row">
                        <div class="info-label">File Name</div>
                        <div class="info-value">{Path(bdf_file).name}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Format</div>
                        <div class="info-value">{metadata['file_format']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Recording Date</div>
                        <div class="info-value">{metadata['recording_date']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Total Channels</div>
                        <div class="info-value"><strong>{metadata['n_channels']}</strong></div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Device Type</div>
                        <div class="info-value">{metadata['device_type']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Device Model</div>
                        <div class="info-value">{metadata['device_model']}</div>
                    </div>
                    <div class="info-row">
                        <div class="info-label">Montage Applied</div>
                        <div class="info-value"><strong>{montage_name}</strong></div>
                    </div>
                </div>
            </div>

            {'<div class="info-box" style="margin-top: 15px;"><strong>Channel Type Breakdown:</strong> ' + ', '.join([f'{k}: {v}' for k, v in metadata["channel_types"].items()]) + '</div>' if metadata['channel_types'] else ''}
        </div>

        <!-- PRIORITY 4: COMPLETE CHANNEL TABLE (Multi-column layout) -->
        <div class="section">
            <h2>5. Raw Channel Information (Pre-Processing)</h2>
            <p style="color: #666; margin-bottom: 15px; font-size: 12px;">
                <strong>Original channel names as they appear in the file BEFORE any renaming or montage application.</strong>
                This section provides full transparency for debugging channel naming issues.
                Table shows channels in {n_columns} columns for space efficiency.
            </p>

            {f'<div class="info-box" style="margin-bottom: 15px;"><strong>Transformation Summary:</strong> {len(rename_map)} channels renamed for standardization. See table below for complete mapping and match status.</div>' if rename_map else '<div class="info-box" style="margin-bottom: 15px;"><strong>No channel renaming was applied.</strong> Channels are shown exactly as they appear in the file.</div>'}

            <div class="multi-column-table">"""

    # Generate multi-column channel tables
    for col_idx in range(n_columns):
        start_idx = col_idx * channels_per_column
        end_idx = min(start_idx + channels_per_column, total_channels)
        col_channels = channel_info[start_idx:end_idx]

        html += f"""
                <div class="channel-column">
                    <table>
                        <thead>
                            <tr>
                                <th>Idx</th>
                                <th>Raw Name</th>
                                <th>Std Name</th>
                                <th>Match</th>
                                <th>Type</th>
                                <th>X</th>
                                <th>Y</th>
                                <th>Z</th>
                            </tr>
                        </thead>
                        <tbody>"""

        for ch in col_channels:
            # Determine row styling based on match status
            row_class = ""
            if ch['matched_montage'] is True:
                row_class = ' style="background: #e8f5e9;"'  # Light green for matched
            elif ch['matched_montage'] is False:
                row_class = ' style="background: #fff8e1;"'  # Light yellow for unmatched

            html += f"""
                            <tr{row_class}>
                                <td>{ch['index']}</td>
                                <td><strong>{ch['raw_name']}</strong></td>
                                <td>{'<span style="color: #1976d2;">' + ch['standard_name'] + '</span>' if ch['will_rename'] else '<span style="color: #999;">—</span>'}</td>
                                <td>{'<span class="badge success">✓</span>' if ch['matched_montage'] else '<span class="badge error">✗</span>' if ch['matched_montage'] is False else '<span class="badge">?</span>'}</td>
                                <td><span class="badge {'success' if ch['type'] == 'EEG' else 'warning' if ch['type'] == 'STIM' else ''}">{ch['type']}</span></td>
                                <td style="font-size: 10px;">{ch['x']}</td>
                                <td style="font-size: 10px;">{ch['y']}</td>
                                <td style="font-size: 10px;">{ch['z']}</td>
                            </tr>"""

        html += """
                        </tbody>
                    </table>
                </div>"""

    html += """
            </div>
        </div>

        <!-- PRIORITY 4: ALTERNATIVE MONTAGE SUGGESTIONS -->"""

    if analysis['match_pct'] < 95:
        html += f"""
        <div class="section">
            <h2>6. Alternative Montage Suggestions</h2>
            <div class="warning-box">
                <strong>Current montage match is below 95%.</strong> Consider these alternatives for better channel matching:
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Montage Name</th>
                            <th>Matched Channels</th>
                            <th>Match %</th>
                            <th>Improvement</th>
                            <th>Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''<tr style="background: {'#e8f5e9' if pct > analysis['match_pct'] else 'inherit'};">
                            <td><strong>{i}</strong></td>
                            <td><strong>{name}</strong></td>
                            <td>{matched}</td>
                            <td><strong>{pct:.1f}%</strong></td>
                            <td>{f"+{pct - analysis['match_pct']:.1f}%" if pct > analysis['match_pct'] else "-"}</td>
                            <td>{'<span class="badge success">TRY THIS FIRST</span>' if pct > analysis['match_pct'] and i <= 3 else '<span class="badge warning">Consider</span>' if pct > analysis['match_pct'] else ''}</td>
                        </tr>''' for i, (name, matched, pct) in enumerate(suggestions[:10], 1))}
                    </tbody>
                </table>
            </div>
        </div>"""

    html += f"""
        <div class="footer">
            <div style="font-weight: bold; margin-bottom: 5px;">EEG Channel Validation Report v5.0 (EEG Research Optimized)</div>
            <div>Generated by AutoClean EEG Pipeline | Compliant with ACNS Guidelines</div>
            <div style="margin-top: 8px; font-size: 10px;">
                Report includes: Montage validation, electrode position verification, channel mapping analysis, quality metrics
            </div>
        </div>
    </div>
</body>
</html>"""

    output_file.write_text(html)
    console.print(f"\n✅ [bold green]HTML report saved:[/bold green] {output_file}")


def main():
    """Main execution."""

    bdf_file = "/Users/ernie/Downloads/Example EEGs/st101as.bdf"
    montage_name = "biosemi64"
    output_dir = "."

    if len(sys.argv) > 1:
        bdf_file = sys.argv[1]
    if len(sys.argv) > 2:
        montage_name = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]    BDF Visual Validation Report v5.1 (Transparency)[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    # Load and process
    console.print(f"[info]Loading:[/info] {Path(bdf_file).name}")
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, stim_channel="auto", exclude=[], verbose=False)

    # Determine rename map first
    rename_map = {}
    for ch in raw.ch_names:
        if ch != "Status" and "_" in ch:
            _, std_name = ch.split("_", 1)
            if std_name == "Afz":
                std_name = "AFz"
            rename_map[ch] = std_name

    # Load montage to get montage channels
    console.print(f"[info]Loading montage:[/info] {montage_name}")
    montage = mne.channels.make_standard_montage(montage_name)
    montage_channels = set(montage.get_positions()['ch_pos'].keys())

    # Extract RAW channel info BEFORE any renaming (for transparency)
    console.print("[info]Extracting raw channel information...[/info]")
    raw_channel_info = extract_channel_info(raw, rename_map, montage_channels)

    # Now apply renaming
    if rename_map:
        raw.rename_channels(rename_map)
        console.print(f"[success]✓[/success] Renamed {len(rename_map)} channels")

    raw.set_montage(montage, match_case=False, on_missing="warn")

    # Analyze
    console.print("[info]Analyzing channels...[/info]")
    analysis = analyze_channels(raw, montage)

    console.print("[info]Testing alternative montages...[/info]")
    suggestions = suggest_montages(analysis['file_channels'])

    # Extract metadata
    console.print("[info]Extracting file metadata...[/info]")
    metadata = extract_file_metadata(raw, bdf_file)

    # Generate visualizations
    console.print("[info]Generating visualizations...[/info]")
    topomap_b64 = create_elegant_topomap(analysis, f"{Path(bdf_file).name} - {montage_name}")
    plot3d_b64 = create_3d_plot(analysis, f"3D Electrode Positions - {montage_name}")
    stats_b64 = create_stats_plot(analysis)

    # Generate HTML report
    html_file = output_dir / f"{Path(bdf_file).stem}_validation_report_v5.html"
    console.print("[info]Generating HTML report...[/info]")
    generate_html_report(bdf_file, montage_name, analysis, suggestions,
                        html_file, topomap_b64, plot3d_b64, stats_b64,
                        metadata, raw_channel_info, rename_map)

    # Summary
    console.print("\n[bold green]════════════════════════════════════════════════════════════[/bold green]")
    console.print(f"[bold green]  Report Complete![/bold green]")
    console.print(f"[bold green]════════════════════════════════════════════════════════════[/bold green]")
    console.print(f"\n  Match: [{'green' if analysis['match_pct'] >= 95 else 'yellow'}]{analysis['match_pct']:.1f}%[/]")
    console.print(f"  Positioned: {analysis['n_positioned']}/{len(analysis['file_channels'])}")
    console.print(f"  HTML Report: [cyan]{html_file}[/cyan]\n")


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
