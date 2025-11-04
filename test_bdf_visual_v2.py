#!/usr/bin/env python3
"""
BDF Import Visual Diagnostic Report - Enhanced v2

Professional-grade validation report with HTML output, interactive visualizations,
and comprehensive montage analysis.

Usage: uv run test_bdf_visual_v2.py [bdf_file] [montage_name] [output_dir]
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from datetime import datetime
import base64
from io import BytesIO

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


def extract_channel_info(raw: mne.io.Raw) -> List[Dict]:
    """Extract detailed channel-by-channel information.

    Works across all file formats supported by MNE.
    """
    channel_info = []

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

        channel_info.append({
            'name': ch_name,
            'index': idx,
            'type': ch_type_name,
            'unit': ch.get('unit_mul', 1),
            'has_position': has_position,
            'x': f"{loc[0]:.4f}" if has_position else "N/A",
            'y': f"{loc[1]:.4f}" if has_position else "N/A",
            'z': f"{loc[2]:.4f}" if has_position else "N/A",
            'cal': ch.get('cal', 'N/A')
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
    channel_info: List[Dict]
):
    """Generate professional HTML report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Determine verdict
    issues = []
    if analysis['match_pct'] < 90:
        issues.append(f"Low match percentage ({analysis['match_pct']:.1f}%)")
    if analysis['duplicates']:
        issues.append(f"{len(analysis['duplicates'])} duplicate position(s)")
    if analysis['outliers']:
        issues.append(f"{len(analysis['outliers'])} position outlier(s)")

    verdict_class = "success" if not issues else "warning"
    verdict_icon = "✓" if not issues else "⚠"
    verdict_title = "VALIDATED" if not issues else "NEEDS REVIEW"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EEG Channel Validation Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        @page {{
            margin: 1in;
        }}

        body {{
            font-family: 'Times New Roman', Times, serif;
            background: #f5f5f5;
            padding: 40px 20px;
            color: #000;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}

        .header {{
            background: #fff;
            border-bottom: 3px solid #000;
            padding: 40px 50px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
            font-weight: bold;
            color: #000;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .header .subtitle {{
            font-size: 16px;
            color: #333;
            font-style: italic;
            margin-bottom: 15px;
        }}

        .header .meta {{
            font-size: 12px;
            color: #666;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ddd;
        }}

        .verdict {{
            margin: 0;
            padding: 30px 50px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            border-bottom: 2px solid #ddd;
        }}

        .verdict.success {{
            background: #f0f8f0;
            color: #2d5016;
            border-left: 5px solid #4caf50;
        }}

        .verdict.warning {{
            background: #fff8e1;
            color: #7a5c00;
            border-left: 5px solid #ff9800;
        }}

        .section {{
            margin: 0;
            padding: 40px 50px;
            border-bottom: 1px solid #ddd;
        }}

        .section:last-child {{
            border-bottom: none;
        }}

        .section h2 {{
            color: #000;
            margin-bottom: 20px;
            font-size: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 25px 0;
        }}

        .stat-card {{
            background: #fafafa;
            padding: 15px;
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
            font-size: 18px;
            font-weight: bold;
            color: #000;
        }}

        .stat-card.success .value {{ color: #2d5016; }}
        .stat-card.warning .value {{ color: #7a5c00; }}
        .stat-card.error .value {{ color: #8b0000; }}

        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}

        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background: #34495e;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}

        .badge.success {{ background: #d5f4e6; color: #27ae60; }}
        .badge.warning {{ background: #ffeaa7; color: #e67e22; }}
        .badge.error {{ background: #ffcccc; color: #e74c3c; }}

        .channel-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 15px 0;
        }}

        .channel-chip {{
            background: #3498db;
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 500;
        }}

        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 14px;
        }}

        .info-box {{
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}

        .warning-box {{
            background: #fef5e7;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}

        .error-box {{
            background: #fadbd8;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 BDF Channel Validation Report</h1>
            <div class="subtitle">Electrode Position & Montage Analysis</div>
            <div class="subtitle" style="margin-top: 10px; font-size: 14px; opacity: 0.7;">
                Generated: {timestamp}
            </div>
        </div>

        <div class="verdict {verdict_class}">
            <div style="font-size: 48px; margin-bottom: 10px;">{verdict_icon}</div>
            <div>{verdict_title}</div>
            <div style="font-size: 16px; margin-top: 10px; opacity: 0.9;">
                Match: {analysis['match_pct']:.1f}% | Positioned: {analysis['n_positioned']}/{len(analysis['file_channels'])}
            </div>
        </div>

        <div class="section">
            <h2>File Information</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">BDF File</div>
                    <div class="value" style="font-size: 18px;">{Path(bdf_file).name}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Selected Montage</div>
                    <div class="value" style="font-size: 18px;">{montage_name}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Channels in File</div>
                    <div class="value">{len(analysis['file_channels'])}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Channels in Montage</div>
                    <div class="value">{len(analysis['montage_channels'])}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Recording Metadata</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="label">File Format</div>
                    <div class="value" style="font-size: 18px;">{metadata['file_format']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Recording Date</div>
                    <div class="value" style="font-size: 14px;">{metadata['recording_date']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Sampling Rate</div>
                    <div class="value" style="font-size: 18px;">{metadata['sampling_rate']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Duration</div>
                    <div class="value" style="font-size: 16px;">{metadata['duration']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total Channels</div>
                    <div class="value">{metadata['n_channels']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Total Samples</div>
                    <div class="value" style="font-size: 16px;">{metadata['n_samples']:,}</div>
                </div>
            </div>

            <div class="stats-grid" style="margin-top: 20px;">
                <div class="stat-card">
                    <div class="label">Device Type</div>
                    <div class="value" style="font-size: 14px;">{metadata['device_type']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Device Model</div>
                    <div class="value" style="font-size: 14px;">{metadata['device_model']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Highpass Filter</div>
                    <div class="value" style="font-size: 16px;">{metadata['highpass_filter']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Lowpass Filter</div>
                    <div class="value" style="font-size: 16px;">{metadata['lowpass_filter']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Line Frequency</div>
                    <div class="value" style="font-size: 16px;">{metadata['line_freq']}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Reference</div>
                    <div class="value" style="font-size: 14px;">{metadata['reference']}</div>
                </div>
            </div>

            {'<div class="info-box" style="margin-top: 20px;"><strong>Channel Types:</strong> ' + ', '.join([f'{k}: {v}' for k, v in metadata["channel_types"].items()]) + '</div>' if metadata['channel_types'] else ''}
            {'<div class="info-box"><strong>Subject ID:</strong> ' + str(metadata['subject_id']) + '</div>' if metadata.get('subject_id') and metadata['subject_id'] != 'Not available' else ''}
        </div>

        <div class="section">
            <h2>Raw Channel Information</h2>
            <p style="color: #7f8c8d; margin-bottom: 20px;">
                Detailed channel-by-channel information extracted from the raw file.
                Channels are shown as they appear in the file before montage application.
            </p>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Index</th>
                            <th>Channel Name</th>
                            <th>Type</th>
                            <th>Has Position</th>
                            <th>X</th>
                            <th>Y</th>
                            <th>Z</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f'''
                        <tr>
                            <td>{ch['index']}</td>
                            <td><strong>{ch['name']}</strong></td>
                            <td><span class="badge {'success' if ch['type'] == 'EEG' else 'warning' if ch['type'] == 'STIM' else ''}">{ch['type']}</span></td>
                            <td>{'<span class="badge success">✓</span>' if ch['has_position'] else '<span class="badge error">✗</span>'}</td>
                            <td>{ch['x']}</td>
                            <td>{ch['y']}</td>
                            <td>{ch['z']}</td>
                        </tr>
                        ''' for ch in channel_info])}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Match Analysis</h2>
            <div class="stats-grid">
                <div class="stat-card success">
                    <div class="label">Matched Channels</div>
                    <div class="value">{len(analysis['matched'])}</div>
                </div>
                <div class="stat-card {'success' if analysis['match_pct'] >= 95 else 'warning' if analysis['match_pct'] >= 85 else 'error'}">
                    <div class="label">Match Percentage</div>
                    <div class="value">{analysis['match_pct']:.1f}%</div>
                </div>
                <div class="stat-card {'warning' if len(analysis['unmatched_file']) > 0 else 'success'}">
                    <div class="label">Unmatched (File)</div>
                    <div class="value">{len(analysis['unmatched_file'])}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Unmatched (Montage)</div>
                    <div class="value">{len(analysis['unmatched_montage'])}</div>
                </div>
            </div>

            {f'''<div class="warning-box">
                <strong>⚠ Unmatched channels in file:</strong><br>
                <div class="channel-list">
                    {' '.join(f'<span class="channel-chip" style="background: #e67e22;">{ch}</span>' for ch in sorted(analysis['unmatched_file']))}
                </div>
            </div>''' if analysis['unmatched_file'] else ''}
        </div>

        <div class="section">
            <h2>Position Quality</h2>
            <div class="stats-grid">
                <div class="stat-card {'success' if analysis['n_positioned'] == len(analysis['file_channels']) else 'warning'}">
                    <div class="label">Channels with Positions</div>
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
                <div class="stat-card">
                    <div class="label">Mean Distance</div>
                    <div class="value" style="font-size: 20px;">{analysis['mean_distance']:.4f}m</div>
                </div>
            </div>

            {f'''<div class="error-box">
                <strong>✗ DUPLICATE POSITIONS DETECTED!</strong><br>
                This usually indicates <strong>WRONG MONTAGE</strong> selected!
            </div>''' if analysis['duplicates'] else ''}

            {f'''<div class="warning-box">
                <strong>⚠ Position outliers detected:</strong><br>
                {', '.join(analysis['outliers'])}
            </div>''' if analysis['outliers'] else ''}
        </div>

        <div class="section">
            <h2>Statistical Analysis</h2>
            <div class="image-container">
                <img src="{stats_b64}" alt="Statistical Analysis">
            </div>
        </div>

        <div class="section">
            <h2>2D Electrode Positions</h2>
            <div class="image-container">
                <img src="{topomap_b64}" alt="2D Topomap">
            </div>
            <div class="info-box">
                <strong>ℹ Legend:</strong>
                Green circles = Matched channels with positions |
                Orange squares = Unmatched channels
            </div>
        </div>

        <div class="section">
            <h2>3D Electrode Positions</h2>
            <div class="image-container">
                <img src="{plot3d_b64}" alt="3D Positions">
            </div>
        </div>

        {f'''<div class="section">
            <h2>Alternative Montage Suggestions</h2>
            <div class="warning-box">
                Current montage match is below 95%. Consider these alternatives:
            </div>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Montage Name</th>
                            <th>Matched Channels</th>
                            <th>Match %</th>
                            <th>Recommendation</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''<tr>
                            <td>{i}</td>
                            <td><strong>{name}</strong></td>
                            <td>{matched}</td>
                            <td>{pct:.1f}%</td>
                            <td>{'<span class="badge success">✓ Try this</span>' if pct > analysis['match_pct'] else ''}</td>
                        </tr>''' for i, (name, matched, pct) in enumerate(suggestions[:10], 1))}
                    </tbody>
                </table>
            </div>
        </div>''' if analysis['match_pct'] < 95 else ''}

        <div class="section">
            <h2>Channel Status Table</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Channel</th>
                            <th>In File</th>
                            <th>In Montage</th>
                            <th>Has Position</th>
                            <th>Distance (m)</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(f'''<tr>
                            <td><strong>{ch}</strong></td>
                            <td><span class="badge success">✓</span></td>
                            <td><span class="badge {'success' if data['in_montage'] else 'warning'}">{'✓' if data['in_montage'] else '✗'}</span></td>
                            <td><span class="badge {'success' if data['has_position'] else 'error'}">{'✓' if data['has_position'] else '✗'}</span></td>
                            <td>{f"{data['distance_from_origin']:.4f}" if data['distance_from_origin'] else "N/A"}</td>
                            <td><span class="badge {'success' if data['matched'] and data['has_position'] else 'warning' if not data['in_montage'] else 'error'}">
                                {'✓ Good' if data['matched'] and data['has_position'] else 'Not in montage' if not data['in_montage'] else 'No position'}
                            </span></td>
                        </tr>''' for ch, data in sorted(analysis['channel_data'].items())[:30])}
                        {f'<tr><td colspan="6" style="text-align: center; font-style: italic;">... +{len(analysis["channel_data"]) - 30} more channels</td></tr>' if len(analysis['channel_data']) > 30 else ''}
                    </tbody>
                </table>
            </div>
        </div>

        {f'''<div class="section">
            <h2>Issues Detected</h2>
            {''.join(f'<div class="error-box"><strong>✗</strong> {issue}</div>' for issue in issues)}
            <div class="warning-box">
                <strong>Recommendations:</strong>
                <ul style="margin: 10px 0 0 20px;">
                    <li>Review the alternative montage suggestions above</li>
                    <li>Check the visual plots for position anomalies</li>
                    <li>Verify you selected the correct montage for this file</li>
                    <li>Ensure channel names match expected format</li>
                </ul>
            </div>
        </div>''' if issues else ''}

        <div class="footer">
            <div>BDF Channel Validation Report v2.0</div>
            <div style="margin-top: 5px; opacity: 0.7;">Generated by AutoClean EEG Pipeline</div>
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
    console.print("[bold cyan]    BDF Visual Validation Report v2[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    # Load and process
    console.print(f"[info]Loading:[/info] {Path(bdf_file).name}")
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, stim_channel="auto", exclude=[], verbose=False)

    # Rename channels (from plugin logic)
    rename_map = {}
    for ch in raw.ch_names:
        if ch != "Status" and "_" in ch:
            _, std_name = ch.split("_", 1)
            if std_name == "Afz":
                std_name = "AFz"
            rename_map[ch] = std_name

    if rename_map:
        raw.rename_channels(rename_map)
        console.print(f"[success]✓[/success] Renamed {len(rename_map)} channels")

    console.print(f"[info]Loading montage:[/info] {montage_name}")
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn")

    # Analyze
    console.print("[info]Analyzing channels...[/info]")
    analysis = analyze_channels(raw, montage)

    console.print("[info]Testing alternative montages...[/info]")
    suggestions = suggest_montages(analysis['file_channels'])

    # Extract metadata
    console.print("[info]Extracting file metadata...[/info]")
    metadata = extract_file_metadata(raw, bdf_file)
    channel_info = extract_channel_info(raw)

    # Generate visualizations
    console.print("[info]Generating visualizations...[/info]")
    topomap_b64 = create_elegant_topomap(analysis, f"{Path(bdf_file).name} - {montage_name}")
    plot3d_b64 = create_3d_plot(analysis, f"3D Electrode Positions - {montage_name}")
    stats_b64 = create_stats_plot(analysis)

    # Generate HTML report
    html_file = output_dir / f"{Path(bdf_file).stem}_validation_report.html"
    console.print("[info]Generating HTML report...[/info]")
    generate_html_report(bdf_file, montage_name, analysis, suggestions,
                        html_file, topomap_b64, plot3d_b64, stats_b64,
                        metadata, channel_info)

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
