"""
Montage Validation Utilities

Provides channel validation, position analysis, and HTML report generation
for EEG montage quality assurance. Works across all MNE-supported file formats.
"""

import base64
import math
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch, Wedge

# Configure matplotlib for non-interactive use
matplotlib.use('Agg')
sns.set_palette("husl")
plt.style.use('seaborn-v0_8-darkgrid')


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

    Args:
        raw: MNE Raw object
        original_filename: Original file path for metadata

    Returns:
        Dictionary containing file and recording metadata
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


def detect_coordinate_scale(positions: np.ndarray) -> Tuple[str, float]:
    """Detect if coordinates are at mouse-scale (micrometers) or human-scale (meters).

    Args:
        positions: Nx3 array of electrode positions in meters

    Returns:
        Tuple of (scale_type, visualization_scale_factor)
        - scale_type: 'mouse' or 'human'
        - visualization_scale_factor: multiplier to make coordinates visible (1.0 for human, >1 for mouse)
    """
    if len(positions) == 0:
        return 'unknown', 1.0

    # Calculate the maximum extent in any dimension
    max_extent = np.max(np.ptp(positions, axis=0))

    # Typical scales:
    # - Mouse probes: < 2mm = 0.002m
    # - Human scalp EEG: ~30cm = 0.3m
    # Threshold at 10mm (0.01m)

    if max_extent < 0.01:  # Less than 10mm
        # Mouse/micro scale - need to scale up for visualization
        # Scale to ~10cm spread for human-scale MNE plotting
        target_spread = 0.10  # 10cm
        scale_factor = target_spread / max_extent if max_extent > 0 else 100.0
        return 'mouse', scale_factor
    else:
        # Human scale - use as-is
        return 'human', 1.0


def extract_channel_info(raw: mne.io.Raw, rename_map: Dict = None, montage_channels: Set = None) -> List[Dict]:
    """Extract detailed channel-by-channel information.

    Works across all file formats supported by MNE.

    Args:
        raw: MNE Raw object (before any renaming)
        rename_map: Dictionary of original_name -> standardized_name
        montage_channels: Set of channel names in the montage

    Returns:
        List of dictionaries containing per-channel details
    """
    channel_info = []
    rename_map = rename_map or {}
    montage_channels = montage_channels or set()

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


def analyze_channels(raw: mne.io.Raw, montage: mne.channels.DigMontage, montage_name: str = None) -> Dict:
    """Comprehensive channel analysis.

    Args:
        raw: MNE Raw object (after renaming and montage application)
        montage: MNE montage object
        montage_name: Name of the montage (for detecting mouse-scale probes)

    Returns:
        Dictionary containing analysis results
    """
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

    # Detect coordinate scale
    # Check montage name first for known mouse probes
    if montage_name and ('mouse' in montage_name.lower() or 'mea' in montage_name.lower()):
        # Known mouse-scale montage - force mouse detection
        scale_type = 'mouse'
        scale_factor = 1.0  # No scaling needed since we'll work in micrometers
    else:
        # Detect from coordinates
        scale_type, scale_factor = detect_coordinate_scale(positions)

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
        'positions': positions,
        'scale_type': scale_type,
        'scale_factor': scale_factor,
        'montage_name': montage_name
    }


def suggest_montages(file_channels: Set[str], top_n: int = 10) -> List[Tuple[str, int, float]]:
    """Test and rank alternative montages.

    Args:
        file_channels: Set of channel names from the file
        top_n: Number of top suggestions to return

    Returns:
        List of tuples (montage_name, matched_count, match_percentage)
    """
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


def create_3d_plot(analysis: Dict, title: str) -> str:
    """Create 3D electrode position visualization with automatic scale detection.

    Args:
        analysis: Analysis results from analyze_channels()
        title: Plot title

    Returns:
        Base64-encoded PNG image
    """
    scale_type = analysis.get('scale_type', 'human')
    scale_factor = analysis.get('scale_factor', 1.0)

    # For mouse scale, use flat 2D grid instead of 3D views
    if scale_type == 'mouse':
        return _create_mouse_flat_plot(analysis)

    # Human scale: use standard 3D views
    fig = plt.figure(figsize=(20, 5))
    views = [
        (30, 45, 'Perspective'),
        (90, 0, 'Top View'),
        (0, 0, 'Back View'),
        (0, 90, 'Side View')
    ]

    for idx, (elev, azim, view_title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 4, idx, projection='3d')
        ax.set_title(view_title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        # Plot channels with labels
        for ch, data in analysis['channel_data'].items():
            if data['has_position']:
                pos = data['position']
                color = '#27ae60' if data['matched'] else '#e67e22'
                marker = 'o' if data['matched'] else 's'
                size = 60 if data['matched'] else 50

                # Plot electrode marker
                ax.scatter(*pos, c=color, marker=marker, s=size, alpha=0.8,
                         edgecolors='black', linewidths=0.5)

                # Calculate radial offset for label (away from origin)
                norm = np.linalg.norm(pos)
                if norm > 0:
                    offset_dist = 0.015
                    offset_vec = (pos / norm) * offset_dist
                    label_pos = pos + offset_vec
                else:
                    label_pos = pos + np.array([0.01, 0.01, 0.01])

                # Add channel name label with background
                ax.text(label_pos[0], label_pos[1], label_pos[2], ch,
                       fontsize=7, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                edgecolor='none', alpha=0.8),
                       zorder=10)

        # Head sphere for human scale
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = 0.095 * np.outer(np.cos(u), np.sin(v))
        y = 0.095 * np.outer(np.sin(u), np.sin(v))
        z = 0.095 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.15, color='gray')

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim([-0.12, 0.12])
        ax.set_ylim([-0.12, 0.12])
        ax.set_zlim([-0.12, 0.12])
        ax.set_box_aspect([1,1,1])

    plt.tight_layout()
    return fig_to_base64(fig)


def _create_mouse_flat_plot(analysis: Dict) -> str:
    """Create 3D and 2D grid visualization for mouse-scale probes.

    Args:
        analysis: Analysis results from analyze_channels()

    Returns:
        Base64-encoded PNG image
    """
    positions = analysis['positions']

    if len(positions) == 0:
        # Return empty plot if no positions
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.text(0.5, 0.5, 'No positioned channels', ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig_to_base64(fig)

    # Use MNE-normalized coordinates directly (MNE scales custom montages)
    montage_name = analysis.get('montage_name', 'Mouse Probe')

    # Create figure with 3 subplots: 3D view, Top view, Side view
    fig = plt.figure(figsize=(24, 8))

    fig.suptitle(f'{montage_name} - Electrode Layout (3D + Orthogonal Views)',
                fontsize=14, fontweight='bold', color='#2c3e50')

    # Subplot 1: 3D perspective view (3/4 view)
    ax_3d = fig.add_subplot(1, 3, 1, projection='3d')
    ax_3d.set_title('3D Perspective View', fontsize=12, fontweight='bold')
    ax_3d.set_xlabel('X Position', fontsize=10)
    ax_3d.set_ylabel('Y Position', fontsize=10)
    ax_3d.set_zlabel('Z Position', fontsize=10)

    # Plot electrodes in 3D
    for ch, data in analysis['channel_data'].items():
        if data['has_position']:
            pos = data['position']
            color = '#27ae60' if data['matched'] else '#e67e22'
            marker = 'o' if data['matched'] else 's'
            size = 100

            ax_3d.scatter(pos[0], pos[1], pos[2], c=color, marker=marker, s=size,
                         alpha=0.8, edgecolors='black', linewidths=2, zorder=3)

            # Add channel label
            ax_3d.text(pos[0], pos[1], pos[2], ch, fontsize=7, ha='center', va='center',
                      fontweight='bold', color='white', zorder=4)

    # Set 3/4 view angle (azimuth=45, elevation=30)
    ax_3d.view_init(elev=30, azim=45)

    # Set equal aspect ratio for 3D plot
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    z_range = np.ptp(positions[:, 2])
    max_range = max(x_range, y_range, z_range)

    x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
    y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
    z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2

    ax_3d.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax_3d.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax_3d.set_zlim(z_center - max_range/2, z_center + max_range/2)
    ax_3d.set_box_aspect([1,1,1])

    # Add grid
    ax_3d.grid(True, alpha=0.3)

    # Subplot 2: Top view (X-Y)
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.set_title('Top View (X-Y Plane)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Position', fontsize=10)
    ax1.set_ylabel('Y Position', fontsize=10)

    # Plot electrodes
    for ch, data in analysis['channel_data'].items():
        if data['has_position']:
            pos = data['position']
            color = '#27ae60' if data['matched'] else '#e67e22'
            marker = 'o' if data['matched'] else 's'
            size = 150

            ax1.scatter(pos[0], pos[1], c=color, marker=marker, s=size,
                       alpha=0.7, edgecolors='black', linewidths=2, zorder=3)

            # Add channel label
            ax1.text(pos[0], pos[1], ch, fontsize=7, ha='center', va='center',
                    fontweight='bold', color='white', zorder=4)

    # Add grid
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.set_aspect('equal')

    # Add axis at origin
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Set limits with padding
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    x_pad = x_range * 0.15
    y_pad = y_range * 0.15
    ax1.set_xlim(positions[:, 0].min() - x_pad, positions[:, 0].max() + x_pad)
    ax1.set_ylim(positions[:, 1].min() - y_pad, positions[:, 1].max() + y_pad)

    # Subplot 3: Side view (X-Z)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.set_title('Side View (X-Z Plane)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X Position', fontsize=10)
    ax2.set_ylabel('Z Position', fontsize=10)

    # Plot electrodes
    for ch, data in analysis['channel_data'].items():
        if data['has_position']:
            pos = data['position']
            color = '#27ae60' if data['matched'] else '#e67e22'
            marker = 'o' if data['matched'] else 's'
            size = 150

            ax2.scatter(pos[0], pos[2], c=color, marker=marker, s=size,
                       alpha=0.7, edgecolors='black', linewidths=2, zorder=3)

            # Add channel label
            ax2.text(pos[0], pos[2], ch, fontsize=7, ha='center', va='center',
                    fontweight='bold', color='white', zorder=4)

    # Add grid
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_aspect('equal')

    # Add axis at origin
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Set limits with padding
    z_range = np.ptp(positions[:, 2])
    z_pad = max(z_range * 0.15, x_pad) if z_range > 0 else x_pad
    ax2.set_xlim(positions[:, 0].min() - x_pad, positions[:, 0].max() + x_pad)
    ax2.set_ylim(positions[:, 2].min() - z_pad, positions[:, 2].max() + z_pad)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='Matched channels'),
        Patch(facecolor='#e67e22', edgecolor='black', label='Unmatched channels')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_stats_plot(analysis: Dict) -> str:
    """Create position distribution analysis plots - top and side views with auto-scaling.

    Args:
        analysis: Analysis results from analyze_channels()

    Returns:
        Base64-encoded PNG image
    """
    scale_type = analysis.get('scale_type', 'human')
    scale_factor = analysis.get('scale_factor', 1.0)

    # For mouse scale, use distance histogram instead
    if scale_type == 'mouse':
        return _create_mouse_stats_plot(analysis)

    # Human scale: standard distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    fig.suptitle('Position Distribution Analysis', fontsize=16, fontweight='bold')

    if len(analysis['positions']) > 0:
        positions = analysis['positions']
        distances = np.linalg.norm(positions, axis=1)

        # Plot 1: Top view (X-Y) colored by distance
        ax1.set_title('Top View (X-Y) - Colored by Distance from Origin', fontsize=13, fontweight='bold')
        x = positions[:, 0]
        y = positions[:, 1]
        scatter1 = ax1.scatter(x, y, c=distances, cmap='viridis', s=120,
                            alpha=0.7, edgecolors='black', linewidths=1.5)
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Distance (m)')

        # Add channel labels
        for ch, data in analysis['channel_data'].items():
            if data['has_position']:
                pos = data['position']
                angle = np.arctan2(pos[1], pos[0])
                offset_dist = 0.010
                text_x = pos[0] + offset_dist * np.cos(angle)
                text_y = pos[1] + offset_dist * np.sin(angle)

                ax1.text(text_x, text_y, ch, fontsize=8, ha='center', va='center',
                        fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                 edgecolor='none', alpha=0.85))

        ax1.set_xlabel('X (meters)', fontsize=11)
        ax1.set_ylabel('Y (meters)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')

        # Add head circle and nose
        circle1 = Circle((0, 0), 0.095, fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(circle1)
        nose1 = Wedge((0, 0.095), 0.015, 60, 120, facecolor='black', alpha=0.5)
        ax1.add_patch(nose1)

        ax1.set_xlim(-0.12, 0.12)
        ax1.set_ylim(-0.12, 0.12)

        # Plot 2: Side view (Y-Z) colored by distance
        ax2.set_title('Side View (Y-Z) - Colored by Distance from Origin', fontsize=13, fontweight='bold')
        y = positions[:, 1]
        z = positions[:, 2]
        scatter2 = ax2.scatter(y, z, c=distances, cmap='viridis', s=120,
                            alpha=0.7, edgecolors='black', linewidths=1.5)
        cbar2 = plt.colorbar(scatter2, ax=ax2, label='Distance (m)')

        # Add channel labels for side view
        for ch, data in analysis['channel_data'].items():
            if data['has_position']:
                pos = data['position']
                angle = np.arctan2(pos[2], pos[1])
                offset_dist = 0.010
                text_y = pos[1] + offset_dist * np.cos(angle)
                text_z = pos[2] + offset_dist * np.sin(angle)

                ax2.text(text_y, text_z, ch, fontsize=8, ha='center', va='center',
                        fontweight='bold', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                                 edgecolor='none', alpha=0.85))

        ax2.set_xlabel('Y (meters)', fontsize=11)
        ax2.set_ylabel('Z (meters)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        # Add head circle
        circle2 = Circle((0, 0), 0.095, fill=False, edgecolor='black', linewidth=2)
        ax2.add_patch(circle2)

        ax2.set_xlim(-0.12, 0.12)
        ax2.set_ylim(-0.12, 0.12)

    plt.tight_layout()
    return fig_to_base64(fig)


def _create_mouse_stats_plot(analysis: Dict) -> str:
    """Create statistics plots for mouse-scale probes.

    Args:
        analysis: Analysis results from analyze_channels()

    Returns:
        Base64-encoded PNG image
    """
    positions = analysis['positions']
    montage_name = analysis.get('montage_name', 'Mouse Probe')

    if len(positions) == 0:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.text(0.5, 0.5, 'No positioned channels', ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig_to_base64(fig)

    # Use normalized coordinates (MNE scales custom montages)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle(f'{montage_name} - Statistical Analysis', fontsize=14, fontweight='bold', color='#2c3e50')

    # Plot 1: Distance histogram
    ax1 = fig.add_subplot(gs[0, 0])
    distances = np.linalg.norm(positions, axis=1)
    ax1.hist(distances, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Distance from Origin (normalized)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distance Distribution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.4f}')
    ax1.legend()

    # Plot 2: X-coordinate distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(positions[:, 0], bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('X Position (normalized)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('X-Coordinate Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(positions[:, 0]), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(positions[:, 0]):.4f}')
    ax2.legend()

    # Plot 3: Y-coordinate distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(positions[:, 1], bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Y Position (normalized)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Y-Coordinate Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(np.mean(positions[:, 1]), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(positions[:, 1]):.4f}')
    ax3.legend()

    # Plot 4: Summary statistics table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    stats_data = [
        ['Statistic', 'X', 'Y', 'Z', 'Distance'],
        ['Mean', f'{np.mean(positions[:, 0]):.4f}', f'{np.mean(positions[:, 1]):.4f}',
         f'{np.mean(positions[:, 2]):.4f}', f'{np.mean(distances):.4f}'],
        ['Std Dev', f'{np.std(positions[:, 0]):.4f}', f'{np.std(positions[:, 1]):.4f}',
         f'{np.std(positions[:, 2]):.4f}', f'{np.std(distances):.4f}'],
        ['Min', f'{np.min(positions[:, 0]):.4f}', f'{np.min(positions[:, 1]):.4f}',
         f'{np.min(positions[:, 2]):.4f}', f'{np.min(distances):.4f}'],
        ['Max', f'{np.max(positions[:, 0]):.4f}', f'{np.max(positions[:, 1]):.4f}',
         f'{np.max(positions[:, 2]):.4f}', f'{np.max(distances):.4f}'],
        ['Range', f'{np.ptp(positions[:, 0]):.4f}', f'{np.ptp(positions[:, 1]):.4f}',
         f'{np.ptp(positions[:, 2]):.4f}', f'{np.ptp(distances):.4f}'],
    ]

    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 6):
        for j in range(5):
            if j == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
                table[(i, j)].set_text_props(weight='bold')
            else:
                table[(i, j)].set_facecolor('white')

    ax4.set_title('Statistical Summary (Normalized Coordinates)', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_html_report(
    eeg_file: str,
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
    """Generate enhanced HTML report optimized for EEG researchers.

    Args:
        eeg_file: Path to EEG file
        montage_name: Name of the montage used
        analysis: Analysis results from analyze_channels()
        suggestions: Alternative montage suggestions
        output_file: Path where HTML report will be saved
        topomap_b64: Base64-encoded 2D topomap image (optional, can be None)
        plot3d_b64: Base64-encoded 3D plot image
        stats_b64: Base64-encoded statistics plot image
        metadata: File metadata from extract_file_metadata()
        channel_info: Channel information from extract_channel_info()
        rename_map: Channel rename mapping
    """
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
    verdict_status = "PASS" if not issues else "FAIL" if analysis['match_pct'] < 90 or analysis['duplicates'] else "WARNING"

    # Calculate number of columns for channel table
    total_channels = len(channel_info)
    if total_channels <= 40:
        n_columns = 2
    elif total_channels <= 90:
        n_columns = 3
    else:
        n_columns = 4

    channels_per_column = math.ceil(total_channels / n_columns)

    # Import CSS from external file or embed directly
    # For now, embedding directly for portability
    from pathlib import Path as P

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EEG Channel Validation Report</title>
    <style>
        {_get_report_css(n_columns)}
    </style>
</head>
<body>
    {_generate_report_body(eeg_file, montage_name, timestamp, verdict_class, verdict_status, analysis, issues, recommendations, plot3d_b64, stats_b64, metadata, channel_info, channels_per_column, n_columns, rename_map, suggestions)}
</body>
</html>"""

    output_file.write_text(html)


def _get_report_css(n_columns: int) -> str:
    """Return CSS stylesheet for the HTML report."""
    # Read CSS from file if available, otherwise use embedded version
    try:
        css_file = Path(__file__).parent.parent.parent / "tmp" / "academic_css.txt"
        if css_file.exists():
            return css_file.read_text()
    except:
        pass

    # Embedded CSS (fallback)
    return f"""
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
            margin-bottom: 15px;
        }}

        .header .meta {{
            font-size: 14px;
            color: #333;
            font-weight: 600;
        }}

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

        .footer {{
            background: #f5f5f5;
            color: #666;
            text-align: center;
            padding: 20px;
            font-size: 11px;
            border-top: 2px solid #000;
        }}

        @media print {{
            body {{ padding: 0; background: white; }}
            .container {{ box-shadow: none; max-width: none; }}
            .section {{ page-break-inside: avoid; }}
        }}
    """


def _generate_report_body(eeg_file, montage_name, timestamp, verdict_class, verdict_status,
                           analysis, issues, recommendations, plot3d_b64, stats_b64, metadata,
                           channel_info, channels_per_column, n_columns, rename_map, suggestions):
    """Generate the HTML body content for the report."""
    # Due to size, this would typically be in a template file
    # For now, returning a simplified version
    # Full implementation would include all sections from test_bdf_visual_v5.py

    return f"""
    <div class="container">
        <div class="header">
            <h1>EEG Channel Validation Report</h1>
            <div class="subtitle">Electrode Position and Montage Analysis</div>
            <div class="meta">
                File: {Path(eeg_file).name} &nbsp;|&nbsp; Montage: {montage_name} &nbsp;|&nbsp; Generated: {timestamp}
            </div>
        </div>

        <div class="verdict {verdict_class}">
            <div class="verdict-title">{verdict_status}</div>
            <div class="verdict-details">
                Match: <strong>{analysis['match_pct']:.1f}%</strong> ({len(analysis['matched'])}/{len(analysis['file_channels'])}) &nbsp;|&nbsp;
                Positioned: <strong>{analysis['n_positioned']}/{len(analysis['file_channels'])}</strong> &nbsp;|&nbsp;
                Unmatched: <strong>{len(analysis['unmatched_file'])}</strong> &nbsp;|&nbsp;
                Outliers: <strong>{len(analysis['outliers'])}</strong> &nbsp;|&nbsp;
                Duplicates: <strong>{len(analysis['duplicates'])}</strong>
            </div>
        </div>

        {_generate_quality_issues_section(issues, analysis, recommendations) if issues else ''}

        <div class="section">
            <h2>3D Electrode Positions</h2>
            <p style="font-size: 12px; color: #666; margin-bottom: 15px;">
                Four-perspective view of electrode placement in 3D space.
                <strong>Green circles = matched channels</strong> | <strong>Orange squares = unmatched channels</strong>
            </p>
            <div class="image-container full">
                <img src="{plot3d_b64}" alt="3D Positions" style="max-width: 95%;">
                <div class="image-caption">3D Electrode Positions (Top, Back, Side, and Perspective Views)</div>
            </div>
        </div>

        <div class="section">
            <h2>Statistical Analysis</h2>
            <div class="image-container full">
                <img src="{stats_b64}" alt="Statistical Analysis" style="max-width: 95%;">
                <div class="image-caption">Position distribution analysis</div>
            </div>
        </div>

        {_generate_metadata_section(metadata, eeg_file, montage_name)}

        <div class="footer">
            <div style="font-weight: bold; margin-bottom: 5px;">EEG Channel Validation Report</div>
            <div>Generated by AutoClean EEG Pipeline</div>
        </div>
    </div>
    """


def _generate_quality_issues_section(issues, analysis, recommendations):
    """Generate the quality issues section HTML."""
    return f"""
        <div class="section">
            <h2>Quality Issues</h2>
            {f'<div class="error-box"><strong>CRITICAL:</strong> {len(analysis["duplicates"])} duplicate positions detected - likely WRONG MONTAGE selected!</div>' if analysis['duplicates'] else ''}
            {f'<div class="warning-box"><strong>Position outliers ({len(analysis["outliers"])}):</strong> {", ".join(analysis["outliers"])}</div>' if analysis['outliers'] else ''}

            <div class="two-col-grid" style="margin-top: 15px;">
                {f'<div class="warning-box" style="margin: 0;"><strong>Unmatched channels ({len(analysis["unmatched_file"])}):</strong><br>{", ".join(sorted(analysis["unmatched_file"]))}</div>' if analysis['unmatched_file'] else '<div></div>'}
                {f'<div class="info-box" style="margin: 0;"><strong>Recommendations:</strong><br>{"<br>".join(f"• {rec}" for rec in recommendations)}</div>' if recommendations else '<div></div>'}
            </div>
        </div>
    """


def _generate_metadata_section(metadata, eeg_file, montage_name):
    """Generate the metadata section HTML."""
    return f"""
        <div class="section">
            <h2>Recording Metadata</h2>
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
                        <div class="info-value">{Path(eeg_file).name}</div>
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
    """
