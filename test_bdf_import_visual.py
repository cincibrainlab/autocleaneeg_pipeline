#!/usr/bin/env python3
"""
BDF Import Visual Diagnostic Report

Creates comprehensive visual validation of channel positions and montage matching.
Generates plots, tables, and analysis to verify correct channel mapping.

Run with: uv run test_bdf_import_visual.py [bdf_file] [montage_name]
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter

import numpy as np

try:
    import mne
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import Axes3D
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError as e:
    print(f"ERROR: Required package not installed: {e}")
    print("Install with: uv pip install mne matplotlib rich")
    sys.exit(1)


console = Console()


def analyze_channel_matching(
    raw: mne.io.Raw,
    montage: mne.channels.DigMontage,
    montage_name: str
) -> Dict:
    """Analyze how well channels match the montage."""

    file_channels = set(ch for ch in raw.ch_names if ch != 'Status')
    montage_pos = montage.get_positions()
    montage_channels = set(montage_pos['ch_pos'].keys())

    # Analyze matching
    matched = file_channels & montage_channels
    in_file_not_montage = file_channels - montage_channels
    in_montage_not_file = montage_channels - file_channels

    # Get position status for each channel
    channel_status = {}
    for ch_name in raw.ch_names:
        if ch_name == 'Status':
            continue

        ch_idx = raw.ch_names.index(ch_name)
        ch_type = raw.info['chs'][ch_idx]['kind']

        if ch_type == mne.io.constants.FIFF.FIFFV_EEG_CH:
            loc = raw.info['chs'][ch_idx]['loc'][:3]
            has_position = not np.any(np.isnan(loc))

            channel_status[ch_name] = {
                'in_file': True,
                'in_montage': ch_name in montage_channels,
                'has_position': has_position,
                'position': loc if has_position else None,
                'matched': ch_name in matched
            }

    return {
        'file_channels': file_channels,
        'montage_channels': montage_channels,
        'matched': matched,
        'in_file_not_montage': in_file_not_montage,
        'in_montage_not_file': in_montage_not_file,
        'channel_status': channel_status,
        'match_percentage': len(matched) / len(file_channels) * 100 if file_channels else 0
    }


def check_position_quality(raw: mne.io.Raw, channel_status: Dict) -> Dict:
    """Check quality of channel positions."""

    positions = []
    position_dict = {}

    for ch_name, status in channel_status.items():
        if status['has_position']:
            pos = status['position']
            positions.append(pos)
            # Round to avoid floating point comparison issues
            pos_tuple = tuple(np.round(pos, 6))
            if pos_tuple not in position_dict:
                position_dict[pos_tuple] = []
            position_dict[pos_tuple].append(ch_name)

    positions = np.array(positions)

    # Find duplicates
    duplicates = {pos: channels for pos, channels in position_dict.items() if len(channels) > 1}

    # Check position bounds (reasonable head size: ~0.1m radius)
    if len(positions) > 0:
        distances = np.linalg.norm(positions, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        outliers = [
            ch_name for ch_name, status in channel_status.items()
            if status['has_position'] and
            np.linalg.norm(status['position']) > mean_distance + 3 * std_distance
        ]
    else:
        mean_distance = 0
        std_distance = 0
        outliers = []

    # Check for unreasonable coordinates
    unreasonable = []
    if len(positions) > 0:
        for ch_name, status in channel_status.items():
            if status['has_position']:
                pos = status['position']
                # Check if any coordinate is way off
                if np.any(np.abs(pos) > 0.2):  # More than 20cm from origin
                    unreasonable.append(ch_name)

    return {
        'n_positions': len(positions),
        'duplicates': duplicates,
        'outliers': outliers,
        'unreasonable': unreasonable,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'positions_array': positions
    }


def suggest_alternative_montages(file_channels: Set[str]) -> List[Tuple[str, int, float]]:
    """Suggest alternative montages based on channel names."""

    # Common montages to check
    test_montages = [
        'biosemi16', 'biosemi32', 'biosemi64', 'biosemi128', 'biosemi256',
        'standard_1005', 'standard_1020',
        'GSN-HydroCel-129', 'GSN-HydroCel-128', 'GSN-HydroCel-65', 'GSN-HydroCel-64',
        'GSN-HydroCel-32', 'GSN-HydroCel-124', 'GSN-HydroCel-256', 'GSN-HydroCel-257',
    ]

    suggestions = []
    for montage_name in test_montages:
        try:
            montage = mne.channels.make_standard_montage(montage_name)
            montage_channels = set(montage.get_positions()['ch_pos'].keys())
            matched = file_channels & montage_channels
            match_pct = len(matched) / len(file_channels) * 100 if file_channels else 0

            suggestions.append((montage_name, len(matched), match_pct))
        except Exception:
            continue

    # Sort by match percentage
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:5]  # Top 5


def plot_topomap(raw: mne.io.Raw, channel_status: Dict, output_file: str):
    """Create 2D topomap of electrode positions."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: All positions (X-Y plane, top view)
    ax1.set_title('Channel Positions - Top View (X-Y)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Draw head circle
    head_circle = Circle((0, 0), 0.095, fill=False, edgecolor='black', linewidth=2)
    ax1.add_patch(head_circle)

    # Plot channels
    for ch_name, status in channel_status.items():
        if status['has_position']:
            pos = status['position']
            if status['matched']:
                color = 'green'
                marker = 'o'
                size = 100
                label = 'Matched' if 'Matched' not in ax1.get_legend_handles_labels()[1] else ''
            else:
                color = 'orange'
                marker = 's'
                size = 80
                label = 'Not in montage' if 'Not in montage' not in ax1.get_legend_handles_labels()[1] else ''

            ax1.scatter(pos[0], pos[1], c=color, marker=marker, s=size,
                       alpha=0.7, edgecolors='black', linewidths=1, label=label)
            ax1.text(pos[0], pos[1], ch_name, fontsize=7, ha='center', va='center')
        else:
            # Channel without position - plot as red X
            ax1.scatter(0, 0, c='red', marker='x', s=100, label='No position' if 'No position' not in ax1.get_legend_handles_labels()[1] else '')

    ax1.legend(loc='upper right')

    # Plot 2: Side view (Y-Z plane)
    ax2.set_title('Channel Positions - Side View (Y-Z)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Y (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Draw head circle
    head_circle2 = Circle((0, 0), 0.095, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(head_circle2)

    for ch_name, status in channel_status.items():
        if status['has_position']:
            pos = status['position']
            color = 'green' if status['matched'] else 'orange'
            marker = 'o' if status['matched'] else 's'
            size = 100 if status['matched'] else 80

            ax2.scatter(pos[1], pos[2], c=color, marker=marker, s=size,
                       alpha=0.7, edgecolors='black', linewidths=1)
            ax2.text(pos[1], pos[2], ch_name, fontsize=7, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    console.print(f"\n✅ Saved topomap to: [accent]{output_file}[/accent]")
    plt.close()


def plot_3d_positions(raw: mne.io.Raw, channel_status: Dict, quality: Dict, output_file: str):
    """Create 3D scatter plot of electrode positions."""

    fig = plt.figure(figsize=(16, 12))

    # Create 4 subplots for different views
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    axes = [ax1, ax2, ax3, ax4]
    views = [
        (90, 90, 'Top View'),      # Top down
        (0, 90, 'Back View'),      # Back
        (0, 0, 'Side View'),       # Side
        (45, 45, '3D Perspective') # Perspective
    ]

    for ax, (elev, azim, title) in zip(axes, views):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Plot channels
        matched_pos = []
        unmatched_pos = []
        nan_pos = []

        for ch_name, status in channel_status.items():
            if status['has_position']:
                pos = status['position']
                if status['matched']:
                    matched_pos.append(pos)
                    ax.scatter(*pos, c='green', marker='o', s=50, alpha=0.7)
                else:
                    unmatched_pos.append(pos)
                    ax.scatter(*pos, c='orange', marker='s', s=40, alpha=0.7)

        # Draw head sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = 0.095 * np.outer(np.cos(u), np.sin(v))
        y = 0.095 * np.outer(np.sin(u), np.sin(v))
        z = 0.095 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.1, color='gray')

        # Set view
        ax.view_init(elev=elev, azim=azim)

        # Set equal aspect ratio
        max_range = 0.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Matched channels'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Unmatched channels'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    console.print(f"✅ Saved 3D plot to: [accent]{output_file}[/accent]")
    plt.close()


def generate_visual_report(
    bdf_file: str,
    montage_name: str = "biosemi64",
    output_dir: str = "."
):
    """Generate comprehensive visual validation report."""

    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]    BDF Import Visual Validation Report[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")

    bdf_path = Path(bdf_file)
    if not bdf_path.exists():
        console.print(f"[error]ERROR: File not found: {bdf_file}[/error]")
        sys.exit(1)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load BDF file
    console.print(f"[info]Loading BDF file:[/info] {bdf_path.name}")
    raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, stim_channel="auto", exclude=[], verbose=False)

    # Rename channels (from plugins)
    rename_mapping = {}
    for ch_name in raw.ch_names:
        if ch_name == "Status":
            continue
        if "_" in ch_name:
            prefix, standard_name = ch_name.split("_", 1)
            if standard_name == "Afz":
                standard_name = "AFz"
            rename_mapping[ch_name] = standard_name

    if rename_mapping:
        raw.rename_channels(rename_mapping)
        console.print(f"[success]✓[/success] Renamed {len(rename_mapping)} channels")

    # Load montage
    console.print(f"[info]Loading montage:[/info] {montage_name}")
    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, match_case=False, on_missing="warn")
        console.print(f"[success]✓[/success] Applied montage\n")
    except Exception as e:
        console.print(f"[error]ERROR: Failed to load montage: {e}[/error]")
        sys.exit(1)

    # Analyze matching
    analysis = analyze_channel_matching(raw, montage, montage_name)
    quality = check_position_quality(raw, analysis['channel_status'])

    # ===== SECTION 1: OVERALL SUMMARY =====
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  SECTION 1: OVERALL SUMMARY[/bold]")
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("File", bdf_path.name)
    summary_table.add_row("Montage", montage_name)
    summary_table.add_row("Channels in file", str(len(analysis['file_channels'])))
    summary_table.add_row("Channels in montage", str(len(analysis['montage_channels'])))
    summary_table.add_row("Matched channels", f"[green]{len(analysis['matched'])}[/green]")
    summary_table.add_row("Match percentage", f"[{'green' if analysis['match_percentage'] > 90 else 'yellow' if analysis['match_percentage'] > 50 else 'red'}]{analysis['match_percentage']:.1f}%[/]")
    summary_table.add_row("Channels with positions", f"[green]{quality['n_positions']}[/green]")
    summary_table.add_row("Duplicate positions", f"[{'red' if len(quality['duplicates']) > 0 else 'green'}]{len(quality['duplicates'])}[/]")
    summary_table.add_row("Position outliers", f"[{'yellow' if len(quality['outliers']) > 0 else 'green'}]{len(quality['outliers'])}[/]")

    console.print(summary_table)
    console.print()

    # ===== SECTION 2: CHANNEL MATCHING DETAILS =====
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  SECTION 2: CHANNEL MATCHING ANALYSIS[/bold]")
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

    # Matched channels table
    if analysis['matched']:
        console.print(f"[green]✓ {len(analysis['matched'])} channels matched successfully[/green]")
        if len(analysis['matched']) <= 20:
            console.print(f"  {', '.join(sorted(analysis['matched']))}")
        else:
            console.print(f"  {', '.join(sorted(list(analysis['matched']))[:20])}...")

    # Channels in file but not montage
    if analysis['in_file_not_montage']:
        console.print(f"\n[yellow]⚠ {len(analysis['in_file_not_montage'])} channels in file but NOT in montage:[/yellow]")
        for ch in sorted(analysis['in_file_not_montage']):
            console.print(f"  • {ch}")

    # Channels in montage but not file
    if analysis['in_montage_not_file']:
        console.print(f"\n[blue]ℹ {len(analysis['in_montage_not_file'])} channels in montage but NOT in file:[/blue]")
        if len(analysis['in_montage_not_file']) <= 15:
            console.print(f"  {', '.join(sorted(analysis['in_montage_not_file']))}")
        else:
            console.print(f"  {', '.join(sorted(list(analysis['in_montage_not_file']))[:15])}...")

    console.print()

    # ===== SECTION 3: POSITION QUALITY =====
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  SECTION 3: POSITION QUALITY CHECKS[/bold]")
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

    if quality['duplicates']:
        console.print(f"[error]✗ DUPLICATE POSITIONS DETECTED ({len(quality['duplicates'])} locations):[/error]")
        console.print("[yellow]  This usually indicates WRONG MONTAGE selected![/yellow]")
        for pos, channels in list(quality['duplicates'].items())[:5]:
            console.print(f"  • Position {pos}: {', '.join(channels)}")
    else:
        console.print("[green]✓ No duplicate positions (good!)[/green]")

    if quality['outliers']:
        console.print(f"\n[yellow]⚠ OUTLIER POSITIONS ({len(quality['outliers'])} channels):[/yellow]")
        for ch in quality['outliers']:
            console.print(f"  • {ch}")
    else:
        console.print("\n[green]✓ No position outliers[/green]")

    if quality['unreasonable']:
        console.print(f"\n[error]✗ UNREASONABLE COORDINATES ({len(quality['unreasonable'])} channels):[/error]")
        console.print("[yellow]  Positions >20cm from origin (head too large)[/yellow]")
        for ch in quality['unreasonable']:
            console.print(f"  • {ch}")
    else:
        console.print("\n[green]✓ All positions within reasonable bounds[/green]")

    if quality['positions_array'].size > 0:
        console.print(f"\n[info]Position statistics:[/info]")
        console.print(f"  Mean distance from origin: {quality['mean_distance']:.4f}m")
        console.print(f"  Std deviation: {quality['std_distance']:.4f}m")

    console.print()

    # ===== SECTION 4: ALTERNATIVE MONTAGE SUGGESTIONS =====
    if analysis['match_percentage'] < 95:
        console.print("[bold]══════════════════════════════════════════════════════════════[/bold]")
        console.print("[bold]  SECTION 4: ALTERNATIVE MONTAGE SUGGESTIONS[/bold]")
        console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

        console.print("[yellow]Current montage match is below 95%. Checking alternatives...[/yellow]\n")

        suggestions = suggest_alternative_montages(analysis['file_channels'])

        if suggestions:
            suggestion_table = Table(title="Top 5 Alternative Montages", show_header=True)
            suggestion_table.add_column("Rank", style="cyan", width=6)
            suggestion_table.add_column("Montage Name", style="white", width=25)
            suggestion_table.add_column("Matched", style="green", width=10)
            suggestion_table.add_column("Match %", style="yellow", width=10)
            suggestion_table.add_column("Recommendation", style="blue")

            for i, (name, matched, pct) in enumerate(suggestions, 1):
                if pct > analysis['match_percentage']:
                    rec = "✓ Try this"
                    color = "green"
                else:
                    rec = ""
                    color = "white"

                suggestion_table.add_row(
                    str(i),
                    name,
                    str(matched),
                    f"{pct:.1f}%",
                    f"[{color}]{rec}[/{color}]"
                )

            console.print(suggestion_table)
            console.print()

    # ===== SECTION 5: DETAILED CHANNEL TABLE =====
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  SECTION 5: DETAILED CHANNEL STATUS[/bold]")
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

    channel_table = Table(show_header=True, show_lines=True)
    channel_table.add_column("Channel", style="cyan", width=12)
    channel_table.add_column("In File", width=8)
    channel_table.add_column("In Montage", width=11)
    channel_table.add_column("Has Position", width=12)
    channel_table.add_column("Position (X, Y, Z)", width=30)
    channel_table.add_column("Status", width=15)

    # Show first 20 channels
    for ch_name in sorted(analysis['channel_status'].keys())[:20]:
        status = analysis['channel_status'][ch_name]

        in_file = "[green]✓[/green]" if status['in_file'] else "[red]✗[/red]"
        in_montage = "[green]✓[/green]" if status['in_montage'] else "[yellow]✗[/yellow]"
        has_pos = "[green]✓[/green]" if status['has_position'] else "[red]✗[/red]"

        if status['has_position']:
            pos = status['position']
            pos_str = f"{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}"
        else:
            pos_str = "[red]NaN[/red]"

        if status['matched'] and status['has_position']:
            status_str = "[green]✓ Good[/green]"
        elif not status['in_montage']:
            status_str = "[yellow]Not in montage[/yellow]"
        elif not status['has_position']:
            status_str = "[red]No position[/red]"
        else:
            status_str = "[yellow]Check[/yellow]"

        channel_table.add_row(ch_name, in_file, in_montage, has_pos, pos_str, status_str)

    if len(analysis['channel_status']) > 20:
        channel_table.add_row(
            f"... +{len(analysis['channel_status']) - 20} more",
            "", "", "", "", ""
        )

    console.print(channel_table)
    console.print()

    # ===== SECTION 6: VISUAL PLOTS =====
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  SECTION 6: GENERATING VISUAL PLOTS[/bold]")
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

    # Generate plots
    topomap_file = output_dir / f"{bdf_path.stem}_topomap.png"
    plot3d_file = output_dir / f"{bdf_path.stem}_3d_positions.png"

    console.print("[info]Generating 2D topomap...[/info]")
    plot_topomap(raw, analysis['channel_status'], str(topomap_file))

    console.print("[info]Generating 3D position plot...[/info]")
    plot_3d_positions(raw, analysis['channel_status'], quality, str(plot3d_file))

    # ===== FINAL VERDICT =====
    console.print("\n[bold]══════════════════════════════════════════════════════════════[/bold]")
    console.print("[bold]  FINAL VERDICT[/bold]")
    console.print("[bold]══════════════════════════════════════════════════════════════[/bold]\n")

    issues = []
    if analysis['match_percentage'] < 90:
        issues.append(f"Low match percentage ({analysis['match_percentage']:.1f}%)")
    if quality['duplicates']:
        issues.append(f"{len(quality['duplicates'])} duplicate position(s) - WRONG MONTAGE?")
    if quality['unreasonable']:
        issues.append(f"{len(quality['unreasonable'])} unreasonable position(s)")
    if len(analysis['in_file_not_montage']) > len(analysis['matched']) * 0.2:
        issues.append(f"Many unmatched channels ({len(analysis['in_file_not_montage'])})")

    if not issues:
        console.print(Panel(
            "[bold green]✓ PASS: Channel positions look correct![/bold green]\n\n"
            f"• {len(analysis['matched'])} channels matched successfully\n"
            f"• Match percentage: {analysis['match_percentage']:.1f}%\n"
            f"• No duplicate or unreasonable positions\n"
            "• Ready for pipeline processing",
            title="[bold green]SUCCESS[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[bold yellow]⚠ WARNING: Potential issues detected![/bold yellow]\n\n"
            "Issues found:\n" + "\n".join(f"• {issue}" for issue in issues) + "\n\n"
            "[bold]Recommendations:[/bold]\n"
            "• Review the alternative montage suggestions above\n"
            "• Check the visual plots for position anomalies\n"
            "• Verify you selected the correct montage for this file",
            title="[bold yellow]NEEDS REVIEW[/bold yellow]",
            border_style="yellow"
        ))

    console.print(f"\n[info]Report complete! Visual plots saved to:[/info] {output_dir}\n")


if __name__ == "__main__":
    bdf_file = "/Users/ernie/Downloads/Example EEGs/st101as.bdf"
    montage_name = "biosemi64"
    output_dir = "."

    if len(sys.argv) > 1:
        bdf_file = sys.argv[1]
    if len(sys.argv) > 2:
        montage_name = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]

    try:
        generate_visual_report(bdf_file, montage_name, output_dir)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n\n[error]❌ Error: {e}[/error]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
