#!/usr/bin/env python3
"""
Test edge cases for BDF channel validation:
- No channel match (completely wrong montage)
- Missing channels (montage expects more than file has)
- Partial match (some channels match, many don't)
"""

from pathlib import Path
import mne
from test_bdf_visual_v5 import (
    extract_file_metadata,
    extract_channel_info,
    analyze_channels,
    suggest_montages,
    generate_html_report,
    create_3d_plot,
    create_stats_plot
)
from rich.console import Console
from rich.table import Table

console = Console()

def test_montage(bdf_file: str, montage_name: str, output_dir: Path):
    """Test a single montage against the BDF file."""

    console.print(f"\n[bold cyan]Testing: {montage_name}[/bold cyan]")

    # Load data
    raw = mne.io.read_raw_bdf(bdf_file, preload=True, stim_channel="auto", exclude=[], verbose=False)

    # Determine rename map first
    rename_map = {}
    for ch in raw.ch_names:
        if ch != "Status" and "_" in ch:
            _, std_name = ch.split("_", 1)
            if std_name == "Afz":
                std_name = "AFz"
            rename_map[ch] = std_name

    # Get montage and montage channels
    montage = mne.channels.make_standard_montage(montage_name)
    montage_channels = set(montage.get_positions()['ch_pos'].keys())

    # Extract RAW channel info BEFORE any renaming (for transparency)
    raw_channel_info = extract_channel_info(raw, rename_map, montage_channels)

    # Extract metadata
    metadata = extract_file_metadata(raw, bdf_file)

    # Now apply renaming and montage
    if rename_map:
        raw.rename_channels(rename_map)

    raw.set_montage(montage, match_case=False, on_missing="warn")

    # Analyze channels
    analysis = analyze_channels(raw, montage)

    # Get suggestions
    file_channels = {raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)}
    suggestions = suggest_montages(file_channels, top_n=5)

    # Print summary
    table = Table(title=f"Validation Results: {montage_name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Match %", f"{analysis['match_pct']:.1f}%")
    table.add_row("Matched Channels", f"{len(analysis['matched'])}/{len(analysis['file_channels'])}")
    table.add_row("Unmatched (File)", str(len(analysis['unmatched_file'])))
    table.add_row("Unmatched (Montage)", str(len(analysis['unmatched_montage'])))
    table.add_row("Positioned", str(analysis['n_positioned']))
    table.add_row("Duplicates", str(len(analysis['duplicates'])))
    table.add_row("Outliers", str(len(analysis['outliers'])))

    console.print(table)

    if len(analysis['unmatched_file']) > 0:
        console.print(f"[yellow]Unmatched in file:[/yellow] {', '.join(sorted(list(analysis['unmatched_file']))[:10])}")

    # Generate visualizations
    plot3d_b64 = create_3d_plot(analysis, f"3D Electrode Positions - {montage_name}")
    stats_b64 = create_stats_plot(analysis)

    # Generate HTML report
    html_file = output_dir / f"{Path(bdf_file).stem}_validation_{montage_name.replace('-', '_')}.html"
    generate_html_report(bdf_file, montage_name, analysis, suggestions,
                        html_file, None, plot3d_b64, stats_b64,
                        metadata, raw_channel_info, rename_map)

    console.print(f"[green]✓ Report generated: {html_file.name}[/green]")

    return analysis

def main():
    # Configuration
    bdf_file = "/Users/ernie/Downloads/Example EEGs/st101as.bdf"
    output_dir = Path(".")

    # Test cases
    test_cases = [
        ("biosemi64", "Correct montage (baseline)"),
        ("biosemi16", "Too few channels - many unmatched in file"),
        ("biosemi256", "Too many channels - many unmatched in montage"),
        ("easycap-M1", "Completely different system (EasyCap)"),
        ("GSN-HydroCel-128", "Completely different system (EGI)"),
        ("standard_1020", "Generic 10-20 system"),
    ]

    console.print("[bold green]Starting Edge Case Tests[/bold green]")
    console.print(f"BDF File: {bdf_file}")
    console.print(f"Output Directory: {output_dir}")

    results = {}

    for montage_name, description in test_cases:
        try:
            console.print(f"\n[bold]Test: {description}[/bold]")
            analysis = test_montage(bdf_file, montage_name, output_dir)
            results[montage_name] = {
                'match_pct': analysis['match_pct'],
                'matched': len(analysis['matched']),
                'unmatched_file': len(analysis['unmatched_file']),
                'description': description
            }
        except Exception as e:
            console.print(f"[red]✗ Error testing {montage_name}: {e}[/red]")
            results[montage_name] = {'error': str(e), 'description': description}

    # Summary table
    console.print("\n[bold cyan]═══ SUMMARY OF ALL TESTS ═══[/bold cyan]")
    summary_table = Table(title="Edge Case Test Results")
    summary_table.add_column("Montage", style="cyan")
    summary_table.add_column("Description", style="white")
    summary_table.add_column("Match %", style="magenta")
    summary_table.add_column("Matched", style="green")
    summary_table.add_column("Unmatched", style="yellow")

    for montage_name, result in results.items():
        if 'error' in result:
            summary_table.add_row(
                montage_name,
                result['description'],
                "[red]ERROR[/red]",
                "-",
                "-"
            )
        else:
            summary_table.add_row(
                montage_name,
                result['description'],
                f"{result['match_pct']:.1f}%",
                str(result['matched']),
                str(result['unmatched_file'])
            )

    console.print(summary_table)
    console.print("\n[bold green]✓ All edge case tests completed![/bold green]")

if __name__ == "__main__":
    main()
