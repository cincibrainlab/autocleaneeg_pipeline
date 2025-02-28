#!/usr/bin/env python
"""
ASSR Runner Script - Runs ASSR analysis and visualization

This script serves as the main entry point for the ASSR analysis package.
It handles command line arguments and runs the appropriate functions.
"""

import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# Make sure the current directory is in the path so modules can be imported
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the analysis and visualization modules
from assr_analysis import analyze_assr, load_epochs, compute_time_frequency, compute_metrics
from assr_viz import plot_all_figures, plot_global_mean_itc, plot_topomap


def run_complete_analysis(file_path, output_dir=None):
    """Run the complete analysis with all plots"""
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path('results')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running complete analysis on {file_path}")
    print(f"Saving results to {output_dir}")
    
    # Run the analysis
    analysis_results = analyze_assr(file_path, output_dir, save_results=True)
    
    # Generate all plots
    figures = plot_all_figures(
        analysis_results['tf_data'], 
        analysis_results['epochs'],
        output_dir, 
        save_figures=True
    )
    
    print("Analysis and visualization complete!")
    return analysis_results, figures


def run_analysis_only(file_path, output_dir=None):
    """Run just the analysis without generating plots"""
    if output_dir is None:
        output_dir = Path('results')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running analysis only on {file_path}")
    analysis_results = analyze_assr(file_path, output_dir, save_results=True)
    
    print("Analysis complete!")
    return analysis_results


def create_specific_plots(analysis_results, output_dir=None):
    """Create only specific plots from existing analysis results"""
    if output_dir is None:
        output_dir = Path('results')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating selected plots")
    
    # Plot just the global mean ITC
    global_itc_fig = plot_global_mean_itc(
        analysis_results['tf_data'], 
        output_dir, 
        save_figures=True
    )
    
    # Plot just the topographical map
    topo_fig = plot_topomap(
        analysis_results['tf_data'], 
        analysis_results['epochs'],
        time_point=0.3,  # Customize time point
        output_dir=output_dir, 
        save_figures=True
    )
    
    print("Selected plots created!")
    return {'global_itc': global_itc_fig, 'topomap': topo_fig}


def main():
    """Main function to parse arguments and run analysis"""
    parser = argparse.ArgumentParser(description='ASSR Analysis Runner')
    parser.add_argument('file_path', type=str, help='Path to the EEGLAB .set file')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results and figures')
    parser.add_argument('--analysis_type', type=str, choices=['complete', 'analysis_only', 'plots_only'], 
                        default='complete', help='Type of analysis to run')
    
    args = parser.parse_args()
    
    if args.analysis_type == 'complete':
        run_complete_analysis(args.file_path, args.output_dir)
    
    elif args.analysis_type == 'analysis_only':
        run_analysis_only(args.file_path, args.output_dir)
    
    elif args.analysis_type == 'plots_only':
        # First run the analysis to get the results
        analysis_results = analyze_assr(args.file_path, args.output_dir, save_results=False)
        # Then create only specific plots
        create_specific_plots(analysis_results, args.output_dir)
    
    print(f"Completed {args.analysis_type} analysis on {args.file_path}")


if __name__ == "__main__":
    main()