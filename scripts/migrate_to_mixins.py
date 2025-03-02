#!/usr/bin/env python3
"""
Script to migrate task files from using reports.py functions to using mixin methods.

This script will:
1. Find all task files that use reports.py functions
2. Update the imports to use mixins instead
3. Update method calls to use the mixin methods

Usage:
    python migrate_to_mixins.py
"""

import os
import re
from pathlib import Path

def update_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update imports
    content = re.sub(
        r'from autoclean\.step_functions\.reports import\s*\(\s*([^)]*)\s*\)',
        r'# Import the reporting functions directly from the Task class via mixins\n# from autoclean.step_functions.reports import (\n#     \1\n# )',
        content
    )
    
    # Update import on a single line
    content = re.sub(
        r'from autoclean\.step_functions\.reports import (.*)',
        r'# Import the reporting functions directly from the Task class via mixins\n# from autoclean.step_functions.reports import \1',
        content
    )
    
    # Update method calls
    content = content.replace('step_plot_raw_vs_cleaned_overlay(', 'self.plot_raw_vs_cleaned_overlay(')
    content = content.replace('step_plot_ica_full(', 'self.plot_ica_full(')
    content = content.replace('step_generate_ica_reports(', 'self.plot_ica_components(')
    content = content.replace('step_psd_topo_figure(', 'self.psd_topo_figure(')
    
    # Fix any generate_ica_reports calls to use the new method signature
    content = re.sub(
        r'self\.plot_ica_components\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*duration=([^)]+)\s*\)',
        r'self.plot_ica_components(\n            \1.ica2, \2, \3, \1, duration=\4\n        )',
        content
    )
    
    # Add comments about mixin methods
    content = content.replace(
        '# Plot raw vs cleaned overlay', 
        '# Plot raw vs cleaned overlay using mixin method'
    )
    content = content.replace(
        '# Plot ICA components', 
        '# Plot ICA components using mixin method'
    )
    content = content.replace(
        '# Generate ICA reports', 
        '# Generate ICA reports using mixin method'
    )
    content = content.replace(
        '# Create PSD topography figure', 
        '# Create PSD topography figure using mixin method'
    )
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path}")

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    tasks_dir = project_root / 'src' / 'autoclean' / 'tasks'
    
    # Find all task files
    for file_path in tasks_dir.glob('*.py'):
        if file_path.name == '__init__.py':
            continue
            
        # Check if the file uses reports.py
        with open(file_path, 'r') as f:
            content = f.read()
            
        if 'from autoclean.step_functions.reports import' in content:
            update_file(file_path)

if __name__ == "__main__":
    main()
