#!/usr/bin/env python
"""
Example: Batch Processing of EEG Files

This example demonstrates how to process multiple EEG files in batch using
the AutoClean pipeline. It includes parallel processing, error handling,
and progress tracking.
"""

import sys
import os
import time
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autoclean.core.pipeline import Pipeline
from autoclean.tasks.resting_eyes_open import RestingEyesOpenTask
from autoclean.utils.file_system import find_eeg_files
from autoclean.utils.logging import setup_logger


def process_single_file(file_path, output_dir, task_name="resting_eyes_open"):
    """Process a single EEG file using the specified task.
    
    Parameters
    ----------
    file_path : str
        Path to the EEG file
    output_dir : str
        Path to the output directory
    task_name : str, default="resting_eyes_open"
        Name of the task to use for processing
    
    Returns
    -------
    dict
        Result dictionary containing:
        - status: 'success' or 'error'
        - message: Description of result or error
        - metadata: Pipeline metadata if successful
        - processing_time: Time taken to process the file in seconds
    """
    logger = setup_logger(f"autoclean.{Path(file_path).stem}")
    logger.info(f"Processing file: {file_path}")
    start_time = time.time()
    
    try:
        # Create the pipeline
        pipeline = Pipeline(verbose=True)
        
        # Create the output directory for this file
        file_output_dir = Path(output_dir) / Path(file_path).stem
        file_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up the task
        if task_name == "resting_eyes_open":
            task = RestingEyesOpenTask()
        else:
            # Import dynamically based on task_name
            task_module = __import__(f"autoclean.tasks.{task_name}", fromlist=[""])
            task_class = getattr(task_module, task_name.split('_')[-1].capitalize() + "Task")
            task = task_class()
        
        # Load the raw data
        pipeline.raw = pipeline.load_raw(file_path)
        
        # Run the task
        pipeline = task.run(pipeline)
        
        # Save results
        if hasattr(pipeline, 'epochs') and pipeline.epochs is not None:
            epochs_file = file_output_dir / f"{Path(file_path).stem}_epochs.fif"
            pipeline.epochs.save(epochs_file, overwrite=True)
            logger.info(f"Saved epochs to {epochs_file}")
        
        if hasattr(pipeline, 'psd') and pipeline.psd is not None:
            # Save PSD as CSV
            psd_file = file_output_dir / f"{Path(file_path).stem}_psd.csv"
            if isinstance(pipeline.psd, tuple) and len(pipeline.psd) >= 2:
                psd_data, freqs = pipeline.psd
                psd_df = pd.DataFrame(psd_data, columns=freqs)
                psd_df.index.name = 'channel'
                psd_df.to_csv(psd_file)
                logger.info(f"Saved PSD data to {psd_file}")
            
            # Create and save PSD plot
            psd_plot_file = file_output_dir / f"{Path(file_path).stem}_psd.png"
            if isinstance(pipeline.psd, tuple) and len(pipeline.psd) >= 2:
                psd_data, freqs = pipeline.psd
                plt.figure(figsize=(10, 6))
                plt.semilogy(freqs, psd_data.T)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power Spectral Density (µV²/Hz)')
                plt.title(f'Power Spectrum - {Path(file_path).stem}')
                plt.savefig(psd_plot_file)
                plt.close()
                logger.info(f"Saved PSD plot to {psd_plot_file}")
        
        if hasattr(pipeline, 'ica') and pipeline.ica is not None:
            # Save ICA components plot
            ica_plot_file = file_output_dir / f"{Path(file_path).stem}_ica_components.png"
            fig = pipeline.ica.plot_components(show=False)
            plt.savefig(ica_plot_file)
            plt.close()
            logger.info(f"Saved ICA components plot to {ica_plot_file}")
        
        # Generate a comprehensive report
        from autoclean.step_functions.reports import generate_report
        report_file = file_output_dir / f"{Path(file_path).stem}_report.html"
        generate_report(
            pipeline, 
            report_type='html',
            file_path=report_file,
            include_psd=True,
            include_ica=True,
            include_evoked=hasattr(pipeline, 'evoked')
        )
        logger.info(f"Generated report at {report_file}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'status': 'success',
            'message': f"Successfully processed {file_path}",
            'metadata': pipeline.metadata,
            'processing_time': processing_time,
            'output_dir': str(file_output_dir)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing {file_path}: {str(e)}\n{error_details}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            'status': 'error',
            'message': str(e),
            'error_details': error_details,
            'processing_time': processing_time
        }


async def process_files_parallel(file_paths, output_dir, max_workers=4, task_name="resting_eyes_open"):
    """Process multiple EEG files in parallel.
    
    Parameters
    ----------
    file_paths : list
        List of paths to EEG files
    output_dir : str
        Path to the output directory
    max_workers : int, default=4
        Maximum number of parallel workers
    task_name : str, default="resting_eyes_open"
        Name of the task to use for processing
    
    Returns
    -------
    list
        List of result dictionaries
    """
    loop = asyncio.get_event_loop()
    results = []
    
    # Create a process pool executor for parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create tasks for each file
        future_to_file = {
            loop.run_in_executor(
                executor, process_single_file, file_path, output_dir, task_name
            ): file_path for file_path in file_paths
        }
        
        # Process futures as they complete
        for i, future in enumerate(asyncio.as_completed(future_to_file)):
            file_path = future_to_file[future]
            try:
                result = await future
                results.append(result)
                
                # Print progress
                print(f"Progress: {i+1}/{len(file_paths)} files processed - {file_path}")
                
                if result['status'] == 'success':
                    print(f"  ✓ Success: {result['message']}")
                    print(f"  ⏱ Processing time: {result['processing_time']:.2f} seconds")
                    print(f"  📁 Output directory: {result['output_dir']}")
                else:
                    print(f"  ✗ Error: {result['message']}")
                    print(f"  ⏱ Processing time: {result['processing_time']:.2f} seconds")
                
            except Exception as e:
                print(f"Error with {file_path}: {str(e)}")
                results.append({
                    'status': 'error',
                    'file_path': file_path,
                    'message': str(e)
                })
    
    return results


def summarize_results(results, output_dir):
    """Create a summary of processing results.
    
    Parameters
    ----------
    results : list
        List of result dictionaries
    output_dir : str
        Path to the output directory
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the summary
    """
    summary_data = []
    for result in results:
        file_path = result.get('file_path', '')
        if not file_path and 'message' in result:
            # Extract file path from message
            msg_parts = result['message'].split()
            if len(msg_parts) > 1 and 'processed' in result['message']:
                file_path = msg_parts[-1]
        
        data = {
            'file_name': Path(file_path).name if file_path else 'Unknown',
            'status': result.get('status', 'unknown'),
            'processing_time': result.get('processing_time', 0),
        }
        
        # Add metadata if available
        if 'metadata' in result:
            metadata = result['metadata']
            data.update({
                'n_channels': metadata.get('n_channels', 0),
                'sampling_rate': metadata.get('sampling_rate', 0),
                'recording_duration': metadata.get('recording_duration', 0),
                'bad_channels': ', '.join(metadata.get('bad_channels', [])),
                'ica_components_removed': ', '.join(map(str, metadata.get('ica_components_removed', []))),
                'n_epochs': metadata.get('n_epochs', 0),
                'n_epochs_dropped': metadata.get('n_epochs_dropped', 0),
            })
        
        summary_data.append(data)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_file = Path(output_dir) / 'processing_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")
    
    # Create visual summary
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Processing time by file
    if 'processing_time' in summary_df.columns:
        summary_df_sorted = summary_df.sort_values('processing_time', ascending=False)
        bar = axs[0].bar(
            summary_df_sorted['file_name'], 
            summary_df_sorted['processing_time'],
            color=['green' if status == 'success' else 'red' for status in summary_df_sorted['status']]
        )
        axs[0].set_xlabel('File')
        axs[0].set_ylabel('Processing Time (seconds)')
        axs[0].set_title('Processing Time by File')
        axs[0].tick_params(axis='x', rotation=90)
        
        # Add success/error labels
        for i, rect in enumerate(bar):
            status = summary_df_sorted.iloc[i]['status']
            label = '✓' if status == 'success' else '✗'
            axs[0].text(
                rect.get_x() + rect.get_width()/2, 
                rect.get_height() + 0.5, 
                label,
                ha='center', va='bottom'
            )
    
    # Success/Error count
    if 'status' in summary_df.columns:
        status_counts = summary_df['status'].value_counts()
        axs[1].pie(
            status_counts, 
            labels=status_counts.index,
            autopct='%1.1f%%',
            colors=['green', 'red'] if 'success' in status_counts.index and 'error' in status_counts.index else None
        )
        axs[1].set_title('Processing Status')
    
    plt.tight_layout()
    summary_plot_file = Path(output_dir) / 'processing_summary.png'
    plt.savefig(summary_plot_file)
    plt.close()
    print(f"Summary plot saved to {summary_plot_file}")
    
    return summary_df


def create_simulated_data(output_dir, n_files=5):
    """Create simulated EEG data files for demonstration.
    
    Parameters
    ----------
    output_dir : str
        Path to output directory for simulated files
    n_files : int, default=5
        Number of files to generate
    
    Returns
    -------
    list
        List of paths to generated files
    """
    from mne import create_info
    from mne.io import RawArray
    from mne.channels import make_standard_montage
    
    # Create output directory if it doesn't exist
    sim_dir = Path(output_dir) / 'simulated_data'
    sim_dir.mkdir(exist_ok=True, parents=True)
    
    file_paths = []
    
    for i in range(n_files):
        # Create simulated EEG data
        sfreq = 250  # Sampling frequency (Hz)
        duration = 300  # Duration (seconds)
        n_channels = 32  # Number of channels
        
        # Create channel names based on 10-20 system
        montage = make_standard_montage('standard_1020')
        ch_names = list(montage.ch_names)[:n_channels-1] + ['STI 014']
        
        # Create info object
        ch_types = ['eeg'] * (n_channels - 1) + ['stim']
        info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Generate random data
        n_samples = int(duration * sfreq)
        data = np.random.randn(n_channels, n_samples) * 1e-6  # Random noise in Volts
        
        # Set stim channel to zeros
        data[-1, :] = 0
        
        # Add alpha oscillations (8-12 Hz)
        t = np.arange(n_samples) / sfreq
        alpha_power = np.random.uniform(0.5, 2.0)  # Vary alpha power between subjects
        alpha_freq = np.random.uniform(9.0, 11.0)  # Vary alpha frequency between subjects
        alpha = np.sin(2 * np.pi * alpha_freq * t) * alpha_power * 1e-6
        # Apply to posterior channels (O1, O2, P3, P4, etc.)
        posterior_chs = [i for i, ch in enumerate(ch_names) if ch[0] in 'OP' and i < n_channels-1]
        for ch in posterior_chs:
            data[ch, :] += alpha
        
        # Add line noise (50 or 60 Hz)
        line_freq = 50 if i % 2 == 0 else 60  # Alternate between 50 Hz and 60 Hz
        line_noise = np.sin(2 * np.pi * line_freq * t) * 0.2 * 1e-6
        data[:-1, :] += line_noise  # Add to all EEG channels
        
        # Create RawArray object
        raw = RawArray(data, info)
        
        # Set channel locations
        raw.set_montage(montage)
        
        # Add subject metadata
        raw.info['subject_info'] = {
            'id': f"sub-{i+1:03d}",
            'age': np.random.randint(20, 65),
            'sex': np.random.choice([0, 1, 2])  # 0=unknown, 1=male, 2=female
        }
        
        # Save as FIF file
        file_path = sim_dir / f"sub-{i+1:03d}_task-rest_eeg.fif"
        raw.save(file_path, overwrite=True)
        print(f"Created simulated file: {file_path}")
        file_paths.append(str(file_path))
    
    return file_paths


async def main(input_dir=None, output_dir=None, task_name="resting_eyes_open", max_workers=4, use_simulated=False, n_simulated=5):
    """Main function to run batch processing on EEG files.
    
    Parameters
    ----------
    input_dir : str, optional
        Directory containing EEG files to process. If None, simulated data will be used.
    output_dir : str, optional
        Directory to save output files. If None, './batch_output' will be used.
    task_name : str, default="resting_eyes_open"
        Name of the task to use for processing
    max_workers : int, default=4
        Maximum number of parallel workers
    use_simulated : bool, default=False
        Whether to use simulated data instead of reading from input_dir
    n_simulated : int, default=5
        Number of simulated files to generate if use_simulated is True
    """
    # Set up output directory
    if output_dir is None:
        output_dir = Path('./batch_output')
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get input files
    if use_simulated or input_dir is None:
        print("Creating simulated data for demonstration...")
        file_paths = create_simulated_data(output_dir, n_files=n_simulated)
    else:
        # Find all EEG files in the input directory
        input_dir = Path(input_dir)
        file_paths = find_eeg_files(input_dir)
        
        if not file_paths:
            print(f"No EEG files found in {input_dir}")
            return
    
    print(f"Found {len(file_paths)} EEG files to process")
    for i, file_path in enumerate(file_paths):
        print(f"{i+1}. {file_path}")
    
    print(f"\nProcessing files with task: {task_name}")
    print(f"Using {max_workers} parallel workers")
    print(f"Output will be saved to: {output_dir}")
    
    # Process files in parallel
    start_time = time.time()
    results = await process_files_parallel(file_paths, output_dir, max_workers, task_name)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / len(file_paths)
    
    print(f"\nProcessing complete!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per file: {avg_time:.2f} seconds")
    
    # Create summary
    summary_df = summarize_results(results, output_dir)
    
    # Print summary statistics
    success_count = len([r for r in results if r.get('status') == 'success'])
    print(f"\nSuccessfully processed {success_count} out of {len(file_paths)} files.")
    
    if 'processing_time' in summary_df.columns:
        print(f"Average processing time: {summary_df['processing_time'].mean():.2f} seconds")
        print(f"Fastest file: {summary_df.loc[summary_df['processing_time'].idxmin()]['file_name']} "
              f"({summary_df['processing_time'].min():.2f} seconds)")
        print(f"Slowest file: {summary_df.loc[summary_df['processing_time'].idxmax()]['file_name']} "
              f"({summary_df['processing_time'].max():.2f} seconds)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process EEG files using the AutoClean pipeline")
    parser.add_argument("--input-dir", type=str, help="Directory containing EEG files to process")
    parser.add_argument("--output-dir", type=str, default="./batch_output", 
                        help="Directory to save output files")
    parser.add_argument("--task", type=str, default="resting_eyes_open",
                        help="Name of the task to use for processing")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of parallel workers")
    parser.add_argument("--simulated", action="store_true",
                        help="Use simulated data instead of reading from input-dir")
    parser.add_argument("--n-simulated", type=int, default=5,
                        help="Number of simulated files to generate if --simulated is used")
    
    args = parser.parse_args()
    
    # Run the main function
    if sys.platform == 'win32':
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task_name=args.task,
        max_workers=args.max_workers,
        use_simulated=args.simulated,
        n_simulated=args.n_simulated
    )) 