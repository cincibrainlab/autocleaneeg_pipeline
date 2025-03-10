from autoclean.step_functions.io import (
    save_stc_to_file,
)
import matplotlib
import PyQt5
import mne

def estimate_source_function_raw(raw: mne.io.Raw, config: dict = None):
    """
    Perform source localization on continuous resting-state EEG data using an identity matrix
    for noise covariance, keeping it as raw data.
    """
    # --------------------------------------------------------------------------
    # Preprocessing for Source Localization
    # --------------------------------------------------------------------------
    matplotlib.use("Qt5Agg")
  
    raw.set_eeg_reference("average", projection=True)
    print("Set EEG reference to average")

    noise_cov = mne.make_ad_hoc_cov(raw.info)
    print("Using an identity matrix for noise covariance")

    # --------------------------------------------------------------------------
    # Source Localization Setup
    # --------------------------------------------------------------------------
    from mne.datasets import fetch_fsaverage

    fs_dir = fetch_fsaverage()
    subjects_dir = fs_dir.parent
    subject = 'fsaverage'
    trans = 'fsaverage'
    src = mne.read_source_spaces(f'{fs_dir}/bem/fsaverage-ico-5-src.fif')
    bem = mne.read_bem_solution(f'{fs_dir}/bem/fsaverage-5120-5120-5120-bem-sol.fif')

    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=10
    )
    print("Created forward solution")

    inv = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov)
    print("Created inverse operator with identity noise covariance")

    stc = mne.minimum_norm.apply_inverse_raw(
        raw, inv, lambda2=1./9., method='MNE', pick_ori='normal', verbose=True
    )

    # mne.viz.plot_alignment(
    #     raw.info,
    #     src=src,
    #     eeg=["original", "projected"],
    #     trans='fsaverage',
    #     show_axes=True,
    #     mri_fiducials=True,
    #     dig="fiducials",
    # )
    print("Computed continuous source estimates using MNE with identity noise covariance")

    if config is not None:
        save_stc_to_file(stc, config, stage="post_source_localization")

    return stc

def calculate_source_psd(stc, subjects_dir=None, subject='fsaverage', n_jobs=4, output_dir=None, subject_id=None):
    """
    Calculate power spectral density (PSD) from resting-state source estimates using Welch's method,
    average into anatomical ROIs using the Desikan-Killiany atlas, and save to structured files.
    
    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate PSD from
    subjects_dir : str | None
        Path to the freesurfer subjects directory. If None, uses the environment variable
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
        
    Returns
    -------
    psd_df : DataFrame
        DataFrame containing ROI-averaged PSD values
    file_path : str
        Path to the saved file
    """
    import os
    import numpy as np
    import pandas as pd
    import mne
    from scipy import signal
    from mne.parallel import parallel_func
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    
    print(f"Calculating PSD for {subject_id}...")
    
    # Define frequency parameters 
    fmin = 1.0
    fmax = 45.0
    
    # Get data array and sampling frequency
    data = stc.data
    sfreq = stc.sfreq
    
    print(f"Source data shape: {data.shape}")
    print(f"Computing source PSD using Welch's method for {len(data)} vertices...")
    
    # Welch parameters for resting-state data
    window_length = int(4 * sfreq)  # 4-second windows
    n_overlap = window_length // 2  # 50% overlap
    
    # For multitaper, we'll use mne's implementation directly which handles this correctly
    from mne.time_frequency import psd_array_multitaper
    
    # First, calculate frequencies that will be generated
    # We'll use a helper call to get the frequency axis
    freqs = np.fft.rfftfreq(window_length, 1/sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[freq_mask]
    
    # Initialize PSD array
    n_vertices = data.shape[0]
    psd = np.zeros((n_vertices, len(freqs)))
    
    # Function to process a batch of vertices
    def process_batch(vertices_idx):
        batch_psd = np.zeros((len(vertices_idx), len(freqs)))
        
        for i, vertex_idx in enumerate(vertices_idx):
            # Use standard Welch method for stability
            f, Pxx = signal.welch(
                data[vertex_idx], 
                fs=sfreq, 
                window='hann',  # Hann window is more stable than multitaper for this
                nperseg=window_length,
                noverlap=n_overlap,
                nfft=None,
                scaling='density'
            )
            
            # Store PSD for frequencies in our range
            batch_psd[i] = Pxx[freq_mask]
            
        return batch_psd
    
    # Process in batches to avoid memory issues with large source spaces
    batch_size = 1000
    n_batches = int(np.ceil(n_vertices / batch_size))
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)
        
        print(f"Processing batch {batch_idx+1}/{n_batches}, vertices {start_idx}-{end_idx}")
        
        # Get vertices for this batch
        vertices_idx = range(start_idx, end_idx)
        
        # Process batch
        batch_psd = process_batch(vertices_idx)
        
        # Store in main PSD array
        psd[start_idx:end_idx] = batch_psd
    
    print(f"PSD calculation complete. Shape: {psd.shape}, frequencies: {freqs.shape}")
    print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    
    # Load Desikan-Killiany atlas labels
    print("Loading Desikan-Killiany atlas labels...")
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    
    # Remove 'unknown' labels
    labels = [label for label in labels if 'unknown' not in label.name]
    
    # Initialize dataframe for ROI-averaged PSDs
    roi_psds = []
    
    # Process each label/ROI
    print("Averaging PSD within anatomical ROIs...")
    for label in labels:
        # Get vertices in this label
        label_verts = label.get_vertices_used()
        
        # Find indices of these vertices in the stc
        if label.hemi == 'lh':
            # Left hemisphere
            stc_idx = np.where(np.in1d(stc.vertices[0], label_verts))[0]
        else:
            # Right hemisphere
            stc_idx = np.where(np.in1d(stc.vertices[1], label_verts))[0] + len(stc.vertices[0])
        
        # Skip if no vertices found
        if len(stc_idx) == 0:
            print(f"Warning: No vertices found for label {label.name}")
            continue
            
        # Calculate mean PSD across vertices in this ROI
        roi_psd = np.mean(psd[stc_idx, :], axis=0)
        
        # Add to dataframe
        for freq_idx, freq in enumerate(freqs):
            roi_psds.append({
                'subject': subject_id,
                'roi': label.name,
                'hemisphere': label.hemi,
                'frequency': freq,
                'psd': roi_psd[freq_idx]
            })
    
    # Create dataframe
    psd_df = pd.DataFrame(roi_psds)
    
    # Save to file
    file_path = os.path.join(output_dir, f"{subject_id}_roi_psd.parquet")
    psd_df.to_parquet(file_path)
    print(f"Saved ROI-averaged PSD to {file_path}")
    
    # Also save a summary CSV with band averages
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    band_psds = []
    
    # For each ROI, calculate band power
    for roi in psd_df['roi'].unique():
        roi_data = psd_df[psd_df['roi'] == roi]
        for band_name, (band_min, band_max) in bands.items():
            # Get data in this frequency band
            band_data = roi_data[(roi_data['frequency'] >= band_min) & 
                                (roi_data['frequency'] < band_max)]
            
            # Calculate mean band power
            band_power = band_data['psd'].mean()
            
            # Add to band_psds
            band_psds.append({
                'subject': subject_id,
                'roi': roi,
                'hemisphere': roi_data['hemisphere'].iloc[0],
                'band': band_name,
                'band_start_hz': band_min,
                'band_end_hz': band_max,
                'power': band_power
            })
    
    # Create dataframe for band averages
    band_df = pd.DataFrame(band_psds)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{subject_id}_roi_bands.csv")
    band_df.to_csv(csv_path, index=False)
    print(f"Saved frequency band summary to {csv_path}")
    
    return psd_df, file_path

def visualize_psd_results(psd_df, output_dir=None, subject_id=None):
    """
    Create visualization plots for PSD data to confirm spectral analysis results.
    
    Parameters
    ----------
    psd_df : DataFrame
        DataFrame containing ROI-averaged PSD values, with columns:
        subject, roi, hemisphere, frequency, psd
    output_dir : str or None
        Directory to save output files. If None, current directory is used.
    subject_id : str or None
        Subject identifier for plot titles and filenames.
        If None, extracted from the data.
    
    Returns
    -------
    fig : matplotlib Figure
        Figure containing the visualization
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("notebook", font_scale=1.1)
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    if subject_id is None:
        subject_id = psd_df['subject'].iloc[0]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Define frequency bands
    bands = {
        'delta': (1, 4, '#1f77b4'),
        'theta': (4, 8, '#ff7f0e'),
        'alpha': (8, 13, '#2ca02c'),
        'beta': (13, 30, '#d62728'),
        'gamma': (30, 45, '#9467bd')
    }
    
    # 1. Plot PSD for selected regions
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Select a subset of interesting regions to plot
    regions_to_plot = ['precentral-lh', 'postcentral-lh', 'superiorparietal-lh', 
                       'lateraloccipital-lh', 'superiorfrontal-lh']
    
    # If some regions aren't in the data, use what's available
    available_rois = psd_df['roi'].unique()
    regions_to_plot = [r for r in regions_to_plot if r in available_rois]
    
    # If none of the selected regions are available, use the first 5 available
    if not regions_to_plot:
        regions_to_plot = list(available_rois)[:5]
    
    # Plot each region with linear scale
    for roi in regions_to_plot:
        roi_data = psd_df[psd_df['roi'] == roi]
        ax1.plot(roi_data['frequency'], roi_data['psd'], linewidth=2, alpha=0.8, 
                label=roi.split('-')[0])
    
    # Add frequency band backgrounds
    y_min, y_max = ax1.get_ylim()
    for band_name, (fmin, fmax, color) in bands.items():
        ax1.axvspan(fmin, fmax, color=color, alpha=0.1)
        
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density')
    ax1.set_title('PSD for Selected Regions (Left Hemisphere)')
    ax1.legend(loc='upper right')
    ax1.set_xlim(1, 45)
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # 2. Plot left vs right hemisphere comparison
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Average across all regions in each hemisphere
    for hemi, color, label in zip(['lh', 'rh'], ['#1f77b4', '#d62728'], 
                                 ['Left Hemisphere', 'Right Hemisphere']):
        hemi_data = psd_df[psd_df['hemisphere'] == hemi].groupby('frequency')['psd'].mean().reset_index()
        ax2.plot(hemi_data['frequency'], hemi_data['psd'], linewidth=2.5, 
                color=color, label=label)
    
    # Add frequency band backgrounds
    for band_name, (fmin, fmax, color) in bands.items():
        ax2.axvspan(fmin, fmax, color=color, alpha=0.1)
        
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title('Left vs Right Hemisphere Average')
    ax2.legend(loc='upper right')
    ax2.set_xlim(1, 45)
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    
    # 3. Frequency band power comparison across regions
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate average power in each frequency band for each ROI
    band_powers = []
    for roi in psd_df['roi'].unique():
        roi_base = roi.split('-')[0]
        hemi = roi.split('-')[1]
        
        roi_data = psd_df[psd_df['roi'] == roi]
        
        for band_name, (fmin, fmax, _) in bands.items():
            band_data = roi_data[(roi_data['frequency'] >= fmin) & 
                                (roi_data['frequency'] <= fmax)]
            band_power = band_data['psd'].mean()
            
            band_powers.append({
                'ROI': roi_base,
                'Hemisphere': 'Left' if hemi == 'lh' else 'Right',
                'Band': band_name.capitalize(),
                'Power': band_power
            })
    
    band_power_df = pd.DataFrame(band_powers)
    
    # Select a subset of regions for clarity
    regions_for_bands = ['precentral', 'postcentral', 'superiorfrontal', 'lateraloccipital']
    available_base_rois = band_power_df['ROI'].unique()
    regions_for_bands = [r for r in regions_for_bands if r in available_base_rois]
    
    if not regions_for_bands:
        regions_for_bands = list(available_base_rois)[:4]
    
    plot_data = band_power_df[band_power_df['ROI'].isin(regions_for_bands) & 
                             (band_power_df['Hemisphere'] == 'Left')]
    
    # Normalize powers within each region for better visualization
    for roi in plot_data['ROI'].unique():
        roi_mask = plot_data['ROI'] == roi
        max_power = plot_data.loc[roi_mask, 'Power'].max()
        plot_data.loc[roi_mask, 'Normalized Power'] = plot_data.loc[roi_mask, 'Power'] / max_power
    
    # Plot band powers
    sns.barplot(x='ROI', y='Normalized Power', hue='Band', data=plot_data, ax=ax3, 
               palette=[bands[b.lower()][2] for b in plot_data['Band'].unique()])
    
    ax3.set_xlabel('Brain Region')
    ax3.set_ylabel('Normalized Band Power')
    ax3.set_title('Frequency Band Distribution Across Regions')
    ax3.legend(title='Frequency Band')
    
    # 4. Alpha/Beta ratio across regions
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate alpha/beta ratio for each ROI
    alpha_beta_data = []
    
    for roi_base in band_power_df['ROI'].unique():
        for hemi in ['Left', 'Right']:
            alpha_power = band_power_df[(band_power_df['ROI'] == roi_base) & 
                                      (band_power_df['Hemisphere'] == hemi) & 
                                      (band_power_df['Band'] == 'Alpha')]['Power'].values
            
            beta_power = band_power_df[(band_power_df['ROI'] == roi_base) & 
                                     (band_power_df['Hemisphere'] == hemi) & 
                                     (band_power_df['Band'] == 'Beta')]['Power'].values
            
            if len(alpha_power) > 0 and len(beta_power) > 0 and beta_power[0] > 0:
                ratio = alpha_power[0] / beta_power[0]
                
                alpha_beta_data.append({
                    'ROI': roi_base,
                    'Hemisphere': hemi,
                    'Alpha/Beta Ratio': ratio
                })
    
    ratio_df = pd.DataFrame(alpha_beta_data)
    
    # Select regions for visualization
    if len(regions_for_bands) > 0:
        ratio_plot = ratio_df[ratio_df['ROI'].isin(regions_for_bands)]
    else:
        # If no specific regions, use top 4 by ratio
        ratio_plot = ratio_df.sort_values('Alpha/Beta Ratio', ascending=False).head(8)
    
    sns.barplot(x='ROI', y='Alpha/Beta Ratio', hue='Hemisphere', data=ratio_plot, ax=ax4,
               palette=['#1f77b4', '#d62728'])
    
    ax4.set_xlabel('Brain Region')
    ax4.set_ylabel('Alpha/Beta Power Ratio')
    ax4.set_title('Alpha/Beta Ratio by Region')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.legend(title='Hemisphere')
    
    # Final adjustments
    plt.suptitle(f'Power Spectral Density Analysis: {subject_id}', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    # Save figure
    output_path = os.path.join(output_dir, f"{subject_id}_psd_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved PSD visualization to {output_path}")
    
    return fig

def calculate_source_connectivity(stc, labels=None, subjects_dir=None, subject='fsaverage', 
                                 n_jobs=4, output_dir=None, subject_id=None, sfreq=None,
                                 epoch_length=4.0, n_epochs=40):
    import os
    import numpy as np
    import pandas as pd
    import mne
    from mne_connectivity import spectral_connectivity_time
    import itertools
    
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "connectivity"), exist_ok=True)
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    if sfreq is None:
        sfreq = stc.sfreq
    
    print(f"Calculating connectivity for {subject_id} with {n_epochs} {epoch_length}-second epochs (sfreq={sfreq} Hz)...")
    
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    conn_methods = ['wpli']
    
    if labels is None:
        print("Loading Desikan-Killiany atlas labels...")
        labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
        labels = [label for label in labels if 'unknown' not in label.name]
    
    selected_rois = [
        'precentral-lh', 'precentral-rh', 'postcentral-lh', 'postcentral-rh',
        'paracentral-lh', 'paracentral-rh', 'caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh'
    ]
    label_names = [label.name for label in labels]
    selected_labels = [labels[label_names.index(roi)] for roi in selected_rois if roi in label_names]
    if not selected_labels:
        print("No selected ROIs found, using all available labels")
        selected_labels = labels
        selected_rois = label_names
    print(f"Using {len(selected_labels)} selected ROIs")
    
    roi_pairs = list(itertools.combinations(range(len(selected_rois)), 2))
    
    print("Extracting ROI time courses...")
    roi_time_courses = [stc.extract_label_time_course(label, src=None, mode='mean', verbose=False)[0] 
                        for label in selected_labels]
    roi_data = np.array(roi_time_courses)
    print(f"ROI data shape: {roi_data.shape}")
    
    n_times = roi_data.shape[1]
    samples_per_epoch = int(epoch_length * sfreq)
    max_epochs = n_times // samples_per_epoch
    if max_epochs < n_epochs:
        print(f"Warning: Requested {n_epochs} epochs, but only {max_epochs} possible. Using {max_epochs}.")
        n_epochs = max_epochs
    
    epoch_starts = np.random.choice(max_epochs, size=n_epochs, replace=False) * samples_per_epoch
    epoched_data = np.stack([roi_data[:, start:start + samples_per_epoch] 
                            for start in epoch_starts], axis=0)
    print(f"Epoched data shape: {epoched_data.shape}")
    
    connectivity_data = []
    print("Calculating connectivity metrics...")
    for method in conn_methods:
        for band_name, band_range in bands.items():
            print(f"  Computing {method} connectivity in {band_name} band...")
            con = spectral_connectivity_time(
                epoched_data, freqs=np.arange(band_range[0], band_range[1] + 1),
                method=method, sfreq=sfreq, mode='multitaper', faverage=True,
                average=True, n_jobs=n_jobs, verbose=False, n_cycles=2
            )
            con_matrix = con.get_data(output='dense').squeeze()
            if con_matrix.shape != (len(selected_rois), len(selected_rois)):
                raise ValueError(f"Unexpected con_matrix shape: {con_matrix.shape}")
            
            # Debug: Print con_matrix to verify
            print(f"{method} {band_name} con_matrix:\n", con_matrix)
            
            con_df = pd.DataFrame(con_matrix, columns=selected_rois, index=selected_rois)
            matrix_filename = os.path.join(output_dir, f"{subject_id}_{method}_{band_name}_matrix.csv")
            con_df.to_csv(matrix_filename)
            print(f"  Saved connectivity matrix to {matrix_filename}")
            
            for i, (roi1_idx, roi2_idx) in enumerate(roi_pairs):
                # Use lower triangle by swapping indices
                value = con_matrix[roi2_idx, roi1_idx]  # Changed from [roi1_idx, roi2_idx]
                connectivity_data.append({
                    'subject': subject_id, 'method': method, 'band': band_name,
                    'roi1': selected_rois[roi1_idx], 'roi2': selected_rois[roi2_idx],
                    'connectivity': value
                })
    
    # Debug: Print sample connectivity_data
    print("Sample connectivity_data entries:", connectivity_data[:5])
    conn_df = pd.DataFrame(connectivity_data)
    summary_path = os.path.join(output_dir, f"{subject_id}_connectivity_summary.csv")
    conn_df.to_csv(summary_path, index=False)
    print(f"Saved connectivity summary to {summary_path}")
    
    # Calculate graph metrics for each connectivity method and band
    print("Calculating graph metrics...")
    graph_metrics_data = []

    import itertools
    import networkx as nx
    from bct import clustering_coef_wu, efficiency_wei, charpath
    from networkx.algorithms.community import louvain_communities, modularity


    graph_metrics_data = []
    for method in conn_methods:
        for band_name in bands.keys():
            subset_df = conn_df[(conn_df['method'] == method) & (conn_df['band'] == band_name)]

            G = nx.Graph()
            for _, row in subset_df.iterrows():
                G.add_edge(row['roi1'], row['roi2'], weight=row['connectivity'])

            adj_matrix = nx.to_numpy_array(G, nodelist=selected_rois, weight='weight')

            clustering = np.mean(clustering_coef_wu(adj_matrix))
            global_efficiency = efficiency_wei(adj_matrix)
            char_path_length, _, _, _, _ = charpath(adj_matrix)
            communities = louvain_communities(G, resolution=1.0)
            modularity_score = modularity(G, communities, weight='weight')
            strength = np.mean(np.sum(adj_matrix, axis=1))

            graph_metrics_data.append({
                'subject': subject_id,
                'method': method,
                'band': band_name,
                'clustering': clustering,
                'global_efficiency': global_efficiency,
                'char_path_length': char_path_length,
                'modularity': modularity_score,
                'strength': strength
            })

    graph_metrics_df = pd.DataFrame(graph_metrics_data)
    metrics_path = os.path.join(output_dir, f"{subject_id}_graph_metrics.csv")
    graph_metrics_df.to_csv(metrics_path, index=False)



    return conn_df, summary_path

def calculate_source_pac(stc, labels=None, subjects_dir=None, subject='fsaverage', 
                         n_jobs=4, output_dir=None, subject_id=None, sfreq=None):
    """
    Calculate phase-amplitude coupling (PAC) from source-localized data with specific focus
    on ALS-relevant coupling and regions.
    
    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate PAC from
    labels : list of Labels | None
        List of ROI labels to use. If None, will load Desikan-Killiany atlas
    subjects_dir : str | None
        Path to the freesurfer subjects directory. If None, uses the environment variable
    subject : str
        Subject name in the subjects_dir (default: 'fsaverage')
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files. If None, saves in current directory
    subject_id : str | None
        Subject identifier for file naming
    sfreq : float | None
        Sampling frequency. If None, will use stc.sfreq
        
    Returns
    -------
    pac_df : DataFrame
        DataFrame containing PAC values for all ROIs and frequency band pairs
    file_path : str
        Path to the saved summary file
    """
    import os
    import numpy as np
    import pandas as pd
    import mne
    from scipy.signal import hilbert
    from joblib import Parallel, delayed
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pac"), exist_ok=True)
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    
    if sfreq is None:
        sfreq = stc.sfreq
    
    print(f"Calculating ALS-focused phase-amplitude coupling for {subject_id}...")
    
    # Define frequency bands - narrower beta band definition for better ALS sensitivity
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'lowbeta': (13, 20),   # Lower beta associated with motor control
        'highbeta': (20, 30),  # Higher beta associated with inhibitory control
        'gamma': (30, 45)
    }
    
    # Define PAC pairs focused on ALS-relevant couplings
    # Priority on beta-gamma coupling which is most relevant for motor dysfunction in ALS
    coupling_pairs = [
        ('lowbeta', 'gamma'),    # Primary focus: motor system dysfunction
        ('highbeta', 'gamma'),   # Secondary focus: inhibitory control issues in ALS
        ('alpha', 'lowbeta'),    # Changed from 'alpha', 'beta'
        ('theta', 'lowbeta'),    # Changed from 'theta', 'beta'
    ]
    # Load labels if not provided
    if labels is None:
        print("Loading Desikan-Killiany atlas labels...")
        labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
        labels = [label for label in labels if 'unknown' not in label.name]
    
    # Focus on ALS-specific ROIs (motor network emphasis)
    selected_rois = [
        'precentral-lh', 'precentral-rh',         # Primary motor cortex (key for ALS)
        'postcentral-lh', 'postcentral-rh',       # Primary sensory cortex
        'paracentral-lh', 'paracentral-rh',       # Supplementary motor area
        'caudalmiddlefrontal-lh', 'caudalmiddlefrontal-rh',  # Premotor cortex
        'superiorfrontal-lh', 'superiorfrontal-rh',  # Executive function (affected in ~50% of ALS cases)
        'inferiorparietal-lh', 'inferiorparietal-rh'  # Sensorimotor integration
    ]
    
    # Filter labels to keep only selected ROIs
    label_names = [label.name for label in labels]
    selected_labels = []
    selected_roi_names = []
    
    for roi in selected_rois:
        if roi in label_names:
            selected_labels.append(labels[label_names.index(roi)])
            selected_roi_names.append(roi)
        else:
            print(f"Warning: ROI {roi} not found in the available labels")
    
    # If no ROIs matched, use a subset of all labels
    if not selected_labels:
        print("No selected ROIs found, using a subset of available labels")
        # Take a reasonable subset to avoid excessive computation
        selected_labels = labels[:16]  # First 16 labels
        selected_roi_names = label_names[:16]
    else:
        print(f"Using {len(selected_labels)} selected ROIs for ALS-focused PAC analysis")
    
    # Function to segment data into epochs to speed up computation
    def epoch_data(data, epoch_length=4, n_epochs=40, sfreq=250):
        """Split continuous data into epochs"""
        samples_per_epoch = int(epoch_length * sfreq)
        total_samples = len(data)
        
        # If we have enough data, take the first n_epochs
        if total_samples >= n_epochs * samples_per_epoch:
            epochs = data[:n_epochs * samples_per_epoch].reshape(n_epochs, samples_per_epoch)
        else:
            # If not enough data, use as many complete epochs as possible
            max_complete_epochs = total_samples // samples_per_epoch
            print(f"Warning: Only {max_complete_epochs} complete epochs available (requested {n_epochs})")
            epochs = data[:max_complete_epochs * samples_per_epoch].reshape(max_complete_epochs, samples_per_epoch)
        
        return epochs
    
    # Function to calculate PAC for a single time series
    def phase_amplitude_coupling(signal, phase_band, amp_band, fs):
        """Calculate phase-amplitude coupling between two frequency bands"""
        # Segment into epochs for faster processing
        epoch_length = 4  # seconds
        n_epochs = 40     # enough for reliable PAC estimate while keeping computation manageable
        
        try:
            signal_epochs = epoch_data(signal, epoch_length=epoch_length, n_epochs=n_epochs, sfreq=fs)
            n_actual_epochs = signal_epochs.shape[0]
            
            # Calculate PAC for each epoch and average
            epoch_mis = []
            
            for epoch_idx in range(n_actual_epochs):
                epoch = signal_epochs[epoch_idx]
                
                # Filter signal for phase frequency band
                phase_signal = mne.filter.filter_data(
                    epoch, fs, phase_band[0], phase_band[1], method='iir')
                
                # Filter signal for amplitude frequency band
                amp_signal = mne.filter.filter_data(
                    epoch, fs, amp_band[0], amp_band[1], method='iir')
                
                # Extract phase and amplitude using Hilbert transform
                phase = np.angle(hilbert(phase_signal))
                amplitude = np.abs(hilbert(amp_signal))
                
                # Calculate modulation index (MI)
                n_bins = 18
                phase_bins = np.linspace(-np.pi, np.pi, n_bins+1)
                mean_amp = np.zeros(n_bins)
                
                for bin_idx in range(n_bins):
                    bin_mask = np.logical_and(phase >= phase_bins[bin_idx], 
                                             phase < phase_bins[bin_idx+1])
                    if np.any(bin_mask):  # Check if the bin has any data points
                        mean_amp[bin_idx] = np.mean(amplitude[bin_mask])
                
                # Normalize mean amplitude
                if np.sum(mean_amp) > 0:  # Avoid division by zero
                    mean_amp /= np.sum(mean_amp)
                
                    # Calculate MI using Kullback-Leibler divergence
                    uniform = np.ones(n_bins) / n_bins
                    # Avoid log(0) by adding a small epsilon
                    epsilon = 1e-10
                    mi = np.sum(mean_amp * np.log((mean_amp + epsilon) / (uniform + epsilon)))
                    epoch_mis.append(mi)
                else:
                    epoch_mis.append(0)
            
            # Average MI across epochs
            if epoch_mis:
                return np.mean(epoch_mis)
            else:
                return 0
                
        except Exception as e:
            print(f"Error calculating PAC: {e}")
            return 0
    
    # Extract time courses for each ROI
    print("Extracting ROI time courses...")
    roi_time_courses = {}
    
    for i, label in enumerate(selected_labels):
        # Extract time course for this label
        tc = stc.extract_label_time_course(label, src=None, mode='mean', verbose=False)
        roi_time_courses[selected_roi_names[i]] = tc[0]
    
    # Initialize data storage for PAC values
    pac_data = []
    
    # Function to process a single ROI and coupling pair
    def process_pac(roi, phase_band_name, amp_band_name):
        signal = roi_time_courses[roi]
        phase_range = bands[phase_band_name]
        amp_range = bands[amp_band_name]
        
        # Calculate PAC
        mi = phase_amplitude_coupling(signal, phase_range, amp_range, sfreq)
        
        return {
            'roi': roi,
            'phase_band': phase_band_name,
            'amp_band': amp_band_name,
            'mi': mi
        }
    
    # Process all ROIs and coupling pairs in parallel
    print("Calculating PAC for all ROIs and frequency band pairs...")
    
    # Create task list for parallel processing with priority for motor regions and beta-gamma coupling
    tasks = []
    
    # Add high-priority tasks first (beta-gamma in motor areas)
    motor_regions = ['precentral-lh', 'precentral-rh', 'postcentral-lh', 'postcentral-rh']
    beta_gamma_pairs = [pair for pair in coupling_pairs if 'beta' in pair[0] and 'gamma' in pair[1]]
    
    for roi in selected_roi_names:
        # Prioritize motor regions with beta-gamma coupling
        if roi in motor_regions:
            for phase_band, amp_band in beta_gamma_pairs:
                tasks.append((roi, phase_band, amp_band))
    
    # Then add all other combinations
    for roi in selected_roi_names:
        for phase_band, amp_band in coupling_pairs:
            if not ((roi in motor_regions) and ((phase_band, amp_band) in beta_gamma_pairs)):
                tasks.append((roi, phase_band, amp_band))
    
    # Run tasks in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pac)(roi, phase_band, amp_band) 
        for roi, phase_band, amp_band in tasks
    )
    
    # Add all results to data storage
    for result in results:
        result['subject'] = subject_id
        pac_data.append(result)
    
    # Create and save PAC dataframe
    pac_df = pd.DataFrame(pac_data)
    
    # Save to CSV
    file_path = os.path.join(output_dir, f"{subject_id}_als_pac_summary.csv")
    pac_df.to_csv(file_path, index=False)
    print(f"Saved ALS-focused PAC summary to {file_path}")
    
    # Also save a more readable pivot table version
    pivot_data = pac_df.pivot_table(
        index=['roi', 'phase_band'],
        columns='amp_band',
        values='mi'
    ).reset_index()
    
    pivot_path = os.path.join(output_dir, f"{subject_id}_als_pac_pivot.csv")
    pivot_data.to_csv(pivot_path, index=False)
    print(f"Saved PAC pivot table to {pivot_path}")
    
    # Generate a summary with special focus on motor-region beta-gamma coupling
    coupling_summary = []
    
    # Calculate motor-specific summaries for beta-gamma coupling
    motor_regions = ['precentral-lh', 'precentral-rh', 'postcentral-lh', 'postcentral-rh']
    
    # First, summarize beta-gamma coupling in motor regions (most relevant for ALS)
    motor_beta_gamma = pac_df[
        (pac_df['roi'].isin(motor_regions)) & 
        (pac_df['phase_band'].isin(['lowbeta', 'highbeta'])) & 
        (pac_df['amp_band'] == 'gamma')
    ]
    
    if not motor_beta_gamma.empty:
        motor_mean = motor_beta_gamma['mi'].mean()
        motor_max = motor_beta_gamma['mi'].max()
        motor_max_roi = motor_beta_gamma.loc[motor_beta_gamma['mi'].idxmax(), 'roi'] if len(motor_beta_gamma) > 0 else 'none'
        motor_max_band = motor_beta_gamma.loc[motor_beta_gamma['mi'].idxmax(), 'phase_band'] if len(motor_beta_gamma) > 0 else 'none'
        
        coupling_summary.append({
            'subject': subject_id,
            'coupling_name': 'motor_beta_gamma',
            'description': 'Beta-gamma coupling in motor regions (ALS primary focus)',
            'mean_mi': motor_mean,
            'max_mi': motor_max,
            'max_roi': motor_max_roi,
            'max_phase_band': motor_max_band
        })
    
    # Then add summaries for each coupling pair across all regions
    for phase_band, amp_band in coupling_pairs:
        pair_data = pac_df[(pac_df['phase_band'] == phase_band) & 
                           (pac_df['amp_band'] == amp_band)]
        
        # Calculate statistics for this coupling pair
        mean_mi = pair_data['mi'].mean()
        max_mi = pair_data['mi'].max()
        max_roi = pair_data.loc[pair_data['mi'].idxmax(), 'roi'] if len(pair_data) > 0 else 'none'
        
        coupling_summary.append({
            'subject': subject_id,
            'coupling_name': f"{phase_band}-{amp_band}",
            'description': f"{phase_band}-{amp_band} coupling across all regions",
            'mean_mi': mean_mi,
            'max_mi': max_mi,
            'max_roi': max_roi,
            'max_phase_band': phase_band
        })
    
    # Save coupling summary
    summary_df = pd.DataFrame(coupling_summary)
    summary_path = os.path.join(output_dir, f"{subject_id}_als_pac_coupling_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved ALS-focused coupling summary to {summary_path}")
    
    return pac_df, file_path

def calculate_vertex_level_spectral_power(stc, bands=None, n_jobs=10, output_dir=None, subject_id=None):
    """
    Calculate spectral power at the vertex level across the entire brain.
    
    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course to calculate power from
    bands : dict | None
        Dictionary of frequency bands. If None, uses standard bands
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    
    Returns
    -------
    power_dict : dict
        Dictionary containing power values for each frequency band
        for each vertex
    file_path : str
        Path to the saved vertex power file
    """
    import os
    import numpy as np
    import mne
    from scipy import signal
    import h5py
    from tqdm import tqdm
    from joblib import Parallel, delayed
    
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
        
    if subject_id is None:
        subject_id = 'unknown_subject'
        
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'lowbeta': (13, 20),
            'highbeta': (20, 30),
            'gamma': (30, 45)
        }
    
    # Get data and sampling frequency
    data = stc.data
    sfreq = stc.sfreq
    n_vertices = data.shape[0]
    
    print(f"Calculating vertex-level spectral power for {subject_id}...")
    print(f"Source data shape: {data.shape}")
    
    # Parameters for Welch's method
    window_length = int(4 * sfreq)  # 4-second windows
    n_overlap = window_length // 2  # 50% overlap
    
    # Function to calculate power for a batch of vertices
    def process_vertex_batch(vertex_indices):
        # Initialize a dictionary to store power values for each band
        batch_powers = {band: np.zeros(len(vertex_indices)) for band in bands}
        
        for i, vertex_idx in enumerate(vertex_indices):
            # Calculate PSD using Welch's method
            f, psd = signal.welch(
                data[vertex_idx], 
                fs=sfreq, 
                window='hann',
                nperseg=window_length,
                noverlap=n_overlap,
                nfft=None,
                scaling='density'
            )
            
            # Calculate average power in each frequency band
            for band, (fmin, fmax) in bands.items():
                freq_mask = (f >= fmin) & (f <= fmax)
                if np.any(freq_mask):
                    batch_powers[band][i] = np.mean(psd[freq_mask])
                else:
                    batch_powers[band][i] = 0
        
        return batch_powers, vertex_indices
    
    # Process vertices in batches to manage memory
    batch_size = 1000
    n_batches = int(np.ceil(n_vertices / batch_size))
    all_vertex_batches = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_vertices)
        all_vertex_batches.append(range(start_idx, end_idx))
    
    print(f"Processing {n_vertices} vertices in {n_batches} batches...")
    
    # Run processing in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_vertex_batch)(vertex_batch) 
        for vertex_batch in all_vertex_batches
    )
    
    # Combine results
    power_dict = {band: np.zeros(n_vertices) for band in bands}
    
    for batch_powers, vertex_indices in results:
        for band in bands:
            power_dict[band][vertex_indices] = batch_powers[band]
    
    # Save results to disk
    # HDF5 format is good for large arrays and provides compression
    file_path = os.path.join(output_dir, f"{subject_id}_vertex_power.h5")
    
    with h5py.File(file_path, 'w') as f:
        # Store vertex information
        f.attrs['n_vertices'] = n_vertices
        f.attrs['lh_vertices'] = len(stc.vertices[0])
        f.attrs['rh_vertices'] = len(stc.vertices[1])
        
        # Create a group for each frequency band
        for band, power_values in power_dict.items():
            f.create_dataset(band, data=power_values, compression="gzip", compression_opts=9)
    
    print(f"Saved vertex-level spectral power to {file_path}")
    
    return power_dict, file_path

def apply_spatial_smoothing(power_dict, stc, smoothing_steps=5, subject_id=None, output_dir=None):
    """
    Apply spatial smoothing to vertex-level power data.
    
    Parameters
    ----------
    power_dict : dict
        Dictionary containing power values for each frequency band
    stc : instance of SourceEstimate
        The source time course object (needed for vertices info)
    smoothing_steps : int
        Number of smoothing steps to apply
    subject_id : str | None
        Subject identifier for file naming
    output_dir : str | None
        Directory to save output files
        
    Returns
    -------
    smoothed_dict : dict
        Dictionary containing smoothed power values
    file_path : str
        Path to the saved smoothed data file
    """
    import os
    import numpy as np
    import mne
    import h5py
    
    if output_dir is None:
        output_dir = os.getcwd()
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    
    print(f"Applying spatial smoothing (steps={smoothing_steps}) to vertex data...")
    
    # Create a source space for smoothing
    # We need the same source space that was used to create the stc
    src = mne.source_space.SourceSpaces([
        mne.source_space.SourceSpace(vertices=stc.vertices[0], hemisphere=0, coord_frame=5),
        mne.source_space.SourceSpace(vertices=stc.vertices[1], hemisphere=1, coord_frame=5)
    ])
    
    smoothed_dict = {}
    
    # Apply smoothing to each frequency band
    for band, power_values in power_dict.items():
        print(f"Smoothing {band} band data...")
        
        # Create a temporary SourceEstimate to use MNE's smoothing function
        temp_stc = mne.SourceEstimate(
            power_values[:, np.newaxis],  # Add a time dimension
            vertices=stc.vertices,
            tmin=0,
            tstep=1
        )
        
        # Apply spatial smoothing
        smoothed_stc = mne.spatial_src_adjacency(temp_stc, src, n_steps=smoothing_steps)
        
        # Store the smoothed data
        smoothed_dict[band] = smoothed_stc.data[:, 0]  # Remove the time dimension
    
    # Save the smoothed data
    file_path = os.path.join(output_dir, f"{subject_id}_smoothed_vertex_power.h5")
    
    with h5py.File(file_path, 'w') as f:
        # Store vertex information
        f.attrs['n_vertices'] = len(smoothed_dict[next(iter(smoothed_dict))])
        f.attrs['lh_vertices'] = len(stc.vertices[0])
        f.attrs['rh_vertices'] = len(stc.vertices[1])
        f.attrs['smoothing_steps'] = smoothing_steps
        
        # Create a group for each frequency band
        for band, power_values in smoothed_dict.items():
            f.create_dataset(band, data=power_values, compression="gzip", compression_opts=9)
    
    print(f"Saved smoothed vertex-level spectral power to {file_path}")
    
    return smoothed_dict, file_path

def calculate_fooof_aperiodic(stc, freq_range=None, n_jobs=10, output_dir=None, subject_id=None, aperiodic_mode='knee'):
    """
    Calculate FOOOF aperiodic parameters from source-localized data and save results.
    
    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course containing spectral data
    freq_range : tuple | None
        Frequency range to fit (fmin, fmax). If None, will use (1, 45) Hz
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    aperiodic_mode : str
        Aperiodic mode for FOOOF ('fixed' or 'knee')
        
    Returns
    -------
    aperiodic_df : DataFrame
        DataFrame containing aperiodic parameters for each vertex
    file_path : str
        Path to the saved data file
    """
    import os
    import numpy as np
    import pandas as pd
    from fooof import FOOOFGroup
    from joblib import Parallel, delayed
    import gc
    
    # Import FOOOF only if not already imported
    try:
        from fooof import FOOOFGroup
    except ImportError:
        raise ImportError("FOOOF is required for this function. Install with 'pip install fooof'")
    
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    
    if freq_range is None:
        freq_range = (1, 45)
    
    print(f"Calculating FOOOF aperiodic parameters for {subject_id}...")
    
    # Get data from stc
    if hasattr(stc, 'data') and hasattr(stc, 'times'):
        # Assuming stc.data contains PSDs and stc.times contains frequencies
        psds = stc.data
        freqs = stc.times
    else:
        raise ValueError("Input stc must have 'data' and 'times' attributes with PSDs and frequencies")
    
    # Check if frequencies are within the specified range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(freq_mask):
        raise ValueError(f"No frequencies found within the specified range {freq_range}")
    
    # Trim data to specified frequency range
    freqs_to_fit = freqs[freq_mask]
    psds_to_fit = psds[:, freq_mask]
    
    n_vertices = psds.shape[0]
    print(f"Processing FOOOF analysis for {n_vertices} vertices...")
    
    # FOOOF parameters
    fooof_params = {
        'peak_width_limits': [1, 12.0],
        'max_n_peaks': 6,
        'min_peak_height': 0.0,
        'peak_threshold': 2.0,
        'aperiodic_mode': aperiodic_mode,
        'verbose': False
    }
    
    # Function to process a batch of vertices
    def process_batch(vertices):
        # Extract data for these vertices
        batch_psds = psds_to_fit[vertices, :]
        
        # Create FOOOF model and fit
        fg = FOOOFGroup(**fooof_params)
        fg.fit(freqs_to_fit, batch_psds)
        
        # Extract aperiodic parameters
        aperiodic_params = fg.get_params('aperiodic_params')
        r_squared = fg.get_params('r_squared')
        error = fg.get_params('error')
        
        # Process aperiodic parameters
        results = []
        for i, vertex_idx in enumerate(vertices):
            # Parameters depend on aperiodic mode
            if aperiodic_mode == 'knee':
                offset = aperiodic_params[i, 0]
                knee = aperiodic_params[i, 1]
                exponent = aperiodic_params[i, 2]
            else:  # fixed mode
                offset = aperiodic_params[i, 0]
                knee = np.nan
                exponent = aperiodic_params[i, 1]
            
            results.append({
                'vertex': vertex_idx,
                'offset': offset,
                'knee': knee,
                'exponent': exponent,
                'r_squared': r_squared[i],
                'error': error[i]
            })
        
        # Clear memory
        del fg, batch_psds
        gc.collect()
        
        return results
    
    # Process in batches to manage memory
    batch_size = 2000  # Adjust based on memory constraints
    vertex_batches = []
    
    for i in range(0, n_vertices, batch_size):
        vertex_batches.append(range(i, min(i + batch_size, n_vertices)))
    
    print(f"Processing {len(vertex_batches)} batches with {n_jobs} parallel jobs...")
    
    # Run in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in vertex_batches
    )
    
    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]
    
    # Convert to DataFrame
    aperiodic_df = pd.DataFrame(flat_results)
    
    # Add subject_id
    aperiodic_df.insert(0, 'subject', subject_id)
    
    # Save results
    file_path = os.path.join(output_dir, f"{subject_id}_fooof_aperiodic.parquet")
    aperiodic_df.to_csv(os.path.join(output_dir, f"{subject_id}_fooof_aperiodic.csv"), index=False)
    aperiodic_df.to_parquet(file_path)
    
    print(f"Saved FOOOF aperiodic parameters to {file_path}")
    
    return aperiodic_df, file_path

def calculate_fooof_periodic(stc, freq_bands=None, n_jobs=10, output_dir=None, subject_id=None, aperiodic_mode='knee'):
    """
    Calculate FOOOF periodic parameters from source-localized data and save results.
    
    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course containing spectral data
    freq_bands : dict | None
        Dictionary of frequency bands to analyze, e.g., {'alpha': (8, 13)}
        If None, will use default bands: delta, theta, alpha, beta, gamma
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    aperiodic_mode : str
        Aperiodic mode for FOOOF ('fixed' or 'knee')
        
    Returns
    -------
    periodic_df : DataFrame
        DataFrame containing periodic parameters for each vertex and frequency band
    file_path : str
        Path to the saved data file
    """
    import os
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    import gc
    
    # Import FOOOF only if not already imported
    try:
        from fooof import FOOOFGroup
        from fooof.analysis import get_band_peak_fm
    except ImportError:
        raise ImportError("FOOOF is required for this function. Install with 'pip install fooof'")
    
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    
    if freq_bands is None:
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
    
    print(f"Calculating FOOOF oscillatory parameters for {subject_id}...")
    
    # Get data from stc
    if hasattr(stc, 'data') and hasattr(stc, 'times'):
        # Assuming stc.data contains PSDs and stc.times contains frequencies
        psds = stc.data
        freqs = stc.times
    else:
        raise ValueError("Input stc must have 'data' and 'times' attributes with PSDs and frequencies")
    
    # Determine full frequency range
    freq_range = (min([band[0] for band in freq_bands.values()]),
                 max([band[1] for band in freq_bands.values()]))
    
    # Check if frequencies are within the specified range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(freq_mask):
        raise ValueError(f"No frequencies found within the specified range {freq_range}")
    
    # Trim data to specified frequency range
    freqs_to_fit = freqs[freq_mask]
    psds_to_fit = psds[:, freq_mask]
    
    n_vertices = psds.shape[0]
    print(f"Processing FOOOF analysis for {n_vertices} vertices and {len(freq_bands)} frequency bands...")
    
    # FOOOF parameters
    fooof_params = {
        'peak_width_limits': [1, 12.0],
        'max_n_peaks': 6,
        'min_peak_height': 0.0,
        'peak_threshold': 2.0,
        'aperiodic_mode': aperiodic_mode,
        'verbose': False
    }
    
    # Function to process a batch of vertices
    def process_batch(vertices):
        # Extract data for these vertices
        batch_psds = psds_to_fit[vertices, :]
        
        # Create FOOOF model and fit
        fg = FOOOFGroup(**fooof_params)
        fg.fit(freqs_to_fit, batch_psds)
        
        # Extract periodic parameters for each frequency band
        results = []
        
        for i, vertex_idx in enumerate(vertices):
            for band_name, band_range in freq_bands.items():
                # Get FOOOF model for this vertex
                fm = fg.get_fooof(i)
                
                # Extract peak in this band
                peak_params = get_band_peak_fm(fm, band_range, select_highest=True)
                
                if peak_params is not None:
                    cf, pw, bw = peak_params
                else:
                    cf, pw, bw = np.nan, np.nan, np.nan
                
                results.append({
                    'vertex': vertex_idx,
                    'band': band_name,
                    'center_frequency': cf,
                    'power': pw,
                    'bandwidth': bw
                })
        
        # Clear memory
        del fg, batch_psds
        gc.collect()
        
        return results
    
    # Process in batches to manage memory
    batch_size = 2000  # Adjust based on memory constraints
    vertex_batches = []
    
    for i in range(0, n_vertices, batch_size):
        vertex_batches.append(range(i, min(i + batch_size, n_vertices)))
    
    print(f"Processing {len(vertex_batches)} batches with {n_jobs} parallel jobs...")
    
    # Run in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in vertex_batches
    )
    
    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]
    
    # Convert to DataFrame
    periodic_df = pd.DataFrame(flat_results)
    
    # Add subject_id
    periodic_df.insert(0, 'subject', subject_id)
    
    # Save results
    file_path = os.path.join(output_dir, f"{subject_id}_fooof_periodic.parquet")
    periodic_df.to_csv(os.path.join(output_dir, f"{subject_id}_fooof_periodic.csv"), index=False)
    periodic_df.to_parquet(file_path)
    
    print(f"Saved FOOOF periodic parameters to {file_path}")
    
    return periodic_df, file_path

def calculate_vertex_peak_frequencies(stc, freq_range=(6, 12), alpha_range=(6, 12), n_jobs=10, output_dir=None, subject_id=None, smoothing_method='savitzky_golay'):
    """
    Calculate peak frequencies at the vertex level across the source space.
    
    Parameters
    ----------
    stc : instance of SourceEstimate
        The source time course containing spectral data
    freq_range : tuple
        Frequency range for analysis (default: 6-12 Hz)
    alpha_range : tuple
        Alpha band range for peak finding (default: 6-12 Hz)
    n_jobs : int
        Number of parallel jobs to use for computation
    output_dir : str | None
        Directory to save output files
    subject_id : str | None
        Subject identifier for file naming
    smoothing_method : str
        Method for spectral smoothing ('savitzky_golay', 'moving_average', 'gaussian', 'median')
        
    Returns
    -------
    peaks_df : DataFrame
        DataFrame containing peak frequencies and analysis parameters for each vertex
    file_path : str
        Path to the saved data file
    """
    import os
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.signal import find_peaks, savgol_filter
    from scipy.optimize import curve_fit
    from joblib import Parallel, delayed
    import gc
    
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    if subject_id is None:
        subject_id = 'unknown_subject'
    
    print(f"Calculating vertex-level peak frequencies for {subject_id}...")
    
    # Get data from stc
    if hasattr(stc, 'data') and hasattr(stc, 'times'):
        # Assuming stc.data contains PSDs and stc.times contains frequencies
        psds = stc.data
        freqs = stc.times
    else:
        raise ValueError("Input stc must have 'data' and 'times' attributes with PSDs and frequencies")
    
    # Check if frequencies are within the specified range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(freq_mask):
        raise ValueError(f"No frequencies found within the specified range {freq_range}")
    
    # Define Gaussian function for peak fitting
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # Define function to model 1/f trend
    def model_1f_trend(frequencies, powers):
        log_freq = np.log10(frequencies)
        log_power = np.log10(powers)
        slope, intercept, _, _, _ = stats.linregress(log_freq, log_power)
        return slope * log_freq + intercept
    
    # Define smoothing function
    def enhanced_smooth_spectrum(spectrum, method=smoothing_method):
        if method == 'moving_average':
            window_size = 3
            return np.convolve(spectrum, np.ones(window_size)/window_size, mode='same')
        
        elif method == 'gaussian':
            window_size = 3
            sigma = 1.0
            gaussian_window = np.exp(-(np.arange(window_size) - window_size//2)**2 / (2*sigma**2))
            gaussian_window /= np.sum(gaussian_window)
            return np.convolve(spectrum, gaussian_window, mode='same')
        
        elif method == 'savitzky_golay':
            window_length = 5
            poly_order = 2
            return savgol_filter(spectrum, window_length, poly_order)
        
        elif method == 'median':
            window_size = 3
            return np.array([np.median(spectrum[max(0, i-window_size//2):min(len(spectrum), i+window_size//2+1)]) 
                            for i in range(len(spectrum))])
        
        else:
            return spectrum
    
    # Define function to find alpha peak using Dickinson method for a single vertex
    def dickinson_method_vertex(powers, vertex_idx):
        log_powers = np.log10(powers)
        log_trend = model_1f_trend(freqs, powers)
        detrended_log_powers_raw = log_powers - log_trend
        detrended_log_powers = enhanced_smooth_spectrum(log_powers - log_trend)
        
        # Focus on alpha range
        alpha_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        alpha_freqs = freqs[alpha_mask]
        alpha_powers = detrended_log_powers[alpha_mask]
        
        if len(alpha_freqs) == 0:
            return {"vertex": vertex_idx, "peak_freq": np.nan, "peak_power": np.nan, 
                   "r_squared": np.nan, "status": "NO_DATA_IN_RANGE"}
        
        # Find peaks in the detrended spectrum
        peaks, _ = find_peaks(alpha_powers, width=1)
        
        if len(peaks) == 0:
            return {"vertex": vertex_idx, "peak_freq": np.nan, "peak_power": np.nan, 
                   "r_squared": np.nan, "status": "NO_PEAKS_FOUND"}
        
        # Sort peaks by prominence
        peak_prominences = alpha_powers[peaks] - np.min(alpha_powers)
        sorted_peaks = [p for _, p in sorted(zip(peak_prominences, peaks), reverse=True)]
        
        # Try to fit Gaussian to each peak, starting with the most prominent
        for peak_idx in sorted_peaks:
            peak_freq = alpha_freqs[peak_idx]
            if alpha_range[0] <= peak_freq <= alpha_range[1]:
                try:
                    p0 = [alpha_powers[peak_idx], peak_freq, 0.2]
                    popt, pcov = curve_fit(gaussian, alpha_freqs, alpha_powers, p0=p0, maxfev=1000)
                    
                    if alpha_range[0] <= popt[1] <= alpha_range[1]:
                        # Calculate R-squared for the fit
                        fitted_curve = gaussian(alpha_freqs, *popt)
                        ss_tot = np.sum((alpha_powers - np.mean(alpha_powers))**2)
                        ss_res = np.sum((alpha_powers - fitted_curve)**2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        return {"vertex": vertex_idx, "peak_freq": popt[1], "peak_power": popt[0],
                               "peak_width": popt[2], "r_squared": r_squared, "status": "SUCCESS"}
                except:
                    continue
        
        # If no valid fit found, use the max peak in the alpha range
        alpha_range_mask = (alpha_freqs >= alpha_range[0]) & (alpha_freqs <= alpha_range[1])
        if np.any(alpha_range_mask):
            max_idx = np.argmax(alpha_powers[alpha_range_mask])
            max_peak_freq = alpha_freqs[alpha_range_mask][max_idx]
            max_peak_power = alpha_powers[alpha_range_mask][max_idx]
            
            return {"vertex": vertex_idx, "peak_freq": max_peak_freq, "peak_power": max_peak_power,
                   "peak_width": np.nan, "r_squared": np.nan, "status": "MAX_PEAK_USED"}
        else:
            return {"vertex": vertex_idx, "peak_freq": np.nan, "peak_power": np.nan, 
                   "r_squared": np.nan, "status": "NO_VALID_PEAK"}
    
    # Process vertices in batches for memory efficiency
    def process_batch(vertex_indices):
        batch_results = []
        for i in vertex_indices:
            result = dickinson_method_vertex(psds[i], i)
            batch_results.append(result)
        return batch_results
    
    n_vertices = psds.shape[0]
    print(f"Processing peak frequencies for {n_vertices} vertices...")
    
    # Create batches of vertices
    batch_size = 2000  # Adjust based on memory constraints
    vertex_batches = []
    
    for i in range(0, n_vertices, batch_size):
        vertex_batches.append(range(i, min(i + batch_size, n_vertices)))
    
    # Process batches in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in vertex_batches
    )
    
    # Flatten results
    flat_results = [item for sublist in all_results for item in sublist]
    
    # Create DataFrame
    peaks_df = pd.DataFrame(flat_results)
    
    # Add metadata
    peaks_df['subject'] = subject_id
    peaks_df['freq_range'] = f"{freq_range[0]}-{freq_range[1]}"
    peaks_df['alpha_range'] = f"{alpha_range[0]}-{alpha_range[1]}"
    peaks_df['analysis_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save results
    file_path = os.path.join(output_dir, f"{subject_id}_vertex_peak_frequencies.parquet")
    peaks_df.to_csv(os.path.join(output_dir, f"{subject_id}_vertex_peak_frequencies.csv"), index=False)
    peaks_df.to_parquet(file_path)
    
    print(f"Saved vertex-level peak frequencies to {file_path}")
    
    # Create summary statistics
    success_rate = (peaks_df['status'] == 'SUCCESS').mean() * 100
    avg_peak = peaks_df.loc[peaks_df['peak_freq'].notna(), 'peak_freq'].mean()
    
    print(f"Analysis complete: {success_rate:.1f}% of vertices with successful Gaussian fits")
    print(f"Average peak frequency: {avg_peak:.2f} Hz")
    
    return peaks_df, file_path

def visualize_peak_frequencies(peaks_df, stc_template, subjects_dir=None, 
                               subject='fsaverage', output_dir=None, 
                               colormap='viridis', vmin=None, vmax=None):
    """
    Visualize peak frequencies on a brain surface.
    
    Parameters
    ----------
    peaks_df : DataFrame
        DataFrame with peak frequency results from calculate_vertex_peak_frequencies
    stc_template : instance of SourceEstimate
        Template source estimate to use for visualization
    subjects_dir : str | None
        Path to FreeSurfer subjects directory
    subject : str
        Subject name (default: 'fsaverage')
    output_dir : str | None
        Directory to save output files
    colormap : str
        Colormap for visualization
    vmin, vmax : float | None
        Minimum and maximum values for color scaling
        
    Returns
    -------
    brain : instance of Brain
        The visualization object
    """
    import os
    import numpy as np
    import mne
    
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    vertices = peaks_df['vertex'].values
    peak_freqs = peaks_df['peak_freq'].values
    
    # Set default color limits if not provided
    if vmin is None:
        vmin = np.nanpercentile(peak_freqs, 5)
    if vmax is None:
        vmax = np.nanpercentile(peak_freqs, 95)
    
    # Create a data array of the right size, initialized with NaN
    data = np.ones_like(stc_template.data[:, 0]) * np.nan
    
    # Fill in the peak frequency values
    for vertex, freq in zip(vertices, peak_freqs):
        if not np.isnan(freq):
            data[vertex] = freq
    
    # Create a new SourceEstimate with peak frequency data
    stc_viz = mne.SourceEstimate(
        data[:, np.newaxis],
        vertices=stc_template.vertices,
        tmin=0,
        tstep=1
    )
    
    # Visualize
    brain = stc_viz.plot(
        subject=subject,
        surface='pial',
        hemi='both',
        colormap=colormap,
        clim=dict(kind='value', lims=[vmin, (vmin+vmax)/2, vmax]),
        subjects_dir=subjects_dir,
        title='Peak Frequency Distribution'
    )
    
    # Add a colorbar
    brain.add_annotation('aparc', borders=2, alpha=0.7)
    
    # Save images
    brain.save_image(os.path.join(output_dir, 'peak_frequency_lateral.png'))
    brain.show_view('medial')
    brain.save_image(os.path.join(output_dir, 'peak_frequency_medial.png'))
    
    return brain

