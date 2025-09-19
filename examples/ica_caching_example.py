#!/usr/bin/env python3
"""Example demonstrating ICA sources caching for improved performance.

This example shows how the new caching system dramatically improves performance
when generating multiple ICA reports by avoiding redundant source computations.
"""

import time
import numpy as np
import mne
from mne.preprocessing import ICA

# Import the caching functions
from autoclean.mixins.viz._ica_sources_cache import (
    get_cached_ica_sources,
    get_ica_cache_stats,
    clear_ica_cache
)


def create_sample_data():
    """Create sample EEG data for demonstration."""
    # Create synthetic raw data
    sfreq = 500  # Sampling frequency
    n_channels = 32
    n_times = int(60 * sfreq)  # 60 seconds of data
    
    # Generate random data
    data = np.random.randn(n_channels, n_times) * 1e-6
    
    # Create channel names
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    
    # Create info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Create Raw object
    raw = mne.io.RawArray(data, info)
    
    return raw


def create_fitted_ica(raw):
    """Create and fit ICA on the raw data."""
    ica = ICA(n_components=15, method='fastica', random_state=42)
    ica.fit(raw)
    return ica


def benchmark_without_cache(ica, raw, n_components=10):
    """Benchmark ICA sources computation without caching."""
    print("🔄 Benchmarking WITHOUT caching...")
    
    start_time = time.time()
    
    # Simulate generating reports for multiple components
    for i in range(n_components):
        # This is what happens in the old code - repeated get_sources calls
        sources = ica.get_sources(raw)
        component_data = sources.get_data(picks=[i])
        
        # Simulate some processing
        _ = np.mean(component_data)
    
    elapsed = time.time() - start_time
    print(f"   Time without cache: {elapsed:.3f} seconds")
    return elapsed


def benchmark_with_cache(ica, raw, n_components=10):
    """Benchmark ICA sources computation with caching."""
    print("⚡ Benchmarking WITH caching...")
    
    # Clear cache first
    clear_ica_cache()
    
    start_time = time.time()
    
    # Simulate generating reports for multiple components
    for i in range(n_components):
        # This uses the new cached approach
        sources = get_cached_ica_sources(ica, raw)
        component_data = sources.get_data(picks=[i])
        
        # Simulate some processing
        _ = np.mean(component_data)
    
    elapsed = time.time() - start_time
    print(f"   Time with cache: {elapsed:.3f} seconds")
    
    # Show cache statistics
    stats = get_ica_cache_stats()
    print(f"   Cache entries: {stats['entries']}")
    print(f"   Cache size: {stats['total_size_mb']:.1f} MB")
    
    return elapsed


def demonstrate_memory_management():
    """Demonstrate automatic memory management."""
    print("\n🧠 Demonstrating memory management...")
    
    # Clear cache
    clear_ica_cache()
    
    # Create multiple raw objects of different sizes
    raw_small = create_sample_data()  # 60 seconds
    
    # Create larger dataset
    sfreq = 500
    n_channels = 32
    n_times = int(300 * sfreq)  # 5 minutes of data
    data_large = np.random.randn(n_channels, n_times) * 1e-6
    ch_names = [f'EEG{i:03d}' for i in range(n_channels)]
    info_large = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw_large = mne.io.RawArray(data_large, info_large)
    
    ica = create_fitted_ica(raw_small)
    
    print("   Caching small dataset...")
    _ = get_cached_ica_sources(ica, raw_small)
    stats = get_ica_cache_stats()
    print(f"   Cache after small dataset: {stats['total_size_mb']:.1f} MB")
    
    print("   Caching large dataset...")
    _ = get_cached_ica_sources(ica, raw_large)
    stats = get_ica_cache_stats()
    print(f"   Cache after large dataset: {stats['total_size_mb']:.1f} MB")
    print(f"   Cache utilization: {stats['utilization_percent']:.1f}%")


def main():
    """Main demonstration function."""
    print("🔬 ICA Sources Caching Performance Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample EEG data...")
    raw = create_sample_data()
    
    print("🧮 Fitting ICA...")
    ica = create_fitted_ica(raw)
    
    print(f"✅ Created {ica.n_components_} ICA components")
    print(f"📈 Raw data: {raw.n_times} samples, {len(raw.ch_names)} channels")
    
    # Benchmark both approaches
    print("\n🏁 Performance Comparison:")
    time_without = benchmark_without_cache(ica, raw, n_components=10)
    time_with = benchmark_with_cache(ica, raw, n_components=10)
    
    # Calculate speedup
    speedup = time_without / time_with if time_with > 0 else float('inf')
    print(f"\n🚀 Speedup: {speedup:.1f}x faster with caching!")
    
    # Demonstrate memory management
    demonstrate_memory_management()
    
    # Practical usage example
    print("\n💡 Practical Usage in AutoClean:")
    print("""
    # In your task class:
    
    def generate_ica_reports(self):
        # The cache automatically speeds up these operations:
        self.plot_ica_full()              # Uses cached sources
        self.generate_ica_reports()       # Uses cached sources per component
        
        # Monitor cache performance:
        self.log_cache_performance()
        
        # Clear cache if needed:
        self.clear_ica_sources_cache()
    """)
    
    print("🎯 Key Benefits:")
    print("  • Automatic caching with no code changes needed")
    print("  • Smart memory management with LRU eviction")  
    print("  • Thread-safe for parallel processing")
    print("  • Automatic cache invalidation when ICA changes")
    print("  • Configurable memory limits")


if __name__ == "__main__":
    main()
