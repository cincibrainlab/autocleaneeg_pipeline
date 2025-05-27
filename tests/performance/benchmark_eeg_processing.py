"""
Comprehensive EEG processing performance benchmarks.

This module provides detailed benchmarking of core EEG processing operations
including filtering, ICA, artifact rejection, and complete pipeline workflows.
"""

import pytest
import time
import psutil
import os
import json
import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch
import tempfile
import shutil
import yaml

from tests.fixtures.synthetic_data import create_synthetic_raw, create_synthetic_events
from tests.fixtures.test_utils import MockOperations

# Only run benchmarks if dependencies are available
pytest.importorskip("autoclean.core.pipeline")

try:
    from autoclean.core.pipeline import Pipeline
    from autoclean.utils.logging import configure_logger
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


class PerformanceProfiler:
    """Utility class for performance profiling and monitoring."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.results = []
    
    def profile_operation(self, operation_name: str, func, *args, **kwargs):
        """Profile a single operation with memory and time tracking."""
        # Get initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_percent = self.process.cpu_percent()
        
        # Execute operation
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Get final state
        end_time = time.time()
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu_percent = self.process.cpu_percent()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = final_memory - initial_memory
        peak_memory = max(initial_memory, final_memory)
        
        # Store results
        profile_result = {
            "operation": operation_name,
            "execution_time": execution_time,
            "memory_initial": initial_memory,
            "memory_final": final_memory,
            "memory_delta": memory_delta,
            "peak_memory": peak_memory,
            "cpu_percent": final_cpu_percent,
            "success": success,
            "error": error,
            "timestamp": time.time()
        }
        
        self.results.append(profile_result)
        return result, profile_result
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from all profiled operations."""
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful operations to analyze"}
        
        # Calculate aggregate statistics
        total_time = sum(r["execution_time"] for r in successful_results)
        total_memory_delta = sum(r["memory_delta"] for r in successful_results)
        peak_memory = max(r["peak_memory"] for r in successful_results)
        avg_cpu = np.mean([r["cpu_percent"] for r in successful_results])
        
        return {
            "total_operations": len(self.results),
            "successful_operations": len(successful_results),
            "total_execution_time": total_time,
            "total_memory_delta": total_memory_delta,
            "peak_memory_usage": peak_memory,
            "average_cpu_percent": avg_cpu,
            "operations_per_second": len(successful_results) / total_time if total_time > 0 else 0
        }


@pytest.mark.benchmark
@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Pipeline module not available")
class TestEEGProcessingBenchmarks:
    """Comprehensive benchmarks for EEG processing operations."""
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler instance."""
        return PerformanceProfiler()
    
    @pytest.fixture
    def benchmark_workspace(self):
        """Create temporary workspace for benchmark testing."""
        temp_dir = tempfile.mkdtemp(prefix="autoclean_benchmark_")
        workspace = Path(temp_dir)
        
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "config").mkdir()
        
        yield workspace
        
        shutil.rmtree(workspace, ignore_errors=True)
    
    @pytest.fixture
    def benchmark_config(self, benchmark_workspace):
        """Create optimized configuration for benchmarking."""
        config = {
            "eeg_system": {
                "montage": "GSN-HydroCel-129",
                "reference": "average",
                "sampling_rate": 250
            },
            "signal_processing": {
                "filter": {"highpass": 0.1, "lowpass": 50.0},
                "ica": {"n_components": 15, "max_iter": 100},
                "autoreject": {"n_interpolate": [1, 4, 8]}
            },
            "performance": {
                "enable_profiling": True,
                "optimize_memory": True
            }
        }
        
        config_path = benchmark_workspace / "config" / "benchmark_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_synthetic_data_generation_performance(self, profiler, benchmark):
        """Benchmark synthetic EEG data generation performance."""
        def generate_data():
            return create_synthetic_raw(
                montage="GSN-HydroCel-129",
                n_channels=129,
                duration=60.0,
                sfreq=250.0,
                seed=42
            )
        
        # Benchmark with pytest-benchmark
        result = benchmark(generate_data)
        
        # Also profile manually for detailed metrics
        _, profile_result = profiler.profile_operation(
            "synthetic_data_generation", generate_data
        )
        
        # Verify data was generated successfully
        assert result is not None
        assert result.info['nchan'] == 129
        assert len(result.times) == int(60.0 * 250.0)
        
        # Performance assertions
        assert profile_result["execution_time"] < 10.0, "Data generation should complete in <10s"
        assert profile_result["memory_delta"] < 200, "Memory usage should be <200MB"
    
    def test_filtering_performance(self, profiler, benchmark):
        """Benchmark EEG filtering performance."""
        # Create test data
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=60.0,
            sfreq=250.0,
            seed=42
        )
        
        def apply_filter():
            # Copy to avoid modifying original
            raw_copy = raw.copy()
            return raw_copy.filter(l_freq=0.1, h_freq=50.0, verbose=False)
        
        # Benchmark filtering operation
        result = benchmark(apply_filter)
        
        # Profile manually for detailed metrics
        _, profile_result = profiler.profile_operation(
            "eeg_filtering", apply_filter
        )
        
        # Verify filtering was applied
        assert result is not None
        
        # Performance assertions
        assert profile_result["execution_time"] < 30.0, "Filtering should complete in <30s"
        assert profile_result["memory_delta"] < 500, "Memory usage should be reasonable"
    
    def test_ica_performance_mocked(self, profiler, benchmark):
        """Benchmark ICA performance (mocked for speed)."""
        # Create test data
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=60.0,
            sfreq=250.0,
            seed=42
        )
        
        def run_mock_ica():
            # Mock ICA for performance testing
            return MockOperations.mock_ica(raw, n_components=15)
        
        # Benchmark mocked ICA
        result = benchmark(run_mock_ica)
        
        # Profile manually
        _, profile_result = profiler.profile_operation(
            "mock_ica", run_mock_ica
        )
        
        # Verify ICA result
        assert result is not None
        
        # Performance assertions (mocked should be very fast)
        assert profile_result["execution_time"] < 1.0, "Mocked ICA should be very fast"
    
    def test_epochs_creation_performance(self, profiler, benchmark):
        """Benchmark epochs creation performance."""
        # Create test data with events
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=120.0,  # Longer for more epochs
            sfreq=250.0,
            seed=42
        )
        
        # Add events
        n_events = 50
        event_times = np.linspace(5, 115, n_events)  # Events every ~2 seconds
        events = np.array([
            [int(t * raw.info['sfreq']), 0, 1] 
            for t in event_times
        ])
        
        def create_epochs():
            return mne.Epochs(
                raw, events, event_id={'stimulus': 1},
                tmin=-0.2, tmax=0.8, baseline=(-0.2, 0.0),
                verbose=False, preload=True
            )
        
        # Benchmark epochs creation
        result = benchmark(create_epochs)
        
        # Profile manually
        _, profile_result = profiler.profile_operation(
            "epochs_creation", create_epochs
        )
        
        # Verify epochs were created
        assert result is not None
        assert len(result) > 0
        
        # Performance assertions
        assert profile_result["execution_time"] < 60.0, "Epochs creation should complete in <60s"
    
    def test_complete_pipeline_performance(self, profiler, benchmark_workspace, benchmark_config):
        """Benchmark complete pipeline performance (mocked)."""
        # Create input data
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=60.0,
            sfreq=250.0,
            seed=42
        )
        
        input_file = benchmark_workspace / "input" / "benchmark_test.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        
        # Configure logging
        configure_logger(verbose="ERROR", output_dir=benchmark_workspace)
        
        def run_complete_pipeline():
            with patch.multiple(
                'autoclean.mixins.signal_processing.ica.IcaMixin',
                run_ica=MockOperations.mock_ica,
                apply_ica=MockOperations.mock_apply_ica
            ), patch.multiple(
                'autoclean.mixins.signal_processing.autoreject_epochs.AutorejectEpochsMixin',
                run_autoreject=MockOperations.mock_autoreject,
                apply_autoreject=MockOperations.mock_apply_autoreject
            ):
                pipeline = Pipeline(
                    autoclean_dir=benchmark_workspace / "output",
                    autoclean_config=benchmark_config,
                    verbose="ERROR"
                )
                
                return pipeline.process_file(
                    file_path=input_file,
                    task="RestingEyesOpen"
                )
        
        # Profile complete pipeline
        result, profile_result = profiler.profile_operation(
            "complete_pipeline_mocked", run_complete_pipeline
        )
        
        # Verify pipeline completed
        assert result is not None or profile_result["success"], "Pipeline should complete successfully"
        
        # Performance assertions (with mocking should be fast)
        assert profile_result["execution_time"] < 120.0, "Complete pipeline should complete in <2min"
        assert profile_result["memory_delta"] < 1000, "Memory usage should be reasonable"
    
    def test_memory_efficiency_scaling(self, profiler):
        """Test memory efficiency with different data sizes."""
        data_sizes = [
            (30, 30.0),   # Small: 30 channels, 30 seconds
            (64, 60.0),   # Medium: 64 channels, 60 seconds
            (129, 120.0), # Large: 129 channels, 120 seconds
        ]
        
        memory_results = []
        
        for n_channels, duration in data_sizes:
            def create_and_process():
                raw = create_synthetic_raw(
                    montage="GSN-HydroCel-129" if n_channels == 129 else "standard_1020",
                    n_channels=n_channels,
                    duration=duration,
                    sfreq=250.0,
                    seed=42
                )
                
                # Simple processing operation
                return raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
            
            _, profile_result = profiler.profile_operation(
                f"memory_scaling_{n_channels}ch_{duration}s", create_and_process
            )
            
            memory_results.append({
                "n_channels": n_channels,
                "duration": duration,
                "memory_delta": profile_result["memory_delta"],
                "execution_time": profile_result["execution_time"]
            })
        
        # Verify memory scaling is reasonable
        for i in range(1, len(memory_results)):
            prev_result = memory_results[i-1]
            curr_result = memory_results[i]
            
            # Memory should scale reasonably with data size
            size_ratio = (curr_result["n_channels"] * curr_result["duration"]) / \
                        (prev_result["n_channels"] * prev_result["duration"])
            memory_ratio = curr_result["memory_delta"] / max(prev_result["memory_delta"], 1)
            
            # Memory scaling should be less than quadratic
            assert memory_ratio < size_ratio ** 2, f"Memory scaling too aggressive: {memory_ratio} vs {size_ratio}"


@pytest.mark.benchmark
@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Pipeline module not available") 
class TestPerformanceRegression:
    """Tests for detecting performance regressions."""
    
    def test_baseline_performance_metrics(self, tmp_path):
        """Establish baseline performance metrics for regression detection."""
        profiler = PerformanceProfiler()
        
        # Standard benchmark operations
        baseline_operations = [
            ("data_generation", lambda: create_synthetic_raw(
                montage="GSN-HydroCel-129", n_channels=129, duration=60.0, sfreq=250.0
            )),
            ("filtering", lambda: create_synthetic_raw(
                montage="GSN-HydroCel-129", n_channels=129, duration=30.0, sfreq=250.0
            ).filter(l_freq=0.1, h_freq=50.0, verbose=False)),
            ("mock_ica", lambda: MockOperations.mock_ica(
                create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129, duration=30.0, sfreq=250.0),
                n_components=15
            ))
        ]
        
        baseline_results = {}
        
        for op_name, operation in baseline_operations:
            _, profile_result = profiler.profile_operation(op_name, operation)
            baseline_results[op_name] = {
                "execution_time": profile_result["execution_time"],
                "memory_delta": profile_result["memory_delta"],
                "success": profile_result["success"]
            }
        
        # Save baseline metrics
        baseline_file = tmp_path / "performance_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        # Verify all operations completed successfully
        for op_name, result in baseline_results.items():
            assert result["success"], f"Baseline operation {op_name} should succeed"
        
        # Verify reasonable performance bounds
        assert baseline_results["data_generation"]["execution_time"] < 15.0
        assert baseline_results["filtering"]["execution_time"] < 45.0
        assert baseline_results["mock_ica"]["execution_time"] < 2.0
        
        return baseline_results
    
    def test_performance_stability(self):
        """Test performance stability across multiple runs."""
        profiler = PerformanceProfiler()
        
        # Run same operation multiple times
        def test_operation():
            raw = create_synthetic_raw(
                montage="GSN-HydroCel-129",
                n_channels=129,
                duration=30.0,
                sfreq=250.0,
                seed=42
            )
            return raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        
        execution_times = []
        memory_deltas = []
        
        # Run operation 5 times
        for i in range(5):
            _, profile_result = profiler.profile_operation(
                f"stability_test_{i}", test_operation
            )
            execution_times.append(profile_result["execution_time"])
            memory_deltas.append(profile_result["memory_delta"])
        
        # Calculate coefficient of variation (std/mean)
        time_cv = np.std(execution_times) / np.mean(execution_times)
        memory_cv = np.std(memory_deltas) / max(np.mean(memory_deltas), 1)
        
        # Performance should be relatively stable (CV < 50%)
        assert time_cv < 0.5, f"Execution time too variable: CV={time_cv:.2f}"
        assert memory_cv < 0.5, f"Memory usage too variable: CV={memory_cv:.2f}"


def generate_performance_report(profiler: PerformanceProfiler, output_file: Path):
    """Generate comprehensive performance report."""
    summary = profiler.get_summary_stats()
    
    report = {
        "timestamp": time.time(),
        "summary": summary,
        "detailed_results": profiler.results,
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


if __name__ == "__main__":
    import sys
    
    # Run performance benchmarks and generate report
    profiler = PerformanceProfiler()
    
    print("🚀 Running AutoClean EEG Performance Benchmarks...")
    
    # Run key benchmarks
    operations = [
        ("Data Generation", lambda: create_synthetic_raw(
            montage="GSN-HydroCel-129", n_channels=129, duration=60.0, sfreq=250.0
        )),
        ("Filtering", lambda: create_synthetic_raw(
            montage="GSN-HydroCel-129", n_channels=129, duration=30.0, sfreq=250.0
        ).filter(l_freq=0.1, h_freq=50.0, verbose=False)),
        ("Mock ICA", lambda: MockOperations.mock_ica(
            create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129, duration=30.0, sfreq=250.0),
            n_components=15
        ))
    ]
    
    for op_name, operation in operations:
        print(f"  📊 Benchmarking {op_name}...")
        _, result = profiler.profile_operation(op_name.lower().replace(" ", "_"), operation)
        print(f"     ⏱️  Time: {result['execution_time']:.2f}s")
        print(f"     💾 Memory: {result['memory_delta']:.1f}MB")
    
    # Generate report
    report_file = Path("performance_report.json")
    report = generate_performance_report(profiler, report_file)
    
    print(f"\n📈 Performance Report Generated: {report_file}")
    print(f"   Total Operations: {report['summary'].get('total_operations', 0)}")
    print(f"   Success Rate: {report['summary'].get('successful_operations', 0)}/{report['summary'].get('total_operations', 0)}")
    print(f"   Total Time: {report['summary'].get('total_execution_time', 0):.2f}s")
    print(f"   Peak Memory: {report['summary'].get('peak_memory_usage', 0):.1f}MB")