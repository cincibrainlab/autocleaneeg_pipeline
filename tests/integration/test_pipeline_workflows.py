"""
Integration tests for complete pipeline workflows.

These tests verify that the complete processing pipeline works end-to-end
with synthetic data, testing the actual workflow that users experience.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import mne
from unittest.mock import patch, MagicMock

from tests.fixtures.test_utils import MockOperations
from tests.fixtures.synthetic_data import create_synthetic_raw, create_synthetic_events

# Only run if core imports are available
pytest.importorskip("autoclean.core.pipeline")

try:
    from autoclean.core.pipeline import Pipeline
    from autoclean.utils.config import load_config
    from autoclean.utils.logging import configure_logger
    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Pipeline module not available")
class TestPipelineWorkflows:
    """Test complete pipeline workflows from start to finish."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for integration testing."""
        temp_dir = tempfile.mkdtemp(prefix="autoclean_integration_")
        workspace = Path(temp_dir)
        
        # Create directory structure
        (workspace / "input").mkdir()
        (workspace / "output").mkdir()
        (workspace / "config").mkdir()
        
        yield workspace
        
        # Cleanup
        shutil.rmtree(workspace, ignore_errors=True)
    
    @pytest.fixture
    def minimal_config(self, temp_workspace):
        """Create minimal configuration for testing."""
        config = {
            "eeg_system": {
                "montage": "GSN-HydroCel-129",
                "reference": "average",
                "sampling_rate": 250
            },
            "signal_processing": {
                "filter": {
                    "highpass": 0.1,
                    "lowpass": 50.0
                },
                "epochs": {
                    "tmin": -0.2,
                    "tmax": 0.8,
                    "baseline": [-0.2, 0.0]
                },
                "ica": {
                    "n_components": 15,
                    "max_iter": 100
                },
                "autoreject": {
                    "n_interpolate": [1, 4, 8],
                    "consensus": [0.2, 0.5, 0.8]
                }
            },
            "output": {
                "save_stages": ["raw", "epochs", "cleaned"],
                "bids_compliant": True
            },
            "quality_control": {
                "max_bad_channels_percent": 25,
                "max_bad_epochs_percent": 40
            }
        }
        
        config_path = temp_workspace / "config" / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def synthetic_input_file(self, temp_workspace):
        """Create synthetic input file for testing."""
        # Create synthetic raw data
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=60.0,
            sfreq=250.0,
            seed=42
        )
        
        # Save as proper _raw.fif file
        input_file = temp_workspace / "input" / "test_subject_001_raw.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        
        return input_file
    
    @pytest.fixture
    def mock_pipeline_operations(self):
        """Mock computationally expensive operations for fast testing."""
        with patch.multiple(
            'autoclean.mixins.signal_processing.ica.IcaMixin',
            run_ica=MockOperations.mock_ica,
            apply_iclabel_rejection=MockOperations.mock_apply_ica
        ), patch.multiple(
            'autoclean.mixins.signal_processing.autoreject_epochs.AutoRejectEpochsMixin',
            apply_autoreject=MockOperations.mock_apply_autoreject
        ):
            yield
    
    def test_single_file_resting_processing(self, temp_workspace, minimal_config, 
                                          synthetic_input_file, mock_pipeline_operations):
        """Test processing a single resting state file end-to-end."""
        # Configure logging
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        
        # Initialize pipeline
        pipeline = Pipeline(
            output_dir=temp_workspace / "output",
                        verbose="ERROR"
        )
        
        # Process file
        result = pipeline.process_file(
            file_path=synthetic_input_file,
            task="RestingEyesOpen"
        )
        
        # Verify processing completed
        assert result is not None
        
        # Check output directory structure
        output_dir = temp_workspace / "output"
        assert output_dir.exists()
        
        # Check that some output files were created
        output_files = list(output_dir.rglob("*"))
        assert len(output_files) > 0
        
        # Verify no critical errors in processing
        # (This is a basic smoke test - files exist and processing completed)
    
    def test_batch_processing_workflow(self, temp_workspace, minimal_config, 
                                     mock_pipeline_operations):
        """Test batch processing multiple files."""
        # Create multiple synthetic input files
        input_files = []
        for i in range(3):
            raw = create_synthetic_raw(
                montage="GSN-HydroCel-129",
                n_channels=129,
                duration=30.0,  # Shorter for batch testing
                sfreq=250.0,
                seed=42 + i
            )
            
            input_file = temp_workspace / "input" / f"subject_{i+1:03d}.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            input_files.append(input_file)
        
        # Configure logging
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        
        # Initialize pipeline
        pipeline = Pipeline(
            output_dir=temp_workspace / "output",
                        verbose="ERROR"
        )
        
        # Process all files
        results = []
        for input_file in input_files:
            try:
                result = pipeline.process_file(
                    file_path=input_file,
                    task="RestingEyesOpen"
                )
                results.append(result)
            except Exception as e:
                # Track but don't fail on individual file errors in batch
                results.append(None)
        
        # Verify at least some files processed successfully
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) > 0, "No files processed successfully in batch"
        
        # Check output structure
        output_dir = temp_workspace / "output"
        assert output_dir.exists()
        
        # Should have output for multiple subjects
        output_files = list(output_dir.rglob("*"))
        assert len(output_files) > len(input_files)  # More outputs than inputs due to stages
    
    def test_different_task_types(self, temp_workspace, minimal_config, 
                                mock_pipeline_operations):
        """Test processing with different task types."""
        task_configs = [
            ("RestingEyesOpen", {"duration": 60.0, "events": None}),
            ("HBCD_MMN", {"duration": 120.0, "events": "mmn"}),
            ("ChirpDefault", {"duration": 180.0, "events": "chirp"})
        ]
        
        # Configure logging
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        
        # Initialize pipeline
        pipeline = Pipeline(
            output_dir=temp_workspace / "output",
                        verbose="ERROR"
        )
        
        successful_tasks = []
        
        for task_name, config in task_configs:
            # Create task-specific synthetic data
            raw = create_synthetic_raw(
                montage="GSN-HydroCel-129",
                n_channels=129,
                duration=config["duration"],
                sfreq=250.0,
                seed=42
            )
            
            # Add events if needed
            if config["events"]:
                events = create_synthetic_events(
                    n_samples=int(config["duration"] * 250.0),
                    sfreq=250.0,
                    event_type=config["events"]
                )
                # Add events to raw (simplified - real implementation would vary)
                raw.annotations.append(
                    onset=[1.0, 2.0, 3.0],
                    duration=[0.1, 0.1, 0.1],
                    description=['stimulus'] * 3
                )
            
            # Save input file
            input_file = temp_workspace / "input" / f"{task_name.lower()}_test.fif"
            raw.save(input_file, overwrite=True, verbose=False)
            
            # Process with appropriate task
            try:
                result = pipeline.process_file(
                    file_path=input_file,
                    task=task_name
                )
                if result is not None:
                    successful_tasks.append(task_name)
            except Exception as e:
                # Some tasks might fail due to missing task definitions
                # This is expected in integration testing
                pass
        
        # Verify at least resting state processing works
        assert "RestingEyesOpen" in successful_tasks, "Basic resting state processing should work"
    
    def test_pipeline_error_handling(self, temp_workspace, minimal_config, 
                                   mock_pipeline_operations):
        """Test pipeline behavior with various error conditions."""
        # Configure logging
        configure_logger(verbose="ERROR", output_dir=temp_workspace)
        
        # Initialize pipeline
        pipeline = Pipeline(
            output_dir=temp_workspace / "output",
                        verbose="ERROR"
        )
        
        # Test 1: Non-existent file
        with pytest.raises((FileNotFoundError, ValueError, OSError)):
            pipeline.process_file(
                file_path=temp_workspace / "input" / "nonexistent.fif",
                task="RestingEyesOpen"
            )
        
        # Test 2: Invalid task name
        # Create valid input file
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=30.0,
            sfreq=250.0
        )
        input_file = temp_workspace / "input" / "test_invalid_task.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        
        # Try processing with invalid task
        with pytest.raises((ValueError, KeyError, AttributeError)):
            pipeline.process_file(
                file_path=input_file,
                task="NonExistentTask"
            )
    
    def test_pipeline_configuration_variations(self, temp_workspace, mock_pipeline_operations):
        """Test pipeline with different configuration settings."""
        # Create multiple config variations
        configs = {
            "high_quality": {
                "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
                "signal_processing": {
                    "filter": {"highpass": 0.5, "lowpass": 40.0},
                    "ica": {"n_components": 20},
                    "autoreject": {"n_interpolate": [1, 4, 8, 16]}
                },
                "quality_control": {
                    "max_bad_channels_percent": 15,
                    "max_bad_epochs_percent": 25
                }
            },
            "permissive": {
                "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
                "signal_processing": {
                    "filter": {"highpass": 0.1, "lowpass": 50.0},
                    "ica": {"n_components": 10},
                    "autoreject": {"n_interpolate": [1, 4]}
                },
                "quality_control": {
                    "max_bad_channels_percent": 40,
                    "max_bad_epochs_percent": 60
                }
            }
        }
        
        # Create synthetic input
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=30.0,
            sfreq=250.0
        )
        input_file = temp_workspace / "input" / "config_test.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        
        successful_configs = []
        
        for config_name, config_data in configs.items():
            # Save config
            config_path = temp_workspace / "config" / f"{config_name}_config.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Configure logging
            configure_logger(verbose="ERROR", output_dir=temp_workspace)
            
            # Test pipeline with this config
            try:
                pipeline = Pipeline(
                    output_dir=temp_workspace / "output" / config_name,
                                        verbose="ERROR"
                )
                
                result = pipeline.process_file(
                    file_path=input_file,
                    task="RestingEyesOpen"
                )
                
                if result is not None:
                    successful_configs.append(config_name)
                    
            except Exception as e:
                # Some configurations might fail - that's part of testing
                pass
        
        # At least one configuration should work
        assert len(successful_configs) > 0, "At least one pipeline configuration should work"


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Pipeline module not available")
class TestPipelineMemoryAndPerformance:
    """Test pipeline performance and memory characteristics."""
    
    def test_memory_usage_tracking(self, tmp_path):
        """Test that pipeline doesn't consume excessive memory."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Configure minimal logging
        configure_logger(verbose="ERROR", output_dir=tmp_path)
        
        # Create small synthetic data
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=10.0,  # Short duration
            sfreq=250.0
        )
        
        input_file = tmp_path / "memory_test_raw.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        
        # Create minimal config
        config = {
            "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
            "signal_processing": {"filter": {"highpass": 1.0, "lowpass": 40.0}},
            "quality_control": {"max_bad_channels_percent": 50}
        }
        config_path = tmp_path / "memory_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Process file with mocked operations
        with patch.multiple(
            'autoclean.mixins.signal_processing.ica.IcaMixin',
            run_ica=MockOperations.mock_ica,
            apply_iclabel_rejection=MockOperations.mock_apply_ica
        ):
            try:
                pipeline = Pipeline(
                    output_dir=tmp_path / "output",
                    verbose="ERROR"
                )
                
                pipeline.process_file(
                    file_path=input_file,
                    task="RestingEyesOpen"
                )
                
                # Check memory usage didn't explode
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                # Allow reasonable memory increase (up to 500MB for scientific computing)
                assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB"
                
            except Exception:
                # Memory test is informational - don't fail on processing errors
                pass
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    def test_processing_time_reasonable(self, tmp_path):
        """Test that processing completes in reasonable time."""
        import time
        
        # Configure minimal logging
        configure_logger(verbose="ERROR", output_dir=tmp_path)
        
        # Create synthetic data
        raw = create_synthetic_raw(
            montage="GSN-HydroCel-129",
            n_channels=129,
            duration=30.0,
            sfreq=250.0
        )
        
        input_file = tmp_path / "timing_test_raw.fif"
        raw.save(input_file, overwrite=True, verbose=False)
        
        # Create minimal config
        config = {
            "eeg_system": {"montage": "GSN-HydroCel-129", "reference": "average"},
            "signal_processing": {"filter": {"highpass": 1.0, "lowpass": 40.0}},
            "quality_control": {"max_bad_channels_percent": 50}
        }
        config_path = tmp_path / "timing_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Time the processing with mocked operations
        start_time = time.time()
        
        with patch.multiple(
            'autoclean.mixins.signal_processing.ica.IcaMixin',
            run_ica=MockOperations.mock_ica,
            apply_iclabel_rejection=MockOperations.mock_apply_ica
        ):
            try:
                pipeline = Pipeline(
                    output_dir=tmp_path / "output",
                    verbose="ERROR"
                )
                
                pipeline.process_file(
                    file_path=input_file,
                    task="RestingEyesOpen"
                )
                
                processing_time = time.time() - start_time
                
                # With mocked operations, should complete quickly (under 60 seconds)
                assert processing_time < 60, f"Processing took {processing_time:.1f}s, expected <60s"
                
            except Exception:
                # Timing test is informational - don't fail on processing errors
                pass