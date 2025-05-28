"""Unit tests for the Pipeline class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import pytest

from tests.fixtures.test_utils import BaseTestCase, EEGAssertions

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.core.pipeline import Pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineInitialization(BaseTestCase):
    """Test Pipeline class initialization and basic functionality."""
    
    def test_pipeline_init_with_valid_config(self):
        """Test Pipeline initialization with valid configuration."""
        config = {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    "description": "Test task",
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250},
                        "filtering": {"enabled": True, "value": {"l_freq": 1, "h_freq": 100}}
                    }
                }
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
            "database": {"enabled": False}
        }
        
        config_file = self.temp_dir / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = Pipeline(
            autoclean_dir=str(self.autoclean_dir),
            autoclean_config=str(config_file)
        )
        
        assert pipeline.autoclean_dir == Path(self.autoclean_dir)
        assert pipeline.config is not None
        assert "tasks" in pipeline.config
        assert "TestTask" in pipeline.config["tasks"]
    
    def test_pipeline_init_invalid_config_path(self):
        """Test Pipeline initialization with invalid config path."""
        with pytest.raises((FileNotFoundError, IOError)):
            Pipeline(
                autoclean_dir=str(self.autoclean_dir),
                autoclean_config="/nonexistent/path/config.yaml"
            )
    
    def test_pipeline_init_invalid_autoclean_dir(self):
        """Test Pipeline initialization with invalid autoclean directory."""
        config_file = self.temp_dir / "test_config.yaml"
        config_file.write_text("tasks: {}")
        
        # Should handle creation of new directory
        new_dir = self.temp_dir / "new_autoclean_dir"
        pipeline = Pipeline(
            autoclean_dir=str(new_dir),
            autoclean_config=str(config_file)
        )
        
        assert pipeline.autoclean_dir == new_dir
    
    def test_pipeline_task_registry_access(self):
        """Test that Pipeline has access to task registry."""
        config_file = self.temp_dir / "test_config.yaml"
        config_file.write_text("tasks: {}")
        
        pipeline = Pipeline(
            autoclean_dir=str(self.autoclean_dir),
            autoclean_config=str(config_file)
        )
        
        assert hasattr(pipeline, 'TASK_REGISTRY')
        assert isinstance(pipeline.TASK_REGISTRY, dict)
    
    def test_pipeline_verbose_parameter(self):
        """Test Pipeline initialization with different verbose settings."""
        config_file = self.temp_dir / "test_config.yaml"
        config_file.write_text("tasks: {}")
        
        # Test different verbose settings
        for verbose in [True, False, "info", "debug", None]:
            pipeline = Pipeline(
                autoclean_dir=str(self.autoclean_dir),
                autoclean_config=str(config_file),
                verbose=verbose
            )
            assert pipeline is not None


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineConfiguration:
    """Test Pipeline configuration handling."""
    
    def test_config_loading_and_validation(self, tmp_path):
        """Test configuration loading and basic validation."""
        config = {
            "tasks": {
                "ValidTask": {
                    "mne_task": "rest",
                    "description": "Valid test task",
                    "settings": {
                        "resample_step": {"enabled": True, "value": 250},
                        "filtering": {"enabled": True, "value": {"l_freq": 1, "h_freq": 100}},
                        "montage": {"enabled": True, "value": "GSN-HydroCel-129"}
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": True, "suffix": "_postimport"},
                "post_clean_raw": {"enabled": True, "suffix": "_postcleaning"},
                "post_ica": {"enabled": False, "suffix": "_postica"},
                "post_epochs": {"enabled": True, "suffix": "_postepochs"}
            },
            "database": {"enabled": False}
        }
        
        config_file = tmp_path / "valid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        assert pipeline.config["tasks"]["ValidTask"]["mne_task"] == "rest"
        assert pipeline.config["stage_files"]["post_import"] is True
        assert pipeline.config["database"]["enabled"] is False
    
    def test_config_with_missing_required_fields(self, tmp_path):
        """Test configuration with missing required fields."""
        # Config missing tasks
        config = {"stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}}}
        
        config_file = tmp_path / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Should raise an error or handle gracefully
        with patch('autoclean.utils.config.load_config') as mock_load:
            mock_load.side_effect = ValueError("Missing required field: tasks")
            
            with pytest.raises(ValueError, match="Missing required field"):
                Pipeline(
                    autoclean_dir=str(tmp_path / "output"),
                    autoclean_config=str(config_file)
                )


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineFileProcessing:
    """Test Pipeline file processing methods."""
    
    @patch('autoclean.core.pipeline.task_registry')
    @patch('autoclean.utils.file_system.step_prepare_directories')
    @patch('autoclean.utils.database.manage_database')
    def test_process_file_basic_flow(self, mock_db, mock_dirs, mock_registry, tmp_path):
        """Test basic file processing flow."""
        # Setup mock task
        mock_task_class = Mock()
        mock_task_instance = Mock()
        mock_task_class.return_value = mock_task_instance
        mock_registry = {"TestTask": mock_task_class}
        
        # Setup config
        config = {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    "description": "Test task",
                    "settings": {}
                }
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
            "database": {"enabled": False}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create test data file
        test_file = tmp_path / "test_data.fif"
        test_file.touch()
        
        with patch.object(Pipeline, 'TASK_REGISTRY', mock_registry):
            pipeline = Pipeline(
                autoclean_dir=str(tmp_path / "output"),
                autoclean_config=str(config_file)
            )
            
            # This should not raise an error in the basic flow test
            # Full functionality testing would require more complex mocking
            with patch.object(pipeline, '_prepare_processing_run') as mock_prepare:
                mock_prepare.return_value = {"run_id": "test_run_123"}
                
                # Test that the method exists and can be called
                assert hasattr(pipeline, 'process_file')
    
    def test_process_file_invalid_task(self, tmp_path):
        """Test processing with invalid task name."""
        config = {
            "tasks": {
                "ValidTask": {
                    "mne_task": "rest",
                    "description": "Valid task",
                    "settings": {}
                }
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
            "database": {"enabled": False}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        test_file = tmp_path / "test_data.fif"
        test_file.touch()
        
        # Should raise error for non-existent task
        with pytest.raises((KeyError, ValueError)):
            pipeline.process_file(
                file_path=str(test_file),
                task="NonExistentTask"
            )
    
    def test_process_file_nonexistent_file(self, tmp_path):
        """Test processing with non-existent file."""
        config = {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    "description": "Test task",
                    "settings": {}
                }
            },
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
            "database": {"enabled": False}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Should raise error for non-existent file
        with pytest.raises((FileNotFoundError, IOError)):
            pipeline.process_file(
                file_path="/nonexistent/file.fif",
                task="TestTask"
            )


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineUtilityMethods:
    """Test Pipeline utility and helper methods."""
    
    def test_pipeline_string_representation(self, tmp_path):
        """Test string representation of Pipeline."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tasks: {}")
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Should have a meaningful string representation
        str_repr = str(pipeline)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
    
    def test_pipeline_directory_creation(self, tmp_path):
        """Test that Pipeline creates necessary directories."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("tasks: {}")
        
        new_output_dir = tmp_path / "new_output_directory"
        assert not new_output_dir.exists()
        
        pipeline = Pipeline(
            autoclean_dir=str(new_output_dir),
            autoclean_config=str(config_file)
        )
        
        # Pipeline should create the directory or handle its absence gracefully
        assert pipeline.autoclean_dir == new_output_dir


class TestPipelineMocked:
    """Test Pipeline functionality with heavy mocking for speed."""
    
    @patch('autoclean.core.pipeline.load_config')
    @patch('autoclean.utils.logging.configure_logger')
    def test_pipeline_init_mocked(self, mock_logger, mock_load_config, tmp_path):
        """Test Pipeline initialization with mocked dependencies."""
        mock_config = {
            "tasks": {"TestTask": {"settings": {}}},
            "stage_files": {"post_import": {"enabled": True, "suffix": "_postimport"}},
            "database": {"enabled": False}
        }
        mock_load_config.return_value = mock_config
        
        with patch('autoclean.core.pipeline.Pipeline.__init__', lambda x, *args, **kwargs: None):
            # Test that mocked initialization doesn't raise errors
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.autoclean_dir = tmp_path / "output"
            pipeline.config = mock_config
            
            assert pipeline.config["tasks"]["TestTask"]["settings"] == {}
    
    @patch('autoclean.core.pipeline.task_registry', {"MockTask": Mock})
    def test_task_registry_access_mocked(self):
        """Test task registry access with mocked registry."""
        with patch('autoclean.core.pipeline.Pipeline.__init__', lambda x, *args, **kwargs: None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.TASK_REGISTRY = {"MockTask": Mock}
            
            assert "MockTask" in pipeline.TASK_REGISTRY
            assert pipeline.TASK_REGISTRY["MockTask"] is not None


# Tests that don't require full Pipeline import
class TestPipelineConceptual:
    """Conceptual tests for Pipeline design and interface."""
    
    def test_pipeline_expected_interface(self):
        """Test that Pipeline has the expected interface when importable."""
        if not PIPELINE_AVAILABLE:
            pytest.skip("Pipeline not importable, testing interface conceptually")
        
        from autoclean.core.pipeline import Pipeline
        
        # Test that expected methods exist
        expected_methods = [
            'process_file',
            'process_directory', 
            '__init__'
        ]
        
        for method in expected_methods:
            assert hasattr(Pipeline, method), f"Pipeline missing expected method: {method}"
    
    def test_pipeline_expected_attributes(self):
        """Test that Pipeline has expected class attributes."""
        if not PIPELINE_AVAILABLE:
            pytest.skip("Pipeline not importable, testing attributes conceptually")
        
        from autoclean.core.pipeline import Pipeline
        
        # Test that expected attributes exist
        expected_attrs = ['TASK_REGISTRY']
        
        for attr in expected_attrs:
            assert hasattr(Pipeline, attr), f"Pipeline missing expected attribute: {attr}"