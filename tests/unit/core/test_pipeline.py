"""Unit tests for the Pipeline class."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
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
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_pipeline_init_with_valid_config(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db):
        """Test Pipeline initialization with valid configuration."""
        # Use the test config file
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(self.autoclean_dir),
            autoclean_config=str(config_file)
        )
        
        # Test basic attributes
        assert pipeline.autoclean_dir == Path(self.autoclean_dir).absolute()
        assert pipeline.autoclean_config == Path(config_file).absolute()
        assert hasattr(pipeline, 'autoclean_dict')
        assert hasattr(pipeline, 'TASK_REGISTRY')
        assert hasattr(pipeline, 'participants_tsv_lock')
        
        # Test config loaded correctly
        assert "tasks" in pipeline.autoclean_dict
        assert "TestResting" in pipeline.autoclean_dict["tasks"]
        assert "stage_files" in pipeline.autoclean_dict
        
        # Verify database setup was called
        mock_set_db.assert_called_once_with(pipeline.autoclean_dir)
        mock_manage_db.assert_called_once_with(operation="create_collection")
    
    def test_pipeline_init_invalid_config_path(self):
        """Test Pipeline initialization with invalid config path."""
        with pytest.raises((FileNotFoundError, IOError)):
            Pipeline(
                autoclean_dir=str(self.autoclean_dir),
                autoclean_config="/nonexistent/path/config.yaml"
            )
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_pipeline_init_new_directory(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db):
        """Test Pipeline initialization creates new directory."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        # Use a new directory that doesn't exist yet
        new_dir = self.temp_dir / "new_autoclean_dir"
        
        pipeline = Pipeline(
            autoclean_dir=str(new_dir),
            autoclean_config=str(config_file)
        )
        
        assert pipeline.autoclean_dir == new_dir.absolute()
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_pipeline_task_registry_access(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db):
        """Test that Pipeline has access to task registry."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(self.autoclean_dir),
            autoclean_config=str(config_file)
        )
        
        assert hasattr(pipeline, 'TASK_REGISTRY')
        assert isinstance(pipeline.TASK_REGISTRY, dict)
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_pipeline_verbose_parameter(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db):
        """Test Pipeline initialization with different verbose settings."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        # Test different verbose settings
        for verbose in [True, False, "info", "debug", None]:
            pipeline = Pipeline(
                autoclean_dir=str(self.autoclean_dir),
                autoclean_config=str(config_file),
                verbose=verbose
            )
            assert pipeline is not None
            assert pipeline.verbose == verbose


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineUtilityMethods:
    """Test Pipeline utility and helper methods."""
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_list_tasks(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test listing available tasks."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        tasks = pipeline.list_tasks()
        assert isinstance(tasks, list)
        # Tasks come from TASK_REGISTRY which is imported from autoclean.tasks
        assert len(tasks) > 0
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_list_stage_files(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test listing stage files."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        stage_files = pipeline.list_stage_files()
        assert isinstance(stage_files, list)
        expected_stages = ["post_import", "post_basic_steps", "post_clean_raw", "post_ica", "post_epochs", "post_comp"]
        for stage in expected_stages:
            assert stage in stage_files


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineValidation:
    """Test Pipeline validation methods."""
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_validate_task_valid(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test task validation with valid task."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Test with a valid task from our config
        result = pipeline._validate_task("TestResting")
        assert result == "TestResting"
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_validate_task_invalid(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test task validation with invalid task."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Test with an invalid task
        with pytest.raises(ValueError, match="Task 'NonExistentTask' not found in configuration"):
            pipeline._validate_task("NonExistentTask")
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_validate_file_valid(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test file validation with valid file."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Create a test file
        test_file = tmp_path / "test.fif"
        test_file.touch()
        
        result = pipeline._validate_file(str(test_file))
        assert result == test_file
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_validate_file_invalid(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test file validation with non-existent file."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError, match="File not found"):
            pipeline._validate_file("/nonexistent/file.fif")


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline module not available for import")
class TestPipelineString:
    """Test Pipeline string representation."""
    
    @patch('autoclean.utils.database.manage_database')
    @patch('autoclean.utils.database.set_database_path')
    @patch('autoclean.utils.logging.configure_logger')
    @patch('mne.set_log_level')
    def test_pipeline_string_representation(self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path):
        """Test that Pipeline has a string representation."""
        config_file = Path(__file__).parent.parent.parent / "fixtures" / "configs" / "test_config.yaml"
        
        pipeline = Pipeline(
            autoclean_dir=str(tmp_path / "output"),
            autoclean_config=str(config_file)
        )
        
        # Should have a meaningful string representation
        str_repr = str(pipeline)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0


# Tests that can run without full dependencies 
class TestPipelineInterface:
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
            'process_directory_async',
            'list_tasks',
            'list_stage_files',
            '_validate_task',
            '_validate_file'
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

