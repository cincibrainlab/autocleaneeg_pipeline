"""Unit tests for the Pipeline class."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.test_utils import BaseTestCase

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.core.pipeline import Pipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    Pipeline = None


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelineInitialization(BaseTestCase):
    """Test Pipeline class initialization and basic functionality."""

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_pipeline_init_with_valid_config(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db
    ):
        """Test Pipeline initialization with output directory."""
        pipeline = Pipeline(output_dir=str(self.autoclean_dir))

        # Test basic attributes
        assert pipeline.output_dir == Path(self.autoclean_dir).absolute()
        assert hasattr(pipeline, "TASK_REGISTRY")
        assert hasattr(pipeline, "session_task_registry")
        assert hasattr(pipeline, "participants_tsv_lock")

        # Verify database setup was called
        mock_set_db.assert_called_once_with(pipeline.output_dir)
        mock_manage_db.assert_called_once_with(operation="create_collection")

    def test_pipeline_init_invalid_output_path(self):
        """Test Pipeline initialization with invalid output path."""
        # Invalid output path should be handled gracefully or raise appropriate error
        invalid_path = "/nonexistent/path/that/should/not/exist"
        # Pipeline should either handle this gracefully or raise a clear error
        try:
            pipeline = Pipeline(output_dir=invalid_path)
            # If it doesn't raise an error, it should handle it gracefully
            assert hasattr(pipeline, "output_dir")
        except (FileNotFoundError, IOError, PermissionError):
            # This is acceptable behavior
            pass

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_pipeline_init_new_directory(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db
    ):
        """Test Pipeline initialization with new directory."""
        # Use a new directory that doesn't exist yet
        new_dir = self.temp_dir / "new_output_dir"

        pipeline = Pipeline(output_dir=str(new_dir))

        assert pipeline.output_dir == new_dir.absolute()

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_pipeline_task_registry_access(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db
    ):
        """Test that Pipeline has access to task registry."""
        pipeline = Pipeline(output_dir=str(self.autoclean_dir))

        assert hasattr(pipeline, "TASK_REGISTRY")
        assert isinstance(pipeline.TASK_REGISTRY, dict)

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_pipeline_verbose_parameter(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db
    ):
        """Test Pipeline initialization with different verbose settings."""
        config_file = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "configs"
            / "test_config.yaml"
        )

        # Test different verbose settings
        for verbose in [True, False, "info", "debug", None]:
            pipeline = Pipeline(output_dir=str(self.autoclean_dir), verbose=verbose)
            assert pipeline is not None
            assert pipeline.verbose == verbose


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelineUtilityMethods:
    """Test Pipeline utility and helper methods."""

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_list_tasks(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test listing available tasks."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        tasks = pipeline.list_tasks()
        assert isinstance(tasks, list)
        # Tasks come from TASK_REGISTRY which is imported from autoclean.tasks
        assert len(tasks) > 0

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_list_stage_files(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test listing stage files (deprecated functionality)."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        # Note: stage_files functionality has been simplified in the new implementation
        # This test checks if the method exists, and if so, validates basic functionality
        if hasattr(pipeline, "list_stage_files"):
            stage_files = pipeline.list_stage_files()
            assert isinstance(stage_files, list)
        else:
            # Skip this test if the method doesn't exist in the new implementation
            pytest.skip(
                "list_stage_files method not available in simplified implementation"
            )


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelineValidation:
    """Test Pipeline validation methods."""

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_validate_task_valid(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test task validation with valid task."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        # Test with a valid task from the TASK_REGISTRY
        # Get any available task from the registry
        available_tasks = pipeline.list_tasks()
        if available_tasks:
            test_task = available_tasks[0]
            result = pipeline._validate_task(test_task)
            assert result == test_task
        else:
            pytest.skip("No tasks available in TASK_REGISTRY for testing")

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_validate_task_invalid(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test task validation with invalid task."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        # Test with an invalid task
        with pytest.raises(ValueError, match="Task .* not found"):
            pipeline._validate_task("NonExistentTask")

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_validate_file_valid(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test file validation with valid file."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        # Create a test file
        test_file = tmp_path / "test.fif"
        test_file.touch()

        result = pipeline._validate_file(str(test_file))
        assert result == test_file

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_validate_file_invalid(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test file validation with non-existent file."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        # Test with non-existent file
        with pytest.raises(FileNotFoundError, match="File not found"):
            pipeline._validate_file("/nonexistent/file.fif")


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestPipelineString:
    """Test Pipeline string representation."""

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    def test_pipeline_string_representation(
        self, mock_mne_log, mock_logger, mock_set_db, mock_manage_db, tmp_path
    ):
        """Test that Pipeline has a string representation."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

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

        # Pipeline already imported at module level

        # Test that expected methods exist
        expected_methods = [
            "process_file",
            "process_directory",
            "process_directory_async",
            "list_tasks",
            "list_stage_files",
            "_validate_task",
            "_validate_file",
        ]

        for method in expected_methods:
            assert hasattr(
                Pipeline, method
            ), f"Pipeline missing expected method: {method}"

    def test_pipeline_expected_attributes(self):
        """Test that Pipeline has expected class attributes."""
        if not PIPELINE_AVAILABLE:
            pytest.skip("Pipeline not importable, testing attributes conceptually")

        # Pipeline already imported at module level

        # Test that expected attributes exist
        expected_attrs = ["TASK_REGISTRY"]

        for attr in expected_attrs:
            assert hasattr(
                Pipeline, attr
            ), f"Pipeline missing expected attribute: {attr}"
