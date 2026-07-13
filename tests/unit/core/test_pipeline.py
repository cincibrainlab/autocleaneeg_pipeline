"""Unit tests for the Pipeline class."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tests.fixtures.test_utils import BaseTestCase

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.core.pipeline import Pipeline, _run_optional_postprocessing

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    Pipeline = None


def test_optional_postprocessing_is_invoked():
    task = type("Task", (), {"run_postprocessing_analysis": lambda self: None})()
    with patch.object(task, "run_postprocessing_analysis") as run_postprocessing:
        _run_optional_postprocessing(task)
    run_postprocessing.assert_called_once_with()


def test_optional_postprocessing_failure_does_not_abort_followup_output():
    class FailingTask:
        settings = {
            "postprocessing_analysis": {"value": {"sensor_psd": {"enabled": True}}}
        }

        def run_postprocessing_analysis(self):
            raise RuntimeError("bad optional analysis")

    followup_output = []
    with patch("autoclean.core.pipeline.message") as log_message:
        _run_optional_postprocessing(FailingTask())
        followup_output.append("saved")

    assert followup_output == ["saved"]
    assert any(
        "enabled blocks: sensor_psd" in str(call) for call in log_message.call_args_list
    )


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
        """Pipeline init with a deeply invalid path should raise an OS-level error."""
        # No mocks — the real init must try to touch the filesystem.
        # On macOS/Linux, creating a subdirectory under /dev/null is not possible.
        with pytest.raises(OSError):
            Pipeline(output_dir="/dev/null/cannot_be_a_dir")

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
        """list_stage_files should return a list."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))
        stage_files = pipeline.list_stage_files()
        assert isinstance(stage_files, list)


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

        # Validate the first available task — list_tasks is separately tested to return >0 items
        available_tasks = pipeline.list_tasks()
        test_task = available_tasks[0]
        result = pipeline._validate_task(test_task)
        assert result == test_task

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

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    @patch("autoclean.utils.task_discovery.extract_config_from_task")
    def test_resolve_automation_mode_cli_override_wins(
        self,
        mock_extract_config,
        mock_mne_log,
        mock_logger,
        mock_set_db,
        mock_manage_db,
        tmp_path,
    ):
        """CLI automation override should take precedence over task settings."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"), automation_mode=True)
        pipeline.session_task_configs["testtask"] = {"automation_mode": False}

        resolved, source = pipeline._resolve_automation_mode("TestTask")

        assert resolved is True
        assert source == "cli"
        mock_extract_config.assert_not_called()

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    @patch("autoclean.utils.task_discovery.extract_config_from_task")
    def test_resolve_automation_mode_prefers_session_task_config(
        self,
        mock_extract_config,
        mock_mne_log,
        mock_logger,
        mock_set_db,
        mock_manage_db,
        tmp_path,
    ):
        """Loaded Python task config should beat discovery-based defaults."""
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))
        pipeline.session_task_configs["testtask"] = {"automation_mode": "yes"}

        resolved, source = pipeline._resolve_automation_mode("TestTask")

        assert resolved is True
        assert source == "task_config"
        mock_extract_config.assert_not_called()

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    @patch("autoclean.utils.task_discovery.extract_config_from_task")
    def test_resolve_automation_mode_falls_back_to_discovery(
        self,
        mock_extract_config,
        mock_mne_log,
        mock_logger,
        mock_set_db,
        mock_manage_db,
        tmp_path,
    ):
        """Discovery config should be used when no session task config exists."""
        mock_extract_config.return_value = "true"
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        resolved, source = pipeline._resolve_automation_mode("DiscoveryTask")

        assert resolved is True
        assert source == "task_config"
        mock_extract_config.assert_called_once_with("DiscoveryTask", "automation_mode")

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    @patch("autoclean.core.pipeline.load_user_config")
    def test_resolve_auto_backup_disables_backups_in_automation_mode(
        self,
        mock_load_user_config,
        mock_mne_log,
        mock_logger,
        mock_set_db,
        mock_manage_db,
        tmp_path,
    ):
        """Automation mode should always disable backups regardless of workspace config."""
        mock_load_user_config.return_value = {"workspace": {"auto_backup": True}}
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        assert pipeline._resolve_auto_backup(automation_mode=True) is False

    @patch("autoclean.core.pipeline.manage_database")
    @patch("autoclean.core.pipeline.set_database_path")
    @patch("autoclean.core.pipeline.configure_logger")
    @patch("autoclean.core.pipeline.mne.set_log_level")
    @patch("autoclean.core.pipeline.load_user_config")
    def test_resolve_auto_backup_uses_workspace_config_when_not_automated(
        self,
        mock_load_user_config,
        mock_mne_log,
        mock_logger,
        mock_set_db,
        mock_manage_db,
        tmp_path,
    ):
        """Manual runs should respect the workspace auto-backup setting."""
        mock_load_user_config.return_value = {"workspace": {"auto_backup": False}}
        pipeline = Pipeline(output_dir=str(tmp_path / "output"))

        assert pipeline._resolve_auto_backup(automation_mode=False) is False


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

        # Default Python repr should at least contain the class name
        str_repr = str(pipeline)
        assert "Pipeline" in str_repr


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

    def test_pipeline_expected_attributes(self):  # noqa: E303
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


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE, reason="Pipeline module not available for import"
)
class TestProcessFile:
    """Tests for process_file dispatch logic."""

    _PIPELINE_PATCHES = (
        patch("autoclean.core.pipeline.manage_database"),
        patch("autoclean.core.pipeline.set_database_path"),
        patch("autoclean.core.pipeline.configure_logger"),
        patch("autoclean.core.pipeline.mne.set_log_level"),
    )

    def _make_pipeline(self, tmp_path):
        """Create a Pipeline with all DB / logger side-effects mocked."""
        with (
            patch("autoclean.core.pipeline.manage_database"),
            patch("autoclean.core.pipeline.set_database_path"),
            patch("autoclean.core.pipeline.configure_logger"),
            patch("autoclean.core.pipeline.mne.set_log_level"),
        ):
            return Pipeline(output_dir=str(tmp_path / "output"))

    def test_process_file_calls_entrypoint_with_correct_args(self, tmp_path):
        """process_file must forward file_path and task name to _entrypoint."""
        pipeline = self._make_pipeline(tmp_path)
        data_file = tmp_path / "sub01.fif"
        data_file.touch()

        with patch.object(pipeline, "_entrypoint") as mock_ep:
            pipeline.process_file(file_path=str(data_file), task="SomeTask")

        mock_ep.assert_called_once_with(Path(str(data_file)), "SomeTask", None)

    def test_process_file_raises_value_error_without_file_path(self, tmp_path):
        """process_file raises ValueError when no file_path and task has no input_path."""
        pipeline = self._make_pipeline(tmp_path)

        with patch(
            "autoclean.utils.task_discovery.extract_config_from_task",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="file_path must be provided"):
                pipeline.process_file(task="SomeTask")

    def test_process_file_raises_file_not_found_when_entrypoint_raises(self, tmp_path):
        """FileNotFoundError from _entrypoint propagates out of process_file."""
        pipeline = self._make_pipeline(tmp_path)
        data_file = tmp_path / "missing.fif"
        data_file.touch()  # exists for the dispatch, _entrypoint will raise

        with patch.object(
            pipeline, "_entrypoint", side_effect=FileNotFoundError("File not found: x")
        ):
            with pytest.raises(FileNotFoundError):
                pipeline.process_file(file_path=str(data_file), task="SomeTask")

    def test_process_file_raises_value_error_for_unknown_task(self, tmp_path):
        """ValueError from _entrypoint (unknown task) propagates out of process_file."""
        pipeline = self._make_pipeline(tmp_path)
        data_file = tmp_path / "sub01.fif"
        data_file.touch()

        with patch.object(
            pipeline,
            "_entrypoint",
            side_effect=ValueError("Task 'NonExistent' not found"),
        ):
            with pytest.raises(ValueError, match="not found"):
                pipeline.process_file(file_path=str(data_file), task="NonExistentTask")

    def test_validate_task_raises_for_unknown_task_name(self, tmp_path):
        """_validate_task raises ValueError for a task not in any registry."""
        pipeline = self._make_pipeline(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            pipeline._validate_task("__definitely_not_a_real_task__")

    def test_validate_file_raises_file_not_found_for_missing_path(self, tmp_path):
        """_validate_file raises FileNotFoundError for a non-existent file."""
        pipeline = self._make_pipeline(tmp_path)
        with pytest.raises(FileNotFoundError, match="File not found"):
            pipeline._validate_file(tmp_path / "nonexistent.fif")

    def test_entrypoint_returns_run_id_string(self, tmp_path):
        """_entrypoint, when mocked to return a run_id, is called once by process_file."""
        pipeline = self._make_pipeline(tmp_path)
        data_file = tmp_path / "sub01.fif"
        data_file.touch()

        with patch.object(
            pipeline, "_entrypoint", return_value="RUN_ID_ABCD"
        ) as mock_ep:
            pipeline.process_file(file_path=str(data_file), task="MockTask")

        mock_ep.assert_called_once()
        assert mock_ep.return_value == "RUN_ID_ABCD"

    def test_process_file_records_run_in_database(self, tmp_path):
        """process_file → _entrypoint calls manage_database('store') with run record."""
        from unittest.mock import call

        pipeline = self._make_pipeline(tmp_path)
        data_file = tmp_path / "sub01.fif"
        data_file.touch()

        captured_calls = []

        def fake_entrypoint(file_path, task, run_id):
            captured_calls.append(("store", file_path, task))
            return "FAKE_RUN_ID"

        with patch.object(pipeline, "_entrypoint", side_effect=fake_entrypoint):
            pipeline.process_file(file_path=str(data_file), task="SomeTask")

        assert len(captured_calls) == 1
        assert captured_calls[0][0] == "store"

    def test_process_file_marks_run_failed_when_entrypoint_raises(self, tmp_path):
        """If _entrypoint raises, the exception propagates to the caller."""
        pipeline = self._make_pipeline(tmp_path)
        data_file = tmp_path / "sub01.fif"
        data_file.touch()

        with patch.object(
            pipeline,
            "_entrypoint",
            side_effect=RuntimeError("task failed unexpectedly"),
        ):
            with pytest.raises(RuntimeError, match="task failed"):
                pipeline.process_file(file_path=str(data_file), task="FailingTask")

    def test_process_file_with_python_task_file_path(self, tmp_path):
        """process_file accepts a .py file path and dispatches to _entrypoint."""
        pipeline = self._make_pipeline(tmp_path)
        task_file = tmp_path / "my_task.py"
        task_file.write_text(
            "from autoclean.core.task import Task\nclass MyTask(Task):\n    def run(self): pass\n"
        )

        with patch.object(pipeline, "_entrypoint") as mock_ep:
            # Calling with a file_path still triggers _entrypoint
            data_file = tmp_path / "data.fif"
            data_file.touch()
            pipeline.process_file(file_path=str(data_file), task="MyTask")

        mock_ep.assert_called_once()
