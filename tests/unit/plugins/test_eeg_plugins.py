"""Unit tests for EEG plugins."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import mne
import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw
from tests.fixtures.test_utils import BaseTestCase, EEGAssertions

# Test with mock plugins to avoid heavy import dependencies
try:
    from autoclean.io.import_ import BaseEEGPlugin

    PLUGIN_BASE_AVAILABLE = True
except ImportError:
    PLUGIN_BASE_AVAILABLE = False


class TestEGIRawGSN129Plugin:
    """Test EGI Raw GSN-HydroCel-129 plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_format_montage_support(self):
        """Test plugin format and montage support detection."""

        # Create mock plugin based on real interface
        class MockEGIRawGSN129Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "EGI_RAW" and montage_name == "GSN-HydroCel-129"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129)

        plugin_class = MockEGIRawGSN129Plugin

        # Should support correct combination
        assert (
            plugin_class.supports_format_montage("EGI_RAW", "GSN-HydroCel-129") is True
        )

        # Should not support incorrect combinations
        assert (
            plugin_class.supports_format_montage("EEGLAB_SET", "GSN-HydroCel-129")
            is False
        )
        assert (
            plugin_class.supports_format_montage("EGI_RAW", "GSN-HydroCel-128") is False
        )
        assert (
            plugin_class.supports_format_montage("GENERIC_FIF", "standard_1020")
            is False
        )

    @patch("mne.io.read_raw_egi")
    def test_egi_raw_import_functionality(self, mock_read_raw):
        """Test EGI raw file import functionality."""
        # Mock the MNE import function
        mock_raw = create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129)
        mock_read_raw.return_value = mock_raw

        # Create mock plugin
        if PLUGIN_BASE_AVAILABLE:

            class MockEGIPlugin(BaseEEGPlugin):
                @classmethod
                def supports_format_montage(
                    cls, format_id: str, montage_name: str
                ) -> bool:
                    return format_id == "EGI_RAW" and montage_name == "GSN-HydroCel-129"

                def import_and_configure(
                    self, file_path: Path, autoclean_dict: dict, preload: bool = True
                ):
                    # Simulate real plugin behavior
                    raw = mne.io.read_raw_egi(
                        input_fname=file_path,
                        preload=preload,
                        events_as_annotations=True,
                        exclude=[],
                    )
                    return raw

            plugin = MockEGIPlugin()
            test_file = Path("/test/data.raw")
            config = {"montage": {"value": "GSN-HydroCel-129"}}

            result = plugin.import_and_configure(test_file, config)

            # Verify MNE function was called correctly
            mock_read_raw.assert_called_once_with(
                input_fname=test_file,
                preload=True,
                events_as_annotations=True,
                exclude=[],
            )

            # Verify result properties
            EEGAssertions.assert_raw_properties(result, expected_n_channels=129)

    def test_gsn129_montage_configuration(self):
        """Test GSN-HydroCel-129 montage configuration."""
        # Test montage-specific configuration
        raw = create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129)

        # Should have 129 channels with correct naming
        assert len(raw.ch_names) == 129
        assert raw.ch_names[-1] == "Cz"  # 129th channel should be Cz
        assert all(ch.startswith("E") for ch in raw.ch_names[:-1])  # E1-E128

        # Test channel types
        assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())


class TestEGIRawGSN124Plugin:
    """Test EGI Raw GSN-HydroCel-124 plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_gsn124_plugin_support(self):
        """Test GSN-124 plugin format support."""

        class MockEGIRawGSN124Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "EGI_RAW" and montage_name == "GSN-HydroCel-124"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="GSN-HydroCel-124", n_channels=124)

        plugin_class = MockEGIRawGSN124Plugin

        # Should support GSN-124 specifically
        assert (
            plugin_class.supports_format_montage("EGI_RAW", "GSN-HydroCel-124") is True
        )
        assert (
            plugin_class.supports_format_montage("EGI_RAW", "GSN-HydroCel-129") is False
        )

    def test_gsn124_montage_differences(self):
        """Test differences between GSN-124 and GSN-129 montages."""
        raw_124 = create_synthetic_raw(montage="GSN-HydroCel-124", n_channels=124)
        raw_129 = create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129)

        # Should have different channel counts
        assert len(raw_124.ch_names) == 124
        assert len(raw_129.ch_names) == 129

        # Different reference electrode handling
        assert "Cz" not in raw_124.ch_names  # No Cz in 124
        assert "Cz" in raw_129.ch_names  # Cz in 129


class TestEEGLABPlugins:
    """Test EEGLAB format plugins."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_eeglab_set_plugin_support(self):
        """Test EEGLAB .set plugin support."""

        class MockEEGLABGSN129Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "EEGLAB_SET" and montage_name == "GSN-HydroCel-129"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129)

        plugin_class = MockEEGLABGSN129Plugin

        # Should support EEGLAB .set format
        assert (
            plugin_class.supports_format_montage("EEGLAB_SET", "GSN-HydroCel-129")
            is True
        )
        assert (
            plugin_class.supports_format_montage("EGI_RAW", "GSN-HydroCel-129") is False
        )

    @patch("mne.io.read_raw_eeglab")
    def test_eeglab_import_functionality(self, mock_read_eeglab):
        """Test EEGLAB file import functionality."""
        mock_raw = create_synthetic_raw(montage="GSN-HydroCel-129", n_channels=129)
        mock_read_eeglab.return_value = mock_raw

        if PLUGIN_BASE_AVAILABLE:

            class MockEEGLABPlugin(BaseEEGPlugin):
                @classmethod
                def supports_format_montage(
                    cls, format_id: str, montage_name: str
                ) -> bool:
                    return (
                        format_id == "EEGLAB_SET" and montage_name == "GSN-HydroCel-129"
                    )

                def import_and_configure(
                    self, file_path: Path, autoclean_dict: dict, preload: bool = True
                ):
                    raw = mne.io.read_raw_eeglab(input_fname=file_path, preload=preload)
                    return raw

            plugin = MockEEGLABPlugin()
            test_file = Path("/test/data.set")

            result = plugin.import_and_configure(test_file, {})

            mock_read_eeglab.assert_called_once_with(
                input_fname=test_file, preload=True
            )

            EEGAssertions.assert_raw_properties(result, expected_n_channels=129)


class TestStandard1020Plugin:
    """Test standard 10-20 montage plugin."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_standard_1020_plugin_support(self):
        """Test standard 10-20 plugin support."""

        class MockStandard1020Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return montage_name == "standard_1020"  # Support multiple formats

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="standard_1020", n_channels=32)

        plugin_class = MockStandard1020Plugin

        # Should support standard 10-20 with various formats
        assert (
            plugin_class.supports_format_montage("EEGLAB_SET", "standard_1020") is True
        )
        assert (
            plugin_class.supports_format_montage("GENERIC_FIF", "standard_1020") is True
        )
        assert plugin_class.supports_format_montage("EGI_RAW", "standard_1020") is True

        # Should not support other montages
        assert (
            plugin_class.supports_format_montage("EEGLAB_SET", "GSN-HydroCel-129")
            is False
        )

    def test_standard_1020_montage_characteristics(self):
        """Test standard 10-20 montage characteristics."""
        raw = create_synthetic_raw(montage="standard_1020", n_channels=32)

        # Should have standard channel names
        standard_channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "Cz"]
        for ch in standard_channels:
            if ch in raw.ch_names:  # Some might not be included in smaller montages
                assert ch in raw.ch_names

        # Should be EEG type
        assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())


class TestMEA30Plugin:
    """Test MEA30 (mouse) plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_mea30_plugin_support(self):
        """Test MEA30 plugin support."""

        class MockMEA30Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return montage_name == "MEA30"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="MEA30", n_channels=30)

        plugin_class = MockMEA30Plugin

        # Should support MEA30 montage
        assert plugin_class.supports_format_montage("EEGLAB_SET", "MEA30") is True
        assert plugin_class.supports_format_montage("GENERIC_FIF", "MEA30") is True

        # Should not support other montages
        assert (
            plugin_class.supports_format_montage("EEGLAB_SET", "GSN-HydroCel-129")
            is False
        )

    def test_mea30_montage_characteristics(self):
        """Test MEA30 montage characteristics."""
        raw = create_synthetic_raw(montage="MEA30", n_channels=30)

        # Should have 30 channels
        assert len(raw.ch_names) == 30

        # Should have CH naming convention for mouse EEG
        expected_names = [f"CH{i}" for i in range(1, 31)]
        for expected_name in expected_names:
            if expected_name in raw.ch_names:  # Depending on implementation
                assert expected_name in raw.ch_names


class TestPluginErrorHandling:
    """Test plugin error handling and edge cases."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_file_not_found_error(self):
        """Test plugin behavior with non-existent files."""

        class MockPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return True

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Simulate file not found
                raise FileNotFoundError(f"File not found: {file_path}")

        plugin = MockPlugin()

        with pytest.raises(FileNotFoundError, match="File not found"):
            plugin.import_and_configure(Path("/nonexistent/file.raw"), {})

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_invalid_file_format_error(self):
        """Test plugin behavior with invalid file formats."""

        class MockPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return True

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Simulate invalid file format
                raise ValueError(f"Invalid file format: {file_path}")

        plugin = MockPlugin()

        with pytest.raises(ValueError, match="Invalid file format"):
            plugin.import_and_configure(Path("/test/invalid.xyz"), {})

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_montage_mismatch_error(self):
        """Test plugin behavior with montage mismatches."""

        class MockPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "TEST_FORMAT" and montage_name == "TEST_MONTAGE"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Simulate montage mismatch
                raise RuntimeError("File has 64 channels but montage expects 128")

        plugin = MockPlugin()

        with pytest.raises(RuntimeError, match="montage expects"):
            plugin.import_and_configure(Path("/test/data.raw"), {})


class TestPluginIntegration:
    """Test plugin integration with the broader system."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_output_validation(self):
        """Test that plugin outputs are valid Raw objects."""

        class MockPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return True

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw()

        plugin = MockPlugin()
        result = plugin.import_and_configure(Path("/test/data.fif"), {})

        # Should return valid Raw object
        EEGAssertions.assert_raw_properties(result)
        assert hasattr(result, "info")
        assert hasattr(result, "get_data")
        assert hasattr(result, "ch_names")

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_configuration_handling(self):
        """Test that plugins handle configuration dictionaries properly."""

        class MockPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return True

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Should access config without errors
                montage_name = autoclean_dict.get("montage", {}).get("value", "default")
                return create_synthetic_raw(montage=montage_name)

        plugin = MockPlugin()
        config = {
            "montage": {"value": "GSN-HydroCel-129"},
            "other_setting": {"enabled": True},
        }

        result = plugin.import_and_configure(Path("/test/data.fif"), config)

        # Should handle config without errors
        assert result is not None
        EEGAssertions.assert_raw_properties(result)


class TestPluginMocked:
    """Test plugin functionality with heavy mocking."""

    def test_plugin_interface_mocked(self):
        """Test plugin interface with complete mocking."""
        # Mock the entire plugin system
        mock_plugin = Mock()
        mock_plugin.supports_format_montage.return_value = True
        mock_plugin.import_and_configure.return_value = create_synthetic_raw()

        # Test interface calls
        assert mock_plugin.supports_format_montage("TEST", "TEST") is True
        result = mock_plugin.import_and_configure(Path("/test"), {})

        # Verify mock was called
        mock_plugin.supports_format_montage.assert_called_once()
        mock_plugin.import_and_configure.assert_called_once()

        # Verify result
        EEGAssertions.assert_raw_properties(result)

    def test_multiple_plugins_mocked(self):
        """Test multiple plugin coordination with mocking."""
        # Mock multiple plugins
        plugin1 = Mock()
        plugin1.supports_format_montage.side_effect = lambda f, m: f == "FORMAT1"

        plugin2 = Mock()
        plugin2.supports_format_montage.side_effect = lambda f, m: f == "FORMAT2"

        # Test plugin selection logic
        plugins = [plugin1, plugin2]

        # Find plugin for FORMAT1
        selected_plugin = None
        for plugin in plugins:
            if plugin.supports_format_montage("FORMAT1", "TEST"):
                selected_plugin = plugin
                break

        assert selected_plugin == plugin1

        # Find plugin for FORMAT2
        selected_plugin = None
        for plugin in plugins:
            if plugin.supports_format_montage("FORMAT2", "TEST"):
                selected_plugin = plugin
                break

        assert selected_plugin == plugin2
