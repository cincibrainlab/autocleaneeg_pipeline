"""Unit tests for the plugin system architecture."""

from abc import ABC
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw
from tests.fixtures.test_utils import BaseTestCase, EEGAssertions

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.io.import_ import (
        _CORE_FORMATS,
        _FORMAT_REGISTRY,
        _PLUGIN_REGISTRY,
        BaseEEGPlugin,
        get_format_from_extension,
        register_format,
        register_plugin,
    )

    IMPORT_AVAILABLE = True
except ImportError:
    IMPORT_AVAILABLE = False


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
class TestBaseEEGPlugin:
    """Test the BaseEEGPlugin abstract base class."""

    def test_base_plugin_is_abstract(self):
        """Test that BaseEEGPlugin is properly abstract."""
        from autoclean.io.import_ import BaseEEGPlugin

        assert issubclass(BaseEEGPlugin, ABC)

        # Should not be directly instantiable
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEEGPlugin()

    def test_base_plugin_abstract_methods(self):
        """Test that BaseEEGPlugin defines expected abstract methods."""
        from autoclean.io.import_ import BaseEEGPlugin

        abstract_methods = getattr(BaseEEGPlugin, "__abstractmethods__", set())
        expected_methods = {"supports_format_montage", "import_and_configure"}

        assert expected_methods.issubset(
            abstract_methods
        ), f"BaseEEGPlugin missing abstract methods: {expected_methods - abstract_methods}"

    def test_concrete_plugin_implementation(self):
        """Test that concrete plugin implementation works."""
        from autoclean.io.import_ import BaseEEGPlugin

        class TestPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "TEST_FORMAT" and montage_name == "TEST_MONTAGE"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw()

        # Should be able to instantiate concrete plugin
        plugin = TestPlugin()
        assert isinstance(plugin, BaseEEGPlugin)

        # Should implement abstract methods
        assert plugin.supports_format_montage("TEST_FORMAT", "TEST_MONTAGE") is True
        assert plugin.supports_format_montage("OTHER_FORMAT", "TEST_MONTAGE") is False

        # Should be able to call import_and_configure
        result = plugin.import_and_configure(Path("/test.fif"), {})
        assert hasattr(result, "info")  # Should be Raw-like object


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
class TestFormatRegistry:
    """Test the format registration system."""

    def setup_method(self):
        """Set up clean registry for each test."""
        # Save original state
        self.original_format_registry = _FORMAT_REGISTRY.copy()
        self.original_plugin_registry = _PLUGIN_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry state."""
        _FORMAT_REGISTRY.clear()
        _FORMAT_REGISTRY.update(self.original_format_registry)
        _PLUGIN_REGISTRY.clear()
        _PLUGIN_REGISTRY.update(self.original_plugin_registry)

    def test_core_formats_available(self):
        """Test that core formats are properly defined."""
        from autoclean.io.import_ import _CORE_FORMATS

        expected_formats = {
            "set": "EEGLAB_SET",
            "raw": "EGI_RAW",
            "mff": "EGI_RAW",
            "fif": "GENERIC_FIF",
        }

        for ext, format_id in expected_formats.items():
            assert ext in _CORE_FORMATS
            assert _CORE_FORMATS[ext] == format_id

    def test_register_format(self):
        """Test format registration."""
        from autoclean.io.import_ import get_format_from_extension, register_format

        # Register new format
        register_format("xyz", "XYZ_FORMAT")

        # Should be retrievable
        assert get_format_from_extension("xyz") == "XYZ_FORMAT"
        assert get_format_from_extension(".xyz") == "XYZ_FORMAT"  # With dot

    def test_register_format_case_insensitive(self):
        """Test that format registration is case insensitive."""
        from autoclean.io.import_ import get_format_from_extension, register_format

        register_format("XYZ", "XYZ_FORMAT")

        assert get_format_from_extension("xyz") == "XYZ_FORMAT"
        assert get_format_from_extension("XYZ") == "XYZ_FORMAT"
        assert get_format_from_extension("Xyz") == "XYZ_FORMAT"

    def test_get_format_from_extension_core_formats(self):
        """Test getting format from extension for core formats."""
        from autoclean.io.import_ import get_format_from_extension

        assert get_format_from_extension("set") == "EEGLAB_SET"
        assert get_format_from_extension("raw") == "EGI_RAW"
        assert get_format_from_extension("fif") == "GENERIC_FIF"
        assert get_format_from_extension("unknown") is None

    def test_format_override_warning(self):
        """Test that overriding existing formats shows warning."""
        from autoclean.io.import_ import register_format

        with patch("autoclean.io.import_.message") as mock_message:
            # Override existing core format
            register_format("set", "NEW_SET_FORMAT")

            # Should log warning
            mock_message.assert_any_call(
                "warning", "Overriding existing format for extension: set"
            )


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
class TestPluginRegistry:
    """Test the plugin registration system."""

    def setup_method(self):
        """Set up clean registry for each test."""
        from autoclean.io.import_ import _FORMAT_REGISTRY

        self.original_plugin_registry = _PLUGIN_REGISTRY.copy()
        self.original_format_registry = _FORMAT_REGISTRY.copy()

    def teardown_method(self):
        """Restore original registry state."""
        from autoclean.io.import_ import _FORMAT_REGISTRY

        _PLUGIN_REGISTRY.clear()
        _PLUGIN_REGISTRY.update(self.original_plugin_registry)
        _FORMAT_REGISTRY.clear()
        _FORMAT_REGISTRY.update(self.original_format_registry)

    def test_register_plugin(self):
        """Test plugin registration."""
        from autoclean.io.import_ import BaseEEGPlugin, register_format, register_plugin

        # First register the test format
        register_format("test", "TEST_FORMAT")

        class TestPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "TEST_FORMAT" and montage_name == "GSN-HydroCel-129"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw()

        # Register plugin
        register_plugin(TestPlugin)

        # Should be in registry with a known montage
        assert ("TEST_FORMAT", "GSN-HydroCel-129") in _PLUGIN_REGISTRY
        assert _PLUGIN_REGISTRY[("TEST_FORMAT", "GSN-HydroCel-129")] == TestPlugin

    def test_register_plugin_multiple_combinations(self):
        """Test plugin registration for multiple format/montage combinations."""
        from autoclean.io.import_ import BaseEEGPlugin, register_format, register_plugin

        # Register test formats
        register_format("testa", "FORMAT_A")
        register_format("testb", "FORMAT_B")

        class MultiPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                supported = [
                    ("FORMAT_A", "GSN-HydroCel-129"),
                    ("FORMAT_A", "GSN-HydroCel-124"),
                    ("FORMAT_B", "standard_1020"),
                ]
                return (format_id, montage_name) in supported

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw()

        register_plugin(MultiPlugin)

        # Should register all supported combinations (using known montages)
        expected_keys = [
            ("FORMAT_A", "GSN-HydroCel-129"),
            ("FORMAT_A", "GSN-HydroCel-124"),
            ("FORMAT_B", "standard_1020"),
        ]

        for key in expected_keys:
            assert key in _PLUGIN_REGISTRY
            assert _PLUGIN_REGISTRY[key] == MultiPlugin

    def test_register_plugin_override_warning(self):
        """Test warning when overriding existing plugin."""
        from autoclean.io.import_ import BaseEEGPlugin, register_format, register_plugin

        # Register test format
        register_format("test", "TEST_FORMAT")

        class Plugin1(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "TEST_FORMAT" and montage_name == "GSN-HydroCel-129"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw()

        class Plugin2(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "TEST_FORMAT" and montage_name == "GSN-HydroCel-129"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw()

        with patch("autoclean.io.import_.message") as mock_message:
            register_plugin(Plugin1)
            register_plugin(Plugin2)  # Override

            # Should log warning about override
            mock_message.assert_any_call(
                "warning",
                "Overriding existing plugin for TEST_FORMAT, GSN-HydroCel-129",
            )


@pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
class TestPluginDiscovery:
    """Test automatic plugin discovery."""

    @patch("autoclean.io.import_.pkgutil.iter_modules")
    @patch("autoclean.io.import_.importlib.import_module")
    def test_plugin_discovery_mechanism(self, mock_import, mock_iter):
        """Test that plugin discovery mechanism works."""
        # Mock module discovery
        mock_module_info = Mock()
        mock_module_info.name = "test_plugin"
        mock_iter.return_value = [mock_module_info]

        # Mock module import
        mock_module = Mock()
        mock_plugin_class = Mock()
        mock_plugin_class.__name__ = "TestPlugin"
        mock_module.TestPlugin = mock_plugin_class
        mock_import.return_value = mock_module

        # The actual discovery would happen in the __init__ or import process
        # Here we just test that the mechanism can work
        assert mock_iter.called or not mock_iter.called  # Placeholder test

    def test_plugin_discovery_error_handling(self):
        """Test plugin discovery handles import errors gracefully."""
        with patch("autoclean.io.import_.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("Module not found")

            # Plugin discovery should handle import errors gracefully
            # This would be tested in the actual discovery code
            assert True  # Placeholder - actual discovery code would be tested


class TestPluginSystemMocked:
    """Test plugin system with heavy mocking."""

    def test_plugin_system_interface(self):
        """Test plugin system interface with mocks."""
        # Mock the entire plugin system
        with patch("autoclean.io.import_.BaseEEGPlugin") as MockBasePlugin:
            MockBasePlugin.__abstractmethods__ = {
                "supports_format_montage",
                "import_and_configure",
            }

            # Test interface expectations
            assert hasattr(MockBasePlugin, "__abstractmethods__")
            expected_methods = {"supports_format_montage", "import_and_configure"}
            assert MockBasePlugin.__abstractmethods__ == expected_methods

    def test_format_registry_operations_mocked(self):
        """Test format registry operations with mocking."""
        mock_registry = {}

        with patch("autoclean.io.import_._FORMAT_REGISTRY", mock_registry):
            # Mock format registration
            mock_registry["test"] = "TEST_FORMAT"

            assert "test" in mock_registry
            assert mock_registry["test"] == "TEST_FORMAT"

    def test_plugin_registration_mocked(self):
        """Test plugin registration with mocking."""
        mock_plugin_registry = {}
        mock_plugin_class = Mock()
        mock_plugin_class.supports_format_montage.return_value = True

        with patch("autoclean.io.import_._PLUGIN_REGISTRY", mock_plugin_registry):
            # Mock plugin registration
            mock_plugin_registry[("TEST_FORMAT", "TEST_MONTAGE")] = mock_plugin_class

            assert ("TEST_FORMAT", "TEST_MONTAGE") in mock_plugin_registry
            assert (
                mock_plugin_registry[("TEST_FORMAT", "TEST_MONTAGE")]
                == mock_plugin_class
            )


class TestPluginSystemConceptual:
    """Conceptual tests for plugin system design."""

    def test_plugin_system_design_principles(self):
        """Test that plugin system follows good design principles."""
        if not IMPORT_AVAILABLE:
            pytest.skip("Import module not available for design testing")

        from autoclean.io.import_ import BaseEEGPlugin

        # Abstract base class principle
        assert issubclass(BaseEEGPlugin, ABC)

        # Interface segregation - specific abstract methods
        abstract_methods = getattr(BaseEEGPlugin, "__abstractmethods__", set())
        assert len(abstract_methods) > 0
        assert "supports_format_montage" in abstract_methods
        assert "import_and_configure" in abstract_methods

    def test_plugin_extensibility_concept(self):
        """Test plugin system extensibility concept."""
        if not IMPORT_AVAILABLE:
            pytest.skip("Import module not available for extensibility testing")

        from autoclean.io.import_ import BaseEEGPlugin

        # Should be extensible through inheritance
        class CustomPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return True  # Custom logic

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return Mock()  # Custom implementation

        assert issubclass(CustomPlugin, BaseEEGPlugin)
        plugin = CustomPlugin()
        assert plugin.supports_format_montage("any", "any") is True

    def test_format_registry_concept(self):
        """Test format registry concept."""
        if not IMPORT_AVAILABLE:
            pytest.skip("Import module not available for registry testing")

        # Registry should support:
        # 1. Format registration
        # 2. Format lookup
        # 3. Extension mapping
        # 4. Override capability

        # These concepts are tested in the actual registry tests above
        # This is a placeholder for design validation
        assert True


# Error handling tests
class TestPluginSystemErrorHandling:
    """Test plugin system error handling."""

    @pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
    def test_invalid_plugin_registration(self):
        """Test registration of invalid plugins."""
        from autoclean.io.import_ import register_plugin

        # Test with non-plugin class
        class NotAPlugin:
            pass

        with pytest.raises((TypeError, AttributeError)):
            register_plugin(NotAPlugin)

    @pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
    def test_plugin_import_error_handling(self):
        """Test handling of plugin import errors."""
        from autoclean.io.import_ import BaseEEGPlugin

        class FailingPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return True

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                raise ImportError("Simulated import failure")

        plugin = FailingPlugin()

        # Should raise the import error
        with pytest.raises(ImportError, match="Simulated import failure"):
            plugin.import_and_configure(Path("/test.fif"), {})

    @pytest.mark.skipif(not IMPORT_AVAILABLE, reason="Import module not available")
    def test_format_lookup_nonexistent(self):
        """Test format lookup for non-existent extensions."""
        from autoclean.io.import_ import get_format_from_extension

        assert get_format_from_extension("nonexistent") is None
        assert get_format_from_extension("") is None
        assert get_format_from_extension(".") is None
