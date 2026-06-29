"""Unit tests for BioSemi BDF plugins."""

from pathlib import Path
from unittest.mock import Mock, patch

import mne
import numpy as np
import pytest

from autoclean.io.import_ import (
    _PLUGIN_REGISTRY,
    find_plugin_for_combination,
    get_format_from_extension,
    normalize_montage_name,
    register_plugin,
)
from autoclean.plugins.eeg_plugins.bdf_biosemi32_plugin import BDFBiosemi32Plugin
from autoclean.plugins.eeg_plugins.bdf_biosemi64_plugin import BDFBiosemi64Plugin
from autoclean.plugins.eeg_plugins.bdf_biosemi128_plugin import BDFBiosemi128Plugin
from autoclean.plugins.eeg_plugins.bdf_biosemi256_plugin import BDFBiosemi256Plugin
from tests.fixtures.synthetic_data import create_synthetic_raw
from tests.fixtures.test_utils import EEGAssertions

# Test with mock plugins to avoid heavy import dependencies
try:
    from autoclean.io.import_ import BaseEEGPlugin

    PLUGIN_BASE_AVAILABLE = True
except ImportError:
    PLUGIN_BASE_AVAILABLE = False


def _make_biosemi_raw_with_externals() -> mne.io.RawArray:
    """Create a compact BioSemi-like Raw with EEG, mastoids, externals, and status."""
    ch_names = [
        "Fp1",
        "Fp2",
        "C3",
        "C4",
        "O1",
        "O2",
        "LM",
        "RM",
        "LVE",
        "RVE",
        "LHE",
        "RHE",
        "EXG7",
        "EXG8",
        "Status",
    ]
    ch_types = ["eeg"] * 14 + ["misc"]
    data = np.vstack(
        [
            np.full(8, 10.0),
            np.full(8, 12.0),
            np.full(8, 14.0),
            np.full(8, 16.0),
            np.full(8, 18.0),
            np.full(8, 20.0),
            np.full(8, 1.0),
            np.full(8, 3.0),
            np.full(8, 0.5),
            np.full(8, 0.75),
            np.full(8, 1.25),
            np.full(8, 1.5),
            np.full(8, 2.0),
            np.full(8, 2.5),
            np.arange(8, dtype=float),
        ]
    )
    info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types=ch_types)
    return mne.io.RawArray(data, info, verbose=False)


class TestBDFBiosemi32Plugin:
    """Test BioSemi BDF 32-channel plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_format_montage_support(self):
        """Test BDF biosemi32 plugin format and montage support detection."""

        # Create mock plugin based on real interface
        class MockBDFBiosemi32Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF" and montage_name == "biosemi32"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="biosemi32", n_channels=32)

        plugin_class = MockBDFBiosemi32Plugin

        # Should support correct combination
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi32") is True

        # Should not support incorrect combinations
        assert plugin_class.supports_format_montage("EGI_RAW", "biosemi32") is False
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi64") is False
        assert plugin_class.supports_format_montage("EEGLAB_SET", "biosemi32") is False

    @patch("mne.io.read_raw_bdf")
    def test_bdf_biosemi32_import_functionality(self, mock_read_bdf):
        """Test BDF biosemi32 file import functionality."""
        # Mock the MNE import function
        mock_raw = create_synthetic_raw(montage="biosemi32", n_channels=32)
        mock_read_bdf.return_value = mock_raw

        # Create mock plugin
        if PLUGIN_BASE_AVAILABLE:

            class MockBDFPlugin(BaseEEGPlugin):
                @classmethod
                def supports_format_montage(
                    cls, format_id: str, montage_name: str
                ) -> bool:
                    return format_id == "BIOSEMI_BDF" and montage_name == "biosemi32"

                def import_and_configure(
                    self, file_path: Path, autoclean_dict: dict, preload: bool = True
                ):
                    # Simulate real plugin behavior
                    raw = mne.io.read_raw_bdf(
                        input_fname=file_path,
                        preload=preload,
                        stim_channel="auto",
                        exclude=[],
                    )
                    return raw

            plugin = MockBDFPlugin()
            test_file = Path("/test/data.bdf")
            config = {"montage": {"value": "biosemi32"}}

            result = plugin.import_and_configure(test_file, config)

            # Verify MNE function was called correctly
            mock_read_bdf.assert_called_once_with(
                input_fname=test_file,
                preload=True,
                stim_channel="auto",
                exclude=[],
            )

            # Verify result properties
            EEGAssertions.assert_raw_properties(result, expected_n_channels=32)

    def test_biosemi32_montage_configuration(self):
        """Test biosemi32 montage configuration."""
        # Test montage-specific configuration
        raw = create_synthetic_raw(montage="biosemi32", n_channels=32)

        # Should have 32 channels
        assert len(raw.ch_names) == 32

        # Test channel types
        assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())


class TestBDFBiosemi64Plugin:
    """Test BioSemi BDF 64-channel plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_format_montage_support(self):
        """Test BDF biosemi64 plugin format and montage support detection."""

        class MockBDFBiosemi64Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF" and montage_name == "biosemi64"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="biosemi64", n_channels=64)

        plugin_class = MockBDFBiosemi64Plugin

        # Should support correct combination
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi64") is True

        # Should not support incorrect combinations
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi32") is False
        assert (
            plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi128") is False
        )
        assert plugin_class.supports_format_montage("EGI_RAW", "biosemi64") is False

    @patch("mne.io.read_raw_bdf")
    def test_bdf_biosemi64_import_functionality(self, mock_read_bdf):
        """Test BDF biosemi64 file import functionality."""
        mock_raw = create_synthetic_raw(montage="biosemi64", n_channels=64)
        mock_read_bdf.return_value = mock_raw

        if PLUGIN_BASE_AVAILABLE:

            class MockBDFPlugin(BaseEEGPlugin):
                @classmethod
                def supports_format_montage(
                    cls, format_id: str, montage_name: str
                ) -> bool:
                    return format_id == "BIOSEMI_BDF" and montage_name == "biosemi64"

                def import_and_configure(
                    self, file_path: Path, autoclean_dict: dict, preload: bool = True
                ):
                    raw = mne.io.read_raw_bdf(
                        input_fname=file_path,
                        preload=preload,
                        stim_channel="auto",
                        exclude=[],
                    )
                    return raw

            plugin = MockBDFPlugin()
            test_file = Path("/test/data.bdf")
            config = {"montage": {"value": "biosemi64"}}

            result = plugin.import_and_configure(test_file, config)

            # Verify MNE function was called correctly
            mock_read_bdf.assert_called_once_with(
                input_fname=test_file,
                preload=True,
                stim_channel="auto",
                exclude=[],
            )

            # Verify result properties
            EEGAssertions.assert_raw_properties(result, expected_n_channels=64)

    def test_biosemi64_montage_configuration(self):
        """Test biosemi64 montage configuration."""
        raw = create_synthetic_raw(montage="biosemi64", n_channels=64)

        # Should have 64 channels
        assert len(raw.ch_names) == 64

        # Test channel types
        assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())


class TestBDFBiosemi128Plugin:
    """Test BioSemi BDF 128-channel plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_format_montage_support(self):
        """Test BDF biosemi128 plugin format and montage support detection."""

        class MockBDFBiosemi128Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF" and montage_name == "biosemi128"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="biosemi128", n_channels=128)

        plugin_class = MockBDFBiosemi128Plugin

        # Should support correct combination
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi128") is True

        # Should not support incorrect combinations
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi64") is False
        assert (
            plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi256") is False
        )

    @patch("mne.io.read_raw_bdf")
    def test_bdf_biosemi128_import_functionality(self, mock_read_bdf):
        """Test BDF biosemi128 file import functionality."""
        mock_raw = create_synthetic_raw(montage="biosemi128", n_channels=128)
        mock_read_bdf.return_value = mock_raw

        if PLUGIN_BASE_AVAILABLE:

            class MockBDFPlugin(BaseEEGPlugin):
                @classmethod
                def supports_format_montage(
                    cls, format_id: str, montage_name: str
                ) -> bool:
                    return format_id == "BIOSEMI_BDF" and montage_name == "biosemi128"

                def import_and_configure(
                    self, file_path: Path, autoclean_dict: dict, preload: bool = True
                ):
                    raw = mne.io.read_raw_bdf(
                        input_fname=file_path,
                        preload=preload,
                        stim_channel="auto",
                        exclude=[],
                    )
                    return raw

            plugin = MockBDFPlugin()
            test_file = Path("/test/data.bdf")
            config = {"montage": {"value": "biosemi128"}}

            result = plugin.import_and_configure(test_file, config)

            # Verify MNE function was called correctly
            mock_read_bdf.assert_called_once_with(
                input_fname=test_file,
                preload=True,
                stim_channel="auto",
                exclude=[],
            )

            # Verify result properties
            EEGAssertions.assert_raw_properties(result, expected_n_channels=128)

    def test_biosemi128_montage_configuration(self):
        """Test biosemi128 montage configuration."""
        raw = create_synthetic_raw(montage="biosemi128", n_channels=128)

        # Should have 128 channels
        assert len(raw.ch_names) == 128

        # Test channel types
        assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())


class TestBDFBiosemi256Plugin:
    """Test BioSemi BDF 256-channel plugin functionality."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_plugin_format_montage_support(self):
        """Test BDF biosemi256 plugin format and montage support detection."""

        class MockBDFBiosemi256Plugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF" and montage_name == "biosemi256"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="biosemi256", n_channels=256)

        plugin_class = MockBDFBiosemi256Plugin

        # Should support correct combination
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi256") is True

        # Should not support incorrect combinations
        assert (
            plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi128") is False
        )
        assert plugin_class.supports_format_montage("EGI_RAW", "biosemi256") is False

    @patch("mne.io.read_raw_bdf")
    def test_bdf_biosemi256_import_functionality(self, mock_read_bdf):
        """Test BDF biosemi256 file import functionality."""
        mock_raw = create_synthetic_raw(montage="biosemi256", n_channels=256)
        mock_read_bdf.return_value = mock_raw

        if PLUGIN_BASE_AVAILABLE:

            class MockBDFPlugin(BaseEEGPlugin):
                @classmethod
                def supports_format_montage(
                    cls, format_id: str, montage_name: str
                ) -> bool:
                    return format_id == "BIOSEMI_BDF" and montage_name == "biosemi256"

                def import_and_configure(
                    self, file_path: Path, autoclean_dict: dict, preload: bool = True
                ):
                    raw = mne.io.read_raw_bdf(
                        input_fname=file_path,
                        preload=preload,
                        stim_channel="auto",
                        exclude=[],
                    )
                    return raw

            plugin = MockBDFPlugin()
            test_file = Path("/test/data.bdf")
            config = {"montage": {"value": "biosemi256"}}

            result = plugin.import_and_configure(test_file, config)

            # Verify MNE function was called correctly
            mock_read_bdf.assert_called_once_with(
                input_fname=test_file,
                preload=True,
                stim_channel="auto",
                exclude=[],
            )

            # Verify result properties
            EEGAssertions.assert_raw_properties(result, expected_n_channels=256)

    def test_biosemi256_montage_configuration(self):
        """Test biosemi256 montage configuration."""
        raw = create_synthetic_raw(montage="biosemi256", n_channels=256)

        # Should have 256 channels
        assert len(raw.ch_names) == 256

        # Test channel types
        assert all(ch_type == "eeg" for ch_type in raw.get_channel_types())


class TestBDFMontageComparison:
    """Test differences between BioSemi montages."""

    def test_biosemi_montage_channel_count_differences(self):
        """Test that different BioSemi montages have different channel counts."""
        raw_32 = create_synthetic_raw(montage="biosemi32", n_channels=32)
        raw_64 = create_synthetic_raw(montage="biosemi64", n_channels=64)
        raw_128 = create_synthetic_raw(montage="biosemi128", n_channels=128)
        raw_256 = create_synthetic_raw(montage="biosemi256", n_channels=256)

        # Should have different channel counts
        assert len(raw_32.ch_names) == 32
        assert len(raw_64.ch_names) == 64
        assert len(raw_128.ch_names) == 128
        assert len(raw_256.ch_names) == 256

        # Channel counts should be strictly increasing
        assert len(raw_32.ch_names) < len(raw_64.ch_names)
        assert len(raw_64.ch_names) < len(raw_128.ch_names)
        assert len(raw_128.ch_names) < len(raw_256.ch_names)


class TestBDFPluginErrorHandling:
    """Test BDF plugin error handling and edge cases."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_bdf_plugin_file_not_found_error(self):
        """Test BDF plugin behavior with non-existent files."""

        class MockBDFPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Simulate file not found
                raise FileNotFoundError(f"BDF file not found: {file_path}")

        plugin = MockBDFPlugin()

        with pytest.raises(FileNotFoundError, match="BDF file not found"):
            plugin.import_and_configure(Path("/nonexistent/file.bdf"), {})

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_bdf_plugin_invalid_status_channel_error(self):
        """Test BDF plugin behavior with invalid status channel."""

        class MockBDFPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Simulate status channel error
                raise RuntimeError("Could not detect BDF status channel")

        plugin = MockBDFPlugin()

        with pytest.raises(RuntimeError, match="status channel"):
            plugin.import_and_configure(Path("/test/invalid.bdf"), {})

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_bdf_plugin_montage_mismatch_error(self):
        """Test BDF plugin behavior with montage mismatches."""

        class MockBDFPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF" and montage_name == "biosemi64"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Simulate montage mismatch
                raise RuntimeError(
                    "BDF file has 128 channels but biosemi64 montage expects 64"
                )

        plugin = MockBDFPlugin()

        with pytest.raises(RuntimeError, match="montage expects"):
            plugin.import_and_configure(Path("/test/data.bdf"), {})


class TestBDFPluginIntegration:
    """Test BDF plugin integration with the broader system."""

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_bdf_plugin_output_validation(self):
        """Test that BDF plugin outputs are valid Raw objects."""

        class MockBDFPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                return create_synthetic_raw(montage="biosemi64", n_channels=64)

        plugin = MockBDFPlugin()
        result = plugin.import_and_configure(Path("/test/data.bdf"), {})

        # Should return valid Raw object
        EEGAssertions.assert_raw_properties(result)
        assert hasattr(result, "info")
        assert hasattr(result, "get_data")
        assert hasattr(result, "ch_names")
        assert hasattr(result, "annotations")  # BDF files have annotations

    @pytest.mark.skipif(not PLUGIN_BASE_AVAILABLE, reason="Plugin base not available")
    def test_bdf_plugin_status_channel_handling(self):
        """Test that BDF plugins properly handle status channels."""

        class MockBDFPlugin(BaseEEGPlugin):
            @classmethod
            def supports_format_montage(cls, format_id: str, montage_name: str) -> bool:
                return format_id == "BIOSEMI_BDF"

            def import_and_configure(
                self, file_path: Path, autoclean_dict: dict, preload: bool = True
            ):
                # Create raw with both EEG and STIM channels
                raw = create_synthetic_raw(montage="biosemi64", n_channels=64)
                # Note: In real BDF files, stim_channel='auto' would add status channel
                return raw

        plugin = MockBDFPlugin()
        result = plugin.import_and_configure(Path("/test/data.bdf"), {})

        # Should have valid channel structure
        assert len(result.ch_names) > 0
        EEGAssertions.assert_raw_properties(result)


class TestBDFPluginMocked:
    """Test BDF plugin functionality with heavy mocking."""

    def test_bdf_plugin_interface_mocked(self):
        """Test BDF plugin interface with complete mocking."""
        # Mock the entire BDF plugin system
        mock_plugin = Mock()
        mock_plugin.supports_format_montage.return_value = True
        mock_plugin.import_and_configure.return_value = create_synthetic_raw(
            montage="biosemi64", n_channels=64
        )

        # Test interface calls
        assert mock_plugin.supports_format_montage("BIOSEMI_BDF", "biosemi64") is True
        result = mock_plugin.import_and_configure(Path("/test/data.bdf"), {})

        # Verify mock was called
        mock_plugin.supports_format_montage.assert_called_once()
        mock_plugin.import_and_configure.assert_called_once()

        # Verify result
        EEGAssertions.assert_raw_properties(result)

    def test_multiple_biosemi_plugins_mocked(self):
        """Test multiple BioSemi plugin coordination with mocking."""
        # Mock multiple BioSemi plugins
        plugin_32 = Mock()
        plugin_32.supports_format_montage.side_effect = (
            lambda f, m: f == "BIOSEMI_BDF" and m == "biosemi32"
        )

        plugin_64 = Mock()
        plugin_64.supports_format_montage.side_effect = (
            lambda f, m: f == "BIOSEMI_BDF" and m == "biosemi64"
        )

        plugin_128 = Mock()
        plugin_128.supports_format_montage.side_effect = (
            lambda f, m: f == "BIOSEMI_BDF" and m == "biosemi128"
        )

        # Test plugin selection logic
        plugins = [plugin_32, plugin_64, plugin_128]

        # Find plugin for biosemi64
        selected_plugin = None
        for plugin in plugins:
            if plugin.supports_format_montage("BIOSEMI_BDF", "biosemi64"):
                selected_plugin = plugin
                break

        assert selected_plugin == plugin_64

        # Find plugin for biosemi128
        selected_plugin = None
        for plugin in plugins:
            if plugin.supports_format_montage("BIOSEMI_BDF", "biosemi128"):
                selected_plugin = plugin
                break

        assert selected_plugin == plugin_128


def test_biosemi_bdf_format_and_plugin_registry_routes_all_supported_montages():
    """BioSemi BDF should route explicitly to each supported montage plugin."""
    original_registry = _PLUGIN_REGISTRY.copy()
    original_discovered = __import__(
        "autoclean.io.import_", fromlist=["_PLUGINS_DISCOVERED"]
    )._PLUGINS_DISCOVERED
    import_module = __import__("autoclean.io.import_", fromlist=["_PLUGINS_DISCOVERED"])
    import_module._PLUGINS_DISCOVERED = True
    _PLUGIN_REGISTRY.clear()
    try:
        for plugin_class in (
            BDFBiosemi32Plugin,
            BDFBiosemi64Plugin,
            BDFBiosemi128Plugin,
            BDFBiosemi256Plugin,
        ):
            register_plugin(plugin_class)

        assert get_format_from_extension(".bdf") == "BIOSEMI_BDF"
        assert _PLUGIN_REGISTRY[("BIOSEMI_BDF", "biosemi32")] is BDFBiosemi32Plugin
        assert _PLUGIN_REGISTRY[("BIOSEMI_BDF", "biosemi64")] is BDFBiosemi64Plugin
        assert _PLUGIN_REGISTRY[("BIOSEMI_BDF", "biosemi128")] is BDFBiosemi128Plugin
        assert _PLUGIN_REGISTRY[("BIOSEMI_BDF", "biosemi256")] is BDFBiosemi256Plugin
        assert (
            find_plugin_for_combination("BIOSEMI_BDF", "BioSemi-256").__class__
            is BDFBiosemi256Plugin
        )
    finally:
        _PLUGIN_REGISTRY.clear()
        _PLUGIN_REGISTRY.update(original_registry)
        import_module._PLUGINS_DISCOVERED = original_discovered


def test_biosemi_montage_aliases_accept_user_facing_names():
    """User-facing BioSemi-32/64/128/256 names should map to plugin keys."""
    assert normalize_montage_name("BioSemi-32") == "biosemi32"
    assert normalize_montage_name("BioSemi-64") == "biosemi64"
    assert normalize_montage_name("BioSemi-128") == "biosemi128"
    assert normalize_montage_name("BioSemi-256") == "biosemi256"
    assert normalize_montage_name("biosemi256") == "biosemi256"
    assert normalize_montage_name("GSN-HydroCel-129") == "GSN-HydroCel-129"


def test_biosemi_bdf_registry_rejects_unsupported_montage():
    """Unsupported BioSemi BDF montages should fail instead of using a wrong plugin."""
    original_registry = _PLUGIN_REGISTRY.copy()
    import_module = __import__("autoclean.io.import_", fromlist=["_PLUGINS_DISCOVERED"])
    original_discovered = import_module._PLUGINS_DISCOVERED
    import_module._PLUGINS_DISCOVERED = True
    _PLUGIN_REGISTRY.clear()
    try:
        for plugin_class in (
            BDFBiosemi32Plugin,
            BDFBiosemi64Plugin,
            BDFBiosemi128Plugin,
            BDFBiosemi256Plugin,
        ):
            register_plugin(plugin_class)

        assert find_plugin_for_combination("BIOSEMI_BDF", "biosemi16") is None
    finally:
        _PLUGIN_REGISTRY.clear()
        _PLUGIN_REGISTRY.update(original_registry)
        import_module._PLUGINS_DISCOVERED = original_discovered


def test_biosemi_process_events_falls_back_to_status_channel():
    """BioSemi plugins should read triggers from the Status stim channel."""
    info = mne.create_info(
        ch_names=["Fp1", "Fp2", "Status"],
        sfreq=100.0,
        ch_types=["eeg", "eeg", "stim"],
    )
    data = np.zeros((3, 40))
    data[2, 10:12] = 7
    data[2, 25:27] = 9
    raw = mne.io.RawArray(data, info, verbose=False)

    events, event_id, events_df = BDFBiosemi64Plugin().process_events(raw)

    assert events[:, 0].tolist() == [10, 25]
    assert events[:, 2].tolist() == [7, 9]
    assert event_id == {"Status-7": 7, "Status-9": 9}
    assert events_df["type"].tolist() == ["Status-7", "Status-9"]


def test_biosemi_process_events_does_not_swallow_unexpected_errors(monkeypatch):
    """Unexpected event-processing errors should remain visible to callers."""
    info = mne.create_info(
        ch_names=["Fp1", "Status"], sfreq=100.0, ch_types=["eeg", "stim"]
    )
    raw = mne.io.RawArray(np.zeros((2, 10)), info, verbose=False)

    def raise_unexpected(*args, **kwargs):
        raise TypeError("unexpected parser failure")

    monkeypatch.setattr(mne, "events_from_annotations", raise_unexpected)

    with pytest.raises(TypeError, match="unexpected parser failure"):
        BDFBiosemi64Plugin().process_events(raw)


@pytest.mark.parametrize(
    ("plugin_class", "montage_name"),
    [
        (BDFBiosemi32Plugin, "biosemi32"),
        (BDFBiosemi64Plugin, "biosemi64"),
        (BDFBiosemi128Plugin, "biosemi128"),
        (BDFBiosemi256Plugin, "biosemi256"),
    ],
)
@patch("mne.io.read_raw_bdf")
def test_biosemi_plugins_rereference_to_mastoids_on_import(
    mock_read_bdf,
    plugin_class,
    montage_name,
):
    """BioSemi BDF imports should immediately rereference to mastoids when present."""
    mock_read_bdf.return_value = _make_biosemi_raw_with_externals()

    result = plugin_class().import_and_configure(
        Path("/test/data.bdf"),
        {"eeg_system": montage_name},
    )

    expected = np.array([8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
    observed = result.get_data(picks=["Fp1", "Fp2", "C3", "C4", "O1", "O2"])[:, 0]
    np.testing.assert_allclose(observed, expected)
    assert "LM" not in result.ch_names
    assert "RM" not in result.ch_names
    assert "Status" in result.ch_names


@patch("mne.io.read_raw_bdf")
def test_biosemi_plugin_can_keep_reference_and_external_channels(mock_read_bdf):
    """BioSemi import options should allow retaining reference and external channels."""
    mock_read_bdf.return_value = _make_biosemi_raw_with_externals()

    result = BDFBiosemi64Plugin().import_and_configure(
        Path("/test/data.bdf"),
        {
            "eeg_system": "biosemi64",
            "biosemi_import": {
                "keep_reference_channels": True,
                "keep_external_channels": True,
            },
        },
    )

    for channel in ["LM", "RM", "LVE", "RVE", "LHE", "RHE", "EXG7", "EXG8", "Status"]:
        assert channel in result.ch_names

    assert result.get_channel_types(["LVE"])[0] == "eog"
    assert result.get_channel_types(["EXG7"])[0] == "misc"
    assert result.get_channel_types(["Status"])[0] == "stim"
