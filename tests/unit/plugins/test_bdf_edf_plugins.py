from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.fixtures.synthetic_data import create_synthetic_raw
from tests.fixtures.test_utils import EEGAssertions

try:
    from autoclean.plugins.eeg_plugins.bdf_biosemi64_plugin import BiosemiBDFPlugin
    from autoclean.plugins.eeg_plugins.edf_standard1020_plugin import (
        EDFStandard1020Plugin,
    )

    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False


@pytest.mark.skipif(not PLUGINS_AVAILABLE, reason="Plugins not available")
class TestBDFPlugin:
    """Tests for the Biosemi BDF plugin."""

    def test_supports_format_montage(self):
        plugin_class = BiosemiBDFPlugin
        assert plugin_class.supports_format_montage("BIOSEMI_BDF", "biosemi64")
        assert not plugin_class.supports_format_montage(
            "BIOSEMI_BDF", "standard_1020"
        )
        assert not plugin_class.supports_format_montage("OTHER", "biosemi64")

    @patch("mne.io.read_raw_bdf")
    def test_import_and_configure(self, mock_read_bdf):
        mock_raw = create_synthetic_raw(montage="standard_1020", n_channels=64)
        mock_raw.set_montage = Mock()
        mock_read_bdf.return_value = mock_raw

        plugin = BiosemiBDFPlugin()
        result = plugin.import_and_configure(Path("/test.bdf"), {})

        mock_read_bdf.assert_called_once_with(
            Path("/test.bdf"), preload=True, verbose=True
        )
        mock_raw.set_montage.assert_called_once()
        EEGAssertions.assert_raw_properties(result, expected_n_channels=64)


@pytest.mark.skipif(not PLUGINS_AVAILABLE, reason="Plugins not available")
class TestEDFPlugin:
    """Tests for the EDF standard 10-20 plugin."""

    def test_supports_format_montage(self):
        plugin_class = EDFStandard1020Plugin
        assert plugin_class.supports_format_montage("EDF_FORMAT", "standard_1020")
        assert not plugin_class.supports_format_montage("EDF_FORMAT", "biosemi64")
        assert not plugin_class.supports_format_montage("BIOSEMI_BDF", "standard_1020")

    @patch("mne.io.read_raw_edf")
    def test_import_and_configure(self, mock_read_edf):
        mock_raw = create_synthetic_raw(montage="standard_1020", n_channels=32)
        mock_read_edf.return_value = mock_raw

        plugin = EDFStandard1020Plugin()
        result = plugin.import_and_configure(Path("/test.edf"), {})

        mock_read_edf.assert_called_once_with(
            Path("/test.edf"), preload=True, verbose=True
        )
        EEGAssertions.assert_raw_properties(result, expected_n_channels=32)
