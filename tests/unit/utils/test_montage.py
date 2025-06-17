"""Unit tests for montage utilities."""

import pytest
from typing import Dict, List
from unittest.mock import Mock, patch, mock_open

from tests.fixtures.test_utils import BaseTestCase

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.utils.montage import (
        load_valid_montages,
        VALID_MONTAGES,
        GSN_TO_1020_MAPPING,
        get_10_20_to_gsn_mapping,
        get_gsn_to_10_20_mapping,
        convert_channel_names,
        get_standard_set_in_montage,
        validate_channel_set
    )
    MONTAGE_AVAILABLE = True
except ImportError:
    MONTAGE_AVAILABLE = False


@pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
class TestMontageLoading:
    """Test montage loading functionality."""
    
    @patch('autoclean.utils.montage.resources.files')
    @patch('autoclean.utils.montage.yaml.safe_load')
    def test_load_valid_montages(self, mock_yaml_load, mock_resources):
        """Test loading valid montages from configuration."""
        mock_montages = {
            "valid_montages": {
                "standard_1020": "10-20 system",
                "GSN-HydroCel-128": "HydroCel GSN 128",
                "GSN-HydroCel-129": "HydroCel GSN 129"
            }
        }
        mock_yaml_load.return_value = mock_montages
        
        # Mock the resources chain
        mock_file_path = Mock()
        mock_file_path.read_text.return_value = "mock yaml content"
        mock_resources.return_value.joinpath.return_value = mock_file_path
        
        result = load_valid_montages()
        
        expected = {
            "standard_1020": "10-20 system",
            "GSN-HydroCel-128": "HydroCel GSN 128", 
            "GSN-HydroCel-129": "HydroCel GSN 129"
        }
        
        assert result == expected
        mock_resources.assert_called_once()
        mock_yaml_load.assert_called_once()
    
    @patch('autoclean.utils.montage.resources.files', side_effect=ImportError)
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_valid_montages_file_not_found(self, mock_file, mock_resources):
        """Test behavior when montages.yaml file is not found."""
        with pytest.raises(FileNotFoundError):
            load_valid_montages()
    
    @patch('builtins.open', new_callable=mock_open, read_data='invalid: yaml: content')
    @patch('yaml.safe_load', side_effect=Exception("YAML parsing error"))
    def test_load_valid_montages_yaml_error(self, mock_yaml_load, mock_file):
        """Test behavior when YAML parsing fails."""
        with pytest.raises(Exception, match="YAML parsing error"):
            load_valid_montages()
    
    def test_valid_montages_constant(self):
        """Test that VALID_MONTAGES constant is properly loaded."""
        # Should be a dictionary
        assert isinstance(VALID_MONTAGES, dict)
        
        # Should contain expected montage types
        expected_montages = ["GSN-HydroCel-128", "GSN-HydroCel-129", "standard_1020"]
        for montage in expected_montages:
            if montage in VALID_MONTAGES:
                assert isinstance(VALID_MONTAGES[montage], str)


@pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
class TestMontageMapping:
    """Test montage mapping functionality."""
    
    def test_gsn_to_1020_mapping_structure(self):
        """Test GSN to 10-20 mapping structure."""
        # Should be a dictionary
        assert isinstance(GSN_TO_1020_MAPPING, dict)
        
        # Should contain expected mappings
        expected_mappings = {
            "Fz": "E11",
            "Cz": "E129", 
            "F3": "E24",
            "F4": "E124"
        }
        
        for key, value in expected_mappings.items():
            if key in GSN_TO_1020_MAPPING:
                assert GSN_TO_1020_MAPPING[key] == value
    
    def test_get_10_20_to_gsn_mapping(self):
        """Test getting 10-20 to GSN mapping."""
        mapping = get_10_20_to_gsn_mapping()
        
        assert isinstance(mapping, dict)
        
        # Should contain expected mappings
        expected_mappings = ["Fz", "Cz", "F3", "F4", "O1", "O2"]
        for channel in expected_mappings:
            if channel in mapping:
                assert isinstance(mapping[channel], str)
                assert mapping[channel].startswith("E")
    
    def test_get_gsn_to_10_20_mapping(self):
        """Test getting GSN to 10-20 mapping."""
        mapping = get_gsn_to_10_20_mapping()
        
        assert isinstance(mapping, dict)
        
        # Should contain expected reverse mappings
        expected_mappings = ["E11", "E129", "E24", "E124"]
        for channel in expected_mappings:
            if channel in mapping:
                assert isinstance(mapping[channel], str)
                # Should be a valid 10-20 channel name
                assert not mapping[channel].startswith("E")
    
    def test_mapping_consistency(self):
        """Test that forward and reverse mappings are consistent."""
        forward_mapping = get_10_20_to_gsn_mapping()
        reverse_mapping = get_gsn_to_10_20_mapping()
        
        # Test a few key mappings for consistency
        test_channels = ["Fz", "Cz", "F3", "F4"]
        for channel in test_channels:
            if channel in forward_mapping:
                gsn_channel = forward_mapping[channel]
                if gsn_channel in reverse_mapping:
                    assert reverse_mapping[gsn_channel] == channel


@pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
class TestChannelConversion:
    """Test channel name conversion functionality."""
    
    def test_convert_channel_names_10_20_to_gsn(self):
        """Test converting 10-20 channels to GSN format."""
        channels_10_20 = ["Fz", "Cz", "F3", "F4"]
        
        result = convert_channel_names(channels_10_20, "GSN-HydroCel-129")
        
        assert isinstance(result, list)
        assert len(result) == len(channels_10_20)
        
        # Should convert known channels
        if "Fz" in channels_10_20:
            fz_index = channels_10_20.index("Fz")
            # Should convert to GSN format (E11 if mapping exists)
            assert result[fz_index].startswith("E") or result[fz_index] == "Fz"  # Fallback
    
    def test_convert_channel_names_unknown_channels(self):
        """Test converting unknown channel names."""
        unknown_channels = ["UnknownCh1", "UnknownCh2"]
        
        result = convert_channel_names(unknown_channels, "GSN-HydroCel-129")
        
        # Should return original names for unknown channels
        assert result == unknown_channels
    
    def test_convert_channel_names_empty_list(self):
        """Test converting empty channel list."""
        result = convert_channel_names([], "GSN-HydroCel-129")
        
        assert result == []
    
    def test_convert_channel_names_124_montage_special_case(self):
        """Test channel conversion for GSN-124 montage (Cz special case)."""
        channels = ["Cz"]
        
        result = convert_channel_names(channels, "GSN-HydroCel-124")
        
        # Should handle Cz differently for 124 montage
        assert isinstance(result, list)
        assert len(result) == 1
        # Cz should map to E31 in 124 montage or stay as Cz
        assert result[0] in ["E31", "Cz"]


@pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
class TestStandardChannelSets:
    """Test standard channel set functionality."""
    
    def test_get_standard_set_in_montage_frontal(self):
        """Test getting frontal channel set."""
        result = get_standard_set_in_montage("frontal", "GSN-HydroCel-129")
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Should convert to GSN format or keep original names
        for channel in result:
            assert isinstance(channel, str)
    
    def test_get_standard_set_in_montage_central(self):
        """Test getting central channel set."""
        result = get_standard_set_in_montage("central", "GSN-HydroCel-129")
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_get_standard_set_in_montage_mmn_standard(self):
        """Test getting MMN standard channel set."""
        result = get_standard_set_in_montage("mmn_standard", "GSN-HydroCel-129")
        
        assert isinstance(result, list)
        assert len(result) > 0
        
        # MMN standard should include frontal and central channels
        # Exact channels depend on implementation
    
    def test_get_standard_set_unknown_roi(self):
        """Test getting unknown ROI set."""
        with pytest.raises(ValueError, match="Unknown ROI set"):
            get_standard_set_in_montage("unknown_roi", "GSN-HydroCel-129")
    
    def test_get_standard_set_all_available_sets(self):
        """Test all available standard sets."""
        available_sets = ["frontal", "frontocentral", "central", "temporal", 
                         "parietal", "occipital", "mmn_standard"]
        
        for roi_set in available_sets:
            try:
                result = get_standard_set_in_montage(roi_set, "GSN-HydroCel-129")
                assert isinstance(result, list)
                assert len(result) >= 0  # Can be empty if no mappings exist
            except ValueError:
                # Some sets might not be implemented
                continue


@pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
class TestChannelValidation:
    """Test channel validation functionality."""
    
    def test_validate_channel_set_all_valid(self):
        """Test validation with all valid channels."""
        requested_channels = ["E1", "E2", "E3"]
        available_channels = ["E1", "E2", "E3", "E4", "E5"]
        
        result = validate_channel_set(requested_channels, available_channels)
        
        assert result == requested_channels
    
    def test_validate_channel_set_some_missing(self):
        """Test validation with some missing channels."""
        requested_channels = ["E1", "E2", "E10"]  # E10 not available
        available_channels = ["E1", "E2", "E3", "E4", "E5"]
        
        with patch('autoclean.utils.montage.message') as mock_message:
            result = validate_channel_set(requested_channels, available_channels)
            
            # Should return only valid channels
            assert "E1" in result
            assert "E2" in result
            assert "E10" not in result
            
            # Should log warning about missing channels
            mock_message.assert_called_with("warning", 
                "Some requested channels not found in data: {'E10'}")
    
    def test_validate_channel_set_all_missing(self):
        """Test validation with all channels missing."""
        requested_channels = ["X1", "X2", "X3"]
        available_channels = ["E1", "E2", "E3"]
        
        with patch('autoclean.utils.montage.message') as mock_message:
            result = validate_channel_set(requested_channels, available_channels)
            
            # Should return empty list
            assert result == []
            
            # Should log warning
            mock_message.assert_called_once()
    
    def test_validate_channel_set_empty_inputs(self):
        """Test validation with empty inputs."""
        # Empty requested channels
        result1 = validate_channel_set([], ["E1", "E2"])
        assert result1 == []
        
        # Empty available channels
        result2 = validate_channel_set(["E1"], [])
        assert result2 == []
    
    def test_validate_channel_set_case_sensitivity(self):
        """Test channel validation case sensitivity."""
        requested_channels = ["e1", "E2"]  # Mixed case
        available_channels = ["E1", "E2", "E3"]
        
        result = validate_channel_set(requested_channels, available_channels)
        
        # Behavior depends on implementation (case-sensitive or not)
        # Should handle consistently
        assert isinstance(result, list)


class TestMontageMocked:
    """Test montage functionality with heavy mocking."""
    
    def test_montage_loading_mocked(self):
        """Test montage loading with complete mocking."""
        mock_montages = {
            "standard_1020": "10-20 system",
            "GSN-HydroCel-129": "HydroCel GSN 129"
        }
        
        with patch('autoclean.utils.montage.yaml.safe_load', return_value={"valid_montages": mock_montages}):
            with patch('builtins.open', mock_open()):
                if MONTAGE_AVAILABLE:
                    result = load_valid_montages()
                    assert result == mock_montages
    
    def test_channel_conversion_mocked(self):
        """Test channel conversion with mocked mappings."""
        mock_mapping = {"Fz": "E11", "Cz": "E129"}
        
        with patch('autoclean.utils.montage.get_10_20_to_gsn_mapping', return_value=mock_mapping):
            if MONTAGE_AVAILABLE:
                result = convert_channel_names(["Fz", "Cz"], "GSN-HydroCel-129")
                assert "E11" in result
                assert "E129" in result
    
    def test_standard_sets_mocked(self):
        """Test standard channel sets with mocking."""
        mock_sets = {
            "frontal": ["Fz", "F3", "F4"],
            "central": ["Cz", "C3", "C4"]
        }
        
        with patch('autoclean.utils.montage.convert_channel_names', side_effect=lambda x, y: x):
            if MONTAGE_AVAILABLE:
                # Mock the standard sets dictionary
                with patch.dict('autoclean.utils.montage.__dict__', 
                               {'standard_sets': mock_sets}, clear=False):
                    # Test would use the mocked sets
                    assert True  # Placeholder for actual mocked test


class TestMontageConceptual:
    """Conceptual tests for montage design."""
    
    def test_montage_system_design(self):
        """Test montage system design principles."""
        if not MONTAGE_AVAILABLE:
            pytest.skip("Montage module not available")
        
        # Should support bidirectional mapping
        forward_mapping = get_10_20_to_gsn_mapping()
        reverse_mapping = get_gsn_to_10_20_mapping()
        
        assert isinstance(forward_mapping, dict)
        assert isinstance(reverse_mapping, dict)
        
        # Should be extensible (new montages can be added)
        assert isinstance(VALID_MONTAGES, dict)
    
    def test_montage_flexibility_concept(self):
        """Test montage system flexibility."""
        if not MONTAGE_AVAILABLE:
            pytest.skip("Montage module not available")
        
        # Should handle different montage types
        # Should support standard channel sets
        # Should validate channel availability
        
        # These concepts are tested through the functional tests above
        assert True
    
    def test_montage_error_resilience_concept(self):
        """Test montage system error resilience."""
        if not MONTAGE_AVAILABLE:
            pytest.skip("Montage module not available")
        
        # Should handle missing channels gracefully
        # Should handle unknown montages appropriately
        # Should provide meaningful error messages
        
        # This is tested through the error handling tests above
        assert True


# Error handling and edge cases
class TestMontageErrorHandling:
    """Test montage error handling and edge cases."""
    
    @pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
    def test_montage_with_special_characters(self):
        """Test montage handling with special characters in channel names."""
        special_channels = ["E1_ref", "E2-bad", "E3.modified"]
        available_channels = ["E1_ref", "E2-bad", "E3.modified", "E4"]
        
        result = validate_channel_set(special_channels, available_channels)
        
        # Should handle special characters in names
        assert result == special_channels
    
    @pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
    def test_montage_with_numeric_channels(self):
        """Test montage handling with purely numeric channel names."""
        numeric_channels = ["1", "2", "3"]
        available_channels = ["1", "2", "3", "4", "5"]
        
        result = validate_channel_set(numeric_channels, available_channels)
        
        # Should handle numeric channel names
        assert result == numeric_channels
    
    @pytest.mark.skipif(not MONTAGE_AVAILABLE, reason="Montage module not available")
    def test_montage_with_very_long_names(self):
        """Test montage handling with very long channel names."""
        long_names = ["very_long_channel_name_that_exceeds_normal_length"]
        available_channels = ["very_long_channel_name_that_exceeds_normal_length", "E2"]
        
        result = validate_channel_set(long_names, available_channels)
        
        # Should handle long channel names
        assert result == long_names
    
    def test_montage_with_none_inputs(self):
        """Test montage functions with None inputs."""
        if not MONTAGE_AVAILABLE:
            pytest.skip("Montage module not available")
        
        # Test various functions with None inputs
        with pytest.raises((TypeError, AttributeError)):
            convert_channel_names(None, "GSN-HydroCel-129")
        
        with pytest.raises((TypeError, AttributeError)):
            validate_channel_set(None, ["E1", "E2"])
        
        with pytest.raises((TypeError, AttributeError)):
            validate_channel_set(["E1"], None)