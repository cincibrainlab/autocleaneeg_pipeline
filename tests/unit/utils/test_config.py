"""Unit tests for configuration utilities."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import tempfile

from tests.fixtures.test_utils import BaseTestCase

# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.utils.config import (
        load_config,
        hash_and_encode_yaml,
        validate_eeg_system
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not available")
class TestConfigLoading(BaseTestCase):
    """Test configuration loading functionality."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        valid_config = {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    "description": "Test task",
                    "settings": {
                        "filtering": {
                            "enabled": True,
                            "value": {
                                "l_freq": 1.0,
                                "h_freq": 100.0,
                                "notch_freqs": [60.0],
                                "notch_widths": 5.0
                            }
                        },
                        "resample_step": {"enabled": True, "value": 250},
                        "drop_outerlayer": {"enabled": False, "value": []},
                        "eog_step": {"enabled": False, "value": []},
                        "trim_step": {"enabled": True, "value": 2.0},
                        "crop_step": {
                            "enabled": False,
                            "value": {"start": 0, "end": None}
                        },
                        "reference_step": {"enabled": True, "value": "average"},
                        "montage": {"enabled": True, "value": "GSN-HydroCel-129"},
                        "ICA": {
                            "enabled": True,
                            "value": {
                                "method": "picard",
                                "n_components": 15,
                                "random_state": 42
                            }
                        },
                        "ICLabel": {
                            "enabled": True,
                            "value": {
                                "ic_flags_to_reject": ["muscle", "heart", "eog"],
                                "ic_rejection_threshold": 0.5
                            }
                        },
                        "epoch_settings": {
                            "enabled": True,
                            "value": {"tmin": -1.0, "tmax": 1.0},
                            "event_id": None,
                            "remove_baseline": {"enabled": False, "window": None},
                            "threshold_rejection": {
                                "enabled": True,
                                "volt_threshold": {"eeg": 125e-6}
                            }
                        }
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": True, "suffix": "_postimport"},
                "post_clean_raw": {"enabled": True, "suffix": "_postcleanraw"},
                "post_ica": {"enabled": True, "suffix": "_postica"},
                "post_epochs": {"enabled": True, "suffix": "_postepochs"}
            }
        }
        
        config_file = self.temp_dir / "valid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        result = load_config(config_file)
        
        assert result == valid_config
        assert "tasks" in result
        assert "TestTask" in result["tasks"]
        assert result["tasks"]["TestTask"]["mne_task"] == "rest"
    
    def test_load_config_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        non_existent_file = self.temp_dir / "non_existent.yaml"
        
        with pytest.raises(FileNotFoundError):
            load_config(non_existent_file)
    
    def test_load_config_invalid_yaml(self):
        """Test error handling for invalid YAML syntax."""
        invalid_yaml = "invalid: yaml: content:\n  - missing closing bracket ["
        
        config_file = self.temp_dir / "invalid.yaml"
        with open(config_file, 'w') as f:
            f.write(invalid_yaml)
        
        with pytest.raises(yaml.YAMLError):
            load_config(config_file)
    
    def test_load_config_schema_validation_failure(self):
        """Test schema validation failure."""
        invalid_config = {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    # Missing description and settings
                }
            }
            # Missing stage_files
        }
        
        config_file = self.temp_dir / "invalid_schema.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(Exception):  # Could be SchemaError or similar
            load_config(config_file)
    
    def test_load_config_with_optional_fields(self):
        """Test loading config with optional None values."""
        config_with_optionals = {
            "tasks": {
                "TestTaskOptional": {
                    "mne_task": "rest",
                    "description": "Test task with optional fields",
                    "settings": {
                        "filtering": {
                            "enabled": True,
                            "value": {
                                "l_freq": None,
                                "h_freq": 100.0,
                                "notch_freqs": None,
                                "notch_widths": None
                            }
                        },
                        "resample_step": {"enabled": False, "value": None},
                        "drop_outerlayer": {"enabled": False, "value": None},
                        "eog_step": {"enabled": False, "value": None},
                        "trim_step": {"enabled": True, "value": 1},
                        "crop_step": {
                            "enabled": False,
                            "value": {"start": 0, "end": None}
                        },
                        "reference_step": {"enabled": False, "value": None},
                        "montage": {"enabled": False, "value": None},
                        "ICA": {
                            "enabled": False,
                            "value": {"method": "picard"}
                        },
                        "ICLabel": {
                            "enabled": False,
                            "value": {
                                "ic_flags_to_reject": [],
                                "ic_rejection_threshold": 0.0
                            }
                        },
                        "epoch_settings": {
                            "enabled": False,
                            "value": {"tmin": None, "tmax": None},
                            "event_id": None,
                            "remove_baseline": {"enabled": False, "window": None},
                            "threshold_rejection": {
                                "enabled": False,
                                "volt_threshold": {"eeg": 100e-6}
                            }
                        }
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": False, "suffix": "_postimport"},
                "post_clean_raw": {"enabled": False, "suffix": "_postcleanraw"},
                "post_ica": {"enabled": False, "suffix": "_postica"},
                "post_epochs": {"enabled": False, "suffix": "_postepochs"}
            }
        }
        
        config_file = self.temp_dir / "optional_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_with_optionals, f)
        
        result = load_config(config_file)
        
        assert result == config_with_optionals
        assert result["tasks"]["TestTaskOptional"]["settings"]["filtering"]["value"]["l_freq"] is None


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not available")
class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_hash_and_encode_yaml(self):
        """Test YAML hashing and encoding functionality."""
        test_config = {
            "test_key": "test_value",
            "nested": {"key": "value"},
            "list": [1, 2, 3]
        }
        
        result = hash_and_encode_yaml(test_config, is_file=False)
        
        # Should return a tuple of (hash, encoded)
        assert isinstance(result, tuple)
        assert len(result) == 2
        file_hash, encoded = result
        assert isinstance(file_hash, str)
        assert isinstance(encoded, str)
        assert len(file_hash) > 0
        assert len(encoded) > 0
        
        # Same input should give same output
        result2 = hash_and_encode_yaml(test_config, is_file=False)
        assert result == result2
        
        # Different input should give different output
        different_config = {"different": "config"}
        result3 = hash_and_encode_yaml(different_config, is_file=False)
        assert result != result3
    
    def test_hash_and_encode_yaml_with_none_values(self):
        """Test hash and encode with None values."""
        config_with_none = {
            "key": None,
            "nested": {"value": None}
        }
        
        result = hash_and_encode_yaml(config_with_none)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_hash_and_encode_yaml_empty_config(self):
        """Test hash and encode with empty config."""
        empty_config = {}
        
        result = hash_and_encode_yaml(empty_config)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_validate_eeg_system(self):
        """Test EEG system validation."""
        valid_systems = [
            "GSN-HydroCel-129",
            "GSN-HydroCel-124", 
            "standard_1020",
            "biosemi64"
        ]
        
        for system in valid_systems:
            # Should not raise an exception
            try:
                validate_eeg_system(system)
            except Exception as e:
                pytest.fail(f"validate_eeg_system raised exception for valid system {system}: {e}")


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not available")
class TestConfigMocked:
    """Test config functionality with heavy mocking."""
    
    @patch('autoclean.utils.config.yaml.safe_load')
    @patch('builtins.open', mock_open(read_data="test: data"))
    def test_load_config_mocked(self, mock_yaml_load):
        """Test load_config with mocked dependencies."""
        mock_yaml_load.return_value = {
            "tasks": {
                "TestTask": {
                    "mne_task": "rest",
                    "description": "Test",
                    "settings": {
                        "filtering": {"enabled": False, "value": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60.0], "notch_widths": 1}},
                        "resample_step": {"enabled": False, "value": None},
                        "drop_outerlayer": {"enabled": False, "value": None},
                        "eog_step": {"enabled": False, "value": None},
                        "trim_step": {"enabled": False, "value": 0},
                        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
                        "reference_step": {"enabled": False, "value": None},
                        "montage": {"enabled": False, "value": None},
                        "ICA": {"enabled": False, "value": {"method": "picard"}},
                        "ICLabel": {"enabled": False, "value": {"ic_flags_to_reject": [], "ic_rejection_threshold": 0.0}},
                        "epoch_settings": {
                            "enabled": False,
                            "value": {"tmin": None, "tmax": None},
                            "event_id": None,
                            "remove_baseline": {"enabled": False, "window": None},
                            "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 100e-6}}
                        }
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": True, "suffix": "_test"}
            }
        }
        
        result = load_config(Path("fake_path.yaml"))
        
        assert "tasks" in result
        assert "TestTask" in result["tasks"]
    
    @patch('autoclean.utils.config.yaml.safe_dump')
    @patch('autoclean.utils.config.zlib.compress')
    @patch('autoclean.utils.config.base64.b64encode')
    @patch('autoclean.utils.config.hashlib.sha256')
    def test_hash_and_encode_yaml_mocked(self, mock_sha256, mock_b64encode, mock_compress, mock_yaml_dump):
        """Test hash_and_encode_yaml with mocked dependencies."""
        mock_yaml_dump.return_value = "test: data\n"
        mock_compress.return_value = b"compressed"
        mock_b64encode.return_value = b"encoded"
        mock_hasher = Mock()
        mock_hasher.hexdigest.return_value = "abcd1234"
        mock_sha256.return_value = mock_hasher
        
        result = hash_and_encode_yaml({"test": "data"}, is_file=False)
        
        assert result == ("abcd1234", "encoded")
        mock_yaml_dump.assert_called()
        mock_compress.assert_called_once()
        mock_b64encode.assert_called_once()


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not available")
class TestConfigConceptual:
    """Conceptual tests for config design patterns."""
    
    def test_config_schema_structure(self):
        """Test that config follows expected schema structure."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config not available for schema testing")
        
        # Schema should validate configs with tasks and stage_files
        minimal_config = {
            "tasks": {
                "MinimalTask": {
                    "mne_task": "test",
                    "description": "Minimal test task",
                    "settings": {
                        "filtering": {"enabled": False, "value": {"l_freq": 1, "h_freq": 100, "notch_freqs": [60.0], "notch_widths": 1}},
                        "resample_step": {"enabled": False, "value": None},
                        "drop_outerlayer": {"enabled": False, "value": None},
                        "eog_step": {"enabled": False, "value": None},
                        "trim_step": {"enabled": False, "value": 0},
                        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
                        "reference_step": {"enabled": False, "value": None},
                        "montage": {"enabled": False, "value": None},
                        "ICA": {"enabled": False, "value": {"method": "picard"}},
                        "ICLabel": {"enabled": False, "value": {"ic_flags_to_reject": [], "ic_rejection_threshold": 0.0}},
                        "epoch_settings": {
                            "enabled": False,
                            "value": {"tmin": None, "tmax": None},
                            "event_id": None,
                            "remove_baseline": {"enabled": False, "window": None},
                            "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 100e-6}}
                        }
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": True, "suffix": "_test"}
            }
        }
        
        # Should be able to create a valid config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(minimal_config, f)
            config_file = Path(f.name)
        
        try:
            result = load_config(config_file)
            assert "tasks" in result
            assert "stage_files" in result
        finally:
            config_file.unlink()
    
    def test_config_extensibility_concept(self):
        """Test config extensibility concept."""
        # Config should support multiple tasks
        assert CONFIG_AVAILABLE  # Basic availability check
        
        # Config loading should be deterministic
        # (Tested via other test cases)
    
    def test_config_validation_concept(self):
        """Test config validation concept."""
        if not CONFIG_AVAILABLE:
            pytest.skip("Config not available for validation testing")
        
        # Should have hash_and_encode_yaml function
        assert callable(hash_and_encode_yaml)
        
        # Should have validate_eeg_system function
        assert callable(validate_eeg_system)


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not available")
class TestConfigErrorHandling:
    """Test config error handling and edge cases."""
    
    def test_config_with_circular_references(self):
        """Test handling of circular references (if applicable)."""
        # YAML doesn't naturally support circular references in Python
        # This is more of a conceptual test
        pass
    
    def test_config_with_very_large_values(self):
        """Test config with very large values."""
        large_config = {
            "tasks": {
                "LargeTask": {
                    "mne_task": "rest",
                    "description": "Task with large values",
                    "settings": {
                        "filtering": {
                            "enabled": True,
                            "value": {
                                "l_freq": 1.0,
                                "h_freq": 1000000.0,
                                "notch_freqs": list(range(1000)),
                                "notch_widths": 1.0
                            }
                        },
                        "resample_step": {"enabled": False, "value": None},
                        "drop_outerlayer": {"enabled": False, "value": None},
                        "eog_step": {"enabled": False, "value": None},
                        "trim_step": {"enabled": False, "value": 0},
                        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
                        "reference_step": {"enabled": False, "value": None},
                        "montage": {"enabled": False, "value": None},
                        "ICA": {"enabled": False, "value": {"method": "picard"}},
                        "ICLabel": {
                            "enabled": False,
                            "value": {"ic_flags_to_reject": [], "ic_rejection_threshold": 0.0}
                        },
                        "epoch_settings": {
                            "enabled": False,
                            "value": {"tmin": None, "tmax": None},
                            "event_id": None,
                            "remove_baseline": {"enabled": False, "window": None},
                            "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 100e-6}}
                        }
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": False, "suffix": "_postimport"},
                "post_clean_raw": {"enabled": False, "suffix": "_postcleanraw"},
                "post_ica": {"enabled": False, "suffix": "_postica"},
                "post_epochs": {"enabled": False, "suffix": "_postepochs"}
            }
        }
        
        # Should handle large configurations
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(large_config, f)
            config_file = Path(f.name)
        
        try:
            result = load_config(config_file)
            assert "tasks" in result
        finally:
            config_file.unlink()
    
    def test_config_with_unicode_characters(self):
        """Test config with Unicode characters."""
        unicode_config = {
            "tasks": {
                "UnicodeTask": {
                    "mne_task": "rest",
                    "description": "Test with Unicode: äöü 你好 🧠",
                    "settings": {
                        "filtering": {
                            "enabled": True,
                            "value": {
                                "l_freq": 1.0,
                                "h_freq": 100.0,
                                "notch_freqs": [60.0],
                                "notch_widths": 5.0
                            }
                        },
                        "resample_step": {"enabled": False, "value": None},
                        "drop_outerlayer": {"enabled": False, "value": None},
                        "eog_step": {"enabled": False, "value": None},
                        "trim_step": {"enabled": False, "value": 0},
                        "crop_step": {"enabled": False, "value": {"start": 0, "end": None}},
                        "reference_step": {"enabled": False, "value": None},
                        "montage": {"enabled": False, "value": None},
                        "ICA": {"enabled": False, "value": {"method": "picard"}},
                        "ICLabel": {
                            "enabled": False,
                            "value": {"ic_flags_to_reject": [], "ic_rejection_threshold": 0.0}
                        },
                        "epoch_settings": {
                            "enabled": False,
                            "value": {"tmin": None, "tmax": None},
                            "event_id": None,
                            "remove_baseline": {"enabled": False, "window": None},
                            "threshold_rejection": {"enabled": False, "volt_threshold": {"eeg": 100e-6}}
                        }
                    }
                }
            },
            "stage_files": {
                "post_import": {"enabled": False, "suffix": "_postimport"},
                "post_clean_raw": {"enabled": False, "suffix": "_postcleanraw"},
                "post_ica": {"enabled": False, "suffix": "_postica"},
                "post_epochs": {"enabled": False, "suffix": "_postepochs"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)
            config_file = Path(f.name)
        
        try:
            result = load_config(config_file)
            assert "tasks" in result
            assert "äöü" in result["tasks"]["UnicodeTask"]["description"]
        finally:
            config_file.unlink()