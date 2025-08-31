"""Tests for user-defined variables in task configuration."""

import unittest
from pathlib import Path
from unittest.mock import Mock

from autoclean.core.task import Task


class TestUserDefinedConfigVariables(unittest.TestCase):
    """Test user-defined variables in task config system."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            "run_id": "test_run",
            "unprocessed_file": Path("/tmp/test.raw"),
            "task": "test_task",
        }

    def test_config_variables_auto_detection(self):
        """Test that module-level config variables are auto-detected."""
        
        # Create a mock module with config
        mock_module = Mock()
        mock_module.config = {
            "stimulus_duration": 500,
            "trial_count": 120,
            "custom_threshold": 75.0,
            "experiment_conditions": ["A", "B", "control"]
        }
        
        # Create a task class that simulates having a module with config
        class TestTaskWithConfig(Task):
            def run(self):
                pass
        
        # Mock the module inspection to return our mock module
        import autoclean.core.task
        original_getmodule = autoclean.core.task.inspect.getmodule
        
        def mock_getmodule(cls):
            if cls == TestTaskWithConfig:
                return mock_module
            return original_getmodule(cls)
        
        autoclean.core.task.inspect.getmodule = mock_getmodule
        
        try:
            # Initialize task
            task = TestTaskWithConfig(self.base_config)
            
            # Verify settings were loaded from module config
            self.assertIsNotNone(task.settings)
            self.assertEqual(task.settings["stimulus_duration"], 500)
            self.assertEqual(task.settings["trial_count"], 120)
            self.assertEqual(task.settings["custom_threshold"], 75.0)
            self.assertEqual(task.settings["experiment_conditions"], ["A", "B", "control"])
            
        finally:
            # Restore original function
            autoclean.core.task.inspect.getmodule = original_getmodule

    def test_config_variables_access_with_defaults(self):
        """Test accessing config variables with default values."""
        
        # Create a mock module with some config variables
        mock_module = Mock()
        mock_module.config = {
            "stimulus_duration": 500,
            "trial_count": 120
            # Note: missing "custom_threshold" to test defaults
        }
        
        class TestTaskWithPartialConfig(Task):
            def run(self):
                pass
                
            def get_variables_with_defaults(self):
                """Helper method to test variable access."""
                duration = self.settings.get("stimulus_duration", 250)
                count = self.settings.get("trial_count", 100)
                threshold = self.settings.get("custom_threshold", 50.0)  # Not in config, should use default
                missing = self.settings.get("nonexistent_var", "default_value")
                
                return {
                    "duration": duration,
                    "count": count, 
                    "threshold": threshold,
                    "missing": missing
                }
        
        # Mock module inspection
        import autoclean.core.task
        original_getmodule = autoclean.core.task.inspect.getmodule
        
        def mock_getmodule(cls):
            if cls == TestTaskWithPartialConfig:
                return mock_module
            return original_getmodule(cls)
        
        autoclean.core.task.inspect.getmodule = mock_getmodule
        
        try:
            # Initialize task and test variable access
            task = TestTaskWithPartialConfig(self.base_config)
            variables = task.get_variables_with_defaults()
            
            # Verify config variables are used when available
            self.assertEqual(variables["duration"], 500)  # From config
            self.assertEqual(variables["count"], 120)     # From config
            
            # Verify defaults are used when variables are missing
            self.assertEqual(variables["threshold"], 50.0)  # Default used
            self.assertEqual(variables["missing"], "default_value")  # Default used
            
        finally:
            # Restore original function
            autoclean.core.task.inspect.getmodule = original_getmodule

    def test_nested_config_variables(self):
        """Test accessing nested configuration variables."""
        
        mock_module = Mock()
        mock_module.config = {
            "analysis_parameters": {
                "method": "custom_algorithm",
                "window_size": 100,
                "overlap": 50,
                "advanced_settings": {
                    "tolerance": 0.001,
                    "iterations": 1000
                }
            },
            "experiment_conditions": ["condition_A", "condition_B", "control"]
        }
        
        class TestTaskWithNestedConfig(Task):
            def run(self):
                pass
                
            def access_nested_variables(self):
                """Test accessing nested config variables."""
                analysis = self.settings.get("analysis_parameters", {})
                method = analysis.get("method", "standard")
                window_size = analysis.get("window_size", 50)
                
                advanced = analysis.get("advanced_settings", {})
                tolerance = advanced.get("tolerance", 0.01)
                iterations = advanced.get("iterations", 100)
                
                conditions = self.settings.get("experiment_conditions", [])
                
                return {
                    "method": method,
                    "window_size": window_size,
                    "tolerance": tolerance,
                    "iterations": iterations,
                    "conditions": conditions
                }
        
        # Mock module inspection
        import autoclean.core.task
        original_getmodule = autoclean.core.task.inspect.getmodule
        
        def mock_getmodule(cls):
            if cls == TestTaskWithNestedConfig:
                return mock_module
            return original_getmodule(cls)
        
        autoclean.core.task.inspect.getmodule = mock_getmodule
        
        try:
            task = TestTaskWithNestedConfig(self.base_config)
            variables = task.access_nested_variables()
            
            # Verify nested variables are accessible
            self.assertEqual(variables["method"], "custom_algorithm")
            self.assertEqual(variables["window_size"], 100)
            self.assertEqual(variables["tolerance"], 0.001)
            self.assertEqual(variables["iterations"], 1000)
            self.assertEqual(variables["conditions"], ["condition_A", "condition_B", "control"])
            
        finally:
            autoclean.core.task.inspect.getmodule = original_getmodule

    def test_no_config_fallback(self):
        """Test behavior when no module config is present."""
        
        # Create a task class without module config
        class TestTaskWithoutConfig(Task):
            def run(self):
                pass
        
        # Mock module inspection to return module without config
        mock_module = Mock()
        # Note: no config attribute on mock_module
        
        import autoclean.core.task
        original_getmodule = autoclean.core.task.inspect.getmodule
        
        def mock_getmodule(cls):
            if cls == TestTaskWithoutConfig:
                return mock_module
            return original_getmodule(cls)
        
        autoclean.core.task.inspect.getmodule = mock_getmodule
        
        try:
            task = TestTaskWithoutConfig(self.base_config)
            
            # Verify settings is None when no config is found
            self.assertIsNone(task.settings)
            
        finally:
            autoclean.core.task.inspect.getmodule = original_getmodule

    def test_config_variable_types(self):
        """Test that various data types work in config variables."""
        
        mock_module = Mock()
        mock_module.config = {
            # Various Python data types
            "integer_var": 42,
            "float_var": 3.14159,
            "string_var": "test_string",
            "boolean_var": True,
            "list_var": [1, 2, 3, "four"],
            "dict_var": {"key1": "value1", "key2": 2},
            "tuple_var": (1, 2, 3),
            "none_var": None
        }
        
        class TestTaskWithVariousTypes(Task):
            def run(self):
                pass
                
            def get_all_variables(self):
                """Get all config variables to test types."""
                return {
                    "integer": self.settings.get("integer_var"),
                    "float": self.settings.get("float_var"), 
                    "string": self.settings.get("string_var"),
                    "boolean": self.settings.get("boolean_var"),
                    "list": self.settings.get("list_var"),
                    "dict": self.settings.get("dict_var"),
                    "tuple": self.settings.get("tuple_var"),
                    "none": self.settings.get("none_var")
                }
        
        # Mock module inspection
        import autoclean.core.task
        original_getmodule = autoclean.core.task.inspect.getmodule
        
        def mock_getmodule(cls):
            if cls == TestTaskWithVariousTypes:
                return mock_module
            return original_getmodule(cls)
        
        autoclean.core.task.inspect.getmodule = mock_getmodule
        
        try:
            task = TestTaskWithVariousTypes(self.base_config)
            variables = task.get_all_variables()
            
            # Verify all data types are preserved
            self.assertEqual(variables["integer"], 42)
            self.assertEqual(variables["float"], 3.14159)
            self.assertEqual(variables["string"], "test_string")
            self.assertEqual(variables["boolean"], True)
            self.assertEqual(variables["list"], [1, 2, 3, "four"])
            self.assertEqual(variables["dict"], {"key1": "value1", "key2": 2})
            self.assertEqual(variables["tuple"], (1, 2, 3))
            self.assertIsNone(variables["none"])
            
        finally:
            autoclean.core.task.inspect.getmodule = original_getmodule


if __name__ == "__main__":
    unittest.main()