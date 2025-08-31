"""Tests for JSON header variable parsing in task files."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from autoclean.core.task import Task
from autoclean.utils.task_discovery import parse_json_header


class TestJSONHeaderParsing:
    """Test JSON header parsing functionality."""

    def test_parse_json_header_triple_quotes(self):
        """Test parsing JSON from triple-quoted string at file start."""
        content = '''"""
{
  "param1": "value1",
  "param2": 42,
  "param3": [1, 2, 3]
}
"""

from autoclean.core.task import Task

class TestTask(Task):
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is not None
        assert result["param1"] == "value1"
        assert result["param2"] == 42
        assert result["param3"] == [1, 2, 3]

    def test_parse_json_header_block_comment(self):
        """Test parsing JSON from block comment at file start.""" 
        content = '''/*
{
  "setting": "custom_value",
  "number": 123
}
*/

from autoclean.core.task import Task

class TestTask(Task):
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is not None
        assert result["setting"] == "custom_value"
        assert result["number"] == 123

    def test_parse_json_header_with_shebang(self):
        """Test parsing JSON header after shebang and encoding."""
        content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{
  "experiment": "test_exp",
  "trials": 100
}
"""

from autoclean.core.task import Task
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is not None
        assert result["experiment"] == "test_exp"
        assert result["trials"] == 100

    def test_parse_json_header_invalid_json(self):
        """Test handling of invalid JSON in header."""
        content = '''"""
{
  "param1": "value1",
  "param2": 42,
  // This is invalid JSON due to the comment
}
"""

from autoclean.core.task import Task
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        # Should return None for invalid JSON
        assert result is None

    def test_parse_json_header_no_header(self):
        """Test file with no JSON header."""
        content = '''from autoclean.core.task import Task

class TestTask(Task):
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is None

    def test_parse_json_header_docstring_ignored(self):
        """Test that regular docstrings are ignored."""
        content = '''"""
This is a regular docstring that starts with text.
It should not be parsed as JSON.
"""

from autoclean.core.task import Task

class TestTask(Task):
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is None

    def test_parse_json_header_nonexistent_file(self):
        """Test handling of nonexistent file."""
        result = parse_json_header(Path("/nonexistent/file.py"))
        assert result is None


class TestTaskContextIntegration:
    """Test integration of JSON header variables with task_context."""

    def test_task_context_initialization(self):
        """Test that task_context is initialized in Task.__init__."""
        
        # Create a temporary task file with JSON header
        task_content = '''"""
{
  "test_param": "test_value",
  "numeric_param": 42
}
"""

from typing import Any, Dict
from autoclean.core.task import Task

class TestTaskWithHeader(Task):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def run(self):
        pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(task_content)
            f.flush()
            task_file = Path(f.name)
            
        # Import the task dynamically
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("test_module", task_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module"] = module
        spec.loader.exec_module(module)
        
        # Create task instance
        test_config = {
            "run_id": "test_run",
            "unprocessed_file": Path("/tmp/test.raw"),
            "task": "TestTaskWithHeader"
        }
        
        try:
            task = module.TestTaskWithHeader(test_config)
            
            # Check that task_context is properly initialized
            assert hasattr(task, 'task_context')
            assert isinstance(task.task_context, dict)
            assert task.task_context.get("test_param") == "test_value" 
            assert task.task_context.get("numeric_param") == 42
            
        finally:
            # Clean up
            if "test_module" in sys.modules:
                del sys.modules["test_module"]
            task_file.unlink()

    def test_task_context_empty_when_no_header(self):
        """Test task_context is empty dict when no JSON header present."""
        
        task_content = '''
from typing import Any, Dict
from autoclean.core.task import Task

class TestTaskNoHeader(Task):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def run(self):
        pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(task_content)
            f.flush()
            task_file = Path(f.name)
            
        # Import the task dynamically
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("test_module2", task_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module2"] = module
        spec.loader.exec_module(module)
        
        # Create task instance
        test_config = {
            "run_id": "test_run",
            "unprocessed_file": Path("/tmp/test.raw"), 
            "task": "TestTaskNoHeader"
        }
        
        try:
            task = module.TestTaskNoHeader(test_config)
            
            # Check that task_context exists but is empty
            assert hasattr(task, 'task_context')
            assert isinstance(task.task_context, dict)
            assert len(task.task_context) == 0
            
        finally:
            # Clean up
            if "test_module2" in sys.modules:
                del sys.modules["test_module2"]
            task_file.unlink()

    def test_task_context_access_pattern(self):
        """Test recommended access pattern with defaults."""
        
        task_content = '''"""
{
  "stimulus_duration": 500,
  "trial_count": 120
}
"""

from typing import Any, Dict
from autoclean.core.task import Task

class TestTaskAccess(Task):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def run(self):
        pass
    
    def get_duration_with_default(self):
        return self.task_context.get("stimulus_duration", 250)
    
    def get_nonexistent_with_default(self):
        return self.task_context.get("nonexistent_param", "default_value")
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(task_content)
            f.flush()
            task_file = Path(f.name)
            
        # Import the task dynamically
        import importlib.util
        import sys
        
        spec = importlib.util.spec_from_file_location("test_module3", task_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["test_module3"] = module
        spec.loader.exec_module(module)
        
        # Create task instance
        test_config = {
            "run_id": "test_run",
            "unprocessed_file": Path("/tmp/test.raw"),
            "task": "TestTaskAccess"
        }
        
        try:
            task = module.TestTaskAccess(test_config)
            
            # Test accessing existing values
            assert task.get_duration_with_default() == 500
            
            # Test accessing non-existent values with defaults
            assert task.get_nonexistent_with_default() == "default_value"
            
        finally:
            # Clean up
            if "test_module3" in sys.modules:
                del sys.modules["test_module3"]
            task_file.unlink()


class TestComplexJSONStructures:
    """Test parsing of complex JSON structures."""

    def test_nested_json_structures(self):
        """Test parsing nested JSON objects and arrays."""
        content = '''"""
{
  "experiment": {
    "name": "P300_Study",
    "version": "2.1"
  },
  "parameters": {
    "timing": {
      "baseline": [-200, 0],
      "stimulus_duration": 500,
      "isi_range": [800, 1200]
    },
    "filtering": {
      "highpass": 0.1,
      "lowpass": 30.0,
      "notch": [50, 60]
    }
  },
  "conditions": ["target", "standard", "novel"]
}
"""

from autoclean.core.task import Task
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is not None
        assert result["experiment"]["name"] == "P300_Study"
        assert result["experiment"]["version"] == "2.1"
        assert result["parameters"]["timing"]["baseline"] == [-200, 0]
        assert result["parameters"]["filtering"]["notch"] == [50, 60]
        assert result["conditions"] == ["target", "standard", "novel"]

    def test_json_with_various_data_types(self):
        """Test JSON with different data types."""
        content = '''"""
{
  "string_param": "hello",
  "integer_param": 42,
  "float_param": 3.14159,
  "boolean_true": true,
  "boolean_false": false,
  "null_param": null,
  "array_mixed": [1, "two", 3.0, true],
  "empty_object": {},
  "empty_array": []
}
"""

from autoclean.core.task import Task
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f.flush()
            
            result = parse_json_header(Path(f.name))
            
        # Clean up
        Path(f.name).unlink()
        
        assert result is not None
        assert result["string_param"] == "hello"
        assert result["integer_param"] == 42
        assert result["float_param"] == 3.14159
        assert result["boolean_true"] is True
        assert result["boolean_false"] is False
        assert result["null_param"] is None
        assert result["array_mixed"] == [1, "two", 3.0, True]
        assert result["empty_object"] == {}
        assert result["empty_array"] == []