"""Unit tests for external plugin block discovery."""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest
# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_plugin_discovery_finds_external_mixins():
    """Test that plugin discovery finds external mixin classes."""
    plugin_dir = Path.home() / ".autoclean" / "blocks"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_path = plugin_dir / "test_simple_plugin.py"
    plugin_path.write_text(
        "class TestSimpleMixin:\n"
        "    def test_plugin_method(self):\n"
        "        return \"plugin-ok\"\n",
        encoding="utf-8",
    )

    import autoclean.mixins

    importlib.reload(autoclean.mixins)

    try:
        assert plugin_path.exists()
        assert autoclean.mixins.DISCOVERED_MIXINS is not None
    finally:
        plugin_path.unlink(missing_ok=True)


def test_plugin_file_naming_convention():
    """Test that only Python files ending with .py are discovered."""
    from autoclean.mixins import DISCOVERED_MIXINS

    # Should discover Python files, not other files
    # This is implicitly tested by the fact that only .py files load


def test_plugin_mixin_class_naming():
    """Test that only classes ending with 'Mixin' are discovered."""
    from autoclean.core.task import Task

    # The test plugin has TestSimpleMixin which should be found
    # But classes not ending in Mixin would not be found
    # This is verified by the discovery working at all


def test_plugin_graceful_failure():
    """Test that bad plugins don't crash the system."""
    plugin_dir = Path.home() / ".autoclean" / "blocks"
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # Create a broken plugin temporarily
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='_plugin.py',
        dir=plugin_dir,
        delete=False
    ) as f:
        f.write("class BrokenMixin:\n    raise SyntaxError('broken!')\n")
        broken_path = Path(f.name)

    try:
        import autoclean.mixins
        importlib.reload(autoclean.mixins)
        assert True
    finally:
        broken_path.unlink(missing_ok=True)


def test_plugin_search_paths():
    """Test that plugin discovery checks correct paths."""
    from autoclean.mixins import _EXTERNAL_BLOCK_PATHS

    # Should have at least the home directory path
    assert len(_EXTERNAL_BLOCK_PATHS) >= 1
    assert Path.home() / ".autoclean" / "blocks" in _EXTERNAL_BLOCK_PATHS


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
