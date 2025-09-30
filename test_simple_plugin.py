#!/usr/bin/env python3
"""Simple plugin to test discovery system."""

__block_metadata__ = {
    "name": "test_simple",
    "version": "1.0.0",
    "category": "test",
}

class TestSimpleMixin:
    """Simple test mixin that doesn't require any imports."""

    def test_plugin_method(self):
        """Simple test method."""
        return "Plugin loaded successfully!"