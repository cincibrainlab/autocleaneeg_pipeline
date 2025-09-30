"""Test task to verify plugin loading."""

from autoclean.core.task import Task

config = {
    "schema_version": "2025.09",
}

class TestPluginTask(Task):
    """Test task that uses plugin methods."""

    def run(self):
        """Run test - check if plugin method exists."""
        print("=" * 80)
        print("Testing Plugin Discovery System")
        print("=" * 80)

        # Check if plugin method exists
        if hasattr(self, 'test_plugin_method'):
            print("✓ Plugin method 'test_plugin_method' found!")
            result = self.test_plugin_method()
            print(f"✓ Result: {result}")
        else:
            print("✗ Plugin method 'test_plugin_method' NOT found")
            print("Available methods:")
            methods = [m for m in dir(self) if not m.startswith('_') and callable(getattr(self, m))]
            for m in sorted(methods)[:10]:
                print(f"  - {m}")

        print("=" * 80)