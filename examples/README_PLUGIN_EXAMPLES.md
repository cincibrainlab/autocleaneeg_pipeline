# Plugin Examples

This directory contains example plugins demonstrating the **correct** use of the external plugin system introduced in v2.4.0.

## ⚠️ Important: Plugins vs. Mixins

**Plugins are for USER EXTENSIONS, not for duplicating core functionality.**

### When to Use Plugins ✅

1. **Custom Research Methods**
   - Your lab developed a novel artifact detection algorithm
   - You want to add experimental preprocessing steps
   - You need organization-specific workflows

2. **Third-Party Integrations**
   - Integrating with external ML models
   - Adding support for proprietary analysis methods
   - Connecting to custom databases or APIs

3. **Prototyping New Features**
   - Testing experimental algorithms before proposing for pipeline core
   - Rapid iteration without modifying pipeline source code
   - Sharing beta features with collaborators

### When NOT to Use Plugins ❌

1. **Duplicating Existing Mixins**
   - All standard processing methods are already in the pipeline
   - wavelet_threshold, ICA, autoreject, source localization, etc. are built-in
   - Use the existing mixins instead of creating plugins

2. **Core Functionality Replacement**
   - Don't override fundamental pipeline behavior
   - Don't create alternative versions of standard methods

## Example Plugins

### 1. `custom_artifact_detector_plugin.py`

**Purpose:** Demonstrates adding a new ML-based artifact detection method

**What It Shows:**
- Proper plugin structure with metadata
- Importing from pipeline to avoid duplication
- Configuration integration
- Following mixin patterns

**Usage:**
```bash
# Copy to your plugins directory
cp examples/custom_artifact_detector_plugin.py ~/.autoclean/blocks/

# Use in any task
class MyTask(Task):
    def run(self):
        self.import_raw()
        self.apply_ml_artifact_detection()  # Available automatically!
```

**Configuration:**
```python
config = {
    "ml_artifact_detection": {
        "enabled": True,
        "value": {
            "model_path": "/path/to/model.h5",
            "confidence_threshold": 0.8
        }
    }
}
```

## Plugin Development Guide

### Basic Structure

```python
#!/usr/bin/env python3
"""Plugin description."""

from autoclean.utils.logging import message  # Import from pipeline

__block_metadata__ = {
    "name": "my_plugin",
    "version": "1.0.0",
    "description": "What this plugin does",
    "author": "Your Name",
    "license": "MIT",
}

class MyCustomMixin:
    """Mixin providing my custom functionality."""

    def my_custom_method(self):
        """Apply my custom processing."""
        # Your implementation here
        pass
```

### Integration with Configuration

Plugins should check if they're enabled in the task configuration:

```python
def my_custom_method(self):
    # Check if enabled
    is_enabled, settings = self._check_step_enabled("my_custom_step")
    if not is_enabled:
        message("info", "Custom step disabled in configuration")
        return

    # Extract parameters
    params = (settings or {}).get("value", {})
    my_param = params.get("my_param", default_value)

    # Do processing
    ...
```

### Best Practices

1. **Zero Duplication**
   - Import algorithms from `autoclean.calc.*`
   - Import helpers from `autoclean.utils.*`
   - Don't copy-paste existing code

2. **Clear Documentation**
   - Docstrings for all public methods
   - Explain parameters and return values
   - Provide usage examples

3. **Graceful Errors**
   - Use try/except for external dependencies
   - Provide helpful error messages
   - Don't crash the pipeline

4. **Configuration Schema**
   - Provide a descriptor function like `my_plugin_descriptor()`
   - Follow existing configuration patterns
   - Document all parameters

5. **Testing**
   - Test with real EEG data
   - Verify integration with tasks
   - Check error handling

## Installation

### For Personal Use

```bash
# Create plugins directory
mkdir -p ~/.autoclean/blocks/

# Copy plugin file
cp my_plugin.py ~/.autoclean/blocks/

# Reinstall pipeline to pick up plugins
uv tool install autocleaneeg-pipeline --upgrade --force
```

### For Team Distribution

```bash
# Create project plugins directory
mkdir -p /path/to/project/blocks/

# Copy plugin
cp my_plugin.py /path/to/project/blocks/

# Team members run from project directory
cd /path/to/project
autocleaneeg-pipeline process --task MyTask --file data.raw
```

### Via Environment Variable

```bash
# Set task-registry path
export AUTOCLEAN_TASK_REGISTRY_PATH=/path/to/task-registry

# Plugins in task-registry/blocks/ will be auto-discovered
autocleaneeg-pipeline process --task MyTask --file data.raw
```

## Discovery

The pipeline automatically discovers plugins from these locations (in order):

1. `~/.autoclean/blocks/` - User plugins
2. `./blocks/` - Project plugins
3. `$AUTOCLEAN_TASK_REGISTRY_PATH/blocks/` - Registry plugins

All `.py` files containing `*Mixin` classes will be loaded.

## Verification

When plugins load successfully, you'll see:

```
✓ Loaded external block: MyCustomMixin from my_plugin.py
```

If a plugin fails to load, you'll see a warning but the pipeline continues:

```
Warning: Could not load external block from my_plugin.py: <error details>
```

## Contributing

To share your plugin with the community:

1. Ensure it follows best practices above
2. Add comprehensive documentation
3. Include example usage
4. Test with multiple datasets
5. Submit to the plugin registry (coming in v3.0.0)

## Resources

- **Plugin Architecture:** See `PLUGIN_BLOCKS_PLAN.md` for full specification
- **Discovery Implementation:** See `PLUGIN_DISCOVERY_IMPLEMENTATION.md` for technical details
- **Duplication Analysis:** See `BLOCK_DUPLICATION_ANALYSIS.md` for why not to duplicate

## Questions?

- GitHub Issues: https://github.com/cincibrainlab/autocleaneeg_pipeline/issues
- Documentation: https://cincibrainlab.github.io/autoclean_pipeline/