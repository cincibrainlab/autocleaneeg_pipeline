# Migration Guide

This guide helps you migrate between versions of AutoClean EEG Pipeline. For breaking changes, follow the migration steps for your target version.

## Table of Contents

- [Migrating from 1.x to 2.0.0](#migrating-from-1x-to-200)
- [Migrating from 2.0.0 to 3.0.0](#migrating-from-200-to-300)

---

## Migrating from 1.x to 2.0.0

Version 2.0.0 introduced significant architectural changes. Follow these steps to migrate your workflows.

### Prerequisites

- Python 3.10+
- Backup existing configurations and workflows

### 1. Update Installation

```bash
# Using uv (recommended)
uv tool upgrade autocleaneeg-pipeline

# Or using pip
pip install --upgrade autocleaneeg-pipeline
```

### 2. Pipeline API Changes

**Before (v1.x):**
```python
from autoclean import Pipeline

pipeline = Pipeline(
    autoclean_dir="/path/to/output",
    autoclean_config="/path/to/config.yaml"
)
```

**After (v2.0.0+):**
```python
from autoclean import Pipeline

pipeline = Pipeline(
    output_dir="/path/to/output"
    # autoclean_config parameter removed - YAML configs no longer required
)
```

### 3. Task System Migration

**Before (v1.x):**
Tasks were configured via YAML files with full pipeline definitions.

**After (v2.0.0+):**
Tasks are now Python files with embedded configuration. Built-in tasks no longer require YAML configuration.

**Migration Steps:**
1. Review existing YAML task configurations
2. Convert to Python task files if creating custom tasks
3. Built-in tasks (ASSR, Chirp, MMN, Resting State) work without configuration

**Example Task File Structure:**
```python
# custom_task.py
from autoclean.tasks import BaseTask

class MyCustomTask(BaseTask):
    def run(self):
        # Task implementation
        pass
```

### 4. Task Validation Changes

**Before (v1.x):**
Multiple required parameters for task validation.

**After (v2.0.0+):**
Only three fields are required:
- `run_id`
- `unprocessed_file`
- `task`

### 5. Workspace Management

Version 2.0.0 introduced a new workspace setup wizard for first-time users. Existing workspaces remain compatible, but new installations will prompt for workspace setup.

**Manual Workspace Setup:**
If you need to recreate your workspace, the setup wizard will guide you through the process.

### 6. Export Counter System

**Before (v1.x):**
Complex stage file management for tracking exports.

**After (v2.0.0+):**
Simplified export counter system replaces stage file complexity.

**Action Required:**
- Existing exports remain valid
- New processing automatically uses the export counter system
- No manual migration needed

---

## Migrating from 2.0.0 to 3.0.0

Version 3.0.0 maintains API compatibility with 2.0.0. Most changes are internal improvements.

### Prerequisites

- Python 3.11+ (version 3.0.0 requires Python 3.11-3.13)
- Existing 2.0.0 installation

### 1. Python Version Requirement

**Before (v2.0.0):**
Python 3.10+ supported

**After (v3.0.0):**
Python 3.11-3.13 required

**Migration Steps:**
1. Upgrade Python to 3.11 or later
2. Reinstall dependencies:
   ```bash
   uv tool upgrade autocleaneeg-pipeline
   ```

### 2. Dependency Updates

Version 3.0.0 includes updated dependency versions. Reinstall to ensure compatibility:

```bash
uv tool upgrade autocleaneeg-pipeline --force
```

### 3. No Breaking API Changes

The Pipeline API remains unchanged from version 2.0.0. Existing code should work without modification.

---

## General Migration Tips

### Backing Up Your Data

Before migrating, always backup:
- Pipeline output directories
- Custom task files
- Configuration files
- Database files (if using database-backed tracking)

### Testing Your Migration

1. Test with a single run first
2. Verify output structure matches expectations
3. Check that custom tasks still function correctly
4. Review logs for any warnings or errors

### Getting Help

If you encounter issues during migration:
- Check the [GitHub Issues](https://github.com/cincibrainlab/autoclean_pipeline/issues)
- Review the [documentation](https://docs.autocleaneeg.org)
- Consult the [changelog](docs/development/changelog.rst) for detailed changes

---

## Version Compatibility Matrix

| From Version | To Version | Breaking Changes | Migration Required |
|-------------|------------|------------------|-------------------|
| 1.x         | 2.0.0      | Yes              | Yes (see above)   |
| 2.0.0       | 3.0.0      | No               | Python upgrade    |
| 3.0.0+      | Future     | Check changelog  | See release notes |

---

## Related Documentation

- [Changelog](docs/development/changelog.rst) - Detailed version history
- [Contributing Guide](CONTRIBUTING.md) - For developers
- [MNE pick_types() Migration](docs/MNE_PICK_TYPES_MIGRATION.md) - Technical migration guide for developers
